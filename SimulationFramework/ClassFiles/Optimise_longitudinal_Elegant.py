import os, errno, sys, shutil
import numpy as np
import random
# sys.path.append(os.path.abspath(__file__+'/../../../../../'))
from ..Modules.constraints import constraintsClass
from ..Modules.nelder_mead import nelder_mead
import time
import csv
from copy import copy
from functools import partial
from collections import OrderedDict
from shutil import copyfile
from ..Modules.merge_two_dicts import merge_two_dicts
from . import runElegant as runEle
from scipy.optimize import minimize

def saveState(dir, args, n, fitness):
    with open(dir+'/best_solutions_running.csv','a') as out:
        csv_out=csv.writer(out)
        args=list(args)
        args.append(n)
        args.append(fitness)
        csv_out.writerow(args)

def saveParameterFile(best, file='clara_elegant_best.yaml'):
    if POST_INJECTOR:
        allparams = list(zip(*(parameternames)))
    else:
        allparams = list(zip(*(injparameternames+parameternames)))
    output = {}
    for p, k, v in zip(allparams[0], allparams[1], best):
        if p not in output:
            output[p] = {}
        output[p][k] = v
        with open(file,"w") as yaml_file:
            yaml.dump(output, yaml_file, default_flow_style=False)

class Optimise_Elegant(runEle.fitnessFunc):

    injector_parameter_names = [
        ['CLA-HRG1-GUN-CAV', 'phase'],
        ['CLA-HRG1-GUN-SOL', 'field_amplitude'],
        ['CLA-L01-CAV', 'field_amplitude'],
        ['CLA-L01-CAV', 'phase'],
        ['CLA-L01-CAV-SOL-01', 'field_amplitude'],
        ['CLA-L01-CAV-SOL-02', 'field_amplitude'],
    ]
    parameter_names = [
        ['CLA-L02-CAV', 'field_amplitude'],
        ['CLA-L02-CAV', 'phase'],
        ['CLA-L03-CAV', 'field_amplitude'],
        ['CLA-L03-CAV', 'phase'],
        ['CLA-L4H-CAV', 'field_amplitude'],
        ['CLA-L4H-CAV', 'phase'],
        ['CLA-L04-CAV', 'field_amplitude'],
        ['CLA-L04-CAV', 'phase'],
        ['bunch_compressor', 'angle'],
        ['CLA-S07-DCP-01', 'factor'],
    ]

    def __init__(self):
        super(Optimise_Elegant, self).__init__()
        self.cons = constraintsClass()
        self.changes = None
        self.lattice = None
        self.resultsDict = {}
        self.deleteFolders = True
        # ******************************************************************************
        self.CLARA_dir = os.path.relpath(__file__+'/../../CLARA')
        self.POST_INJECTOR = True
        CREATE_BASE_FILES = False
        self.scaling = 5
        self.verbose = False
        self.basefiles = '../../CLARA/basefiles_'+str(self.scaling)+'/'
        # ******************************************************************************
        if not self.POST_INJECTOR:
            best = injector_startingvalues + best
        elif CREATE_BASE_FILES:
            for i in [self.scaling]:
                pass
                # optfunc(injector_startingvalues + best, scaling=scaling, post_injector=False, verbose=False, runGenesis=False, dir='nelder_mead/basefiles_'+str(i))

    def calculate_constraints(self):
        pass

    def set_changes_file(self, changes):
        self.changes = changes

    def set_lattice_file(self, lattice):
        self.lattice_file = lattice

    def set_start_file(self, file):
        self.start_lattice = file

    def OptimisingFunction(self, inputargs, *args, **kwargs):
        if not self.POST_INJECTOR:
            parameternames = self.injector_parameter_names + self.parameter_names
        else:
            parameternames = copy(self.parameter_names)
        self.inputlist = [a[0]+[a[1]] for a in zip(parameternames, inputargs)]

        self.linac_fields = np.array([i[2] for i in self.inputlist if i[1] == 'field_amplitude'])
        self.linac_phases = np.array([i[2] for i in self.inputlist if i[1] == 'phase'])

        if 'iteration' in list(kwargs.keys()):
            self.opt_iteration = kwargs['iteration']
            del kwargs['iteration']

        if 'bestfit' in list(kwargs.keys()):
            self.bestfit = kwargs['bestfit']
            del kwargs['bestfit']

        if 'dir' in list(kwargs.keys()):
            dir = kwargs['dir']
            del kwargs['dir']
            save_state = False
        else:
            save_state = True
            dir = self.optdir+str(self.opt_iteration)

        self.setup_lattice(self.inputlist, dir)
        self.before_tracking()
        fitvalue = self.track()

        if isinstance(self.opt_iteration, int):
            self.opt_iteration += 1

        constraintsList = self.calculate_constraints()
        fitvalue = self.cons.constraints(constraintsList)

        if isinstance(self.opt_iteration, int):
            print('fitvalue[', self.opt_iteration-1, '] = ', fitvalue)
        else:
            print('fitvalue = ', fitvalue)

        if save_state:
            if isinstance(self.opt_iteration, int):
                saveState(self.subdir, inputargs, self.opt_iteration-1, fitvalue)
            else:
                saveState(self.subdir, inputargs, self.opt_iteration, fitvalue)
        if save_state and fitvalue < self.bestfit:
            print(self.cons.constraintsList(constraintsList))
            print('!!!!!!  New best = ', fitvalue, inputargs)
            copyfile(dir+'/changes.yaml', self.best_changes)
            self.bestfit = fitvalue
            self.framework.save_lattice()
        else:
            if self.deleteFolders:
                shutil.rmtree(dir, ignore_errors=True)
        return fitvalue

    def Nelder_Mead(self, best=None, step=0.1, best_changes='./nelder_mead_best_changes.yaml', subdir='nelder_mead'):
        best = np.array(self.best) if best is None else np.array(best)
        self.subdir = subdir
        self.optdir = self.subdir + '/iteration_'
        self.best_changes = best_changes
        print('best = ', best)
        self.bestfit = 1e26

        with open(subdir+'/best_solutions_running.csv','w') as out:
            self.opt_iteration = 0
        res = nelder_mead(self.OptimisingFunction, best, step=step, max_iter=300, no_improv_break=100)
        print(res)

    def Simplex(self, best=None, best_changes='./simplex_best_changes.yaml', subdir='simplex', maxiter=300):
        best = self.best if best is None else best
        self.subdir = subdir
        self.optdir = self.subdir + '/iteration_'
        self.best_changes = best_changes
        print('best = ', best)
        self.bestfit = 1e26

        with open(subdir+'/best_solutions_running.csv','w') as out:
            self.opt_iteration = 0
        res = minimize(self.OptimisingFunction, best, method='nelder-mead', options={'disp': True, 'maxiter': maxiter, 'adaptive': True})
        print(res.x)

    def Example(self, best=None, step=0.1, dir='example'):
        best = np.array(self.best) if best is None else np.array(best)
        self.subdir = dir
        self.optdir = self.subdir
        self.best_changes = './example_best_changes.yaml'
        # print('best = ', best)
        self.bestfit = -1e26

        self.opt_iteration = ''
        # try:
        self.OptimisingFunction(best)
        # except:
            # pass
