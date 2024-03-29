import os
import numpy as np
from .. import constants

def cumtrapz(x=[], y=[]):
    return [np.trapz(x=x[:n], y=y[:n]) for n in range(len(x))]

def read_astra_twiss_files(self, filename, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            self.read_astra_twiss_files(f, reset=False)
    elif os.path.isfile(filename):
        if 'xemit' not in filename.lower():
            filename = filename.replace('Yemit','Xemit').replace('Zemit','Xemit')
        xemit = np.loadtxt(filename, unpack=False) if os.path.isfile(filename) else False
        if 'yemit' not in filename.lower():
            filename = filename.replace('Xemit','Yemit').replace('Zemit','Yemit')
        yemit = np.loadtxt(filename, unpack=False) if os.path.isfile(filename) else False
        if 'zemit' not in filename.lower():
            filename = filename.replace('Xemit','Zemit').replace('Yemit','Zemit')
        zemit = np.loadtxt(filename, unpack=False) if os.path.isfile(filename) else False
        interpret_astra_data(self, xemit, yemit, zemit)


def interpret_astra_data(self, xemit, yemit, zemit):
        z, t, mean_x, rms_x, rms_xp, exn, mean_xxp = np.transpose(xemit)
        z, t, mean_y, rms_y, rms_yp, eyn, mean_yyp = np.transpose(yemit)
        z, t, e_kin, rms_z, rms_e, ezn, mean_zep = np.transpose(zemit)
        e_kin = 1e6 * e_kin
        t = 1e-9 * t
        exn = 1e-6*exn
        eyn = 1e-6*eyn
        rms_x, rms_xp, rms_y, rms_yp, rms_z, rms_e = 1e-3*np.array([rms_x, rms_xp, rms_y, rms_yp, rms_z, rms_e])
        rms_e = 1e6 * rms_e
        self.append('z',z)
        self.append('t',t)
        self.append('kinetic_energy', e_kin)
        gamma = 1 + (e_kin / self.E0_eV)
        self.append('gamma', gamma)
        cp = np.sqrt(e_kin * (2 * self.E0_eV + e_kin)) * constants.elementary_charge
        # self.append('cp', cp)
        self.append('cp', cp / constants.elementary_charge)
        p = cp * self.q_over_c
        self.append('p', p)
        self.append('enx', exn)
        ex = exn / gamma
        self.append('ex', ex)
        self.append('eny', eyn)
        ey = eyn / gamma
        self.append('ey', ey)
        self.append('enz', ezn)
        ez = ezn / gamma
        self.append('ez', ez)
        self.append('beta_x', rms_x**2 / ex)
        self.append('gamma_x', rms_xp**2 / ex)
        self.append('alpha_x', (-1 * np.sign(mean_xxp) * rms_x * rms_xp) / ex)
        # self.append('alpha_x', (-1 * mean_xxp * rms_x) / ex)
        self.append('beta_y', rms_y**2 / ey)
        self.append('gamma_y', rms_yp**2 / ey)
        self.append('alpha_y', (-1 * np.sign(mean_yyp) * rms_y * rms_yp) / ey)
        self.append('beta_z', rms_z**2 / ez)
        self.append('gamma_z', rms_e**2 / ez)
        self.append('alpha_z', (-1 * np.sign(mean_zep) * rms_z * rms_e) / ez)
        self.append('sigma_x', rms_x)
        self.append('sigma_y', rms_y)
        self.append('sigma_z', rms_z)
        self.append('mean_x', mean_x)
        self.append('mean_y', mean_y)
        beta = np.sqrt(1-(gamma**-2))
        self.append('sigma_t', rms_z / (beta * constants.speed_of_light))
        self.append('sigma_p', (1e-3*rms_e / (e_kin+self.E0_eV)))
        # self.append('sigma_cp', (rms_e / e_kin) * p)
        self.append('sigma_cp', 1e3*(rms_e))
        # print('astra = ', (rms_e)[-1)
        self.append('mux', cumtrapz(x=self['z'], y=1/self['beta_x']))
        self.append('muy', cumtrapz(x=self['z'], y=1/self['beta_y']))
        self.append('eta_x', np.zeros(len(z)))
        self.append('eta_xp', np.zeros(len(z)))

        self.append('ecnx', exn)
        self.append('ecny', eyn)
        self.append('element_name', np.full(len(z),''))
        self.append('eta_x_beam', np.zeros(len(z)))
        self.append('eta_xp_beam', np.zeros(len(z)))
        self.append('eta_y_beam', np.zeros(len(z)))
        self.append('eta_yp_beam', np.zeros(len(z)))
        self.append('beta_x_beam', rms_x**2 / ex)
        self.append('beta_y_beam', rms_y**2 / ey)
        self.append('alpha_x_beam', (-1 * np.sign(mean_xxp) * rms_x * rms_xp) / ex)
        self.append('alpha_y_beam', (-1 * np.sign(mean_yyp) * rms_y * rms_yp) / ey)
        self['cp_eV'] = self['cp']
        self['sigma_cp_eV'] = self['sigma_cp']
