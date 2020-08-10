import sys, os, time, math, datetime, copy, re,  h5py
from collections import OrderedDict
import glob
try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except:
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
import pyqtgraph as pg
import numpy as np
sys.path.append(os.path.abspath(os.path.realpath(__file__)+'/../../../'))
import SimulationFramework.Modules.read_beam_file as raf
import SimulationFramework.Modules.read_twiss_file as rtf
sys.path.append(os.path.realpath(__file__)+'/../../../../')

class mainWindow(QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()
        self.resize(1800,900)
        self.centralWidget = QWidget()
        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        self.tab = QTabWidget()
        self.multiPlot = multiPlotWidget()

        self.layout.addWidget(self.multiPlot)

        self.setCentralWidget(self.centralWidget)

        self.setWindowTitle("ASTRA Data Plotter")
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

class multiPlotWidget(QWidget):
    ''' QWidget containing various Twiss plots '''
    # Styles for the plot lines
    colors = [QColor('#F5973A'),QColor('#A95AA1'),QColor('#85C059'),QColor('#0F2080'),QColor('#BDB8AD'), 'r', 'k', 'm', 'g']
    styles = [Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine, Qt.DashDotDotLine]

    # Layout oder for the individual Tiwss plot items
    plotParams = []

    def __init__(self, xmin=None, ymin=None, **kwargs):
        super(multiPlotWidget, self).__init__(**kwargs)
        ''' multiPlotWidget - main pyQtGraph display widgets '''
        self.multiPlotView = pg.GraphicsView(useOpenGL=True)
        self.multiPlotWidget = pg.GraphicsLayout()
        self.multiPlotView.setCentralItem(self.multiPlotWidget)
        ''' multiPlotWidgets - holds the base plotWidgets for each Twiss plot '''
        self.multiPlotWidgets = {}
        ''' curves - a dictionary containing {directory: [individual plotItems for each twiss plot,]} '''
        self.curves = {}
        for n, param in enumerate(self.plotParams):
            if param == 'next_row':
                self.multiPlotWidget.nextRow()
            else:
                ''' p - the relevant plotWidget for each param in plotParams '''
                p = self.multiPlotWidget.addPlot(title=param['label'])
                ''' this just links the horizontal axis for all plots.
                    The first plot is selected to be the master axis '''
                if n == 0:
                    self.linkAxis = p.vb
                else:
                    p.setXLink(self.linkAxis)
                p.showGrid(x=True, y=True)
                p.setLabel('left', text=param['name'], units=param['units'])
                ''' set lower plot limit at 0 for all plots '''
                if xmin is not None and ymin is not None:
                    p.vb.setLimits(xMin=xmin,yMin=ymin)
                ''' set the vertical viewRange according to the plotParams '''
                # p.vb.setYRange(*param['range'])
                ''' record the plotWidget in the dictionary indexed by Twiss item '''
                self.multiPlotWidgets[param['label']] = p

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.multiPlotView)

        ''' used for style cycling '''
        self.plotColor = 0
        self.shadowCurves = []

    def addCurve(self, x, y, name, label, pen):
        ''' adds a curve to the main plot '''
        self.curves[name][label] = self.multiPlotWidgets[label].plot(x=x, y=y, pen=pen)
        self.curves[name][label].curve.setClickable(True)
        self.curves[name][label].sigClicked.connect(lambda: self.highlightPlot(name))

    def updateCurve(self, x, y, name, label):
        ''' adds a curve to the main plot '''
        self.curves[name][label].setData(x=x, y=y, pen=self.curves[name][label].opts['pen'])

    def removeCurve(self, name):
        ''' finds all curves based on a key name, and removes them '''
        if not isinstance(name, (list, tuple)):
            name = [name]
        for n in name:
            if n in self.curves:
                for param in self.plotParams:
                    if not param == 'next_row':
                        ''' Remove the plotItem from the relevant plotWidget '''
                        print('REMOVING curve: ', name)
                        try:
                            self.multiPlotWidgets[param['label']].removeItem(self.curves[n][param['label']])
                        except:
                            pass
                del self.curves[n]

    def clearCurves(self):
        self.removeCurve(self.curves.keys())

    def highlightPlot(self, name):
        ''' highlights a particular plot '''
        # print('highligher clicked! = ', name)
        if not isinstance(name, (list, tuple)):
            name = [name]
        for n in name:
            self.addShadowPen(n)
            for n in self.curves.keys():
                if n in self.shadowCurves or not len(self.shadowCurves) > 0:
                    self.setPenAlpha(n, 255, 3)
                else:
                    self.setPenAlpha(n, 25, 3)

    def addShadowPen(self, name):
        ''' add/remove a shadow pen to a given plot curve '''
        for param in self.plotParams:
            if not param == 'next_row':
                label = param['label']
                # if name in self.curves and label in self.curves[name]:
                curve = self.curves[name][label]
                if curve.opts['shadowPen'] is None:
                    self.shadowCurves.append(name)
                    pen = curve.opts['pen']
                    shadowpencolor = pen.color()
                    shadowpencolor.setAlpha(100)
                    shadowpen = pg.mkPen(color=shadowpencolor, width=(pen.width()+3))
                    curve.setShadowPen(shadowpen)
                else:
                    self.shadowCurves.remove(name)
                    curve.setShadowPen(None)
                    curve.opts['shadowPen'] = None

    def setPenAlpha(self, name, alpha=255, width=3):
        ''' change the alpha channel and width of a curves pen '''
        for param in self.plotParams:
            if not param == 'next_row':
                label = param['label']
                if name in self.curves and label in self.curves[name]:
                    curve = self.curves[name][label]
                    pen = curve.opts['pen']
                    pencolor = pen.color()
                    pencolor.setAlpha(alpha)
                    pen = pg.mkPen(color=pencolor, width=width, style=pen.style())
                    curve.setPen(pen)


pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    ex = mainWindow()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
   main()
