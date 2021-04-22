import time
import numpy as np
from ...units import UnitValue

class slice():

    def __init__(self, beam):
        self.beam = beam
        self._slicelength = 0
        self._slices = 0
        self.time_binned = {'beam': None, 'slices': None, 'slice_length': None}
        # self.bin_time()
    # def __repr__(self):
    #     return repr({p: self.emittance(p) for p in ('x', 'y')})

    def have_we_already_been_binned(self):
        if self.time_binned['beam'] == self.beam and self.time_binned['slices'] == self._slices and self.time_binned['slice_length'] == self._slicelength:
            return True
        return False

    def update_binned_parameters(self):
        self.time_binned = {'beam': self.beam, 'slices': self._slices, 'slice_length': self._slicelength}

    @property
    def slice_length(self):
        return UnitValue(self._slicelength,'s')

    @slice_length.setter
    def slice_length(self, slicelength):
        self._slicelength = slicelength
        self.bin_time()

    @property
    def slices(self):
        return self._slices

    @slices.setter
    def slices(self, slices):
        self.set_slices(slices)

    def set_slices(self, slices):
        twidth = np.ptp(self.beam.t, axis=0)
        # print('twidth = ', twidth)
        if twidth == 0:
            t = self.beam.z / (-1 * self.beam.Bz * constants.speed_of_light)
            twidth = np.ptp(t, axis=0)
        if slices == 0:
            slices = int(twidth / 0.1e-12)
        self._slices = slices
        self._slicelength = twidth / slices
        self.bin_time()

    def bin_time(self):
        if not self.have_we_already_been_binned():
            if hasattr(self.beam, 't') and len(self.beam.t) > 0:
                if not self.slice_length > 0:
                    # print('no slicelength', self.slice_length)
                    self._slice_length = 0
                    # print("Assuming slice length is 100 fs")
                twidth = np.ptp(self.beam.t, axis=0)
                if twidth == 0:
                    t = self.beam.z / (-1 * self.beam.Bz * constants.speed_of_light)
                    twidth = np.ptp(t, axis=0)
                else:
                    t = self.beam.t
                if not self.slice_length > 0.0:
                    self.slice_length = twidth / 20.0
                # print('slicelength =', self.slice_length)
                nbins = max([1,int(np.ceil(twidth / self.slice_length))])+2
                self._hist, binst =  np.histogram(t, bins=nbins, range=(np.min(t)-self.slice_length, np.max(t)+self.slice_length))
                self._t_Bins = binst
                self._t_binned = np.digitize(t, self._t_Bins)
                self._tfbins = [[self._t_binned == i] for i in range(1, len(binst))]
                self._tbins = UnitValue([np.array(self.beam.t)[tuple(tbin)] for tbin in self._tfbins], units='s', dtype=np.ndarray)
                self._cpbins = UnitValue([np.array(self.beam.cp)[tuple(tbin)] for tbin in self._tfbins], units='eV/c', dtype=np.ndarray)
                self.update_binned_parameters()

    def bin_momentum(self, width=10**6):
        pwidth = (max(self.beam.cp) - min(self.beam.cp))
        if width is None:
            self.slice_length_cp = pwidth / self.slices
        else:
            self.slice_length_cp = width
        nbins = max([1,int(np.ceil(pwidth / self.slice_length_cp))])+2
        self._hist, binst =  np.histogram(self.beam.cp, bins=nbins, range=(min(self.beam.cp)-self.slice_length_cp, max(self.beam.cp)+self.slice_length_cp))
        self._cp_Bins = binst
        self._cp_binned = np.digitize(self.beam.cp, self._cp_Bins)
        self._tfbins = [np.array([self._cp_binned == i]) for i in range(1, len(binst))]
        self._cpbins = UnitValue([self.beam.cp[tuple(cpbin)] for cpbin in self._tfbins], units='eV/c')
        self._tbins = UnitValue([self.beam.t[tuple(cpbin)] for cpbin in self._tfbins], units='s')

    @property
    def slice_bins(self):
        if not hasattr(self,'slice'):
            self.bin_time()
        bins = self._t_Bins
        return (bins[:-1] + bins[1:]) / 2
    @property
    def slice_cpbins(self):
        if not hasattr(self,'slice'):
            self.bin_momentum()
        bins = self.sliceProperties['cp_Bins']
        return (bins[:-1] + bins[1:]) / 2


    @property
    def slice_momentum(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        return UnitValue([cpbin.mean() if len(cpbin) > 0 else 0 for cpbin in self._cpbins ], units='eV/c')
    @property
    def slice_momentum_spread(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        return UnitValue([cpbin.std() if len(cpbin) > 0 else 0 for cpbin in self._cpbins], units='eV/c')
    @property
    def slice_relative_momentum_spread(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        return UnitValue([100*cpbin.std()/cpbin.mean() if len(cpbin) > 0 else 0 for cpbin in self._cpbins], units='')

    def slice_data(self, data):
        return UnitValue([data[tuple(tbin)] for tbin in self._tfbins], units=data.units, dtype=object)

    def emitbins(self, x, y):
        xbins = self.slice_data(x)
        ybins = self.slice_data(y)
        return np.array([xbins, ybins, self._cpbins]).T

    @property
    def ex(self):
        return self.slice_ex
    @property
    def ey(self):
        return self.slice_ey
    @property
    def enx(self):
        return self.slice_enx
    @property
    def eny(self):
        return self.slice_eny
    # @property
    # def ecx(self):
    #     return self.horizontal_emittance_corrected
    # @property
    # def ecy(self):
    #     return self.vertical_emittance_corrected
    # @property
    # def ecnx(self):
    #     return self.normalised_horizontal_emittance_corrected
    # @property
    # def ecny(self):
    #     return self.normalised_vertical_emittance_corrected

    @property
    def slice_ex(self):
        return self.slice_horizontal_emittance
    @property
    def slice_ey(self):
        return self.slice_vertical_emittance
    @property
    def slice_enx(self):
        return self.slice_normalized_horizontal_emittance
    @property
    def slice_eny(self):
        return self.slice_normalized_vertical_emittance

    @property
    def slice_t(self):
        return np.array(self.slice_bins)
    @property
    def slice_z(self):
        return np.array(self.slice_bins)

    @property
    def slice_horizontal_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.beam.x, self.beam.xp)
        return UnitValue([self.beam.emittance.emittance_calc(xbin, xpbin) if len(cpbin) > 0 else 0 for xbin, xpbin, cpbin in emitbins], units='m-rad')
    @property
    def slice_vertical_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.beam.y, self.beam.yp)
        return UnitValue([self.beam.emittance.emittance_calc(ybin, ypbin) if len(cpbin) > 0 else 0 for ybin, ypbin, cpbin in emitbins], units='m-rad')
    @property
    def slice_normalized_horizontal_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.beam.x, self.beam.xp)
        return UnitValue([self.beam.emittance.emittance_calc(xbin, xpbin, cpbin) if len(cpbin) > 0 else 0 for xbin, xpbin, cpbin in emitbins], units='m-rad')
    @property
    def slice_normalized_vertical_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.beam.y, self.beam.yp)
        return UnitValue([self.beam.emittance.emittance_calc(ybin, ypbin, cpbin) if len(cpbin) > 0 else 0 for ybin, ypbin, cpbin in emitbins], units='m-rad')
    @property
    def slice_current(self):
        if not hasattr(self,'_hist'):
            self.bin_time()
        absQ = np.abs(self.beam.Q) / len(self.beam.t)
        f = lambda bin: absQ * (len(bin) / np.ptp(bin, axis=0)) if len(bin) > 1 else 0
        # f = lambda bin: len(bin) if len(bin) > 1 else 0
        return UnitValue([f(bin) for bin in self._tbins], units='A')
    @property
    def peak_current(self):
        peakI = self.slice_current
        return UnitValue(max(abs(peakI)), units='A')
    @property
    def slice_max_peak_current_slice(self):
        peakI = self.slice_current
        return UnitValue(list(abs(peakI)).index(max(abs(peakI))), units='A')

    @property
    def beta_x(self):
        return self.slice_beta_x
    @property
    def alpha_x(self):
        return self.slice_alpha_x
    @property
    def gamma_x(self):
        return self.slice_gamma_x
    @property
    def beta_y(self):
        return self.slice_beta_y
    @property
    def alpha_y(self):
        return self.slice_alpha_y
    @property
    def gamma_y(self):
        return self.slice_gamma_y

    @property
    def slice_beta_x(self):
        xbins = self.slice_data(self.beam.x)
        exbins =  self.slice_horizontal_emittance
        emitbins = list(zip(xbins, exbins))
        return UnitValue([self.beam.covariance(x, x)/ex if ex > 0 else 0 for x, ex in emitbins], units='m/rad')
    @property
    def slice_alpha_x(self):
        xbins = self.slice_data(self.beam.x)
        xpbins = self.slice_data(self.beam.xp)
        exbins =  self.slice_horizontal_emittance
        emitbins = list(zip(xbins, xpbins, exbins))
        return UnitValue([-1*self.beam.covariance(x, xp)/ex if ex > 0 else 0 for x, xp, ex in emitbins], units='')
    @property
    def slice_gamma_x(self):
        xpbins = self.slice_data(self.beam.xp)
        exbins =  self.slice_horizontal_emittance
        emitbins = list(zip(xpbins, exbins))
        return UnitValue([self.beam.covariance(xp, xp)/ex if ex > 0 else 0 for xp, ex in emitbins], units='rad/m')

    @property
    def slice_beta_y(self):
        ybins = self.slice_data(self.beam.y)
        eybins =  self.slice_vertical_emittance
        emitbins = list(zip(ybins, eybins))
        return UnitValue([self.beam.covariance(y, y)/ey if ey > 0 else 0 for y, ey in emitbins], units='m/rad')
    @property
    def slice_alpha_y(self):
        ybins = self.slice_data(self.beam.y)
        ypbins = self.slice_data(self.beam.yp)
        eybins =  self.slice_vertical_emittance
        emitbins = list(zip(ybins, ypbins, eybins))
        return UnitValue([-1*self.beam.covariance(y,yp)/ey if ey > 0 else 0 for y, yp, ey in emitbins], units='')
    @property
    def slice_gamma_y(self):
        ypbins = self.slice_data(self.beam.yp)
        eybins =  self.slice_vertical_emittance
        emitbins = list(zip(ypbins, eybins))
        return UnitValue([self.beam.covariance(yp,yp)/ey if ey > 0 else 0 for yp, ey in emitbins], units='rad/m')

    def sliceAnalysis(self, density=False):
        self.slice = {}
        self.bin_time()
        peakIPosition = self.slice_max_peak_current_slice
        slice_density = self.slice_density[peakIPosition] if density else 0
        return self.slice_current[peakIPosition], \
            np.std(self.slice_current), \
            self.slice_relative_momentum_spread[peakIPosition], \
            self.slice_normalized_horizontal_emittance[peakIPosition], \
            self.slice_normalized_vertical_emittance[peakIPosition], \
            self.slice_momentum[peakIPosition], \
            self.slice_density[peakIPosition],

    @property
    def chirp(self):
        self.bin_time()
        slice_current_centroid_indices = []
        slice_momentum_centroid = []
        peakIPosition = self.slice_max_peak_current_slice
        peakI = self.slice_current[peakIPosition]
        slicemomentum = self.slice_momentum
        for index, slice_current in enumerate(self.slice_current):
            if abs(peakI - slice_current) < (peakI * 0.75):
                slice_current_centroid_indices.append(index)
        for index in slice_current_centroid_indices:
            slice_momentum_centroid.append(slicemomentum[index])
        chirp = ((slice_momentum_centroid[-1] - slice_momentum_centroid[0]) / (len(slice_momentum_centroid) * self.slice_length))
        return UnitsFloat(chirp, 'eV/s')
