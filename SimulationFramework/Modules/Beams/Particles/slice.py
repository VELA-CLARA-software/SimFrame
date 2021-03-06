import numpy as np

class slice():

    def __init__(self, beam):
        self.beam = beam
        self._slicelength = 0
        self._slices = 0
        self.bin_time()
    # def __repr__(self):
    #     return repr({p: self.emittance(p) for p in ('x', 'y')})

    @property
    def slice_length(self):
        return self._slicelength

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
        twidth = (max(self.beam.t) - min(self.beam.t))
        # print('twidth = ', twidth)
        if twidth == 0:
            t = self.beam.z / (-1 * self.beam.Bz * constants.speed_of_light)
            twidth = (max(t) - min(t))
        if slices == 0:
            slices = int(twidth / 0.1e-12)
        self._slices = slices
        self._slicelength = twidth / slices
        self.bin_time()

    def bin_time(self):
        if hasattr(self.beam, 't') and len(self.beam.t) > 0:
            if not self.slice_length > 0:
                # print('no slicelength', self.slice_length)
                self._slice_length = 0
                # print("Assuming slice length is 100 fs")
            twidth = (max(self.beam.t) - min(self.beam.t))
            if twidth == 0:
                t = self.beam.z / (-1 * self.beam.Bz * constants.speed_of_light)
                twidth = (max(t) - min(t))
            else:
                t = self.beam.t
            if not self.slice_length > 0.0:
                self.slice_length = twidth / 20.0
            # print('slicelength =', self.slice_length)
            nbins = max([1,int(np.ceil(twidth / self.slice_length))])+2
            self._hist, binst =  np.histogram(t, bins=nbins, range=(min(t)-self.slice_length, max(t)+self.slice_length))
            self._t_Bins = binst
            self._t_binned = np.digitize(t, self._t_Bins)
            self._tfbins = [[self._t_binned == i] for i in range(1, len(binst))]
            self._tbins = [np.array(self.beam.t)[tuple(tbin)] for tbin in self._tfbins]
            self._cpbins = [np.array(self.beam.cp)[tuple(tbin)] for tbin in self._tfbins]

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
        self._cpbins = [self.beam.cp[tuple(cpbin)] for cpbin in self._tfbins]
        self._tbins = [self.beam.t[tuple(cpbin)] for cpbin in self._tfbins]

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
        return np.array([cpbin.mean() if len(cpbin) > 0 else 0 for cpbin in self._cpbins ])
    @property
    def slice_momentum_spread(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        return np.array([cpbin.std() if len(cpbin) > 0 else 0 for cpbin in self._cpbins])
    @property
    def slice_relative_momentum_spread(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        return np.array([100*cpbin.std()/cpbin.mean() if len(cpbin) > 0 else 0 for cpbin in self._cpbins])

    def slice_data(self, data):
        return [data[tuple(tbin)] for tbin in self._tfbins]

    def emitbins(self, x, y):
        xbins = self.slice_data(x)
        ybins = self.slice_data(y)
        return list(zip(*[xbins, ybins, self._cpbins]))

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
    def slice_enx(self):
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
        return np.array([self.beam.emittance.emittance_calc(xbin, xpbin) if len(cpbin) > 0 else 0 for xbin, xpbin, cpbin in emitbins])
    @property
    def slice_vertical_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.beam.y, self.beam.yp)
        return np.array([self.beam.emittance.emittance_calc(ybin, ypbin) if len(cpbin) > 0 else 0 for ybin, ypbin, cpbin in emitbins])
    @property
    def slice_normalized_horizontal_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.beam.x, self.beam.xp)
        return np.array([self.beam.emittance.emittance_calc(xbin, xpbin, cpbin) if len(cpbin) > 0 else 0 for xbin, xpbin, cpbin in emitbins])
    @property
    def slice_normalized_vertical_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.beam.y, self.beam.yp)
        return np.array([self.beam.emittance.emittance_calc(ybin, ypbin, cpbin) if len(cpbin) > 0 else 0 for ybin, ypbin, cpbin in emitbins])
    @property
    def slice_current(self):
        if not hasattr(self,'_hist'):
            self.bin_time()
        f = lambda bin: self.beam.Q / len(self.beam.t) * (len(bin) / (max(bin) - min(bin))) if len(bin) > 1 else 0
        # f = lambda bin: len(bin) if len(bin) > 1 else 0
        return np.array([f(bin) for bin in self._tbins])
    @property
    def slice_max_peak_current_slice(self):
        peakI = self.slice_current
        return list(abs(peakI)).index(max(abs(peakI)))

    @property
    def slice_beta_x(self):
        xbins = self.slice_data(self.beam.x)
        exbins =  self.slice_horizontal_emittance
        emitbins = list(zip(xbins, exbins))
        return np.array([self.beam.covariance(x, x)/ex if ex > 0 else 0 for x, ex in emitbins])
    @property
    def slice_alpha_x(self):
        xbins = self.slice_data(self.beam.x)
        xpbins = self.slice_data(self.beam.xp)
        exbins =  self.slice_horizontal_emittance
        emitbins = list(zip(xbins, xpbins, exbins))
        return np.array([-1*self.beam.covariance(x, xp)/ex if ex > 0 else 0 for x, xp, ex in emitbins])
    @property
    def slice_gamma_x(self):
        xpbins = self.slice_data(self.beam.xp)
        exbins =  self.slice_horizontal_emittance
        emitbins = list(zip(xpbins, exbins))
        return np.array([self.beam.covariance(xp, xp)/ex if ex > 0 else 0 for xp, ex in emitbins])

    @property
    def slice_beta_y(self):
        ybins = self.slice_data(self.beam.y)
        eybins =  self.slice_vertical_emittance
        emitbins = list(zip(ybins, eybins))
        return np.array([self.beam.covariance(y, y)/ey if ey > 0 else 0 for y, ey in emitbins])
    @property
    def slice_alpha_y(self):
        ybins = self.slice_data(self.beam.y)
        ypbins = self.slice_data(self.beam.yp)
        eybins =  self.slice_vertical_emittance
        emitbins = list(zip(ybins, ypbins, eybins))
        return np.array([-1*self.beam.covariance(y,yp)/ey if ey > 0 else 0 for y, yp, ey in emitbins])
    @property
    def slice_gamma_y(self):
        ypbins = self.slice_data(self.beam.yp)
        eybins =  self.slice_vertical_emittance
        emitbins = list(zip(ypbins, eybins))
        return np.array([self.beam.covariance(yp,yp)/ey if ey > 0 else 0 for yp, ey in emitbins])

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
        chirp = (1e-18 * (slice_momentum_centroid[-1] - slice_momentum_centroid[0]) / (len(slice_momentum_centroid) * self.slice_length))
        return chirp
