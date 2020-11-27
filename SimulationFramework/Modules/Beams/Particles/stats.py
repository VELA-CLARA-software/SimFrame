import munch
import numpy as np
import scipy.constants as constants

class stats(munch.Munch):

    def __init__(self, beam):
        self.beam = beam

    @property
    def sigma_x(self):
        return self.Sx
    @property
    def sigma_y(self):
        return self.Sy
    @property
    def sigma_t(self):
        return self.St
    @property
    def sigma_z(self):
        return self.Sz

    @property
    def Sx(self):
        return np.sqrt(self.beam.covariance(self.beam.x, self.beam.x))
    @property
    def Sy(self):
        return np.sqrt(self.beam.covariance(self.beam.y, self.beam.y))
    @property
    def Sz(self):
        return np.sqrt(self.beam.covariance(self.beam.z, self.beam.z))
    @property
    def St(self):
        return np.sqrt(self.beam.covariance(self.beam.t, self.beam.t))
    @property
    def momentum_spread(self):
        return self.beam.cp.std()/np.mean(self.beam.cp)
    @property
    def linear_chirp_z(self):
        return -1*self.beam.rms(self.beam.Bz*constants.speed_of_light*self.t)/self.momentum_spread/100