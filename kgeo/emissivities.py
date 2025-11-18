import numpy as np
import scipy.special as sp
from tqdm import tqdm
from kgeo.kerr_raytracing_utils import my_cbrt, radial_roots, mino_total, is_outside_crit, uplus_uminus
from kgeo.equatorial_lensing import r_equatorial, nmax_equatorial, nmax_poloidal
import time
from mpmath import polylog
from scipy.interpolate import RegularGridInterpolator
import pkg_resources
from kgeo.bfields import Bfield
from kgeo.velocities import Velocity

# kgeo path TODO
KGEOPATH = '/home/achael/RelElectrons/kgeo/kgeo'

# Broken Power Law parameters
P1E_230=-2.0; P2E_230=-0.5; # for  230 GHz
P1E_86=0; P2E_86=-.75;  # for 86 GHz

# GLM model parameters
GAMMAOFF = -1.5
SIGMA_GLM = 0.5
R_RING = 4.5
SIGMA_RING = 0.3

# Thermal model parameters
e  = 4.80320425e-10  # statcoul
me = 9.10938356e-28  # g
c  = 2.99792458e10   # cm/s
kB = 1.380649e-16    # erg/K
h  = 6.62607015e-27  # erg*s

# default observation frequency and spectral index
OBSFREQ = 230.e9
SPECIND = 0.0

class Emissivity(object):
    """ object for rest frame emissivity as a function of r, only in equatorial plane for now (theta=np.pi/2) """
    def __init__(self, emistype="bpl", **kwargs):
        print("Emissivity emistype =", repr(emistype))

        self.emistype = emistype
        self.kwargs = kwargs

        self.emiscut_in = self.kwargs.get('emiscut_in', 0)
        self.emiscut_out = self.kwargs.get('emiscut_out', 1.e10)

        if self.emistype=='thermal':
            self.Rb = float(self.kwargs.get('Rb', 5.0))  
            self.ne0 = float(self.kwargs['ne0']) 
            self.Te0 = float(self.kwargs['Te0']) 
            self.B0 = float(self.kwargs['B0'])
            self.alpha_n = float(self.kwargs.get('alpha_n', 0.7))
            self.alpha_T = float(self.kwargs.get('alpha_T', 1.0))
            self.alpha_B = float(self.kwargs.get('alpha_B', 1.5))
            self.bfield_model = bool(self.kwargs.get('bfield_model', False))

        elif self.emistype=='bpl':
            self.p1 = self.kwargs.get('p1', P1E_230)
            self.p2 = self.kwargs.get('p2', P2E_230)
        elif self.emistype =='ring':
            self.mu_ring = self.kwargs.get('r_ring', R_RING)
            self.gamma_off = self.kwargs.get('gamma_off', 0)
            self.sigma = self.kwargs.get('sigma', SIGMA_RING)
        elif self.emistype=='glm':
            self.mu_ring = False
            self.gamma_off = self.kwargs.get('gamma_off', GAMMAOFF)
            self.sigma = self.kwargs.get('sigma', SIGMA_GLM)
        elif self.emistype == 'constant':
            pass
        else:
            raise Exception("emistype %s not recognized in Emissivity!"%self.emistype)
        

    # power law disk profiles to define thermal variables
    def profiles_plaw(self, r):
        x = np.asarray(r, dtype=float)/self.Rb
        ne = self.ne0 * np.power(x, -self.alpha_n) # cm^-3
        Te = self.Te0 * np.power(x, -self.alpha_T) # K 
        B  = self.B0  * np.power(x, -self.alpha_B) # Gauss
        return ne, Te, B

    def jrest(self, a, r, g=None, sinthetab=None, Bmag=None, nu_obs=OBSFREQ, specind=SPECIND):

        ## Physical models
        if self.emistype=='thermal':
            nu_em = np.asarray(nu_obs) / np.asarray(g)     # emitted frequency
            ne, Te, B = self.profiles_plaw(r)              # local plasma properties

            # option to overwrite default power law field strength with actual |B| from model 
            if self.bfield_model == True:
                Bmag = np.asarray(Bmag, dtype=float)
                B = (Bmag / self.Rb) * self.B0
            
            j = j_nu_thermal(ne, B, Te, nu_em, sinthetab)

        ## Phenomenological models
        elif self.emistype=='constant':
            j = np.ones(r.shape)

        elif self.emistype=='bpl':
            j = emisBPL(a, r, p1=self.p1, p2=self.p2)

        elif self.emistype=='glm' or self.emistype=='ring':
            j = emisGLM(a, r, gamma_off=self.gamma_off, sigma=self.sigma, mu_ring=self.mu_ring)

        else:
            raise Exception("emistype %s not recognized in Emissivity.emis!"%self.emistype)

        # add spectral behavior for non-physical emissivities
        if self.emistype != 'thermal':
            if g is not None:
                j *=  (g**specind)
            if sinthetab is not None:
                j *= (sinthetab**(1+specind))

        return j


def emisBPL(a, r, p1=P1E_230, p2=P2E_230):
    """emissivity at radius r - broken power law model fit to GRMHD"""

    rh = 1 + np.sqrt(1-a**2)
    emis = np.exp(p1*np.log(r/rh) + p2*np.log(r/rh)**2)

    return emis


def emisGLM(a, r, gamma_off=GAMMAOFF, sigma=SIGMA_GLM, mu_ring=False):
    """emissivity at radius r from GLM paper"""

    if mu_ring:
        mu = mu_ring
    else:
        mu = 1 - np.sqrt(1-a**2)
    emis = np.exp(-0.5*(gamma_off+np.arcsinh((r-mu)/sigma))**2) / np.sqrt((r-mu)**2 + sigma**2)
    return emis

# Thermal model function definitions
def II_fit(x):
    return 2.5651 * (1.0 + 1.92*x**(-1.0/3.0) + 0.9977*x**(-2.0/3.0)) * np.exp(-1.8899 * x**(1.0/3.0))

# critical freqneucy (in fluid frame)
def nu_c_fcn(B, Theta_e, sin_thetaB):
    nu_B = e*B/(2.0*np.pi*me*c)
    nu_c = 1.5 * nu_B * Theta_e**2 * sin_thetaB
    return nu_c
    #return (3.0/(4.0*np.pi)) * (e*B*Theta_e**2)/(me*c) * sin_thetaB

# Emissivity j_nu (fluid frame, per unit volume, freq, steradians)
# cgs units: erg s^-1 cm^-3 Hz^-1 sr^-1
def j_nu_thermal(ne, B, Te, nu, sin_thetaB):
    Theta_e = kB*Te/(me*c*c)       # strictly valid only in Theta_e > 3 limit
    nu_c = nu_c_fcn(B, Theta_e, sin_thetaB)
    x = nu/np.maximum(nu_c, 1e-40) # strictly valid only in nu > nu_c limit
    return (ne * e**2 * nu)/(2 * np.sqrt(3) * c * (Theta_e**2)) * II_fit(x)
