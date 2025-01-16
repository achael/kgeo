import numpy as np
import scipy.special as sp
from tqdm import tqdm
from kgeo.kerr_raytracing_utils import my_cbrt, radial_roots, mino_total, is_outside_crit, uplus_uminus
from kgeo.equatorial_lensing import r_equatorial, nmax_equatorial, nmax_poloidal
import time
from mpmath import polylog
from scipy.interpolate import UnivariateSpline
import os

# Fitting function parameters for emissivity and velocity
P1E_230=-2.0; P2E_230=-0.5; # for  230 GHz
P1E_86=0; P2E_86=-.75;  # for 86 GHz


GAMMAOFF = -1.5
SIGMA_GLM = 0.5
R_RING = 4.5
SIGMA_RING = 0.3

class Emissivity(object):
    """ object for rest frame emissivity as a function of r, only in equatorial plane for now """
    
    def __init__(self, emistype="bpl", **kwargs):

        self.emistype = emistype
        self.kwargs = kwargs
        
        self.emiscut_in = self.kwargs.get('emiscut_in', 0)
        self.emiscut_out = self.kwargs.get('emiscut_out', 1.e10)
        
        if self.emistype=='constant':
            pass
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
        else: 
            raise Exception("emistype %s not recognized in Emissivity!"%self.veltype)
    
    def jrest(self, a, r):
        if self.emistype=='constant':
            j = np.ones(r.shape)
            
        elif self.emistype=='bpl':     
            j = emisBPL(a, r, p1=self.p1, p2=self.p2)
            
        elif self.emistype=='glm' or self.emistype=='ring':
            j = emisGLM(a, r, gamma_off=self.gamma_off, sigma=self.sigma, mu_ring=self.mu_ring)
                        
        else: 
            raise Exception("emistype %s not recognized in Emissivity.emis!"%self.veltype)    
            
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


OBSFREQ = 230.e9 # observation frequency, Hz
N0 = 1.e5  # particles/cm^3
T0 = 5.e10 # Kelvin
B0 = 5 # Gauss
ALPHAN = 1
ALPHAT = 1
ALPHAB = 1.5
def emisThermal(a, r, nuemit, thetab,
                alpha_n=ALPHAN, nref=NREF, 
                alpha_T=ALPHAT, Tref=TREF, 
                alpha_B=ALPHAB, Bref=BREF,
                use_consistent_bfield=False, bfield=None):
    """emissivity from thermal electrons following power law distributions, following Desire+24,Dexter+16"""
    
    # nuemit = nu_obs/g
    
    # fluid quantities and emission radius in cgs units
    r_0 = 5.
    n = nref*(r/r_0)**(-alpha_n)
    T = Tref*(r/r_0)**(-alpha_T)
    B = Bref*(r/r_0)**(-alpha_B)
    
    #TODO: consistent bfield
    if use_consistent_bfield:
    
    # dimensionless electron temperature
    thetae = T * 1.68637005e-10
    
    # synchrotron critical frequency
    nu_c = 4.19887e6 * B * thetae * thetae * np.abs(np.sin(thetab))
    
    # thermal emissivity fitting function (Dexter A18)
    x = nuemit/nu_c
    Ii = 2.5651*(1+1.92*x**(-1./3.) + 0.9977*x**(-2./3.))*np.exp(-1.8899*x**(1./3.))
    
    # units?? 
    jnu = (2.22152e-30)*n*(nuemit/thetae**2)*Ii

    return jnu 
    
GAMMAMIN=10
GAMMAMAX=1.e8
PINDEX=2.5
def emisPowerlaw(a, r, nuemit, thetab, p=PINDEX, 
                 alpha_n=ALPHAN, nref=NREF, 
                 gammamin=GAMMAMIN, gammamax=GAMMAMAX, 
                 alpha_B=ALPHAB, Bref=BREF,
                 use_consistent_bfield=False, bfield=None):
    """emissivity from nonthermal electrons following power law distributions, following Dexter+16"""
    
    # nuemit = nu_obs/g
    
    # fluid quantities and emission radius in cgs units
    r_0 = 5.
    n = nref*(r/r_0)**(-alpha_n)
    B = Bref*(r/r_0)**(-alpha_B)

    #TODO: consistent bfield
    if use_consistent_bfield:
      
    # synchrotron critical frequency / gamma^2
    nu_p = 4.19887e6 * B  * np.abs(np.sin(thetab))
   
    # edge term fitting functions
    x1 = nuemit / (gammamin*gammamin*nu_p)
    x2 = nuemit / (gammamax*gammamax*nu_p)
    
    # TODO implment fitting functions
    # for now, approximate
    Gimin = (2**(0.5*p-1.5)) * (p+7./3.) * sp.gamma(0.25*p + 7./12.) * sp.gamma(0.25*p - 1./12.) / (p+1)
    Gimax = 0
    
    # units?? 
    alpha = (p-1)/2.
    nfac = gammamin**(1-p) - gammamax**(1-p)
    jnu = (2.22152e-30)*(n/nfac)*(p-1)*nu_p*((nu/nu_p)**(-alpha))*(Gimin-Gimax)

    return jnu       
    
    
          
