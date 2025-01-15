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


#def emisThermal(a, r, thetab,  
#                alpha_n=ALPHAN, n0=N0, 
#                alpha_T=ALPHAT, T0=T0, 
#                B0=B0):
#    """emissivity from thermal electrons following power law distributions, following Desire+24"""




                
