import numpy as np
import scipy.special as sp
from tqdm import tqdm
from kgeo.kerr_raytracing_utils import my_cbrt, radial_roots, mino_total, is_outside_crit, uplus_uminus
from kgeo.equatorial_lensing import r_equatorial, nmax_equatorial, nmax_poloidal
import time
from mpmath import polylog
from scipy.interpolate import UnivariateSpline, RegularGridInterpolator
import os
import pkg_resources

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

# thermal parameters
OBSFREQ = 230.e9 # observation frequency, Hz
N0 = 1.e5  # particles/cm^3
T0 = 5.e10 # Kelvin
B0 = 5 # Gauss
ALPHAN = 1
ALPHAT = 1
ALPHAB = 1.5

# power law parameters
GAMMAMIN=10
GAMMAMAX=1.e8
PINDEX=2.5

class Emissivity(object):
    """ object for rest frame emissivity as a function of r, only in equatorial plane for now (theta=np.pi/2) """
    
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
        elif self.emistype=='thermal':
            self.alpha_n = self.kwargs.get('alpha_n', ALPHAN)
            self.alpha_T = self.kwargs.get('alpha_T', ALPHAT)
            self.alpha_B = self.kwargs.get('alpha_B', ALPHAB)
            self.nref = self.kwargs.get('nref', N0)
            self.Tref = self.kwargs.get('Tref', T0)
            self.Bref = self.kwargs.get('Bref', B0)  
            self.use_consistent_bfield = self.kwargs.get('use_consistent_bfield', False)
            self.bfield = self.kwargs.get('bfield',None)
            self.velocity = self.kwargs.get('velocity',None)
        elif self.emistype=='powerlaw':
            self.alpha_n = self.kwargs.get('alpha_n', ALPHAN)
            self.alpha_B = self.kwargs.get('alpha_B', ALPHAB)
            self.nref = self.kwargs.get('nref', N0)
            self.Bref = self.kwargs.get('Bref', B0) 
            self.p =self.kwargs.get('p',PINDEX)
            self.gammamin = self.kwargs.get('gammamin',GAMMAMIN)
            self.gammamax = self.kwargs.get('gammamax',GAMMAMAX)            
            self.use_consistent_bfield = self.kwargs.get('use_consistent_bfield', False)
            self.bfield = self.kwargs.get('bfield',None)
            self.velocity = self.kwargs.get('velocity',None)
        else: 
            raise Exception("emistype %s not recognized in Emissivity!"%self.veltype)
    
    def jrest(self, a, r, g=None, sinthetab=None, nu_obs=OBSFREQ):
        if self.emistype=='constant':
            j = np.ones(r.shape)
            
        elif self.emistype=='bpl':     
            j = emisBPL(a, r, p1=self.p1, p2=self.p2)
            
        elif self.emistype=='glm' or self.emistype=='ring':
            j = emisGLM(a, r, gamma_off=self.gamma_off, sigma=self.sigma, mu_ring=self.mu_ring)
         
        elif self.emistype=='thermal':
            j = emisThermal(a, r, g, sinthetab, nu_obs,
                alpha_n=self.alpha_n, nref=self.nref, 
                alpha_T=self.alpha_T, Tref=self.Tref, 
                alpha_B=self.alpha_B, Bref=self.Bref,
                use_consistent_bfield=self.use_consistent_bfield, 
                bfield=self.bfield, velocity=self.velocity)
                
        elif self.emistype=='powerlaw':
            j = emisPowerlaw(a, r, g, sinthetab, nu_obs,
                p=self.p,gammamin=self.gammamin,gammamax=self.gammamax,
                alpha_n=self.alpha_n, nref=self.nref, 
                alpha_B=self.alpha_B, Bref=self.Bref,
                use_consistent_bfield=self.use_consistent_bfield, 
                bfield=self.bfield, velocity=self.velocity)        
                       
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



def emisThermal(a, r, g, sinthetab, nu_obs,
                alpha_n=ALPHAN, nref=N0, 
                alpha_T=ALPHAT, Tref=T0, 
                alpha_B=ALPHAB, Bref=B0,
                use_consistent_bfield=False, bfield=None, velocity=None):
    """emissivity from thermal electrons following power law distributions, following Desire+24,Dexter+16"""

    # TODO check r, g compatible
    
    # redshifted frequency
    nuemit = nu_obs/g
    
    # fluid quantities and emission radius in cgs units
    r_0 = 5.
    n = nref*(r/r_0)**(-alpha_n)
    T = Tref*(r/r_0)**(-alpha_T)

    # magnetic field can either follow simple power law or be based on bfield used for polarization
    if use_consistent_bfield:
        if bfield is None:
            raise Exception("bfield==None with use_consistent_bfield! in emisThermal!")
        if velocity is None:
            raise Exception("velocity==None with use_consistent_bfield! in emisThermal!")        
        bmag = np.sqrt(bfield.bsq(a, r, velocity, th=np.pi/2.))
        bmag_ref = np.sqrt(bfield.bsq(a, r_0, velocity, th=np.pi/2.))
        B = Bref * bmag/bmag_ref
    
    else:
        B = Bref*(r/r_0)**(-alpha_B)
        
    # dimensionless electron temperature
    thetae = T * 1.68637005e-10
    
    # synchrotron critical frequency
    nu_c = 4.19887e6 * B * thetae * thetae * np.abs(sinthetab)
    
    # thermal emissivity fitting function (Dexter A18)
    x = nuemit/nu_c
    x3 = x**(1./3.)
    Ii = 2.5651*(1+1.92/x3 + 0.9977/x3/x3)*np.exp(-1.8899*x3)
    
    # units?? 
    #jnu = (2.22152e-30)*n*nuemit*Ii/thetae/thetae
    jnu = n*nuemit*Ii/thetae/thetae
    
    return jnu 
    

def emisPowerlaw(a, r, g, sinthetab, nu_obs,
                 p=PINDEX,gammamin=GAMMAMIN, gammamax=GAMMAMAX, 
                 alpha_n=ALPHAN, nref=N0, 
                 alpha_B=ALPHAB, Bref=B0,
                 use_consistent_bfield=False, bfield=None, velocity=None):
    """emissivity from nonthermal electrons following power law distributions, following Dexter+16"""
    
    # TODO check r, g compatible
    
    # redshifted frequency
    nuemit = nu_obs/g
    
    # fluid quantities and emission radius in cgs units
    r_0 = 5.
    n = nref*(r/r_0)**(-alpha_n)
    B = Bref*(r/r_0)**(-alpha_B)

    # magnetic field can either follow simple power law or be based on bfield used for polarization
    if use_consistent_bfield:
        if bfield is None:
            raise Exception("bfield==None with use_consistent_bfield! in emisThermal!")
        if velocity is None:
            raise Exception("velocity==None with use_consistent_bfield! in emisThermal!")        
        bmag = np.sqrt(bfield.bsq(a, r, velocity, th=np.pi/2.))
        bmag_ref = np.sqrt(bfield.bsq(a, r_0, velocity, th=np.pi/2.))
        B = Bref * bmag/bmag_ref
    
    else:
        B = Bref*(r/r_0)**(-alpha_B)
      
    # synchrotron critical frequency / gamma^2
    nu_p = 4.19887e6 * B  * np.abs(sinthetab)
   
    # edge term fitting functions
    x1 = nuemit / (gammamin*gammamin*nu_p)
    x2 = nuemit / (gammamax*gammamax*nu_p)
    
    # TODO implment fitting functions
    # for now, approximate
    #Gimin = (2**(0.5*p-1.5)) * (p+7./3.) * sp.gamma(0.25*p + 7./12.) * sp.gamma(0.25*p - 1./12.) / (p+1)
    #Gimax = 0
    Gimin = GIfunc(p,x2)
    Gimax = GIfunc(p,x1)
    
    # units?? 
    alpha = (p-1)/2.
    nfac = gammamin**(1-p) - gammamax**(1-p)
    #jnu = (2.22152e-30)*(n/nfac)*(p-1)*nu_p*((nuemit/nu_p)**(-alpha))*(Gimin-Gimax)
    jnu = (n/nfac)*(p-1)*nu_p*((nuemit/nu_p)**(-alpha))*(Gimin-Gimax)
    
    return jnu       
  

# power law emissivity function from presaved data
datafile = pkg_resources.resource_stream(__name__, 'synchpl_gxfit.csv')
synchpldat = np.loadtxt(datafile, dtype=float,delimiter=',')
#synchpldat = np.loadtxt(KGEOPATH+'/synchpl_gxfit.csv',dtype=float,delimiter=',')
pvals = np.unique(synchpldat[:,0])
logxvals = np.unique(synchpldat[:,1])
gxvals = synchpldat[:,2].reshape((len(pvals), len(logxvals)))
gxinterp = RegularGridInterpolator((pvals,logxvals),gxvals)
del gxvals
def GIfunc(p,x):
    if not isinstance(x, np.ndarray): x = np.array([x]).flatten()
    
    if not(np.min(pvals)<p<np.max(pvals)):
        raise Exception("p=%f is out of bounds in GIfunc!"%p)
    
    logx = np.log10(x)
    
    out = np.empty(x.shape)
    outmin = (2**(0.5*p-1.5))*(p+7./3.)*sp.gamma(0.25*p + 7./12.)*sp.gamma(0.25*p - 1./12.)/(p+1)
    outmax = 0
    
    out[logx<np.min(logxvals)]=outmin
    out[logx>np.max(logxvals)]=outmax  
    
    interpmask = (logx>=np.min(logxvals)) * (logx<=np.max(logxvals))
    out[interpmask]=gxinterp((p,logx[interpmask]))

    return out
    
        
 
    
          
