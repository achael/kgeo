import numpy as np
import scipy.special as sp
from tqdm import tqdm
from kgeo.kerr_raytracing_utils import my_cbrt, radial_roots, mino_total, is_outside_crit, uplus_uminus
from kgeo.equatorial_lensing import r_equatorial, nmax_equatorial, nmax_poloidal
import time
from mpmath import polylog
from scipy.interpolate import UnivariateSpline
import os

RMAXINTERP = 100
NINTERP = 100000
def f(r):
    """BZ monopole f(r) function"""
    r = r + 0j
    out = (polylog(2,2./r) - np.log(1-2./r)*np.log(r/2.))*r*r*(2*r-3)/8.
    out += (1+3*r-6*r*r)*np.log(r/2.)/12.
    out += 11./72. + 1./(3.*r) + r/2. - r*r/2.
    return float(np.real(out))
    
# interpolate f(r)
datafile = os.path.join(os.path.dirname(__file__), 'bz_fr_data.dat')
(rsinterp,fsinterp) = np.loadtxt(datafile)
fi = UnivariateSpline(rsinterp, fsinterp, k=1,s=0, ext=0)
fiprime = fi.derivative()

def Bfield_simple(a, r, coeffs):
    """magnetic field vector in the lab frame in equatorial plane
       TODO: unclear if these definitions work at all spins"""

    # coefficients for each component
    (arad, avert, ator) = coeffs
    
    th = np.pi/2. # TODO equatorial plane only
    Sigma = r**2 + a**2 * np.cos(th)**2    
    gdet = np.abs(np.sin(th))*Sigma
    
    # b-field in lab
    Br  = avert*(r*np.cos(th))/gdet + arad*(np.sin(th)/gdet)
    Bth = avert*(-np.sin(th)/gdet)
    Bph = ator*(1./gdet)
   
    return (Br, Bth, Bph)
    
def Bfield_BZmonopole(a,r):
    """perturbative BZ monopole solution"""
    th = np.pi/2. # TODO equatorial plane only
    a2 = a**2
    r2 = r**2
    th = np.pi/2. # equatorial
    sth = np.sin(th)
    cth = np.cos(th)
    cth2 = cth**2
    sth2 = sth**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2*cth2
    gdet = sth*Sigma
    
    C = 1. 
    
    fr = fi(r)
    fr[r>RMAXINTERP] = 1./(4.*r[r>RMAXINTERP])
    frp = fiprime(r)
    frp[r>RMAXINTERP] = -1./(4.*r2[r>RMAXINTERP])
    
    phi = C*(1 - cth) + C*a2*fr*sth2*cth
    
    dphidtheta = C*sth*(1 + 0.5*a2*fr*(1 + 3*np.cos(2*th)))
    dphidr =  C*a2*sth2*cth*frp
    Br = dphidtheta / gdet
    Bth = -dphidr / gdet
        
    # TODO: sign of current? 
    f2 = (6*np.pi*np.pi - 49.)/72.
    omega2 = 1./32. - sth2*(4*f2 - 1)/64.
    I = -1*C*2*np.pi*sth2*(a/8. + (a**3)*(omega2 + 0.25*fr*cth2))
    Bph = I / (2*np.pi*Delta*sth2)
    

    return(Br, Bth, Bph)
 
def Bfield_BZmagic(a, r):
    """Guess ratio for Bphi/Br from split monopole"""



    th = np.pi/2. # TODO equatorial plane only
    a2 = a**2
    r2 = r**2
    th = np.pi/2. # equatorial
    sth = np.sin(th)
    cth = np.cos(th)
    cth2 = cth**2
    sth2 = sth**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2*cth2
    gdet = sth*Sigma
    Pi = (r2+a2)**2 - a2*Delta*sth2

    rH = 1 + np.sqrt(1-a2)
    OmegaH = a/(2*rH)
    
    # angular velocity guess from split Monopole
    omega0 = 1./8.
    f2 = (6*np.pi*np.pi - 49.)/72.
    omega2 = 1./32. - (4*f2 - 1)*sth2/64.
    Omega = a*omega0 + (a**3)*omega2

    # Magnetic field ratio Bphi/Br
    # sign? 
    Bratioguess = (OmegaH - Omega)*np.sqrt(Pi)/Delta
    
    C = 1. 
    Br = C/r2
    Bph = Bratioguess*Br
    Bth = 0*Br
    
    return(Br, Bth, Bph)
    
