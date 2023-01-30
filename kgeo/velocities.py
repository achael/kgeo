import numpy as np
import scipy.special as sp
from tqdm import tqdm
from kgeo.kerr_raytracing_utils import my_cbrt, radial_roots, mino_total, is_outside_crit, uplus_uminus
from kgeo.equatorial_lensing import r_equatorial, nmax_equatorial, nmax_poloidal
import time
from mpmath import polylog
from scipy.interpolate import UnivariateSpline
import os

# simulation fit factors
ELLISCO =1.; VRISCO = 2;
P1=6.; P2=2.; DD=0.2;  

# gelles model parameters
BETA = 0.3
CHI = -150*np.pi/180.

class Velocity(object):
    """ object for lab frame velocity as a function of r, only in equatorial plane for now """
    
    def __init__(self, veltype="kep", **kwargs):

        self.veltype = veltype
        self.kwargs = kwargs
             
        if self.veltype=='zamo':
            pass
            
        elif self.veltype=='kep' or self.veltype=='cunningham':
            self.retrograde = self.kwargs.get('retrograde', False)
            
        elif self.veltype=='subkep' or self.veltype=='cunningham_subkep':
            self.retrograde = self.kwargs.get('retrograde', False)
            self.fac_subkep = self.kwargs.get('fac_subkep', 1)
            
        elif self.veltype=='gelles':
            self.gelles_beta = self.kwargs.get('gelles_beta', BETA)
            self.gelles_chi = self.kwargs.get('gelles_chi', CHI)

        elif self.veltype=='simfit':
            self.ell_isco = self.kwargs.get('ell_isco', ELLISCO)
            self.vr_isco = self.kwargs.get('vr_isco', VRISCO)
            self.p1 = self.kwargs.get('p1', P1)
            self.p2 = self.kwargs.get('p2', P2)
            self.dd = self.kwargs.get('dd', DD)
                        
        else: 
            raise Exception("veltype %s not recognized in Velocity!"%self.veltype)
            
    def u_lab(self, a, r):  
        if self.veltype=='zamo':
            ucon = u_zamo(a, r)              
        elif self.veltype=='kep' or self.veltype=='cunningham':
            ucon = u_kep(a, r, retrograde=self.retrograde)         
        elif self.veltype=='subkep' or self.veltype=='cunningham_subkep':
            ucon = u_subkep(a, r, retrograde=self.retrograde, fac_subkep=self.fac_subkep)  
        elif self.veltype=='gelles':
            ucon = u_gelles(a, r, beta=self.gelles_beta, chi=self.gelles_chi)
        elif self.veltype=='simfit':                        
            ucon = u_simfit(ell_isco=ell_isco, vr_isco=vr_isco, p1=p1, p2=p2, dd=dd)               
        else: 
            raise Exception("veltype %s not recognized in Velocity.u_lab!"%self.veltype)
            
        return ucon
                                                              
def u_zamo(a, r):
    """velocity for zero angular momentum frame"""
    # checks
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()
    
    # Metric
    th = np.pi/2. # TODO equatorial only
    Delta = r**2 - 2*r + a**2
    Sigma = r**2 + a**2 * np.cos(th)**2
    g00 = -(1-2*r/Sigma)
    g11 = Sigma/Delta
    g22 = Sigma
    g33 = (r**2 + a**2 + 2*r*(a*np.sin(th))**2 / Sigma) * np.sin(th)**2
    g03 = -2*r*a*np.sin(th)**2 / Sigma

    # Velocity Components, fit to GRMHD
    v3 = -g03/g33

    # Compute u0
    aa = g00
    bb = 2*g03*v3
    cc = g33*v3*v3
    u0 = np.sqrt(-1./(aa + bb + cc))

    # Compute the 4-velocity (contravariant)
    u3 = u0*v3
    
    return (u0, 0, 0, u3)


       
def u_kep(a, r, retrograde=False):
    """Cunningham velocity for material on keplerian orbits and infalling inside isco"""
    # checks
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()
    
    if retrograde:
        s = -1
        raise Exception('use u_subkep for retrograde!')
    else:
        s = 1
        
    # isco radius
    z1 = 1 + np.cbrt(1-a**2)*(np.cbrt(1+a) + np.cbrt(1-a))
    z2 = np.sqrt(3*a**2 + z1**2)
    ri = 3 + z2 - s*np.sqrt((3-z1)*(3+z1+2*z2))
    print("r_isco: ",ri)
    
    u0 = np.zeros(r.shape)
    u1 = np.zeros(r.shape)
    u3 = np.zeros(r.shape)
    
    g = np.empty(r.shape)
    iscomask = (r >= ri)
    
    # outside isco
    if np.any(iscomask):
        rr = r[iscomask]

        spin = np.abs(a)
        asign = np.sign(a)
        
        Omega = asign*s / (rr**1.5 + s*spin)
        u0[iscomask] = (rr**1.5 + s*spin) / np.sqrt(rr**3 - 3*rr**2 + 2*s*spin*rr**1.5)
        u3[iscomask] = Omega*u0[iscomask]

    
    # inside isco
    if np.any(~iscomask):
        rr = r[~iscomask]

        spin = np.abs(a)
        asign = np.sign(a)
                 
        # isco conserved quantities
        gam_isco = np.sqrt(1 - 2./(3.*ri)) #nice expression only for keplerian isco
        lam_isco = s*asign*(ri**2 - s*2*spin*np.sqrt(ri) + a**2)/(ri**1.5 - 2*np.sqrt(ri) + s*spin)

        # preliminaries
        Delta = (rr**2 - 2*rr + a**2)
        H = (2*rr - a*lam_isco)/Delta # T/Delta - 1 in Bardeen eq 2.9
        T = H*Delta + 1 
        
        # velocity components
        u0[~iscomask] = gam_isco*(1 + (2/rr)*(1 + H))
        u1[~iscomask] = -np.sqrt(2./(3*ri))*(ri/rr - 1)**1.5 #?? not right for retrograde?
        u3[~iscomask] = gam_isco*(lam_isco + a*H)/(rr**2)
        
    return (u0, u1, 0, u3)

def u_subkep(a, r, fac_subkep=1, retrograde=False):
    """(sub) keplerian velocty and infalling inside isco"""
    # checks
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()
    
    if retrograde:
        s = -1
        print('retrograde!')
    else:
        s = 1
        
    # isco radius
    z1 = 1 + np.cbrt(1-a**2)*(np.cbrt(1+a) + np.cbrt(1-a))
    z2 = np.sqrt(3*a**2 + z1**2)
    ri = 3 + z2 - s*np.sqrt((3-z1)*(3+z1+2*z2))
    print("r_isco:", ri)
    
    # Metric
    a2 = a**2
    r2 = r**2
    th = np.pi/2. # TODO equatorial only
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * cth2

    g00 = -(1 - 2*r/Sigma)
    g11 = Sigma/Delta
    g22 = Sigma
    g33 = (r2 + a2 + 2*r*a2*sth2 / Sigma) * sth2
    g03 = -2*r*a*sth2 / Sigma

    g00_up = -(r2 + a2 + 2*r*a2*sth2/Sigma) / Delta
    g11_up = Delta/Sigma
    g22_up = 1./Sigma
    g33_up = (Delta - a2*sth2)/(Sigma*Delta*sth2)
    g03_up = -(2*r*a)/(Sigma*Delta)

    u0 = np.zeros(r.shape)
    u1 = np.zeros(r.shape)
    u3 = np.zeros(r.shape)

    g = np.empty(r.shape)
    iscomask = (r >= ri)

    # angular momentum ell = u_phi / -(u_t)
    if np.any(iscomask):
        rr = r[iscomask]
                
        spin = np.abs(a)
        asign = np.sign(a)
        ell = asign*fac_subkep*s * (rr**2 + s*spin**2 - s*2*spin*np.sqrt(rr))/(rr**1.5 - 2*np.sqrt(rr) + s*spin)
        gam = np.sqrt(-1./(g00_up[iscomask] + g33_up[iscomask]*ell*ell - 2*g03_up[iscomask]*ell))
        # keplerian case
        #gam =  (rr**1.5 - 2*np.sqrt(r) + s*spin) / np.sqrt(rr**3 - 3*r**2 + s*2*spin*rr**1.5) 
        
        # compute u_cov
        u_0 = -gam
        u_3 = ell*gam

        # raise indices
        u0[iscomask] = g00_up[iscomask]*u_0 + g03_up[iscomask]*u_3
        u3[iscomask] = g33_up[iscomask]*u_3 + g03_up[iscomask]*u_0

    # inside isco
    if np.any(~iscomask):
        rr = r[~iscomask]

        g00_up_isco = -(ri**2 + a2 + 2*a2/ri) / (ri**2 - 2*ri + a2)
        g33_up_isco = (ri - 2)/(ri**3 - 2*ri**2 + a2*ri) 
        g03_up_isco = -(2*a)/(ri**3 - 2*ri**2 + a2*ri)

         
        # isco conserved quantities
        spin = np.abs(a)        
        asign = np.sign(a)
        ell_isco = asign*fac_subkep*s * (ri**2 + s*spin**2 - s*2*spin*np.sqrt(ri))/(ri**1.5 - 2*np.sqrt(ri) + s*spin)
        gam_isco = np.sqrt(-1./(g00_up_isco + g33_up_isco*ell_isco*ell_isco - 2*g03_up_isco*ell_isco))
        print(ell_isco, gam_isco)
        # covariant velocity with conserved ell, gam at non-isco radius
        u_0 = -gam_isco
        u_3 = ell_isco*gam_isco
        u_1 = -1 * np.sqrt((-1 - (g00_up[~iscomask]*u_0*u_0 + g33_up[~iscomask]*u_3*u_3 + 2*g03_up[~iscomask]*u_3*u_0)) / g11_up[~iscomask])

        # raise indices
        u0[~iscomask] = g00_up[~iscomask]*u_0 + g03_up[~iscomask]*u_3
        u1[~iscomask] = g11_up[~iscomask]*u_1
        u3[~iscomask] = g33_up[~iscomask]*u_3 + g03_up[~iscomask]*u_0

    return (u0, u1, 0, u3)
  
def u_gelles(a, r, beta=0.3, chi=-150*(np.pi/180.)):
    """velocity prescription from Gelles+2021, Eq A4"""
    # checks
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()
    
    # Metric
    a2 = a**2
    r2 = r**2
    th = np.pi/2. # TODO equatorial only
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * cth2
    Xi = (r2 + a2)**2 - Delta*a2*sth2
    omegam = 2*a*r/Xi
    
    gamma = 1/np.sqrt(1-beta**2)
    coschi = np.cos(chi)
    sinchi = np.sin(chi)
    
    u0 = (gamma/r)*np.sqrt(Xi/Delta)
    u1 = (beta*gamma*coschi/r)*np.sqrt(Delta)
    u3 = (gamma*omegam/r)*np.sqrt(Xi/Delta) + (r*beta*gamma*sinchi)/np.sqrt(Xi)

    return (u0, u1, 0, u3)            
          
def u_grmhd_fit(a, r, ell_isco=ELLISCO, vr_isco=VRISCO, p1=P1, p2=P2, dd=DD):
    """velocity for power laws fit to grmhd ell, conserved inside isco
       should be timelike throughout equatorial plane
       might not work for all spins
    """
    # checks
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

    # isco radius
    rh = 1 + np.sqrt(1-a**2)
    z1 = 1 + np.cbrt(1-a**2)*(np.cbrt(1+a) + np.cbrt(1-a))
    z2 = np.sqrt(3*a**2 + z1**2)
    r_isco = 3 + z2 - np.sqrt((3-z1)*(3+z1+2*z2))

    # Metric
    a2 = a**2
    r2 = r**2
    th = np.pi/2. # TODO equatorial only
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * cth2

    g00_up = -(r2 + a2 + 2*r*a2*sth2/Sigma) / Delta
    g11_up = Delta/Sigma
    g22_up = 1./Sigma
    g33_up = (Delta - a2*sth2)/(Sigma*Delta*sth2)
    g03_up = -(2*r*a)/(Sigma*Delta)

    # Fitting function should work down to the horizon
    # u_phi/u_t fitting function
    ell = ell_isco*(r/r_isco)**.5 # defined positive
    vr = -vr_isco*((r/r_isco)**(-p1)) * (0.5*(1+(r/r_isco)**(1/dd)))**((p1-p2)*dd)
    gam = np.sqrt(-1./(g00_up + g11_up*vr*vr + g33_up*ell*ell - 2*g03_up*ell))

    # compute u_t
    u_0 = -gam
    u_1 = vr*gam
    u_3 = ell*gam

    # raise indices
    u0 = g00_up*u_0 + g03_up*u_3
    u1 = g11_up*u_1
    u3 = g33_up*u_3 + g03_up*u_0

    return (u0, u1, 0, u3)            
