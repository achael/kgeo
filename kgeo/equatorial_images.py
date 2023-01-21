# Calculate image of an equatorial source
# Gralla, Lupsasca, Marrone and
# Gralla & Lupsasca 10 section VI C
# https://arxiv.org/pdf/1910.12873.pdf

import numpy as np
import scipy.special as sp
from tqdm import tqdm
from kgeo.kerr_raytracing_utils import my_cbrt, radial_roots, mino_total, is_outside_crit, uplus_uminus
from kgeo.equatorial_lensing import r_equatorial, nmax_equatorial, nmax_poloidal
import time
from .bfields import Bfield_simple, Bfield_BZmonopole, Bfield_BZmagic

VELS = ['simfit','cunningham','zamo','subkep','cunningham_subkep','kep']

# Fitting function parameters for emissivity and velocity
ELLISCO =1.; VRISCO = 2;
P1=6.; P2=2.; DD=0.2;  # from the simulation....
P1E=-2.; P2E=-.5; # for  230 GHz
#P1E=0; P2E=-.75;  # for 86 GHz

FAC_SUBKEP = 0.7 # default subkeplerian factor
SPECIND = 1 # (negative) spectral index
BFIELDS = ['rad','vert','tor','bz_monopole','bz_guess']

def make_image(a, r_o, th_o, mbar_max, alpha_min, alpha_max, beta_min, beta_max, psize,
               whichvel='simfit',nmax_only=False, polarization=False, 
               specind=SPECIND, whichb='bzmonopole'):
    """computes an image in range (alpha_min, alpha_max) x (beta_min, beta_max)
      for all orders of m up to mbar_max
      and pixel size psize"""

    #checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    if not (isinstance(r_o,float) and (r_o>=100)):
        raise Exception("r_o should be a float >= 100")
    #if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
    #    raise Exception("th_o should be a float in range (0,pi/2]")
    if not (isinstance(mbar_max,int) and (mbar_max>=0)):
        raise Exception("mbar_max should be an integer >=0!")
    if not whichvel in VELS:
        raise Exception("whichvel not recognized") 
    if not whichb in BFIELDS:
        raise Exception("whichb not recognized") 
            
    # determine pixel grid
    n_alpha = int(np.floor(alpha_max - alpha_min)/psize)
    alphas = np.linspace(alpha_min, alpha_min+n_alpha*psize, n_alpha)
    n_beta = int(np.floor(beta_max - beta_min)/psize)
    betas = np.linspace(beta_min, beta_min+n_beta*psize, n_beta)

    alpha_arr, beta_arr = np.meshgrid(alphas, betas)
    alpha_arr = alpha_arr.flatten()
    beta_arr = beta_arr.flatten()

    # create output arrays
    outarr_I = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_Q = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_U = np.zeros((len(alpha_arr), mbar_max+1))    
    outarr_r = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_t = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_g = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_n = np.zeros((len(alpha_arr)))
    outarr_np = np.zeros((len(alpha_arr)))
    
    if nmax_only:
        # maximum number of equatorial crossings
        print('calculating maximal number of equatorial crossings')
        tstart = time.time()
        outarr_n = nmax_equatorial(a, r_o, th_o, alpha_arr, beta_arr)
        outarr_np = nmax_poloidal(a, r_o, th_o, alpha_arr, beta_arr)
        
        print('done',time.time()-tstart)
    else:
        outarr_np = nmax_poloidal(a, r_o, th_o, alpha_arr, beta_arr) # TODO
            
        # loop over image order mbar
        for mbar in range(mbar_max+1):
            print('image %i...'%mbar, end="\r")
            tstart = time.time()
                  
            (Ipix, Qpix, Upix, g, r_s, Ir, Imax, Nmax) = Iobs(a, r_o, th_o, mbar, 
                                                              alpha_arr, beta_arr, 
                                                              whichvel=whichvel,polarization=polarization,
                                                              specind=specind, whichb=whichb)
            outarr_I[:,mbar] = Ipix
            outarr_Q[:,mbar] = Qpix
            outarr_U[:,mbar] = Upix            
            outarr_r[:,mbar] = r_s
            outarr_t[:,mbar] = Ir
            outarr_g[:,mbar] = g
            outarr_n = Nmax # TODO


            print('image %i...%0.2f s'%(mbar, time.time()-tstart))


    return (outarr_I, outarr_Q, outarr_U, outarr_r, outarr_t, outarr_g, outarr_n, outarr_np)

def Iobs(a, r_o, th_o, mbar, alpha, beta, whichvel='simfit',whichb='bzmonopole',polarization=False,specind=SPECIND):
    """Return (Iobs, g, r_s, Ir, Imax, Nmax) where
       Iobs is Observed intensity for a ring of order mbar, GLM20 Eq 6
       g is the Doppler factor
       r_s is the equatorial emission radius
       Ir is the elapsed Mino time at emission
       Imax is the maximal Mino time on the geodesic
       Nmax is the *maximum* number of equatorial crossings"""

    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    if not (isinstance(r_o,float) and (r_o>=100)):
        raise Exception("r_o should be a float > 100")
    #if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
    #    raise Exception("th_o should be a float in range (0,pi/2]")
    if not (isinstance(mbar,int) and (mbar>=0)):
        raise Exception("mbar should be an integer >=0!")
    if not whichvel in VELS:
        raise Exception("whichvel not recognized") 
    if not whichb in BFIELDS:
        raise Exception("whichb not recognized") 
                
    if not isinstance(alpha, np.ndarray): alpha = np.array([alpha]).flatten()
    if not isinstance(beta, np.ndarray): beta = np.array([beta]).flatten()
    if len(alpha) != len(beta):
        raise Exception("alpha, beta are different lengths!")

    # horizon radius
    rh = 1 + np.sqrt(1-a**2)

    # conserved quantities
    lam = -alpha*np.sin(th_o)
    eta = (alpha**2 - a**2)*np.cos(th_o)**2 + beta**2

    # emission radius & Mino time
    r_s, Ir, Imax, Nmax = r_equatorial(a, r_o, th_o, mbar, alpha, beta)

    # output arrays
    g = np.zeros(alpha.shape)
    Iobs = np.zeros(alpha.shape)
    Qobs = np.zeros(alpha.shape)
    Uobs = np.zeros(alpha.shape)
        
    # No emission if mbar is too large, if we are in voritcal region,
    # or if source radius is below the horizons
    zeromask = (Nmax<mbar) + (Nmax==-1) + (r_s <= rh)

    
    if np.any(~zeromask):
   
        ###################
        # get velocity and redshift
        ###################        
        kr_sign = radial_momentum_sign(a, th_o, alpha[~zeromask], beta[~zeromask], Ir[~zeromask], Imax[~zeromask])
        kth_sign = theta_momentum_sign(th_o, mbar)
        
        # zero angular momentum  
        if whichvel=='zamo':      
            (u0,u1,u2,u3) = u_zamo(a,r_s[~zeromask])

        # keplerian with infall inside        
        elif whichvel=='cunningham' or whichvel=='kep':
            (u0,u1,u2,u3) = u_kep(a,r_s[~zeromask]) 
        
        #subkeplerian with infall inside
        elif whichvel=='cunningham_subkep' or whichvel=='subkep':
            (u0,u1,u2,u3) = u_subkep(a,r_s[~zeromask], fac_subkep=FAC_SUBKEP) 

        # fit to grmhd data
        elif whichvel=='simfit':
            (u0,u1,u2,u3) = u_grmhd_fit(a, r_s[~zeromask])

        else:
            raise Exception("whichvel must be 'simfit', 'cunningham' or 'zamo'!") 


        gg = calc_redshift(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask], kr_sign, u0, u1, u2, u3)   
        g[~zeromask] = gg

        ###################
        # get emissivity in local frame
        ###################
                                        
        #Iemis = emisGLM(a, r_s[~zeromask], gamma=-1.5)
        Iemis = emisP(a, r_s[~zeromask], p=P1E, p2=P2E)

        ###################
        # get polarization quantities
        ###################
        if polarization:
            (sinthb, kappa) = calc_polquantities(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask],
                                                 kr_sign, kth_sign, u0, u1, u2, u3, whichb=whichb)
            (cos2chi, sin2chi) = calc_evpa(a, th_o, alpha[~zeromask], beta[~zeromask], kappa)
        else:
            sinthb = 1
                        
        ###################
        # observed emission
        ###################         

        Iobs[~zeromask] = gg**2 * (Iemis * gg**specind * (sinthb**(1+specind)))       
        #Iobs[~zeromask] = Iemis * (gg**3)
        #Iobs[~zeromask] = Iemis * (gg**2)

        if polarization:
            Qobs[~zeromask] = cos2chi*Iobs[~zeromask]
            Uobs[~zeromask] = sin2chi*Iobs[~zeromask]

    return (Iobs, Qobs, Uobs, g, r_s, Ir, Imax, Nmax)



def radial_momentum_sign(a, th_o, alpha, beta, Ir, Irmax):
    """Determine the sign of the radial component of the photon momentum"""

    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    #if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
    #    raise Exception("th_o should be a float in range (0,pi/2]")

    if not isinstance(alpha, np.ndarray): alpha = np.array([alpha]).flatten()
    if not isinstance(beta, np.ndarray): beta = np.array([beta]).flatten()
    if len(alpha) != len(beta):
        raise Exception("alpha, beta are different lengths!")

    outside_crit = is_outside_crit(a, th_o, alpha, beta).astype(bool)
    inside_crit = ~outside_crit

    sign = np.empty(alpha.shape)
    outgoing_mask = inside_crit + (Ir <= 0.5*Irmax)
    sign[outgoing_mask] = 1
    sign[~outgoing_mask] = -1

    return sign

def theta_momentum_sign(th_o, mbar):
    """Determine the sign of the theta component of the photon momentum
       TODO: this works for equatorial crossings. do this based on mino time instead?"""
    
    # checks
    #if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
    #    raise Exception("th_o should be a float in range (0,pi/2]")
    if not (isinstance(mbar,int) and (mbar>=0)):
        raise Exception("mbar should be a integer >=0 ")
    
    if th_o < 0:
        sign = np.power(-1, np.mod(mbar,2))
    elif th_o > 0:   
        sign = -1*np.power(-1, np.mod(mbar,2))
    return sign
             
def calc_redshift(a, r, lam, eta, kr_sign, u0, u1, u2, u3):
    """ calculate redshift factor"""

    if not isinstance(lam, np.ndarray): lam = np.array([lam]).flatten()
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()
    if not isinstance(kr_sign, np.ndarray): kr_sign = np.array([kr_sign]).flatten()

    if not(len(lam)==len(eta)==len(r)==len(kr_sign)):
        raise Exception("g_grmhd_fit input arrays are different lengths!")
    
    if u2!=0:
        raise Exception("calc_redshift currently only works for u2=0!")
            
    Delta = r**2 - 2*r + a**2
    R = (r**2 + a**2 -a*lam)**2 - Delta*(eta + (lam-a)**2)
    g = 1 / (1*u0 - lam*u3 - np.sign(kr_sign)*u1*np.sqrt(R)/Delta)
    
    return g
    
def u_zamo(a, r):
    """velocity for zero angular momentum frame"""
    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()
    
    # Metric
    th = np.pi/2. # equatorial
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

def u_grmhd_fit(a, r):
    """velocity for power laws fit to grmhd ell, conserved inside isco
       should be timelike throughout equatorial plane
    """
    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

    # isco radius
    rh = 1 + np.sqrt(1-a**2)
    z1 = 1 + np.cbrt(1-a**2)*(np.cbrt(1+a) + np.cbrt(1-a))
    z2 = np.sqrt(3*a**2 + z1**2)
    r_isco = 3 + z2 - np.sqrt((3-z1)*(3+z1+2*z2))

    # Metric
    a2 = a**2
    r2 = r**2
    th = np.pi/2. # equatorial
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
    ell = ELLISCO*(r/r_isco)**.5 # defined positive
    vr = -VRISCO*((r/r_isco)**(-P1)) * (0.5*(1+(r/r_isco)**(1/DD)))**((P1-P2)*DD)
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
       
def u_kep(a, r):
    """Cunningham velocity for material on keplerian orbits and infalling inside isco"""

    # isco radius
    z1 = 1 + np.cbrt(1-a**2)*(np.cbrt(1+a) + np.cbrt(1-a))
    z2 = np.sqrt(3*a**2 + z1**2)
    r_isco = 3 + z2 - np.sqrt((3-z1)*(3+z1+2*z2))

    u0 = np.zeros(r.shape)
    u1 = np.zeros(r.shape)
    u3 = np.zeros(r.shape)
    
    g = np.empty(r.shape)
    iscomask = (r >= r_isco)
    
    # outside isco
    if np.any(iscomask):
        rr = r[iscomask]

        
        Omega = np.sign(a) / (rr**1.5 + np.abs(a))
        u0[iscomask] = (rr**1.5 + np.abs(a)) / np.sqrt(rr**3 - 3*rr**2 + 2*np.abs(a)*rr**1.5)
        u3[iscomask] = Omega*u0[iscomask]

    
    # inside isco
    if np.any(~iscomask):
        rr = r[~iscomask]
         
        # isco conserved quantities
        gam_isco = np.sqrt(1-2./(3.*r_isco)) #nice expression only for keplerian isco
        lam_isco = (r_isco**2 - 2*a*np.sqrt(r_isco) + a**2)/(r_isco**1.5 - 2*np.sqrt(r_isco) + a)

        # preliminaries
        Delta = (rr**2 - 2*rr + a**2)
        H = (2*rr - a*lam_isco)/Delta


        # velocity components
        u0[~iscomask] = gam_isco*(1 + (2/rr)*(1+H))
        u3[~iscomask] = gam_isco*(lam_isco + a*H)/(rr**2)
        u1[~iscomask] = -np.sqrt(2./(3*r_isco))*(r_isco/rr - 1)**1.5

        
    return (u0, u1, 0, u3)

def u_subkep(a, r, fac_subkep=1):
    """(sub) keplerian velocty and infalling inside isco"""

    # isco radius
    z1 = 1 + np.cbrt(1-a**2)*(np.cbrt(1+a) + np.cbrt(1-a))
    z2 = np.sqrt(3*a**2 + z1**2)
    r_isco = 3 + z2 - np.sqrt((3-z1)*(3+z1+2*z2))

    # Metric
    a2 = a**2
    r2 = r**2
    th = np.pi/2. # equatorial
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
    iscomask = (r >= r_isco)

    # angular momentum u_phi / |u_t|
    if np.any(iscomask):
        rr = r[iscomask]
                
        ell = fac_subkep * (rr**2 + a**2 - 2*a*np.sqrt(rr))/(rr**1.5 - 2*np.sqrt(rr) + a)
        gam = np.sqrt(-1./(g00_up[iscomask] + g33_up[iscomask]*ell*ell - 2*g03_up[iscomask]*ell))

        # compute u_cov
        u_0 = -gam
        u_3 = ell*gam

        # raise indices
        u0[iscomask] = g00_up[iscomask]*u_0 + g03_up[iscomask]*u_3
        u3[iscomask] = g33_up[iscomask]*u_3 + g03_up[iscomask]*u_0

    # inside isco
    if np.any(~iscomask):
        rr = r[~iscomask]

        g00_up_isco = -(r_isco**2 + a2 + 2*a2/r_isco) / (r_isco**2 - 2*r_isco + a2)
        g33_up_isco = (r_isco - 2)/(r_isco**3 - 2*r_isco**2 + a2*r_isco) 
        g03_up_isco = -(2*a)/(r_isco**3 - 2*r_isco**2 + a2*r_isco)

         
        # isco conserved quantities
        ell_isco = fac_subkep * (r_isco**2 + a**2 - 2*a*np.sqrt(r_isco))/(r_isco**1.5 - 2*np.sqrt(r_isco) + a)
        gam_isco = np.sqrt(-1./(g00_up_isco + g33_up_isco*ell_isco*ell_isco - 2*g03_up_isco*ell_isco))

        # covariant velocity with conserved ell, gam at non-isco radius
        u_0 = -gam_isco
        u_3 = ell_isco*gam_isco
        u_1 = -1 * np.sqrt((-1 - (g00_up[~iscomask]*u_0*u_0 + g33_up[~iscomask]*u_3*u_3 + 2*g03_up[~iscomask]*u_3*u_0)) / g11_up[~iscomask])

        # raise indices
        u0[~iscomask] = g00_up[~iscomask]*u_0 + g03_up[~iscomask]*u_3
        u1[~iscomask] = g11_up[~iscomask]*u_1
        u3[~iscomask] = g33_up[~iscomask]*u_3 + g03_up[~iscomask]*u_0

    return (u0, u1, 0, u3)        

def emisP(a, r, p=-2., p2=-.5):
    """emissivity at radius r - broken power law model fit to GRMHD"""

    rh = 1 + np.sqrt(1-a**2)
    emis = np.exp(p*np.log(r/rh) + p2*np.log(r/rh)**2)

    return emis
    
    
def emisGLM(a, r, gamma=0.):
    """emissivity at radius r from GLM paper"""

    # GLM model
    mu = 1 - np.sqrt(1-a**2)
    sig = 0.5
    emis = np.exp(-0.5*(gamma+np.arcsinh((r-mu)/sig))**2) / np.sqrt((r-mu)**2 + sig**2)
    
    
def calc_polquantities(a, r, lam, eta, kr_sign, kth_sign, u0, u1, u2, u3, whichb='bzmonopole'):
    """ calculate polarization quantities
        everything assumes u^2 = 0 for now"""

    if not isinstance(lam, np.ndarray): lam = np.array([lam]).flatten()
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()
    if not isinstance(kr_sign, np.ndarray): kr_sign = np.array([kr_sign]).flatten()
    if not whichb in BFIELDS:
        raise Exception("whichb not recognized") 
        
    if not(len(lam)==len(eta)==len(r)==len(kr_sign)):
        raise Exception("g_grmhd_fit input arrays are different lengths!")
    
    if u2!=0:
        raise Exception("calc_redshift currently only works for u2=0!")
            
    # Metric
    a2 = a**2
    r2 = r**2
    th = np.pi/2. # equatorial
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * cth2

    g00 = -(1 - 2*r/Sigma)
    g11 = Sigma/Delta
    g22 = Sigma
    g33 = (r2 + a2 + 2*r*a2*sth2 / Sigma) * sth2
    g03 = -2*r*a*sth2 / Sigma

    # photon momentum
    R = (r2 + a2 -a*lam)**2 - Delta*(eta + (lam-a)**2)
    TH = eta + a2*cth2 - lam*lam*cth2/sth2
    
    k0_l = -1
    k1_l = kr_sign*np.sqrt(R)/Delta
    k2_l = kth_sign*np.sqrt(TH)
    k3_l = lam
    
    k0 = ((r2 + a2)*(r2 + a2 - a*lam)/Delta + a*(lam-a*sth2))/Sigma
    k1 = kr_sign*np.sqrt(R)/Sigma
    k2 = kth_sign*np.sqrt(TH)/Sigma
    k3 = (a*(r2 + a2 - a*lam)/Delta + lam/sth2 -a)/Sigma
    
    # calculate magnetic field in the fluid frame
    if whichb=='bz_monopole':
        (B1, B2, B3) = Bfield_BZmonopole(a, r)  
    elif whichb=='bz_guess':
        (B1, B2, B3) = Bfield_BZmagic(a, r)            
    elif whichb=='rad':
        (B1, B2, B3) = Bfield_simple(a, r, (1,0,0))            
    elif whichb=='vert':
        (B1, B2, B3) = Bfield_simple(a, r, (0,1,0))            
    elif whichb=='tor':
        (B1, B2, B3) = Bfield_simple(a, r, (0,0,1))            
                    
    u0_l = g00*u0 + g03*u3
    u1_l = g11*u1
    u2_l = g22*u2 # should be zero!
    u3_l = g33*u3 + g03*u0
    
    b0 = B1*u1_l + B2*u2_l + B3*u3_l
    b1 = (B1 + b0*u1)/u0
    b2 = (B2 + b0*u2)/u0
    b3 = (B3 + b0*u3)/u0     

    b0_l = g00*b0 + g03*b3
    b1_l = g11*b1
    b2_l = g22*b2
    b3_l = g33*b3 + g03*b0
        
    bsq = b0*b0_l + b1*b1_l + b2*b2_l + b3*b3_l
    
    # transform to comoving frame
    Nr = np.sqrt(-g11*(u0_l*u0 + u3_l*u3))
    Nth = np.sqrt(g22*(1 + u2_l*u2))
    Nph = np.sqrt(-Delta*sth2*(u0_l*u0 + u3_l*u3))        
    
    e0_x = u1_l*u0/Nr
    e1_x = -(u0_l*u0 + u3_l*u3)/Nr
    e2_x = 0
    e3_x = u1_l*u3/Nr
    
    e0_y = u2_l*u0/Nth
    e1_y = u2_l*u1/Nth
    e2_y = (1+u2_l*u2)/Nth
    e3_y = u2_l*u3

    e0_z = u3_l/Nph
    e1_z = 0
    e2_z = 0
    e3_z = -u0_l/Nph
    
    Bp_x = e0_x*b0_l + e1_x*b1_l  + e2_x*b2_l + e3_x*b3_l
    Bp_y = e0_y*b0_l + e1_y*b1_l  + e2_y*b2_l + e3_y*b3_l
    Bp_z = e0_z*b0_l + e1_z*b1_l  + e2_z*b2_l + e3_z*b3_l
    Bp_mag = np.sqrt(Bp_x**2 + Bp_y**2 + Bp_z**2)
    
    kp_x = e0_x*k0_l + e1_x*k1_l  + e2_x*k2_l + e3_x*k3_l
    kp_y = e0_y*k0_l + e1_y*k1_l  + e2_y*k2_l + e3_y*k3_l
    kp_z = e0_z*k0_l + e1_z*k1_l  + e2_z*k2_l + e3_z*k3_l  
    kp_mag = np.sqrt(kp_x**2 + kp_y**2 + kp_z**2)
    
    # local polarization vector and emission angle
    f_x = (kp_y*Bp_z - kp_z*Bp_y)/kp_mag
    f_y = (kp_z*Bp_x - kp_x*Bp_z)/kp_mag
    f_z = (kp_x*Bp_y - kp_y*Bp_x)/kp_mag 
    sinthb = np.sqrt(f_x**2 + f_y**2 + f_z**2)/Bp_mag   
    
    # polarization four vector
    f0 = e0_x*f_x + e0_y*f_y + e0_z*f_z      
    f1 = e1_x*f_x + e1_y*f_y + e1_z*f_z      
    f2 = e2_x*f_x + e2_y*f_y + e2_z*f_z          
    f3 = e3_x*f_x + e3_y*f_y + e3_z*f_z      
    
    # penrose-walker
    A = k0*f1 - k1*f0 + a*sth2*(k1*f3 - k3*f1)
    B = ((r2+a2)*(k3*f2 - k2*f3) - a*(k0*f2 - k2*f0))*np.sin(th)
    kappa = (r - 1j*a*np.cos(th))*(A - 1j*B)

    return (sinthb, kappa)
    
def calc_evpa(a, th_o, alpha, beta, kappa):    

    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    #if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
    #    raise Exception("th_o should be a float in range (0,pi/2]")
    if not isinstance(alpha, np.ndarray): alpha = np.array([alpha]).flatten()
    if not isinstance(beta, np.ndarray): beta = np.array([beta]).flatten()
    if not isinstance(kappa, np.ndarray): kappa = np.array([kappa]).flatten()
    if len(alpha) != len(beta) != len(kappa):
        raise Exception("alpha, beta, kappa are different lengths in calc_QU")
        
    # parallel transport with penrose-walker
    mu = -(alpha + a*np.sin(th_o))
    kappa1 = np.real(kappa)
    kappa2 = np.imag(kappa)
    
    cos2chi = ((beta*kappa1 + mu*kappa2)**2 - (mu*kappa1-beta*kappa2)**2)/((beta**2 + mu**2)*(kappa1**2 + kappa2**2))
    sin2chi = (2*(beta*kappa1 + mu*kappa2)*(mu*kappa1-beta*kappa2))/((beta**2 + mu**2)*(kappa1**2 + kappa2**2))    
    
    return (cos2chi, sin2chi)

