# Calculate image of an equatorial source
# Gralla, Lupsasca, Marrone and
# Gralla & Lupsasca 10 section VI C
# https://arxiv.org/pdf/1910.12873.pdf

import numpy as np
import scipy.special as sp
from tqdm import tqdm
from kerr_raytracing_utils import my_cbrt, radial_roots, mino_total, is_outside_crit, uplus_uminus, nmax_equatorial
from equatorial_lensing import r_equatorial
import time
# Fitting function parameters for emissivity and velocity
ELLISCO =1.; VRISCO = 2;
P1=6.; P2=2.; DD=0.2;  # from the simulation....
P1E=-2.; P2E=-.5; # for  230 GHz
#P1E=0; P2E=-.75;  # for 86 GHz


def make_image(a, r_o, th_o, mbar_max, alpha_min, alpha_max, beta_min, beta_max, psize, nmax_only=False):
    """computes an image in range (alpha_min, alpha_max) x (beta_min, beta_max)
      for all orders of m up to mbar_max
      and pixel size psize"""

    #checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    if not (isinstance(r_o,float) and (r_o>=100)):
        raise Exception("r_o should be a float >= 100")
    if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
        raise Exception("th_o should be a float in range (0,pi/2]")
    if not (isinstance(mbar_max,int) and (mbar_max>=0)):
        raise Exception("mbar_max should be an integer >=0!")

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
    outarr_r = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_t = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_g = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_n = np.zeros((len(alpha_arr)))

    if nmax_only:
        # maximum number of equatorial crossings
        print('calculating maximual number of equatorial crossings')
        tstart = time.time()
        outarr_n = nmax_equatorial(a, r_o, th_o, alpha_arr, beta_arr)
        print('done',time.time()-tstart)
    else:
        # loop over image order mbar
        for mbar in range(mbar_max+1):
            print('image %i...'%mbar, end="\r")
            tstart = time.time()
            (Ipix, g, r_s, Ir, Imax, Nmax) = Iobs(a, r_o, th_o, mbar, alpha_arr, beta_arr)
            outarr_I[:,mbar] = Ipix
            outarr_r[:,mbar] = r_s
            outarr_t[:,mbar] = Ir
            outarr_g[:,mbar] = g
            outarr_n = Nmax # TODO
            print('image %i...%0.2f s'%(mbar, time.time()-tstart))

    return (outarr_I, outarr_r, outarr_t, outarr_g, outarr_n)

def Iobs(a, r_o, th_o, mbar, alpha, beta):
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
    if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
        raise Exception("th_o should be a float in range (0,pi/2]")
    if not (isinstance(mbar,int) and (mbar>=0)):
        raise Exception("mbar should be an integer >=0!")

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
    g = np.empty(alpha.shape)
    Iobs = np.empty(alpha.shape)

    # No emission if mbar is too large, if we are in voritcal region,
    # or if source radius is below the horizons
    zeromask = (Nmax<mbar) + (Nmax==-1) + (r_s <= rh)
    g[zeromask] = 0
    Iobs[zeromask] = 0

    if np.any(~zeromask):

        # get emission in local frame
        #Iemis = emisGLM(a, r_s[~zeromask], gamma=-1.5)
        Iemis = emisP(a, r_s[~zeromask], p=P1E, p2=P2E)

        # get redshift
        sign = radial_momentum_sign(a, th_o, alpha[~zeromask], beta[~zeromask], Ir[~zeromask], Imax[~zeromask])
        #gg = g_zamo(a,r_s[~zeromask],lam[~zeromask]) # zero angular momentum
        #gg = g_kep(a,r_z[~zeromask],lam[~zeromask],eta[~zeromask],ur_sign=sign) # keplerian with infall inside
        #gg = g_subkep(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask], ur_sign=sign, fac_subkep=.75) #subkeplerian with infall insize
        gg = g_grmhd_fit(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask], sign)

        g[~zeromask] = gg

        # observed emission
        #Iobs[~zeromask] = Iemis * (gg**4)
        Iobs[~zeromask] = Iemis * (gg**3)
        #Iobs[~zeromask] = Iemis * (gg**2)

    return (Iobs, g, r_s, Ir, Imax, Nmax)

def radial_momentum_sign(a, th_o, alpha, beta, Ir, Irmax):
    """Determine the sign of the radial component of the photon momentum"""

    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
        raise Exception("th_o should be a float in range (0,pi/2]")

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

# def g_kep(a, r, lam, eta, ur_sign=1):
#     """redshift factor for material on keplerian orbits and infalling inside isco"""
#
#     # isco radius
#     z1 = 1 + np.cbrt(1-a**2)*(np.cbrt(1+a) + np.cbrt(1-a))
#     z2 = np.sqrt(3*a**2 + z1**2)
#     r_isco = 3 + z2 - np.sqrt((3-z1)*(3+z1+2*z2))
#
#     if r>= r_isco:
#         g = np.sqrt(r**3 - 3*r**2 + 2*a*r**1.5)/(r**1.5 + (a-lam))
#     else:
#         # isco conserved
#         gam_isco = np.sqrt(1-2./(3.*r_isco)) #nice expression only for keplerian isco
#         lam_isco = (r_isco**2 - 2*a*np.sqrt(r_isco) + a**2)/(r_isco**1.5 - 2*np.sqrt(r_isco) + a)
#
#         # preliminaries
#         Delta = (r**2 - 2*r + a**2)
#         H = (2*r - a*lam_isco)/Delta
#         R = (r**2 + a**2 -a*lam)**2 - Delta*(eta + (lam-a)**2)
#
#         # velocity components
#         ut = gam_isco*(1 + (2/r)*(1+H))
#         up = gam_isco*(lam_isco + a*H)/(r**2)
#         ur = -np.sqrt(2./(3*r_isco))*(r_isco/r - 1)**1.5
#
#         # redshift
#         ginv = ut - up*lam - np.sign(ur_sign)*ur*np.sqrt(R)/Delta
#         g = 1./ginv
#     return g
#
# def g_subkep(a, r, lam, eta, ur_sign=1, fac_subkep=1):
#     """redshift factor for sub-keplerian orbits"""
#
#     # isco radius
#     rh = 1 + np.sqrt(1-a**2)
#     z1 = 1 + np.cbrt(1-a**2)*(np.cbrt(1+a) + np.cbrt(1-a))
#     z2 = np.sqrt(3*a**2 + z1**2)
#     r_isco = 3 + z2 - np.sqrt((3-z1)*(3+z1+2*z2))
#
#     # Metric
#     a2 = a**2
#     r2 = r**2
#     th = np.pi/2. # equatorial
#     cth2 = np.cos(th)**2
#     sth2 = np.sin(th)**2
#     Delta = r2 - 2*r + a2
#     Sigma = r2 + a2 * cth2
#
#     g00 = -(1 - 2*r/Sigma)
#     g11 = Sigma/Delta
#     g22 = Sigma
#     g33 = (r2 + a2 + 2*r*a2*sth2 / Sigma) * sth2
#     g03 = -2*r*a*sth2 / Sigma
#
#     g00_up = -(r2 + a2 + 2*r*a2*sth2/Sigma) / Delta
#     g11_up = Delta/Sigma
#     g22_up = 1./Sigma
#     g33_up = (Delta - a2*sth2)/(Sigma*Delta*sth2)
#     g03_up = -(2*r*a)/(Sigma*Delta)
#
#     # angular momentum u_phi / |u_t|
#     #print(r,r_isco,rh)
#     if(r>=r_isco):
#         #print('r>=isco')
#
#         ell = fac_subkep * (r**2 - 2*a*np.sqrt(r) + a**2)/(r**1.5 - 2*np.sqrt(r) + a)
#         vr = 0
#         gam = np.sqrt(-1./(g00_up + g11_up*vr*vr + g33_up*ell*ell - 2*g03_up*ell))
#
#         # compute u_t
#         u_0 = -gam
#         u_1 = vr*gam
#         u_3 = ell*gam
#
#         # raise indices
#         u0 = g00_up*u_0 + g03_up*u_3
#         u1 = g11_up*u_1
#         u3 = g33_up*u_3 + g03_up*u_0
#
#         # Redshift
#         R = (r**2 + a**2 -a*lam)**2 - Delta*(eta + (lam-a)**2)
#         g = 1 / (1*u0 - lam*u3 - np.sign(ur_sign)*u1*np.sqrt(R)/Delta)
#
#
#     else:
#
#         #print('r<isco')
#         # isco conserved quantities (equatorial, Sigma=r^2, th=Pi/2)
#         g00_up_isco = -(r_isco**2 + a**2 + 2*a**2/r_isco) / (r_isco**2 - 2*r_isco + a**2)
#         g11_up_isco = (r_isco**2 - 2*r_isco + a2)/(r_isco**2 + a2 * cth2)
#         g33_up_isco = (r_isco - 2)/(r_isco**3 - 2*r_isco**2 + a**2*r_isco)
#         g03_up_isco = -(2*a)/(r_isco**3 - 2*r_isco**2 + a**2*r_isco)
#
#         ell_isco = fac_subkep * (r_isco**2 - 2*a*np.sqrt(r_isco) + a**2)/(r_isco**1.5 - 2*np.sqrt(r_isco) + a)
#         vr_isco = 0
#
#         gam_isco = np.sqrt(-1./(g00_up_isco + g11_up_isco*vr_isco*vr_isco + g33_up_isco*ell_isco*ell_isco - 2*g03_up_isco*ell_isco))
#
#         # 4-velocity components at non-isco radius with conserved ell, gam
#         u_0 = -gam_isco
#         u_3 = ell_isco*gam_isco
#         u_1 = -np.sqrt((-1 - (g00_up*u_0*u_0 + g33_up*u_3*u_3 + 2*g03_up*u_3*u_0)) / g11_up)
#
#         # raise indices
#         u0 = g00_up*u_0 + g03_up*u_3
#         u1 = g11_up*u_1
#         u3 = g33_up*u_3 + g03_up*u_0
#
#         # Redshift
#         R = (r**2 + a**2 -a*lam)**2 - Delta*(eta + (lam-a)**2)
#         g = 1 / (1*u0 - lam*u3 - np.sign(ur_sign)*u1*np.sqrt(R)/Delta)
#
#
#     return g
#
def g_grmhd_fit(a, r, lam, eta, ur_sign):
    """redshift factor for power laws fit to grmhd ell, conserved inside isco
       should be timelike throughout equatorial plane
    """
    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")

    if not isinstance(lam, np.ndarray): lam = np.array([lam]).flatten()
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()
    if not isinstance(ur_sign, np.ndarray): ur_sign = np.array([ur_sign]).flatten()

    if not(len(lam)==len(eta)==len(r)==len(ur_sign)):
        raise Exception("g_grmhd_fit input arrays are different lengths!")

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

    # Redshift
    R = (r**2 + a**2 -a*lam)**2 - Delta*(eta + (lam-a)**2)
    g = 1 / (1*u0 - lam*u3 - np.sign(ur_sign)*u1*np.sqrt(R)/Delta)

    return g

def g_zamo(a, r, lam):
    """redshift factor for zero angular momentum frame"""
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

    # Redshift
    g = -1 / (-1*u0 + lam*u3)
    return g

def emisGLM(a, r, gamma=0.):
    """emissivity at radius r from GLM paper"""

    # GLM model
    mu = 1 - np.sqrt(1-a**2)
    sig = 0.5
    emis = np.exp(-0.5*(gamma+np.arcsinh((r-mu)/sig))**2) / np.sqrt((r-mu)**2 + sig**2)

    return emis

def emisP(a, r, p=-2., p2=-.5):
    """emissivity at radius r - broken power law model fit to GRMHD"""

    rh = 1 + np.sqrt(1-a**2)
    emis = np.exp(p*np.log(r/rh) + p2*np.log(r/rh)**2)

    return emis
