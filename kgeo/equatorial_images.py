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
from kgeo.bfields import Bfield
from kgeo.velocities import Velocity
from kgeo.emissivities import Emissivity

bfield_default = Bfield('rad')
vel_default = Velocity('zamo')
emis_default = Emissivity('bpl')
SPECIND = 1 # default (negative) spectral index
DISKANGLE = 0.1 # default h/r for equatorial disk

def make_image(a, r_o, th_o, mbar_max, alpha_min, alpha_max, beta_min, beta_max, psize,
               nmax_only=False,
               emissivity=emis_default,
               bfield=bfield_default,
               velocity=vel_default, 
               polarization=False, 
               pathlength=False,
               specind=SPECIND,
               diskangle=DISKANGLE):
    """computes an image in range (alpha_min, alpha_max) x (beta_min, beta_max)
      for all orders of m up to mbar_max
      and pixel size psize"""

    #checks
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not (isinstance(r_o,float) and (r_o>=100)):
        raise Exception("r_o should be a float > 100")
    if not (isinstance(th_o,float) and (0<th_o<np.pi) and th_o!=0.5*np.pi):
        raise Exception("th_o should be a float in range (0,pi/2) or (pi/2,pi)")
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
    outarr_Q = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_U = np.zeros((len(alpha_arr), mbar_max+1))    
    outarr_r = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_t = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_g = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_sinthb = np.zeros((len(alpha_arr), mbar_max+1))
    outarr_n = np.zeros((len(alpha_arr)))
    outarr_np = np.zeros((len(alpha_arr)))
    
    if pathlength:
        outarr_lp = np.zeros((len(alpha_arr), mbar_max+1))
        outarr_Ie = np.zeros((len(alpha_arr), mbar_max+1))
        
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
                  
            imdat = Iobs(a, r_o, th_o, mbar, 
                          alpha_arr, beta_arr, 
                          emissivity=emissivity,
                          velocity=velocity,
                          bfield=bfield,
                          polarization=polarization,
                          pathlength=pathlength,
                          specind=specind,
                          diskangle=diskangle)
        
            outarr_I[:,mbar] = imdat[0]
            outarr_Q[:,mbar] = imdat[1]
            outarr_U[:,mbar] = imdat[2] 
            outarr_g[:,mbar] = imdat[3]       
            outarr_r[:,mbar] = imdat[4]
            outarr_sinthb[:,mbar] = imdat[5]            
            outarr_t[:,mbar] = imdat[6]

            outarr_n = imdat[8] # TODO
            if pathlength:
                outarr_lp[:,mbar] = imdat[9]
                outarr_Ie[:,mbar] = imdat[10]
                
                outdat = (outarr_I, outarr_Q, outarr_U, outarr_r, outarr_t, outarr_g, outarr_sinthb, outarr_n, outarr_np, 
                          outarr_lp, outarr_Ie)
            else:
                outdat = (outarr_I, outarr_Q, outarr_U, outarr_r, outarr_t, outarr_g, outarr_sinthb, outarr_n, outarr_np)

            print('image %i...%0.2f s'%(mbar, time.time()-tstart))


    return outdat

def Iobs(a, r_o, th_o, mbar, alpha, beta, 
         emissivity=emis_default, velocity=vel_default, bfield=bfield_default,
         polarization=False, pathlength=False, specind=SPECIND, DISKANGLE=DISKANGLE, th_s=np.pi/2):
    """Return (Iobs, g, r_s, Ir, Imax, Nmax) where
       Iobs is Observed intensity for a ring of order mbar, GLM20 Eq 6
       g is the Doppler factor
       r_s is the equatorial emission radius
       Ir is the elapsed Mino time at emission
       Imax is the maximal Mino time on the geodesic
       Nmax is the *maximum* number of equatorial crossings"""

    # checks
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not (isinstance(r_o,float) and (r_o>=100)):
        raise Exception("r_o should be a float > 100")
    if not (isinstance(th_o,float) and (0<th_o<np.pi) and th_o!=0.5*np.pi):
        raise Exception("th_o should be a float in range (0,pi/2) or (pi/2,pi)")
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
    g = np.zeros(alpha.shape)
    sin_thb = np.zeros(alpha.shape)
    Iobs = np.zeros(alpha.shape)
    Qobs = np.zeros(alpha.shape)
    Uobs = np.zeros(alpha.shape)

    if pathlength:
        Ie = np.zeros(alpha.shape)
        lp = np.zeros(alpha.shape)     
                 
    # No emission if mbar is too large, if we are in voritcal region,
    # or if source radius is below the horizons
    zeromask = (Nmax<mbar) + (Nmax==-1) + (r_s <= rh)

    # manual cuts to emissivity (e.g. for ring model)
    if emissivity.emiscut_in > 0:
        zeromask = zeromask + (r_s < emissivity.emiscut_in)
    if emissivity.emiscut_out > 0:
        zeromask = zeromask + (r_s > emissivity.emiscut_out)    
        
    if np.any(~zeromask):

        ###############################
        # get momentum signs
        ###############################        
        kr_sign = radial_momentum_sign(a, th_o, alpha[~zeromask], beta[~zeromask], Ir[~zeromask], Imax[~zeromask])
        kth_sign = theta_momentum_sign(th_o, mbar)

        ###############################
        # get velocity and redshift
        ###############################        
        (u0,u1,u2,u3) = velocity.u_lab(a, r_s[~zeromask],th=th_s)    
        gg, lp = calc_redshift(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask], kr_sign, kth_sign, u0, u1, u2, u3, th=th_s)   
        g[~zeromask] = gg

        ###############################
        # get emissivity in local frame
        ###############################
        Iemis = emissivity.jrest(a, r_s[~zeromask])

        ###############################
        # get polarization quantities
        # if polarization not used, set sin(theta_b) = 1 everywhere
        ###############################
        if polarization:
            (sinthb, kappa) = calc_polquantities(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask],
                                                 kr_sign, kth_sign, u0, u1, u2, u3, 
                                                 bfield=bfield, th=th_s)
            (cos2chi, sin2chi) = calc_evpa(a, th_o, alpha[~zeromask], beta[~zeromask], kappa)
        else:
            sinthb = 1
        
        sin_thb[~zeromask] = sinthb   
                             
        ###############################
        # observed emission
        ###############################  
        if pathlength:
            llp = calc_pathlength(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask], kr_sign, kth_sign, u0, u1, u2, u3, th=th_s, diskangle=diskangle)
            Iobs[~zeromask] = (gg**3) * (gg**specind) * Iemis * (sinthb**(1+specind)) * np.abs(llp)
            Ie[~zeromask] = Iemis
            lp[~zeromask] = llp
                    
        else:       
            Iobs[~zeromask] = (gg**2) * (gg**specind) * Iemis * (sinthb**(1+specind))       

        if polarization:
            Qobs[~zeromask] = cos2chi*Iobs[~zeromask]
            Uobs[~zeromask] = sin2chi*Iobs[~zeromask]
    else:
        print("masked all pixels in Iobs! m=%i"%mbar)
    
    if pathlength:
        return (Iobs, Qobs, Uobs, g, r_s, sin_thb, Ir, Imax, Nmax, lp, Ie)    
    else:
        return (Iobs, Qobs, Uobs, g, r_s, sin_thb, Ir, Imax, Nmax)

def radial_momentum_sign(a, th_o, alpha, beta, Ir, Irmax):
    """Determine the sign of the radial component of the photon momentum"""

    # checks
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not (isinstance(th_o,float) and (0<th_o<np.pi) and th_o!=0.5*np.pi):
        raise Exception("th_o should be a float in range (0,pi/2) or (pi/2,pi)")

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
    if not (isinstance(th_o,float) and (0<th_o<np.pi) and th_o!=0.5*np.pi):
        raise Exception("th_o should be a float in range (0,pi/2) or (pi/2,pi)")
    if not (isinstance(mbar,int) and (mbar>=0)):
        raise Exception("mbar should be a integer >=0 ")
    

    if th_o < 0.5*np.pi:
        sign = -1*np.power(-1, np.mod(mbar,2))
    elif th_o > 0.5*np.pi:   
        sign = 1*np.power(-1, np.mod(mbar,2))
    return sign
             
def calc_redshift(a, r, lam, eta, kr_sign, kth_sign, u0, u1, u2, u3, th=np.pi/2):
    """ calculate redshift factor"""

    if not isinstance(lam, np.ndarray): lam = np.array([lam]).flatten()
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()
    if not isinstance(kr_sign, np.ndarray): kr_sign = np.array([kr_sign]).flatten()

    if not(len(lam)==len(eta)==len(r)==len(kr_sign)):
        raise Exception("g_grmhd_fit input arrays are different lengths!")
    

    # Metric
    a2 = a**2
    r2 = r**2
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * cth2
    
    # potentials/photon momentum  
    R = (r2 + a2 -a*lam)**2 - Delta*(eta + (lam-a)**2)
    TH = eta + a2*cth2 - lam*lam*cth2/sth2
    
    # redshift
    g = 1 / (1*u0 - lam*u3 - np.sign(kth_sign)*u2*np.sqrt(TH) - np.sign(kr_sign)*u1*np.sqrt(R)/Delta)
    
    return g

def calc_tetrades(a, r, lam, eta, kr_sign, kth_sign, u0, u1, u2, u3, th=np.pi/2):
    """ calculate tetrads for transformation to orthonormal frame"""

    if not isinstance(lam, np.ndarray): lam = np.array([lam]).flatten()
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()
    if not isinstance(kr_sign, np.ndarray): kr_sign = np.array([kr_sign]).flatten()
        
    if not(len(lam)==len(eta)==len(r)==len(kr_sign)):
        raise Exception("g_grmhd_fit input arrays are different lengths!")
    
            
    # Metric
    a2 = a**2
    r2 = r**2
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
    g03_up = -(2*a*r)/(Sigma*Delta)

    # covariant velocity
    u0_l = g00*u0 + g03*u3
    u1_l = g11*u1
    u2_l = g22*u2 
    u3_l = g33*u3 + g03*u0
    
    # define tetrads to comoving frame
    Nr = np.sqrt(-g11*(u0_l*u0 + u3_l*u3)*(1 + u2_l*u2))
    Nth = np.sqrt(g22*(1 + u2_l*u2))
    Nph = np.sqrt(-Delta*sth2*(u0_l*u0 + u3_l*u3))        

    e0_t = -u0
    e1_t = -u1
    e2_t = -u2
    e3_t = -u3
    
    e0_x = u1_l*u0/Nr
    e1_x = -(u0_l*u0 + u3_l*u3)/Nr
    e2_x = 0
    e3_x = u1_l*u3/Nr

    e0_y = u2_l*u0/Nth
    e1_y = u2_l*u1/Nth
    e2_y = (1+u2_l*u2)/Nth
    e3_y = u2_l*u3/Nth

    e0_z = u3_l/Nph
    e1_z = 0
    e2_z = 0
    e3_z = -u0_l/Nph
                                       
    # output tetrades
    tetrades = ((e0_t,e1_t,e2_t,e3_t),(e0_x,e1_x,e2_x,e3_x),(e0_y,e1_y,e2_y,e3_y),(e0_z,e1_z,e2_z,e3_z))
    
    return tetrades
    
def calc_polquantities(a, r, lam, eta, kr_sign, kth_sign, u0, u1, u2, u3, 
                       bfield=bfield_default, th=np.pi/2):
    """ calculate polarization quantities"""

    if not isinstance(lam, np.ndarray): lam = np.array([lam]).flatten()
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()
    if not isinstance(kr_sign, np.ndarray): kr_sign = np.array([kr_sign]).flatten()
        
    if not(len(lam)==len(eta)==len(r)==len(kr_sign)):
        raise Exception("g_grmhd_fit input arrays are different lengths!")
    
            
    # Metric
    a2 = a**2
    r2 = r**2
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
    g03_up = -(2*a*r)/(Sigma*Delta)

    # covarient velocity
    u0_l = g00*u0 + g03*u3
    u1_l = g11*u1
    u2_l = g22*u2 
    u3_l = g33*u3 + g03*u0
                
    # photon momentum
    R = (r2 + a2 -a*lam)**2 - Delta*(eta + (lam-a)**2)
    TH = eta + a2*cth2 - lam*lam*cth2/sth2
    
    k0_l = -1
    k1_l = kr_sign*np.sqrt(R)/Delta
    k2_l = kth_sign*np.sqrt(TH)
    k3_l = lam

    k0 = g00_up * k0_l + g03_up * k3_l
    k1 = g11_up * k1_l
    k2 = g22_up * k2_l
    k3 = g33_up * k3_l + g03_up * k0_l   
    
    # calculate tetrades
    tetrades = calc_tetrades(a, r, lam, eta, kr_sign, kth_sign, u0, u1, u2, u3, th=th)
    ((e0_t,e1_t,e2_t,e3_t),(e0_x,e1_x,e2_x,e3_x),(e0_y,e1_y,e2_y,e3_y),(e0_z,e1_z,e2_z,e3_z)) = tetrades 
                         
    # B-field defined in the lab frame: transform to fluid-frame quantities
    if bfield.fieldframe=='lab':
    
        # get lab frame B^i
        (B1, B2, B3) = bfield.bfield_lab(a, r, th=th)
         
        # here, we assume the field is degenerate and e^\mu = u_\nu F^{\mu\nu} = 0
        # (standard GRMHD assumption)
        b0 = B1*u1_l + B2*u2_l + B3*u3_l
        b1 = (B1 + b0*u1)/u0
        b2 = (B2 + b0*u2)/u0
        b3 = (B3 + b0*u3)/u0     

        b0_l = g00*b0 + g03*b3
        b1_l = g11*b1
        b2_l = g22*b2
        b3_l = g33*b3 + g03*b0
            
        bsq = b0*b0_l + b1*b1_l + b2*b2_l + b3*b3_l
        
        # transform to comoving frame with tetrads
        Bp_x = e0_x*b0_l + e1_x*b1_l  + e2_x*b2_l + e3_x*b3_l
        Bp_y = e0_y*b0_l + e1_y*b1_l  + e2_y*b2_l + e3_y*b3_l
        Bp_z = e0_z*b0_l + e1_z*b1_l  + e2_z*b2_l + e3_z*b3_l
    
 
    # B-field defined directly in comoving frame as in Gelles+2021
    elif bfield.fieldframe=='comoving':
        print('comoving!')
        (Bp_x, Bp_y, Bp_z) = bfield.bfield_comoving(a,r)
    
    else:
        raise Exception("bfield.fluidframe=%s not recognized!"%bfield.fieldframe)
        
    # comvoving frame magnitude    
    Bp_mag = np.sqrt(Bp_x**2 + Bp_y**2 + Bp_z**2)
    
    # wavevector in comoving frame  
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

def calc_pathlength(a, r, lam, eta, kr_sign, kth_sign, u0, u1, u2, u3, th=np.pi/2, diskangle=DISKANGLE):
    """ calculate rest frame path length through equatorial disk (Narayan+2021 eq 13)"""

    if not isinstance(lam, np.ndarray): lam = np.array([lam]).flatten()
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()
    if not isinstance(kr_sign, np.ndarray): kr_sign = np.array([kr_sign]).flatten()
        
    if not(len(lam)==len(eta)==len(r)==len(kr_sign)):
        raise Exception("g_grmhd_fit input arrays are different lengths!")
       
    # get the redshift g = 1/k_t     
    gg = calc_redshift(a, r, lam, eta, kr_sign, kth_sign, u0, u1, u2, u3, th=th)
    kthat = 1/gg
    
    # get the z component of local photon momentum
    # TODO this assumes that u^2 == 0!
    if (isinstance(u2,np.ndarray) and np.any(u2!=0)) or u2!=0:
        raise Exception("calc_pathlength assumes that u2==0!")
    if (isinstance(th,np.ndarray) and np.any(th!=np.pi/2)) or th!=np.pi/2:
        raise Exception("calc_pathlength assumes that th==pi/2!")    
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2
    a2 = a*a
    TH = eta + a2*cth2 - lam*lam*cth2/sth2
    k2_l = kth_sign*np.sqrt(TH)
    kzhat = -k2_l*r
    
    # get path length 
    diskheight = diskangle*r
    lp = diskheight * kthat / kzhat
         
    return lp
      
def calc_evpa(a, th_o, alpha, beta, kappa):    

    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not (isinstance(th_o,float) and (0<th_o<np.pi) and th_o!=0.5*np.pi):
        raise Exception("th_o should be a float in range (0,pi/2) or (pi/2,pi)")
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
