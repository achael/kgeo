# Analytic raytracing from formalism in Gralla+Lupsasa 2019a,b
# 19a: https://arxiv.org/pdf/1910.12881.pdf
# 19b: https://arxiv.org/pdf/1910.12873.pdf

# TODO run in parallel for speedup?
# TODO what to do at spin limit (seems ok down to spin 1.e-6) ? upper spin limit?
#    - spin 0 should work now. spin=1 untested
# TODO what to do at exactly th_o = 0? th_o = pi/2?
#    - th_o=pi/2 should work now. th_o = 0 doesn't.
# TODO what to do if we have double roots r3==r4 / land exactly on the critical curve?
# TODO -- add image plane transformations for a<0, th_o>pi/2?

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

from kgeo.kerr_raytracing_utils import (
    EP,
    MINSPIN_R,
    MINSPIN_TH,
    Geodesics,
    angular_turning,
    check_inclination,
    check_spin,
    mino_total,
    my_sign,
    obs_trig,
    psi_o_amplitude,
    radial_roots,
    upsilon_o_amplitude,
)
from kgeo.scipy_ellip_binding import ellip_pi_scaled, ellipj_m1

SPIN = 0.94
INC = 20*np.pi/180.
ROUT = 100000. #4.e10 # sgra distance in M
NGEO = 250
NPIX = 1000
MAXTAUFRAC = (1. - 1.e-10) # NOTE: if we go exactly to tau_tot t and phi diverge on horizon

alpha_default = np.linspace(-6,6,NPIX)
beta_default = 0*alpha_default

# GSL elliptic functions
SCIPY = True
GSL = False
if SCIPY:
    from .scipy_ellip_binding import ellip_pi_arr
    ellippi_arr = ellip_pi_arr
    #sellippi_arr = np.vectorize(ellip_pi)
elif GSL:
    from .gsl_ellip_binding import ellip_pi_gsl
    ellippi_arr_gsl = np.frompyfunc(ellip_pi_gsl,3,1)
    ellippi_arr = ellippi_arr_gsl
else:
    raise Exception("no elliptic Pi option chosen!")

#TODO -- errors in phi raytracing with alpha=0, beta!=0.
def raytrace_ana(a=SPIN,
                 observer_coords = [0,ROUT,INC,0],
                 image_coords = [alpha_default, beta_default],
                 ngeo=NGEO, maxtaufrac=MAXTAUFRAC,
                 do_phi_and_t=True,
                 savedata=False, plotdata=True):
    """analytically raytrace over the full range of mino times"""
    tstart = time.time()

    [_, r_o, th_o, _] = observer_coords # assumes ph_o = 0
    [alpha, beta] = image_coords

    # checks
    check_spin(a)
    #if (a<0): print("WARNING a<0! Not fully verified!")
    if not (isinstance(r_o,float) and (r_o>=100)):
        raise Exception("r_o should be a float > 100")
    check_inclination(th_o)
    #if (th_o>np.pi/2.): print("WARNING th_o>pi/2! Not fully verified!")

    if not isinstance(alpha, np.ndarray): alpha = np.array([alpha]).flatten()
    if not isinstance(beta, np.ndarray): beta = np.array([beta]).flatten()
    if len(alpha) != len(beta):
        raise Exception("alpha, beta are different lengths!")

    if maxtaufrac >= 1:
        warnings.warn("maxtaufrac >= 1: rays that fall into the black hole end on the horizon r_+, "
                      "where t and phi diverge logarithmically (returned as +-inf or nan there)")

    print('calculating preliminaries...')

    # conserved quantities
    sin_o, cos_o = obs_trig(th_o) # exact at 0, pi/2, pi
    lam = -alpha*sin_o
    eta = (alpha**2 - a**2)*cos_o**2 + beta**2

    # spin zero should have no vortical geodesics
    if(np.abs(a)<MINSPIN_TH and np.any(eta<0)):
        eta[eta<0]=EP # TODO ok?
        print(f"WARNING: there were eta<0 points for spin {a:f}<MINSPIN_TH!")

    # radial roots and radial motion case
    (r1, r2, r3, r4, rclass) = radial_roots(a, lam, eta)

    # total Mino time to infinity
    tau_tot = mino_total(a, r_o, eta, r1, r2, r3, r4)

    # define steps equally spaced in Mino time tau
    # rays have equal numbers of steps -- step size dtau depends on the ray
    # mino time is positive back from screen in GL19b conventions
    dtau = maxtaufrac*tau_tot / (ngeo - 1)
    tausteps = np.linspace(0, maxtaufrac*tau_tot, ngeo)

    # calculate the points along the geodsic at tausteps
    affinesteps, geo_coords = coords_at_tau(a, observer_coords, image_coords, tausteps, do_phi_and_t=True)


    # create Geodesics object
    geos = Geodesics(a, observer_coords, image_coords, tausteps, affinesteps, geo_coords)

    if savedata and do_phi_and_t:
        print('saving data...')
        try:
            geos.savegeos()
        except:
            print("Error saving to file!")

    if plotdata and do_phi_and_t:
        print('plotting data...')
        try:
            geos.plotgeos()
            plt.pause(1.e-3)
            plt.show()
            plt.pause(1.e-3)
        except:
            print("Error plotting data!")

    tstop = time.time()
    print('done!  ', tstop-tstart, ' seconds!')
    return geos


#TODO -- errors in phi raytracing with alpha=0, beta!=0.
def coords_at_tau(a, observer_coords, image_coords, tau, do_phi_and_t=True):
    r""" compute coordinates x^\mu at a particular Mino time"""

    [_, r_o, th_o, _] = observer_coords # assumes ph_o = 0
    [alpha, beta] = image_coords

    # checks
    check_spin(a)
    #if (a<0): print("WARNING a<0! Not fully verified!")
    if not (isinstance(r_o,float) and (r_o>=100)):
        raise Exception("r_o should be a float > 100")
    check_inclination(th_o)
    #if (th_o>np.pi/2.): print("WARNING th_o>pi/2! Not fully verified!")

    if not isinstance(alpha, np.ndarray): alpha = np.array([alpha]).flatten()
    if not isinstance(beta, np.ndarray): beta = np.array([beta]).flatten()
    if len(alpha) != len(beta):
        raise Exception("alpha, beta are different lengths!")
    if len(tau.shape)==1:
        tau = tau.reshape(1,tau.shape[0])
    if not(tau.shape[1]==len(alpha)): #TODO
        raise Exception("tau has incompatible shape in coords_at_tau!")

    # conserved quantities
    sin_o, cos_o = obs_trig(th_o) # exact at 0, pi/2, pi
    lam = -alpha*sin_o
    eta = (alpha**2 - a**2)*cos_o**2 + beta**2

    # spin zero should have no vortical geodesics
    if(np.abs(a)<MINSPIN_TH and np.any(eta<0)):
        eta[eta<0]=EP # TODO ok?
        print(f"WARNING: there were eta<0 points for spin {a:f}<MINSPIN_TH!")

    # sign of final angular momentum
    #s_o = my_sign(beta)
    # beta = 0 counts as + in the north, - in the south, as in r_equatorial,
    # so the reflection (th_o, beta) -> (pi - th_o, -beta) flips s_o at beta = 0 too
    h = 1. if th_o < 0.5*np.pi else -1. # sign(cos th_o), GL19a 54
    s_o = h*my_sign(h*beta)

    # angular turning points and number of equatorial crossings
    (u_plus, u_minus, th_plus, th_minus, thclass) = angular_turning(a, th_o, lam, eta)

    # radial roots and radial motion case
    (r1, r2, r3, r4, rclass) = radial_roots(a, lam, eta)

    # integrate in theta
    print('integrating in theta...',end="\r")
    start = time.time()
    (th_s, lamG_ph, G_t) = th_integrate(a,th_o,s_o,lam,eta,u_plus,u_minus,tau,
                                        do_phi_and_t=do_phi_and_t, beta=beta, alpha=alpha, return_lamG=True)
    stop = time.time()
    print('integrating in theta...%0.2f s'%(stop-start))

    # integrate in r
    print('integrating in r...',end="\r")
    start = time.time()
    (r_s, I_ph, I_t, I_sig) = r_integrate(a,r_o,lam,eta, r1,r2,r3,r4,tau,
                                          do_phi_and_t=do_phi_and_t)
    stop = time.time()
    print('integrating in r...%0.2f s'%(stop-start))

    # combine integrals to get phi, t, and sigma as a function of time
    sig_s = 0 + I_sig + a**2 * G_t # GL19a 15
    t_s   = 0 + I_t + a**2 * G_t   # GL19a 12
    #ph_s  = 0 + I_ph + lam*G_ph    # GL19a 11
    ph_s  = 0 + I_ph + lamG_ph     # GL19a 11, lam*G_ph finite through the poles

    # return
    geo_coords = [t_s,r_s,th_s,ph_s]

    # non-finite values before the end of the ray are numerical failures, not physical divergences
    # (captured rays reach the horizon at tau = tau_tot, where t and phi diverge)
    tau_tot = mino_total(a, r_o, eta, r1, r2, r3, r4)
    bad = np.any(~np.isfinite(np.array(geo_coords + [sig_s])) & (tau < tau_tot), axis=(0, 1))
    if np.any(bad):
        warnings.warn(f"{np.sum(bad)} of {len(bad)} rays have non-finite coordinates before reaching the horizon")

    return sig_s, geo_coords

@np.errstate(divide='ignore', invalid='ignore') # x/0, 0/0 in branches that are computed but not used
def th_integrate(a,th_o, s_o,lam, eta, u_plus, u_minus, tausteps,
                 do_phi_and_t=True, beta=None, alpha=None, return_lamG=False):
    """theta, G_phi, G_t at Mino times tausteps (GL19a 29-68).
       beta at the observer can be optionally passed as an argument,
       used to increase precision when the observer is near a turning point, beta ~ 0;
       if not given, it is recomputed from Theta(th_o) = beta^2 (GL19b 55).
       alpha at the observer can be optionally passed as an argument,
       needed for an observer on the axis (th_o = 0, pi), where lam = 0 for every alpha;
       if not given, it is recomputed from lam = -alpha sin(th_o) (GL19b 55).
       G_phi diverges at each pole passage as lam -> 0, but lam*G_phi stays finite:
       it is what is computed, and returned in place of G_phi if return_lamG
       (otherwise G_phi = lam*G_phi/lam)."""
    check_spin(a)
    check_inclination(th_o)
    if not isinstance(s_o, np.ndarray): s_o = np.array([s_o]).flatten()
    if not isinstance(eta, np.ndarray): eta= np.array([eta]).flatten()
    if not isinstance(lam, np.ndarray): lam= np.array([lam]).flatten()
    if not isinstance(u_plus, np.ndarray): u_plus = np.array([u_plus]).flatten()
    if not isinstance(u_minus, np.ndarray): u_minus = np.array([u_minus]).flatten()

    if beta is None:
        betaabs = np.sqrt(np.maximum(eta + np.cos(th_o)**2*(a**2 - lam**2/np.sin(th_o)**2), 0))
    else:
        if not isinstance(beta, np.ndarray): beta = np.array([beta]).flatten()
        betaabs = np.abs(beta)

    # lamhat = lam/sin(th_o) = -alpha, finite on the axis
    if alpha is None:
        lamhat = lam/np.sin(th_o)
    else:
        if not isinstance(alpha, np.ndarray): alpha = np.array([alpha]).flatten()
        lamhat = -alpha

    if not(len(s_o)==len(eta)==len(lam)==len(u_plus)==len(u_minus)==len(betaabs)==len(lamhat)):
        raise Exception("inputs to th_integrate not the same length!")
    if not(tausteps.shape[-1]==len(s_o)):
        raise Exception("tausteps has incompatible shape in th_integrate!")

    # output arrays
    th_s = np.zeros(tausteps.shape)
    lamG_ph = np.zeros(tausteps.shape) # lam*G_ph
    G_t  = np.zeros(tausteps.shape)

    # ordinary motion:
    if(np.any(eta>0.)):
        mask = eta>0. # eta = 0 handled separately below
        up = u_plus[mask]
        um = u_minus[mask]
        s = s_o[mask]

        # compute factors up/um and a**2 * um (in zero spin limit)
        if(np.abs(a)<MINSPIN_TH):
            uratio = np.zeros(eta.shape)[mask]
            a2um = -(eta+lam**2)[mask]
        else:
            uratio = up/um
            a2um = a**2 * um

        ## compute antideriviatives at the origin
        pref = 1/np.sqrt(-a2um)
        k = uratio # k<0 since um<0 for eta>0
        # Psi_o = arcsin(cos(th_o)/sqrt(up)), the amplitude of GL19a 29 at the observer (cf. 45)
        #elliparg = np.arcsin(np.cos(th_o)/np.sqrt(up)) # old version
        elliparg, N_o, D_o = psi_o_amplitude(a, th_o, betaabs[mask], a2um) # exact at turning points (beta=0)

        # "G-terms" at observer
        F =  sp.ellipkinc(elliparg,k)
        Gth_o = -pref * F #GL19a, 29

        if do_phi_and_t:

            # GL 19a, 30
            #Gph_o = -pref * ellippi_arr(up, elliparg, k)
            # lam*pref*Pi(up; phi|k) = ellip_pi_scaled(lamc, nc, phi, k), finite as lam -> 0 (up -> 1), with
            # lamc = lam*pref = +-sqrt((1-up)(up-k)/up), since 1 - up = lam^2/(a^2 - a^2 um),
            # nc = k/up = 1/um = a^2/a2um (finite as a -> 0)
            # ctan = lamc tan(Psi_o) = lamhat pref N_o/D_o is finite on the axis where lamc -> 0, Psi_o -> +-pi/2
            lamc = lam[mask]*pref
            nc = a**2/a2um
            ctan = (lamhat[mask]*pref*N_o, D_o)
            lamGph_o = -ellip_pi_scaled(lamc, nc, elliparg, k, ctan=ctan)

            # GL 19a, 31
            maskk = (k==0)
            Gt_o = np.zeros(k.shape)
            if np.any(maskk): # limit as k-> 0, occurs when a==0
                Gt_o[maskk] = 2*up*pref*0.125*(-2*elliparg + np.sin(2*elliparg))[maskk]
            if np.any(~maskk):
                Gt_o[~maskk] = 2*up*pref* (sp.ellipeinc(elliparg,k) - F)[~maskk]/(2*k[~maskk]) # GL 19a, 31

        # compute the amplitude Phi_tau, GL 19a, equation 46
        snarg = np.sqrt(-a2um)*(-tausteps[:,mask] + s*Gth_o)
        snarg = snarg.astype(float)

        # initialize zeros
        sinPhi_tau = np.zeros(snarg.shape)
        Phi_tau = np.zeros(snarg.shape)

        # for very small arguments, use lim_x->0 sn(x,k)=lim_x->0 am(x,k)=x
        jmask = np.abs(snarg) < EP # TODO what value cutoff?
        if np.any(jmask):
            sinPhi_tau[jmask] = snarg[jmask]
            Phi_tau[jmask] = snarg[jmask]

        # for other arguments use some elliptic function identities
        # https://functions.wolfram.com/EllipticFunctions/JacobiAmplitude/introductions/JacobiPQs/ShowAll.html
        # TODO test accuracy more
        if np.any(~jmask):
            #m = (k/(k-1)) # real, in [0,1) since k<=0
            #m = np.outer(np.ones(snarg.shape[0]),m)[~jmask]
            #ellipfuns = sp.ellipj(snarg[~jmask]/np.sqrt(1-m), m) #scale snarg by sqrt(1-m) to use identities

            # m = k/(k-1) -> 1 as eta -> 0 (k = u_plus/u_minus -> -inf):
            # m -> 1 causes issues in ellipj, so in new version we work with 1-m = 1/(1-k)
            m1 = 1/(1-k) # 1-m, real, in (0,1] since k<=0
            m1 = np.outer(np.ones(snarg.shape[0]),m1)[~jmask]
            ellipfuns = ellipj_m1(snarg[~jmask]/np.sqrt(m1), m1) #scale snarg by sqrt(1-m) to use identities

            #sn(sqrt(1-m)x | k) = sqrt(1-m)*sn(x|m)/dn(x|m)
            #sinPhi_tau[~jmask] = np.sqrt(1-m) * ellipfuns[0]/ellipfuns[2]
            sinPhi_tau[~jmask] = np.sqrt(m1) * ellipfuns[0]/ellipfuns[2]

            #am(sqrt(1-m)x | k) = pi/2 - am(K(m) - x | m for m <=1
            #Phi_tau[~jmask] = 0.5*np.pi-sp.ellipj(sp.ellipk(m) - snarg[~jmask]/np.sqrt(1-m), m)[3]
            Phi_tau[~jmask] = 0.5*np.pi - ellipj_m1(sp.ellipkm1(m1) - snarg[~jmask]/np.sqrt(m1), m1)[3]

        # at tau = 0, use the exact relation Phi_tau = -s Psi_o.
        # On the axis, Psi_o = +-pi/2 and roundoff would shift j in ellip_pi_scaled
        t0 = tausteps[:,mask]==0
        Phi_tau = np.where(t0, -s*elliparg, Phi_tau)
        sinPhi_tau = np.where(t0, -s*np.sin(elliparg), sinPhi_tau)

        # solution for theta_o, GL19a 49
        th_s[:,mask] = (np.arccos(-s*np.sqrt(up)*sinPhi_tau)).astype(float)

        if do_phi_and_t:

            # G_phi integral GL19a 47
            #G_ph[:,mask] = (pref*ellippi_arr(up, Phi_tau, k) - s*Gph_o).astype(float)
            lamG_ph[:,mask] = (ellip_pi_scaled(lamc, nc, Phi_tau, k) - s*lamGph_o).astype(float)

            # G_t integral GL19a, 48
            maskk = (k==0)
            Gtout = np.zeros(Phi_tau.shape)
            if np.any(maskk): # limit as k-> 0, occurs when a==0
                Gtout[:,maskk] = -2*up*pref*0.125*(-2*Phi_tau + np.sin(2*Phi_tau))[:,maskk]
            if np.any(~maskk):
                Gtout[:,~maskk] = -(2*up*pref* (sp.ellipeinc(Phi_tau,k) - sp.ellipkinc(Phi_tau,k)))[:,~maskk]/(2*k[~maskk]) # GL 19a, 31

            Gtout =  Gtout - s*Gt_o
            G_t[:,mask] = Gtout

    # vortical motion case
    if(np.any(eta<0.)):
        if(np.abs(a)<MINSPIN_TH):
            raise Exception("below MINSPIN_TH but there are eta<0 points in th_integrate!")

        mask = eta<0.
        up = u_plus[mask]
        um = u_minus[mask]
        s = s_o[mask]

        uratio = up/um
        a2um = a**2 * um

        # compute antideriviatives at the observer point
        if th_o <= np.pi/2: h = 1. # sign(cos(th)) GL19a 54, which hemisphere
        else: h = -1

        prefA = 1/np.sqrt(a2um) # u_minus>0 always for eta<0
        prefB = prefA / (1.-um)
        prefC = np.sqrt(um/a**2)
        k = 1 - uratio # again, k<0 always since for eta<0 up>um>0
        upm = (up-um)/(1.-um)

        #Up_o = np.arcsin(np.sqrt((np.cos(th_o)**2 - um)/(up-um))) # GL19a, 59
        Up_o, N_o, D_o = upsilon_o_amplitude(a, th_o, betaabs[mask], up, um) # exact at turning points (beta=0)

        # "G-terms" at observer
        Gth_o = -h*prefA * sp.ellipkinc(Up_o, k) #GL19a 56
        if do_phi_and_t:
            #Gph_o = -h*prefB * ellippi_arr(upm,Up_o,k) #GL19a 57
            # lam*prefB*Pi(upm; phi|k) = ellip_pi_scaled(lamc, nc, phi, k), finite as lam -> 0 (upm -> 1), with
            # lamc = lam*prefB = +-sqrt((1-upm)(upm-k)/upm), since 1 - upm = lam^2/(a^2 (1-um)^2),
            # nc = k/upm = 1 - 1/um
            # ctan = lamc tan(Up_o) = lamhat prefB N_o/D_o is finite on the axis where lamc -> 0, Up_o -> pi/2
            lamc = lam[mask]*prefB
            nc = 1. - 1./um
            ctan = (lamhat[mask]*prefB*N_o, D_o)
            lamGph_o = -h*ellip_pi_scaled(lamc, nc, Up_o, k, ctan=ctan) #GL19a 57
            Gt_o  = -h*prefC * sp.ellipeinc(Up_o,k) #GL19a 58

        # compute the amplitude Up_tau, GL19a equation 66
        snarg = np.sqrt(a2um)*(-tausteps[:,mask] + s*Gth_o)
        snarg = snarg.astype(float)

        sinUp_tau = np.zeros(snarg.shape)
        Up_tau = np.zeros(snarg.shape)

        # for very small arguments, use lim_x->0 sn(x,k)= lim_x->0 am(x,k)= x
        jmask = np.abs(snarg) < EP # TODO what value cutoff?
        if np.any(jmask):
            sinUp_tau[jmask] = snarg[jmask]
            Up_tau[jmask] = snarg[jmask]

        # for other arguments use some elliptic function identities
        # https://functions.wolfram.com/EllipticFunctions/JacobiAmplitude/introductions/JacobiPQs/ShowAll.html
        # TODO test accuracy more
        if np.any(~jmask):
            #m = (k/(k-1)) # real, in [0,1) since k<=0
            #m = np.outer(np.ones(snarg.shape[0]),m)[~jmask]
            #ellipfuns = sp.ellipj(snarg[~jmask]/np.sqrt(1-m), m) #scale snarg by sqrt(1-m) to use identities

            # m = k/(k-1) -> 1 as eta -> 0 (k = 1 - u_plus/u_minus -> -inf):
            # m -> 1 causes issues in ellipj, so in new version we work with 1-m = 1/(1-k) = u_minus/u_plus
            m1 = 1/(1-k) # 1-m, real, in (0,1] since k<=0
            m1 = np.outer(np.ones(snarg.shape[0]),m1)[~jmask]
            ellipfuns = ellipj_m1(snarg[~jmask]/np.sqrt(m1), m1) #scale snarg by sqrt(1-m) to use identities

            #sn(sqrt(1-m)x | k) = sqrt(1-m)*sn(x|m)/dn(x|m)
            #sinUp_tau[~jmask] = np.sqrt(1-m) * ellipfuns[0]/ellipfuns[2]
            sinUp_tau[~jmask] = np.sqrt(m1) * ellipfuns[0]/ellipfuns[2]

            #am(sqrt(1-m)x | k) = pi/2 - am(K(m) - x | m for m <=1
            #Up_tau[~jmask] = 0.5*np.pi-sp.ellipj(sp.ellipk(m) - snarg[~jmask]/np.sqrt(1-m), m)[3]
            Up_tau[~jmask] = 0.5*np.pi - ellipj_m1(sp.ellipkm1(m1) - snarg[~jmask]/np.sqrt(m1), m1)[3]

        # at tau = 0, use the exact relation Up_tau = -s h Up_o.
        # On the axis, Up_o = pi/2 and roundoff would shift j in ellip_pi_scaled
        t0 = tausteps[:,mask]==0
        Up_tau = np.where(t0, -s*h*Up_o, Up_tau)
        sinUp_tau = np.where(t0, -s*h*np.sin(Up_o), sinUp_tau)

        # solution for theta_o, GL19a 69
        th_s[:,mask] = (np.arccos(h*np.sqrt(um + (up-um)*sinUp_tau**2))).astype(float)

        if do_phi_and_t:

            # G_phi integral, GL19a 67
            #G_ph[:,mask] = (prefB * ellippi_arr(upm,Up_tau,k) - s*Gph_o).astype(float)
            lamG_ph[:,mask] = (ellip_pi_scaled(lamc, nc, Up_tau, k) - s*lamGph_o).astype(float)
            # the ray along the axis (th_o = 0, pi, alpha = beta = 0) has up = um = 1 and stays on it,
            # where phi is undefined (prefB = 1/(1-um) is infinite): take lam*G_phi = 0, so phi = I_phi
            lamG_ph[:,mask] = np.where(um==1, 0., lamG_ph[:,mask])

            # G_t integral, GL19a, 68
            G_t[:,mask] = (prefC * sp.ellipeinc(Up_tau,k) - s*Gt_o).astype(float)

    # exact boundary between vortical and regular motion, eta=0, |a|>0
    # TODO added by Claude --> rederive
    # Here u_minus = 0 is a double root of GL19a 18, (1-u)Theta = a^2 u (u_plus - u),
    # so (dcos(th)/dtau)^2 = a^2 cos^2(th) (u_plus - cos^2(th)) is solved by elementary functions:
    # cos(th) = h*sqrt(u_plus)*sech(psi), psi = psi_o + |a| sqrt(u_plus) tau;
    # the ray asymptotes to the equator
    if(np.any(eta==0.) and np.abs(a)>=MINSPIN_TH):
        mask = eta==0.
        up = u_plus[mask]
        s = s_o[mask]
        lamabs = np.abs(lam[mask])

        if th_o <= np.pi/2: h = 1. # sign(cos(th)) GL19a 54, hemisphere
        else: h = -1

        # sign of psi_o so that dcos(th)/dtau has sign s at tau=0, as in GL19a 49
        #psi_o = -h*s*np.arcsinh(np.sqrt(up - np.cos(th_o)**2)/np.abs(np.cos(th_o)))
        # sqrt(up - cos^2 th_o) = |beta| sin(th_o)/(|a| |cos th_o|), from GL19a 18-19 with um = 0
        #psi_o = -h*s*np.arcsinh(betaabs[mask]*np.sin(th_o)/(np.abs(a)*np.cos(th_o)**2))
        # sinh|psi_o| = zhat sin(th_o). On the equator (th_o = pi/2, so beta = 0) the observer sits on
        # the double root u = 0: psi_o = -+inf, and theta = pi/2 throughout
        sin_o, cos_o = obs_trig(th_o) # exact at 0, pi/2, pi
        if cos_o == 0:
            zhat = np.full(betaabs[mask].shape, np.inf)
        else:
            zhat = betaabs[mask]/(np.abs(a)*cos_o**2)
        psi_o = -h*s*np.arcsinh(zhat*sin_o)
        psi = psi_o + np.sqrt(a**2*up)*tausteps[:,mask]

        # solution for theta_o, solves GL19a 25
        th_s[:,mask] = (np.arccos(h*np.sqrt(up)/np.cosh(psi))).astype(float)

        if do_phi_and_t:
            # GL19a 26, 27, with signs flipped since kgeo's tau runs backward from the observer
            # G_phi = -int dtau/sin^2(th), with sqrt(u_plus/(1-u_plus)) = |a| sqrt(u_plus)/|lam| since 1-u_plus = lam^2/a^2
            #c = np.sqrt(a**2*up)/lamabs
            #G_ph[:,mask] = (-tausteps[:,mask] - (np.arctan(c*np.tanh(psi)) - np.arctan(c*np.tanh(psi_o)))/lamabs).astype(float)
            # times lam: finite at lam = 0 (a pole passage at psi = 0)
            clam = np.sqrt(a**2*up)
            #lamsign = my_sign(lam[mask])
            #lamG_ph[:,mask] = (-lam[mask]*tausteps[:,mask]
            #                   -lamsign*(np.arctan2(clam*np.tanh(psi), lamabs) - np.arctan2(clam*np.tanh(psi_o), lamabs))).astype(float)
            # with lamhat = lam/sin(th_o): its sign is that of lam (the th_o -> 0 limit on the axis), and the
            # observer term drops sin(th_o) from both arguments, using tanh(psi_o)/sin(th_o) = -h s zhat/sqrt(1 + z^2)
            lamsign = my_sign(lamhat[mask])
            #zhat = betaabs[mask]/(np.abs(a)*np.cos(th_o)**2)
            #tanh_psio_hat = -h*s*zhat/np.sqrt(1 + (zhat*np.sin(th_o))**2)
            if cos_o == 0: # zhat = inf
                tanh_psio_hat = -h*s/sin_o
            else:
                tanh_psio_hat = -h*s*zhat/np.sqrt(1 + (zhat*sin_o)**2)
            lamG_ph[:,mask] = (-lam[mask]*tausteps[:,mask]
                               -lamsign*(np.arctan2(clam*np.tanh(psi), lamabs)
                                         - np.arctan2(clam*tanh_psio_hat, np.abs(lamhat[mask])))).astype(float)

            # G_t = -int cos^2(th) dtau
            G_t[:,mask] = (-np.sqrt(up/a**2)*(np.tanh(psi) - np.tanh(psi_o))).astype(float)

    # eta = 0 and a = 0
    # (the radial ray lam = 0, a limit of the principal null congruences, or th_o = pi/2)
    if(np.any(eta==0.) and np.abs(a)<MINSPIN_TH):
        mask = eta==0.
        th_s[:,mask] = th_o
        if do_phi_and_t:
            #lamG_ph[:,mask] = -lam[mask]*tausteps[:,mask]/np.sin(th_o)**2
            # 0 for the radial ray on the axis (lam = 0, sin(th_o) = 0)
            lamG_ph[:,mask] = -lam[mask]*tausteps[:,mask]/np.where(lam[mask]==0, 1., np.sin(th_o)**2)
            G_t[:,mask] = -tausteps[:,mask]*np.cos(th_o)**2

    if return_lamG:
        return (th_s, lamG_ph, G_t)
    G_ph = lamG_ph/lam # inf/nan at lam = 0, as before
    return (th_s, G_ph, G_t)


@np.errstate(divide='ignore', invalid='ignore') # x/0, 0/0 in branches that are computed but not used
def r_integrate(a,r_o,lam,eta, r1,r2,r3,r4,tausteps,
                do_phi_and_t=True):

    # Follow G19a, interchange source <--> observer labels and send tau -> -tau

    # checks
    check_spin(a)
    if not (isinstance(r_o,float) and (r_o>=100)):
        raise Exception("r_o should be a float > 100")
    if not isinstance(lam, np.ndarray): lam  = np.array([lam]).flatten()
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if not isinstance(r1, np.ndarray): r1 = np.array([r1]).flatten()
    if not isinstance(r2, np.ndarray): r2 = np.array([r2]).flatten()
    if not isinstance(r3, np.ndarray): r3 = np.array([r3]).flatten()
    if not isinstance(r4, np.ndarray): r4 = np.array([r4]).flatten()
    if not(len(lam)==len(eta)==len(r1)==len(r2)==len(r3)==len(r4)):
        raise Exception("inputs to r_integrate not the same length!")
    if not(tausteps.shape[-1]==len(eta)):
        raise Exception("tausteps has incompatible shape in r_integrate!")

    # horizons
    rplus  = 1 + np.sqrt(1-a**2)
    rminus = 1 - np.sqrt(1-a**2)

    # output arrays
    r_s = np.zeros(tausteps.shape)
    I_p = np.zeros(tausteps.shape)
    I_m = np.zeros(tausteps.shape)
    I_1  = np.zeros(tausteps.shape)
    I_2  = np.zeros(tausteps.shape)

    s = 1 # the final radial sign at the observer screen is POSITIVE

    # TODO what to do if we are exactly on the critical curves of double roots between regions?

    # no real roots -- region IV, case 4
    mask_4 = (np.imag(r1) != 0)
    if np.any(mask_4):
        tau = tausteps[:,mask_4]

        # real and imaginary parts of r4,r2, GL19a B10, B11
        a1 = np.imag(r4)[mask_4] # a1 > 0
        b1 = np.real(r4)[mask_4] # b1 > 0
        a2 = np.imag(r2)[mask_4] # a2 > 0
        b2 = np.real(r2)[mask_4] # b2 < 0

        # parameters for case 4
        CC = np.sqrt((a1-a2)**2 + (b1-b2)**2) # equal to sqrt(r31*r42)>0, GL19a B85
        DD = np.sqrt((a1+a2)**2 + (b1-b2)**2) # equal to sqrt(r32*r41)>0, GL19a B85
        k4 = (4*CC*DD)/((CC+DD)**2) # 0<k4<1, GL19a B87

        x4rp = (rplus  + b1)/a2 # GL19a, B83 at horizon
        x4rm = (rminus + b1)/a2 # GL19a, B83 at inner horizon
        if r_o==np.inf:
            x4ro = np.inf # GL19a, B83 for observer at r_o
        else:
            x4ro = (r_o + b1)/a2 # GL19a, B83 for observer at r_o

        g0 = np.sqrt((4*a2**2 - (CC-DD)**2)/ ((CC+DD)**2 - 4*a2**2)) # 0<g0<1, GL19a B88
        gp = (g0*x4rp - 1.)/(g0 + x4rp) # B96
        gm = (g0*x4rm - 1.)/(g0 + x4rm) # B96

        # Find r_s
        auxarg = np.arctan(x4ro) + np.arctan(g0)
        pref = 2./(CC+DD)
        Ir_o = pref*sp.ellipkinc(auxarg,k4)# GL 19a B101
        X4 = (1./pref)*(s*(-tau) + Ir_o) # GL19a B104

        ellipfuncs = sp.ellipj(X4,k4) # should work fine because 0 < k4 < 1
        #amX4 = np.arcsin(ellipfuncs[0])  # arcsin does not account for full range!
        amX4 = ellipfuncs[3]
        scX4 = ellipfuncs[0]/ellipfuncs[1] # sc = sn/cn
        rs = -a2*(g0 - scX4)/(1 + g0*scX4) - b1 # B109
        r_s[:,mask_4] = rs

        if do_phi_and_t:

            # building blocks of the path integrals
            S1_a_0, S2_a_0 = S1_S2(g0,amX4,k4,ret_s2=True)
            S1_b_0, S2_b_0 = S1_S2(g0,auxarg,k4,ret_s2=True)
            S1_a_p, _ = S1_S2(gp,amX4,k4,ret_s2=False)
            S1_b_p, _ = S1_S2(gp,auxarg,k4,ret_s2=False)
            S1_a_m, _ = S1_S2(gm,amX4,k4,ret_s2=False)
            S1_b_m, _ = S1_S2(gm,auxarg,k4,ret_s2=False)

            aa0 = (a2/g0)*(1+g0**2)
            Pi_1 = s*pref*aa0*(S1_a_0 - S1_b_0) #B115
            Pi_2 = s*pref*(aa0**2)*(S2_a_0 - S2_b_0) # B115
            Pi_p = s*pref*(1+g0**2)/(g0**2 + g0*x4rp)*(S1_a_p - S1_b_p) #B116
            Pi_m = s*pref*(1+g0**2)/(g0**2 + g0*x4rm)*(S1_a_m - S1_b_m) # B116

            # final integrals
            aa1 =  (a2/g0 - b1)
            I1 = aa1*(-tau) - Pi_1 # B112
            I2 = (aa1**2)*(-tau) - 2*aa1*Pi_1 + Pi_2 #B113
            Ip = g0/(a2-a2*g0*x4rp)*(-tau - Pi_p) # B114
            Im = g0/(a2-a2*g0*x4rm)*(-tau - Pi_m)

            # return output
            I_p[:,mask_4] = Ip
            I_m[:,mask_4] = Im
            I_1[:,mask_4] = I1
            I_2[:,mask_4] = I2

    # two roots (r3, r4) complex -- region III, case 3
    mask_3 = (np.imag(r3) != 0) * (~mask_4)
    if np.any(mask_3):
        tau = tausteps[:,mask_3]

        # real and imaginary parts of r4 GL19a B10
        a1 = np.imag(r4)[mask_3] # a1 > 0
        b1 = np.real(r4)[mask_3] # b1 > 0
        rr1 = np.real(r1)[mask_3] # r1 is real
        rr2 = np.real(r2)[mask_3] # r2 is real
        rr21 = rr2 - rr1
        rrp1 = rplus - rr1
        rrp2 = rplus - rr2
        rrm1 = rminus - rr1
        rrm2 = rminus - rr2
        rro1 = r_o - rr1
        rro2 = r_o - rr2

        # parameters for case 3
        AA = np.sqrt(a1**2 + (b1-rr2)**2) # equal to sqrt(r32*r42)>0, GL19a B85
        BB = np.sqrt(a1**2 + (b1-rr1)**2) # equal to sqrt(r31*r41)>0, GL19a B85
        k3 = ((AA + BB)**2 - rr21**2)/(4*AA*BB) # 0<k3<1, GL19a B59

        x3rp = (AA*rrp1 - BB*rrp2)/(AA*rrp1 + BB*rrp2) # GL19a, B55
        x3rm = (AA*rrm1 - BB*rrm2)/(AA*rrm1 + BB*rrm2) # GL19a, B55
        if r_o==np.inf:
            x3ro = (AA - BB)/(AA + BB)  # GL19a, B55 for observer at r_o
        else:
            x3ro = (AA*rro1 - BB*rro2)/(AA*rro1 + BB*rro2)

        alp = -1./x3rp # B66
        alm = -1./x3rm
        al0 = (BB + AA) / (BB - AA) # B58

        # Find r_s
        auxarg = np.arccos(x3ro)
        pref = (1./np.sqrt(AA*BB))
        Ir_o = pref*sp.ellipkinc(auxarg,k3)# GL 19a B101
        X3 = (1./pref)*(-tau + s*Ir_o) # GL19a B74

        ellipfuncs = sp.ellipj(X3,k3) # should work fine because 0 < k3 < 1
        amX3 = ellipfuncs[3]
        cnX3 = ellipfuncs[1]
        rs_num = (BB*rr2 - AA*rr1) + (BB*rr2 + AA*rr1)*cnX3 # B75
        rs_denom = (BB-AA) + (BB+AA)*cnX3
        rs = rs_num / rs_denom
        r_s[:,mask_3] = rs

        if do_phi_and_t:


            # # building blocks of the path integrals
            R1_a_0, R2_a_0 = R1_R2(al0,amX3,k3)
            R1_b_0, R2_b_0 = R1_R2(al0,auxarg,k3)
            #R1_a_p, _ = R1_R2(alp,amX3,k3,ret_r2=False)
            #R1_b_p, _ = R1_R2(alp,auxarg,k3,ret_r2=False)
            #if np.abs(a)>MINSPIN_R: # I_minus term is O(a^2); its R1 degenerates as r_minus -> r2
            #    R1_a_m, _ = R1_R2(alm,amX3,k3,ret_r2=False)
            #    R1_b_m, _ = R1_R2(alm,auxarg,k3,ret_r2=False)
            #else:
            #    R1_a_m = np.zeros(R1_a_p.shape)
            #    R1_b_m = np.zeros(R1_a_p.shape)
            # B82 with s = nu_r = 1 is a difference of R1s, stable as alm -> -1 (r2 -> r_minus).
            # No a = 0 guard needed: there r_minus = r2 = 0, so alm = -1, which R1_diff handles
            R1_d_p = R1_diff(alp,amX3,auxarg,k3)
            R1_d_m = R1_diff(alm,amX3,auxarg,k3)

            Pi_1 = ((2*rr21*np.sqrt(AA*BB))/(BB**2-AA**2)) * (R1_a_0 - s*R1_b_0) # B81
            Pi_2 = ((2*rr21*np.sqrt(AA*BB))/(BB**2-AA**2))**2 * (R2_a_0 - s*R2_b_0) # B81
            #Pi_p = ((2*rr21*np.sqrt(AA*BB))/(BB*rrp2 - AA*rrp1))*(R1_a_p - s*R1_b_p) # B82
            #Pi_m = ((2*rr21*np.sqrt(AA*BB))/(BB*rrm2 - AA*rrm1))*(R1_a_m - s*R1_b_m) # B82
            Pi_p = ((2*rr21*np.sqrt(AA*BB))/(BB*rrp2 - AA*rrp1))*R1_d_p # B82
            Pi_m = ((2*rr21*np.sqrt(AA*BB))/(BB*rrm2 - AA*rrm1))*R1_d_m # B82

            # final integrals
            pref = ((BB*rr2 + AA*rr1)/(BB+AA))
            I1 = pref*(-tau) + Pi_1 # B78
            I2 = pref**2*(-tau) + 2*pref*Pi_1 + np.sqrt(AA*BB)*Pi_2 # B79
            Ip = -((BB+AA)*(-tau) + Pi_p) / (BB*rrp2 + AA*rrp1) # B80
            Im = -((BB+AA)*(-tau) + Pi_m) / (BB*rrm2 + AA*rrm1) # B80

            # return output
            I_p[:,mask_3] = Ip
            I_m[:,mask_3] = Im
            I_1[:,mask_3] = I1
            I_2[:,mask_3] = I2

    # all roots real - case 2, region II OR region I
    mask_2 = (np.imag(r3) == 0.) * (~mask_3)
    if np.any(mask_2):
        tau = tausteps[:,mask_2]

        # roots in this region
        rr1 = np.real(r1)[mask_2]
        rr2 = np.real(r2)[mask_2]
        rr3 = np.real(r3)[mask_2]
        rr4 = np.real(r4)[mask_2]
        rr31 = rr3 - rr1
        rr32 = rr3 - rr2
        rr41 = rr4 - rr1
        rr42 = rr4 - rr2
        rr43 = rr4 - rr3
        rrp3 = rplus - rr3
        rrm3 = rminus - rr3
        rrp4 = rplus - rr4
        rrm4 = rminus - rr4

        # parameters for case 2
        k2 = (rr32*rr41)/(rr31*rr42)# 0<k=k2<1, GL19a B13

        if r_o==np.inf:
            x2ro = np.sqrt(rr31/rr41) # B35, for observer at r_o
        else:
            x2ro = np.sqrt((rr31*(r_o-rr4))/(rr41*(r_o-rr3)))# GL19a, B35

        # Find r_s
        auxarg = np.arcsin(x2ro)
        Ir_o = 2*sp.ellipkinc(auxarg,k2)/np.sqrt(rr31*rr42)# GL 19a B40
        X2 = 0.5*np.sqrt(rr31*rr42)*(-tau + s*Ir_o) # GL19a B45

        ellipfuncs = sp.ellipj(X2,k2) # should work fine because 0 < k3 < 1
        snX2 = ellipfuncs[0]
        cnX2 = ellipfuncs[1]
        dnX2 = ellipfuncs[2]
        amX2 = ellipfuncs[3]
        rs_num = (rr4*rr31 - rr3*rr41*(snX2)**2) # B46
        rs_denom = (rr31 - rr41*(snX2)**2)
        rs = rs_num / rs_denom
        r_s[:,mask_2] = rs

        if do_phi_and_t:

            # building blocks of the path integrals
            dX2dtau = -0.5*np.sqrt(rr31*rr42)
            dsn2dtau = 2*snX2*cnX2*dnX2*dX2dtau
            drsdtau = rr31*rr43*rr41*dsn2dtau / ((rr31 - rr41*snX2**2)**2)
            drsdtau *= -1 # TODO because we take tau -> -tau we need this sign change
            Rpot_o = (r_o-rr1)*(r_o-rr2)*(r_o-rr3)*(r_o-rr4)
            drsdtau_o = s*np.sqrt(Rpot_o)
            H =  drsdtau / (rs - rr3) - drsdtau_o/(r_o - rr3) #B51
            E = np.sqrt(rr31*rr42)*(sp.ellipeinc(amX2,k2) - s*sp.ellipeinc(auxarg,k2)) # B52
            Pi_1 = (2./np.sqrt(rr31*rr42))*(ellippi_arr(rr41/rr31,amX2,k2)-s*ellippi_arr(rr41/rr31,auxarg,k2)) # B53
            Pi_p = (2./np.sqrt(rr31*rr42))*(rr43/(rrp3*rrp4))*(ellippi_arr((rrp3*rr41)/(rrp4*rr31),amX2,k2)-
                                                             s*ellippi_arr((rrp3*rr41)/(rrp4*rr31),auxarg,k2))
            Pi_m = (2./np.sqrt(rr31*rr42))*(rr43/(rrm3*rrm4))*(ellippi_arr((rrm3*rr41)/(rrm4*rr31),amX2,k2)-
                                                             s*ellippi_arr((rrm3*rr41)/(rrm4*rr31),auxarg,k2))

            # final integrals
            I1 = rr3*(-tau) + rr43*Pi_1 # B48
            I2 = H - 0.5*(rr1*rr4 + rr2*rr3)*(-tau) - E # B49
            Ip = tau/rrp3 - Pi_p # B50
            Im = tau/rrm3 - Pi_m # B50

            # return output
            I_p[:,mask_2] = Ip
            I_m[:,mask_2] = Im
            I_1[:,mask_2] = I1
            I_2[:,mask_2] = I2

    # TODO: added by Claude - rederive
    # z = 0: principal null congruence (GL19a 75, 76), incl. the a = 0 radial ray,
    # overriding the degenerate case 2 / 4 values above.
    # sqrt(R) = r^2 + b^2 with b = a|cos(th_0)|, so arctan(r/b) = arctan(r_o/b) - b tau
    mask_0 = (np.real(r4) == 0)
    if np.any(mask_0):
        tau = tausteps[:,mask_0]
        bb = np.imag(r4)[mask_0]
        tanbt = tau*np.sinc(bb*tau/np.pi)/np.cos(bb*tau) # tan(b tau)/b, finite at b = 0
        rs = (1 - bb**2*tanbt/r_o)/(1/r_o + tanbt)
        r_s[:,mask_0] = rs

        if do_phi_and_t:
            # I_n = -int r^n dtau = -int_rs^r_o r^n dr/(r^2 + b^2), and partial fractions for I_p, I_m
            lnrb = 0.5*np.log((r_o**2 + bb**2)/(rs**2 + bb**2))
            I_1[:,mask_0] = -lnrb
            I_2[:,mask_0] = -(r_o - rs) + bb**2*tau
            I_p[:,mask_0] = (rplus*tau - np.log((r_o - rplus)/(rs - rplus)) + lnrb)/(rplus**2 + bb**2)
            # rminus = b = 0 only at a = 0, where I_m = -int_rs^r_o dr/r^3 (and its prefactors vanish)
            rmb2 = rminus**2 + bb**2
            I_m[:,mask_0] = np.where(rmb2==0, -0.5*(1/rs**2 - 1/r_o**2),
                                     (rminus*tau - np.log((r_o - rminus)/(rs - rminus)) + lnrb)/np.where(rmb2==0, 1., rmb2))

    # get I_phi, I_t, I_sigma
    I_0 = -tausteps
    I_phi = (2*a/(rplus-rminus))*((rplus - 0.5*a*lam)*I_p - (rminus - 0.5*a*lam)*I_m) # B1
    I_tA = (4/(rplus-rminus))*((rplus**2 - 0.5*a*lam*rplus)*I_p - (rminus**2 - 0.5*a*lam*rminus)*I_m) # B2
    I_t = I_tA + 4*I_0 + 2*I_1 + I_2
    I_sig = I_2

    return (r_s, I_phi, I_t, I_sig)


# auxillary functions for the radial integrals
# TODO al->0 limit??
@np.errstate(divide='ignore', invalid='ignore') # x/0, 0/0 in branches that are computed but not used
def S1_S2(al,phi,j,ret_s2=True): #B92 and B93 of GL19a
    al2 = al*al
    al2p1 = 1 + al2
    al2j = 1 - j + al2
    s2phi = np.sqrt(1.- j*(np.sin(phi))**2)

    p2 = np.sqrt(al2p1 / al2j)
    f2 = 0.5*p2*np.log(np.abs(((1-p2)*(1+p2*s2phi))/((1+p2)*(1-p2*s2phi))))


    # definitions of functions in this region should keep things in the allowed prange
    F = sp.ellipkinc(phi,j)
    S1 = (F + al2*ellippi_arr(al2p1,phi,j) - al*f2) / al2p1

    if ret_s2:
        E = sp.ellipeinc(phi,j)
        tanphi = np.tan(phi)

        S2a = (1-j)*F + al2*E + al2*s2phi*(al - tanphi)/(1 + al*tanphi) - al2*al
        S2a = -S2a / (al2p1*al2j)
        S2b = S1*(1./al2p1 + (1-j)/al2j)
        S2 = S2a + S2b
    else:
        S2=np.nan

    return (S1,S2)

# TODO al->0 limit??
def R1_R2(al,phi,j,ret_r2=True): #B62 and B65 of GL19a
    al2 = al**2
    s2phi = np.sqrt(1-j*np.sin(phi)**2)
    p1 = np.sqrt((al2 -1)/(j+(1-j)*al2))
    f1 = 0.5*p1*np.log(np.abs((p1*s2phi+np.sin(phi))/(p1*s2phi-np.sin(phi))))
    nn = al2/(al2-1)
    R1 = (ellippi_arr(nn,phi,j) - al*f1)/(1-al2)

    if ret_r2:
        F = sp.ellipkinc(phi,j)
        E = sp.ellipeinc(phi,j)
        R2 = (F - (al2/(j+(1-j)*al2))*(E - al*np.sin(phi)*s2phi/(1+al*np.cos(phi)))) / (al2-1)
        R2 = R2 + (2*j - nn)*R1 / (j + (1-j)*al2)

    else:
        R2=np.nan

    return (R1,R2)

# TODO added by Claude --> rederive
def R1_diff(al,phi_a,phi_b,j):
    """R1(al; phi_a|j) - R1(al; phi_b|j), where R1 is the integral in GL19a B61, for 0 < phi_a, phi_b < pi.

       Near al = -1 each R1 diverges like 1/sqrt(1 + al), but only by a constant that does not depend
       on phi, so it cancels in the difference that B82 needs.

       For al < 0, splitting 1/(1 + al cos t) as in B62, applying DLMF 19.7.8 to its Pi,
       and writing F - Pi in Carlson form (DLMF 19.25.14) gives
         R1(al; phi|j) = const(al) + (j/(3 al^2)) sin^3(phi) R_J(cos^2 phi, Delta^2, 1, 1 + j (1 - al^2) sin^2(phi)/al^2)
                         - (|al|/D0) [T(W, |al| Delta cot(phi)) + T(W, Delta/sin(phi))],
       with Delta^2 = 1 - j sin^2(phi), D0 = j + (1-j) al^2, W = (1 - al^2)/D0,
       and T(W, z) = arctan(sqrt(W) z)/sqrt(W), which is smooth in W (= z at W = 0, an artanh for W < 0).
       For al > 0, t -> pi - t gives R1(al; phi) = const - R1(-al; pi - phi)."""

    def T(W, z): # = z at W = 0
        y = np.sqrt(np.abs(W))*z
        ys = np.where(y==0, 1., y)
        ay = np.abs(ys)
        atanc = np.arctan(ys)/ys
        atanhc = np.where(ay<1, np.arctanh(np.where(ay<1, ys, 0.)), np.arctanh(np.where(ay>1, 1./ys, 0.)))/ys # principal value
        return z*np.where(y==0, 1., np.where(W>=0, atanc, atanhc))

    b = np.abs(al)
    D0 = j + (1-j)*b**2
    W = (1-b**2)/D0
    nu = -j*(1-b**2)/b**2

    def R1_reg(phi): # R1(-b; phi|j) without its constant
        def h(ph): # (F(ph|j) - Pi(nu; ph|j))/(1-b^2) in Carlson form, 0 <= ph <= pi/2
            s, c = np.sin(ph), np.cos(ph)
            return (j/(3*b**2))*s**3*sp.elliprj(c*c, 1-j*s*s, 1., 1-nu*s*s)
        hphi = np.where(phi<=0.5*np.pi, h(np.minimum(phi, 0.5*np.pi)), 2*h(0.5*np.pi) - h(np.pi - np.maximum(phi, 0.5*np.pi)))
        s, c = np.sin(phi), np.cos(phi)
        Delta = np.sqrt(1-j*s*s)
        return hphi - (b/D0)*(T(W, b*Delta*c/s) + T(W, Delta/s))

    sgn = np.where(al<0, 1., -1.)
    phi_a = np.where(al<0, phi_a, np.pi - phi_a)
    phi_b = np.where(al<0, phi_b, np.pi - phi_b)
    return sgn*(R1_reg(phi_a) - R1_reg(phi_b))
