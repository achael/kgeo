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

import numpy as np
import scipy.special as sp
import mpmath
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from kerr_raytracing_utils import *
from gsl_ellip_binding import ellip_pi_gsl
import h5py

ROUT = 1000 #4.e10 # sgra distance in M
NGEO = 200
NPIX = 200
EP = 1.e-12
MAXTAUFRAC = (1. - 1.e-10) # NOTE: if we go exactly to tau_tot t and phi diverge on horizon
MINSPIN = 1.e-6 # minimum spin for full formulas to work before taking limits.

# mpmath elliptic functions are SLOW -- use scipy and GSL
ellipf_arr = np.frompyfunc(mpmath.ellipf,2,1) # SLOW, use sp.ellipkinc
ellipe_arr = np.frompyfunc(mpmath.ellipe,2,1) # SLOW, use sp.ellipeinc
ellippi_arr = np.frompyfunc(mpmath.ellippi,3,1) # SLOW, use ellip_pi_gsl
sn_arr = np.frompyfunc(mpmath.ellipfun('sn'),2,1) # SLOW, use sp.ellipj and transforms
sc_arr = np.frompyfunc(mpmath.ellipfun('sc'),2,1) # SLOW, use sp.ellipj and transforms

# GSL elliptic functions
ellippi_arr_gsl = np.frompyfunc(ellip_pi_gsl,3,1)

def raytrace_ana(a=0.94, th_o=20*np.pi/180., r_o=ROUT,
                 alpha=np.linspace(-8,8,NPIX), beta=0*np.ones(NPIX), ngeo=NGEO,
                 savedata=False, plotdata=True):
    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be float in range [0,1)")
    if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
        raise Exception("th_o should be float in range (0,pi/2]")
    if not isinstance(alpha, np.ndarray): lam = np.array([lam]).flatten()
    if not isinstance(beta, np.ndarray): eta = np.array([eta]).flatten()
    if len(alpha) != len(beta):
        raise Exception("alpha, beta are different lengths!")

    print('calculating preliminaries...')
    # horizon radii
    rplus  = 1 + np.sqrt(1-a**2)
    rminus = 1 - np.sqrt(1-a**2)

    # conserved quantities
    lam = -alpha*np.sin(th_o)
    eta = (alpha**2 - a**2)*np.cos(th_o)**2 + beta**2

    # spin zero should have no voritical geodesics
    if(a<MINSPIN and np.any(eta<0)):
        eta[eta<0]=EP # TODO ok?
        print("WARNING: there were eta<0 points for spin %f<MINSPIN!"%a)

    # angular turning points
    (u_plus, u_minus, th_plus, th_minus, thclass) = angular_turning(a, th_o, lam, eta)

    # sign of final angular momentum
    s_o = my_sign(beta)

    # radial roots and radial motion case
    (r1, r2, r3, r4, rclass) = radial_roots(a, lam, eta)
    #if(a<MINSPIN):
#        r2 = np.zeros(r2.shape) # r2 should be exactly zero in zero spin limit #TODO check

    # total Mino time to infinity
    tau_tot = mino_total(a, r_o, eta, r1, r2, r3, r4)

    # find the steps in tau
    # go to taumax in the same number of steps on each ray -- step dtau depends on the ray
    dtau = MAXTAUFRAC*tau_tot / (ngeo - 1)
    tausteps = np.linspace(0, MAXTAUFRAC*tau_tot, ngeo) # positive back from screen in GL19b conventions


    # find the number of poloidal orbits as a function of time (GL 19b Eq 35)
    # Only applies for normal geodesics eta>0
    if(a<MINSPIN):
        uratio = 0.
        a2u_minus = -(eta+lam**2)
    else:
        uratio = u_plus/u_minus
        a2u_minus = a**2 * u_minus

    K = sp.ellipk(uratio) # gives NaN for eta<0
    n_all = (np.sqrt(-a2u_minus.astype(complex))*tausteps)/(4*K)
    n_all = np.real(n_all.astype(complex))
    n_tot = n_all[-1]

    # fractional number of equatorial crossings
    # Only applies for normal geodesics eta>0
    F_o = sp.ellipkinc(np.arcsin(np.cos(th_o)/np.sqrt(u_plus)), uratio) # gives NaN for eta<0
    Nmax_eq = ((tau_tot*np.sqrt(-a2u_minus.astype(complex)) + s_o*F_o) / (2*K))  + 1
    Nmax_eq[beta>=0] -= 1
    Nmax_eq = np.floor(np.real(Nmax_eq.astype(complex)))
    Nmax_eq[np.isnan(Nmax_eq)] = 0

    # integrate in theta
    print('integrating in theta...',end="\r")
    start = time.time()
    (th_s, G_ph, G_t) = th_integrate(a,th_o,s_o,lam,eta,u_plus,u_minus,tausteps)
    stop = time.time()
    print('integrating in theta...%0.2f s'%(stop-start))

    # integrate in r1
    print('integrating in r...',end="\r")
    start = time.time()
    (r_s, I_ph, I_t, I_sig) = r_integrate(a,r_o,lam,eta, r1,r2,r3,r4,tausteps)
    stop = time.time()
    print('integrating in r...%0.2f s'%(stop-start))

    #old unified formulas (slow)
    #th_s0 = th_integrate_old(a,th_o,s_o,eta,u_plus,u_minus,tausteps)
    #r_s0 = r_integrate_old(r_o, eta, r1,r2,r3,r4,tausteps)

    # combine to get phi, t, and sigma as a function of time
    sig_s = 0 + I_sig + a**2 * G_t # GL19a 15
    t_s = 0 + I_t + a**2 * G_t # GL19a 12
    ph_s = 0 + I_ph + lam*G_ph # GL19a 11

    # put phi in range (-pi,pi)
    #ph_s = np.mod(ph_s - np.pi, 2*np.pi) - np.pi

    if savedata:
        print('saving data...')
        try:
            savegeos(a,th_o,r_o,alpha, beta,n_tot,Nmax_eq,tausteps,t_s,r_s,th_s,ph_s,sig_s)
        except:
            print("Error saving to file!")
    if plotdata:
        print('plotting data...')
        plotgeos(a,th_o,r_o,Nmax_eq,r_s,th_s,ph_s)

    print('done!')
    return(n_tot,Nmax_eq,tausteps,t_s,r_s,th_s,ph_s,sig_s)

def savegeos(a,th_o,r_o,alpha,beta,n_tot,Nmax_eq,tausteps,t_s,r_s,th_s,ph_s,sig_s):
    fname = 'a%0.2f_th%0.2f_geo.h5'%(a,th_o*180/np.pi)
    hf = h5py.File(fname,'w')
    hf.create_dataset('spin',data=a)
    hf.create_dataset('inc',data=th_o)
    hf.create_dataset('alpha',data=alpha)
    hf.create_dataset('beta',data=beta)
    hf.create_dataset('t',data=t_s)
    hf.create_dataset('r',data=r_s)
    hf.create_dataset('theta',data=th_s)
    hf.create_dataset('phi',data=ph_s)
    hf.create_dataset('affine',data=sig_s)
    hf.create_dataset('mino',data=tausteps)
    #hf.create_dataset('eq_crossings',data=Nmax_eq)
    #hf.create_dataset('frac_orbits',data=n_tot)
    hf.close()

def plotgeos(a,th_o,r_o,Nmax_eq,r_s,th_s,ph_s,xlim=10,rmax=12):
    rplus  = 1 + np.sqrt(1-a**2)

    # convert to cartesian for plotting
    x_s = r_s * np.cos(ph_s) * np.sin(th_s)
    y_s = r_s * np.sin(ph_s) * np.sin(th_s)
    z_s = r_s * np.cos(th_s)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    ax.plot_surface(rplus*np.cos(u) * np.sin(v),  rplus*np.sin(u) * np.sin(v),  rplus*np.cos(v), color='black')

    rr, thth = np.mgrid[0:xlim, 0:2*np.pi:20j]
    xx = rr*np.cos(thth); yy = rr*np.sin(thth)
    zz = np.zeros(xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=0.5)
    ax.set_xlim(-xlim,xlim)
    ax.set_ylim(-xlim,xlim)
    ax.set_zlim(-xlim,xlim)
    ax.auto_scale_xyz([-xlim, xlim], [-xlim, xlim], [-xlim, xlim])
    ax.set_axis_off()

    x_o = 1.5*rmax * np.cos(0) * np.sin(th_o)
    y_o = 1.5*rmax * np.sin(0) * np.sin(th_o)
    z_o = 1.5*rmax * np.cos(th_o)
    ax.plot3D([0,x_o],[0,y_o],[0,z_o],'black',ls='dashed')

    maxwraps = int(np.nanmax(Nmax_eq))
    colors = ['k','b','g','orange','r','m']
    NPLOT=50
    for j in range(maxwraps+1):
        mask = (Nmax_eq==j)
        color = colors[j]
        xs = x_s[:,mask];ys = y_s[:,mask];zs = z_s[:,mask];rs = r_s[:,mask];
        nplot = int(np.ceil(NPLOT*xs.shape[-1]/NPIX))
        if xs.shape[-1] < 5 or j>=3:
            geos = range(xs.shape[-1])
        else:
            geos = range(0,xs.shape[-1],xs.shape[-1]//nplot)

        for i in geos:
            x = xs[:,i]; y=ys[:,i]; z=zs[:,i]
            mask = rs[:,i] < rmax
            x = x[mask]; y = y[mask]; z = z[mask]
            ax.plot3D(x,y,z,color)
    return

def th_integrate(a,th_o, s_o,lam, eta, u_plus, u_minus, tausteps):
    if not isinstance(s_o, np.ndarray): s_o = np.array([s_o]).flatten()
    if not isinstance(eta, np.ndarray): eta= np.array([eta]).flatten()
    if not isinstance(u_plus, np.ndarray): u_plus = np.array([u_plus]).flatten()
    if not isinstance(u_minus, np.ndarray): u_minus = np.array([u_minus]).flatten()
    if not(len(s_o)==len(eta)==len(u_plus)==len(u_minus)):
        raise Exception("inputs to th_integrate not the same length!")
    if not(tausteps.shape[1]==len(s_o)):
        raise Exception("tausteps has incompatible shape in th_integrate!")

    # output arrays
    th_s = np.zeros(tausteps.shape)
    G_ph = np.zeros(tausteps.shape)
    G_t  = np.zeros(tausteps.shape)

    # ordinary motion:
    if(np.any(eta>0.)):
        mask = eta>=0.
        up = u_plus[mask]
        um = u_minus[mask]
        s = s_o[mask]

        # compute factors up/um and a**2 * um (in zero spin limit)
        if(a<MINSPIN):
            uratio = np.zeros(eta.shape)
            a2um = -(eta+lam**2)
        else:
            uratio = up/um
            a2um = a**2 * um

        ## compute antideriviatives at the origin
        pref = 1/np.sqrt(-a2um)
        k = uratio # k<0 since um<0 for eta>0
        elliparg = np.arcsin(np.cos(th_o)/np.sqrt(up))

        #Gth_o, GL19a, 29
        F =  sp.ellipkinc(elliparg,k)
        Gth_o = -pref * F

        #Gph_o , GL 19a, 30
        Gph_o = -pref * ellippi_arr_gsl(up, elliparg, k)

        #Gt_o , GL 19a, 31
        maskk = (k==0)
        Gt_o = np.zeros(k.shape)
        if np.any(maskk): # limit as k-> 0, occurs when a==0
            Gt_o[maskk] = 0.125*(-2*elliparg + np.sin(2*elliparg))[maskk]
        if np.any(~maskk):
            Gt_o[~maskk] = 2*up*pref* (sp.ellipeinc(elliparg,k) - F)[~maskk]/(2*k[~maskk]) # GL 19a, 31

        ## compute the amplitude Phi_tau
        snarg = np.sqrt(-a2um)*(-tausteps[:,mask] + s*Gth_o)
        snarg = snarg.astype(float)

        sinPhi_tau = np.zeros(snarg.shape)
        Phi_tau = np.zeros(snarg.shape)

        # for very small arguments, use lim x->0 sn(x,k)=lim am(x,k)=x
        jmask = np.abs(snarg) < EP # TODO what value cutoff?
        if np.any(jmask):
            sinPhi_tau[jmask] = snarg[jmask]
            Phi_tau[jmask] = snarg[jmask]

        # for other arguments use some elliptic function identities
        # https://functions.wolfram.com/EllipticFunctions/JacobiAmplitude/introductions/JacobiPQs/ShowAll.html
        # TODO test accuracy more
        if np.any(~jmask):
            m = (k/(k-1)) # real, in (0,1) since k<0
            m = np.outer(np.ones(snarg.shape[0]),m)[~jmask]
            ellipfuns = sp.ellipj(snarg[~jmask]/np.sqrt(1-m), m)
            #sn(sqrt(1-m)x | k) = sqrt(1-m)*sn(x|m)/dn(x|m)
            sinPhi_tau[~jmask] = np.sqrt(1-m) * ellipfuns[0]/ellipfuns[2] #sn(sqrt(1-m)x | k) = sqrt(1-m)*sn(x|m)/dn(x|m)
            #am(sqrt(1-m)x | k) = pi/2 - am(K(m) - x | m for m <=1
            Phi_tau[~jmask] = 0.5*np.pi-sp.ellipj(sp.ellipk(m) - snarg[~jmask]/np.sqrt(1-m), m)[3]

        ## get the two angular integrals and the solution for theta_o

        # G_phi integral GL19a 47
        G_ph[:,mask] = (pref*ellippi_arr_gsl(up, Phi_tau, k) - s*Gph_o).astype(float)

        # G_t integral GL19a, 48
        maskk = (k==0)
        Gtout = np.zeros(Phi_tau.shape)
        if np.any(maskk): # limit as k-> 0, occurs when a==0
            Gtout[:,maskk] = 0.125*(-2*Phi_tau + np.sin(2*Phi_tau))[:,maskk]
        if np.any(~maskk):
            Gtout[:,~maskk] = (2*up*pref* (sp.ellipeinc(Phi_tau,k) - F))[:,~maskk]/(2*k[~maskk]) # GL 19a, 31
        G_t[:,mask] = Gtout

        # solution for theta_o GL19a 49
        th_s[:,mask] = (np.arccos(-s*np.sqrt(up)*sinPhi_tau)).astype(float)

    # vortical motion cause
    # most eta=0 exact points are a limit of vortical motion
    if(np.any(eta<=0.)):
        if(a<MINSPIN):
            raise Exception("below MINSPIN but there are eta<0 points in th_integrate!")

        mask = eta<0.
        up = u_plus[mask]
        um = u_minus[mask]
        s = s_o[mask]

        uratio = up/um
        a2um = a**2 * um

        ## compute antideriviatives at the origin
        h = 1. # sign(cos(th)) GL19a 54, we always consider northern hemisphere
        prefA = 1/np.sqrt(a2um) # u_minus>0 always for eta<0
        prefB = prefA / (1.-um)
        prefC = np.sqrt(um/a**2)
        Nu_o = np.arcsin(np.sqrt((np.cos(th_o)**2 - um)/(up-um))) # GL19a, 59
        k = 1 - uratio # again, k<0 always since for eta<0 up>um>0
        upm = (up-um)/(1.-um)

        #Gth_o , GL19a 56
        Gth_o = -h*prefA * sp.ellipkinc(Nu_o, k)

        #Gph_o , GL19a 57
        Gph_o = -h*prefB * ellippi_arr_gsl(upm,Nu_o,k)

        #Gt_o  , GL19a 58
        Gt_o  = -h*prefC * sp.ellipeinc(Nu_o,k)

        ## compute the amplitude Nu_tau
        snarg = np.sqrt(a2um)*(-tausteps[:,mask] + s*Gth_o)
        snarg = snarg.astype(float)

        sinNu_tau = np.zeros(snarg.shape)
        Nu_tau = np.zeros(snarg.shape)

        # for very small arguments, use lim x->0 sn(x,k)=lim am(x,k)=x
        jmask = np.abs(snarg) < EP # TODO what value cutoff?
        if np.any(jmask):
            sinNu_tau[jmask] = snarg[jmask]
            Nu_tau[jmask] = snarg[jmask]

        # for other arguments use some elliptic function identities
        # https://functions.wolfram.com/EllipticFunctions/JacobiAmplitude/introductions/JacobiPQs/ShowAll.html
        # TODO test accuracy more
        if np.any(~jmask):
            m = (k/(k-1)) # real, in (0,1) since k<0
            m = np.outer(np.ones(snarg.shape[0]),m)[~jmask]
            ellipfuns = sp.ellipj(snarg[~jmask]/np.sqrt(1-m), m)
            #sn(sqrt(1-m)x | k) = sqrt(1-m)*sn(x|m)/dn(x|m)
            sinNu_tau[~jmask] = np.sqrt(1-m) * ellipfuns[0]/ellipfuns[2] #sn(sqrt(1-m)x | k) = sqrt(1-m)*sn(x|m)/dn(x|m)
            #am(sqrt(1-m)x | k) = pi/2 - am(K(m) - x | m for m <=1
            Nu_tau[~jmask] = 0.5*np.pi-sp.ellipj(sp.ellipk(m) - snarg[~jmask]/np.sqrt(1-m), m)[3]

        ## get the two angular integrals and the solution for theta_o

        # G_phi integral GL19a 67
        G_ph[:,mask] = (prefB * ellippi_arr_gsl(upm,Nu_tau,k) - s*Gph_o).astype(float)

        # G_t integral GL19a, 68
        G_t[:,mask] = (prefC * sp.ellipeinc(Nu_tau,k) - s*Gt_o).astype(float)

        # solution for theta_o GL19a 49
        th_s[:,mask] = (np.arccos(h*np.sqrt(um + (up-um)*sinNu_tau**2))).astype(float)

    return (th_s, G_ph, G_t)


def r_integrate(a,r_o,lam,eta, r1,r2,r3,r4,tausteps):

    # Follow G19a, interchange source <--> observer labels and send tau -> -tau

    # checks
    if not isinstance(lam, np.ndarray): lam  = np.array([lam]).flatten()
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if not isinstance(r1, np.ndarray): r1 = np.array([r1]).flatten()
    if not isinstance(r2, np.ndarray): r2 = np.array([r2]).flatten()
    if not isinstance(r3, np.ndarray): r3 = np.array([r3]).flatten()
    if not isinstance(r4, np.ndarray): r4 = np.array([r4]).flatten()
    if not(len(lam)==len(eta)==len(r1)==len(r2)==len(r3)==len(r4)):
        raise Exception("inputs to r_integrate not the same length!")
    if not(tausteps.shape[1]==len(eta)):
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
        if r_o==np.infty:
            x4ro = np.infty # GL19a, B83 for observer at r_o
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

        # auxillary functions
        def S1_S2(al,phi,j,ret_s2=True): #B92 and B93
            al2 = al*al
            al2p1 = 1 + al2
            al2j = 1 - j + al2
            s2phi = np.sqrt(1.- j*(np.sin(phi))**2)

            p2 = np.sqrt(al2p1 / al2j)
            f2 = 0.5*p2*np.log(np.abs(((1-p2)*(1+p2*s2phi))/((1+p2)*(1-p2*s2phi))))


            # definitions of functions in this region should keep things in the allowed prange
            F = sp.ellipkinc(phi,j)
            S1 = (F + al2*ellippi_arr_gsl(al2p1,phi,j) - al*f2) / al2p1

            if ret_s2:
                E = sp.ellipeinc(phi,j)
                tanphi = np.tan(phi)

                S2a = (1-j)*F + al2*E + al2*s2phi*(al - tanphi)/(1 + al*tanphi) - al2*al
                S2a = -S2a / (al2p1*al2j)
                S2b = S1*(1./al2p1 + (1-j)/al2j)
                S2 = S2a + S2b
            else:
                S2=np.NaN

            return (S1,S2)

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
        if r_o==np.infty:
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
        #amX3 = np.arcsin(ellipfuncs[0])  # arcsin does not account for full range!
        amX3 = ellipfuncs[3]
        cnX3 = ellipfuncs[1]
        rs_num = (BB*rr2 - AA*rr1) + (BB*rr2 + AA*rr1)*cnX3 # B75
        rs_denom = (BB-AA) + (BB+AA)*cnX3
        rs = rs_num / rs_denom
        r_s[:,mask_3] = rs

        # auxillary functions
        # al->0 limit??
        def R1_R2(al,phi,j,ret_r2=True): #B62 and B65
            al2 = al**2
            s2phi = np.sqrt(1-j*np.sin(phi)**2)
            p1 = np.sqrt((al2 -1)/(j+(1-j)*al2))
            f1 = 0.5*p1*np.log(np.abs((p1*s2phi+np.sin(phi))/(p1*s2phi-np.sin(phi))))
            nn = al2/(al2-1)
            R1 = (ellippi_arr_gsl(nn,phi,j) - al*f1)/(1-al2)

            if ret_r2:
                F = sp.ellipkinc(phi,j)
                E = sp.ellipeinc(phi,j)
                R2 = (F - (al2/(j+(1-j)*al2))*(E - al*np.sin(phi)*s2phi/(1+al*np.cos(phi)))) / (al2-1)
                R2 = R2 + (2*j - nn)*R1 / (j + (1-j)*al2)

            else:
                R2=np.NaN

            return (R1,R2)

        # # building blocks of the path integrals
        R1_a_0, R2_a_0 = R1_R2(al0,amX3,k3)
        R1_b_0, R2_b_0 = R1_R2(al0,auxarg,k3)
        R1_a_p, _ = R1_R2(alp,amX3,k3,ret_r2=False)
        R1_b_p, _ = R1_R2(alp,auxarg,k3,ret_r2=False)
        if a>MINSPIN:
            R1_a_m, _ = R1_R2(alm,amX3,k3,ret_r2=False)
            R1_b_m, _ = R1_R2(alm,auxarg,k3,ret_r2=False)
        else:
            R1_a_m = np.zeros(R1_a_p.shape)
            R1_b_m = np.zeros(R1_a_p.shape)

        Pi_1 = ((2*rr21*np.sqrt(AA*BB))/(BB**2-AA**2)) * (R1_a_0 - s*R1_b_0) # B81
        Pi_2 = ((2*rr21*np.sqrt(AA*BB))/(BB**2-AA**2))**2 * (R2_a_0 - s*R2_b_0) # B81
        Pi_p = ((2*rr21*np.sqrt(AA*BB))/(BB*rrp2 - AA*rrp1))*(R1_a_p - s*R1_b_p) # B82
        Pi_m = ((2*rr21*np.sqrt(AA*BB))/(BB*rrm2 - AA*rrm1))*(R1_a_m - s*R1_b_m) # B82

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

        x2rp = np.sqrt((rr31*rrp4)/(rr41*rrp3)) # GL19a, B35
        x2rp = np.sqrt((rr31*rrm4)/(rr41*rrm3)) # GL19a, B35
        if r_o==np.infty:
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
        rs_num = (rr4*rr31 - rr3*rr41*(snX2)**2) # B 46
        rs_denom = (rr31 - rr41*(snX2)**2)
        rs = rs_num / rs_denom
        r_s[:,mask_2] = rs

        # building blocks of the path integrals
        dX2dtau = -0.5*np.sqrt(rr31*rr42)
        dsn2dtau = 2*snX2*cnX2*dnX2*dX2dtau
        drsdtau = rr31*rr43*rr41*dsn2dtau / ((rr31 - rr41*snX2**2)**2)
        drsdtau *= -1 # TODO I *think* because we take tau->-tau we need this sign change
        Rpot_o = (r_o-rr1)*(r_o-rr2)*(r_o-rr3)*(r_o-rr4)
        drsdtau_o = s*np.sqrt(Rpot_o)
        H =  drsdtau / (rs - rr3) - drsdtau_o/(r_o - rr3) #B51 ???
        E = np.sqrt(rr31*rr42)*(sp.ellipeinc(amX2,k2) - s*sp.ellipeinc(auxarg,k2)) # B52
        Pi_1 = (2./np.sqrt(rr31*rr42))*(ellippi_arr_gsl(rr41/rr31,amX2,k2)-s*ellippi_arr_gsl(rr41/rr31,auxarg,k2)) # B53
        Pi_p = (2./np.sqrt(rr31*rr42))*(rr43/(rrp3*rrp4))*(ellippi_arr_gsl((rrp3*rr41)/(rrp4*rr31),amX2,k2)-s*ellippi_arr_gsl((rrp3*rr41)/(rrp4*rr31),auxarg,k2))
        Pi_m = (2./np.sqrt(rr31*rr42))*(rr43/(rrm3*rrm4))*(ellippi_arr_gsl((rrm3*rr41)/(rrm4*rr31),amX2,k2)-s*ellippi_arr_gsl((rrm3*rr41)/(rrm4*rr31),auxarg,k2))

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

    # get I_phi, I_t, I_sigma
    I_0 = -tausteps
    I_phi = (2*a/(rplus-rminus))*((rplus - 0.5*a*lam)*I_p - (rminus - 0.5*a*lam)*I_m) # B1
    I_tA = (4/(rplus-rminus))*((rplus**2 - 0.5*a*lam*rplus)*I_p - (rminus**2 - 0.5*a*lam*rminus)*I_m) # B2
    I_t = I_tA + 4*I_0 + 2*I_1 + I_2
    I_sig = I_2

    return (r_s, I_phi, I_t, I_sig)


def th_integrate_old(a,th_o, s_o, eta, u_plus, u_minus,tausteps):
    if not isinstance(s_o, np.ndarray): s_o = np.array([s_o]).flatten()
    if not isinstance(eta, np.ndarray): eta= np.array([eta]).flatten()
    if not isinstance(u_plus, np.ndarray): u_plus = np.array([u_plus]).flatten()
    if not isinstance(u_minus, np.ndarray): u_minus = np.array([u_minus]).flatten()
    if not(len(s_o)==len(eta)==len(u_plus)==len(u_minus)):
        raise Exception("inputs to th_integrate not the same length!")
    if not(tausteps.shape[1]==len(s_o)):
        raise Exception("tausteps has incompatible shape in th_integrate!")

    # GL 19b Eq 25
    s_eta = my_sign(eta) # GL 19b claim this works for eta<0!
    k = u_plus/u_minus

    xFarg = np.cos(th_o)/np.sqrt(u_plus)
    # need mpmath elliptic function for arbitrary k
    F_o = ellipf_arr(np.arcsin(xFarg),k).astype(complex)
    GG = s_o*a*np.sqrt(-u_minus.astype(complex))

    # compute the angular motion
    # need mpmath elliptic function for arbitrary k
    rhs = sn_arr((F_o + GG*tausteps), k)
    rhs = np.real(rhs.astype(complex))
    th_s = np.arccos(np.sqrt(u_plus)*rhs)

    return th_s


def r_integrate_old(r_o, eta, r1,r2,r3,r4,tausteps):
    if not isinstance(eta, np.ndarray): eta= np.array([eta]).flatten()
    if not isinstance(r1, np.ndarray): r1 = np.array([r1]).flatten()
    if not isinstance(r2, np.ndarray): r2 = np.array([r2]).flatten()
    if not isinstance(r3, np.ndarray): r3 = np.array([r3]).flatten()
    if not isinstance(r4, np.ndarray): r4 = np.array([r4]).flatten()
    if not(len(eta)==len(r1)==len(r2)==len(r3)==len(r4)):
        raise Exception("inputs to r_integrate not the same length!")
    if not(tausteps.shape[1]==len(eta)):
        raise Exception("tausteps has incompatible shape in r_integrate!")

    r31 = r3 - r1
    r32 = r3 - r2
    r41 = r4 - r1
    r42 = r4 - r2
    k = (r32 * r41) / (r31 * r42)

    if r_o==np.infty:
        x2ro = np.sqrt(r31/r41) # B35, for observer at r_o
    else:
        x2ro = np.sqrt((r31*(r_o-r4))/(r41*(r_o-r3)))# GL19a, B35

    # Find r_s
    auxarg = np.arcsin(x2ro)
    F_o = ellipf_arr(auxarg,k).astype(complex)
    #F_o = ellipf_arr(np.arcsin(np.sqrt(r31/r41)),k).astype(complex)
    xx = .5*np.sqrt(r31*r42)

    # compute the radial motion
    sn2 = (sn_arr(xx*tausteps - F_o, k).astype(complex))**2
    r_s = (r4*r31 - r3*r41*sn2) / (r31 - r41*sn2)
    r_s = np.real(r_s)

    return r_s
