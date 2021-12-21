# Calculate lensed curve in image plane of constant radius in source plane
# Gralla & Lupsasca 10 section VI C
# https://arxiv.org/pdf/1910.12873.pdf

import numpy as np
import scipy.special as sp
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
from kerr_raytracing_utils import my_cbrt, radial_roots, mino_total, is_outside_crit, uplus_uminus
from kerr_raytracing_ana import r_integrate
import ehtim.parloop as parloop
import ehtim.observing.obs_helpers as obsh
from multiprocessing import cpu_count, Pool
import os

# TODO -- vectorize / numba!
#from numba import jit

INF = 1.e50
R0 = np.infty
NPROC = 10

def nmax_equatorial(a, r_o, th_o, alpha, beta):
    """Return the maximum number of equatorial crossings"""

    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    if not (isinstance(r_o,float) and (r_o>=100)):
        raise Exception("r_o should be a float > 100")
    if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
        raise Exception("th_o should be a float in range (0,pi/2]")

    if not isinstance(alpha, np.ndarray): alpha = np.array([alpha]).flatten()
    if not isinstance(beta, np.ndarray): beta = np.array([beta]).flatten()
    if len(alpha) != len(beta):
        raise Exception("alpha, beta are different lengths!")

    # conserved quantities
    lam = -alpha*np.sin(th_o)
    eta = (alpha**2 - a**2)*np.cos(th_o)**2 + beta**2

    # output arrays
    Nmax = np.empty(alpha.shape)

    # vortical motion
    vortmask = (eta<=0)
    Nmax[vortmask] = -2

    # regular motion
    if np.any(~vortmask):

        lam_reg = lam[~vortmask]
        eta_reg = eta[~vortmask]
        beta_reg = beta[~vortmask]

        # angular turning points
        (u_plus, u_minus, uratio, a2u_minus) = uplus_uminus(a,th_o,lam_reg,eta_reg)

        # equation 12 for F0
        # checks on xFarg should be handled in uplus_uminus function
        xFarg = np.cos(th_o)/np.sqrt(u_plus)
        F0 = sp.ellipkinc(np.arcsin(xFarg), uratio)

        # equation 17 for K
        K = sp.ellipkinc(0.5*np.pi, uratio)

        # radial roots
        (r1,r2,r3,r4,rclass) = radial_roots(a,lam_reg,eta_reg)

        # total Mino time
        Imax_reg = mino_total(a, r_o, eta_reg, r1, r2, r3, r4)

        #Equation 81 for the number of maximual crossings Nmax_eq
        #Note that eqn C8 of Gralla, Lupsasca, Marrone has a typo!
        #using sign(0) = 1
        Nmax_reg = np.empty(Imax_reg.shape)

        betamask = (beta_reg<=0)
        if np.any(betamask):
            Nmax_reg[betamask] = (np.floor((Imax_reg*np.sqrt(-u_minus*a**2) - F0) / (2*K)))[betamask]
        if np.any(~betamask):
            Nmax_reg[~betamask] = (np.floor((Imax_reg*np.sqrt(-u_minus*a**2) + F0) / (2*K)) - 1)[~betamask]

    # return data
    Nmax[~vortmask] = Nmax_reg

    return Nmax

def r_equatorial(a, r_o, th_o, mbar, alpha, beta):
    """Return (r_s, Ir, Imax, Nmax) where

       r_s is the equatorial emission radius
       Ir is the elapsed Mino time at emission
       Imax is the maximal Mino time on the geodesic
       Nmax is the maximum number of equatorial crossings"""

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

    # conserved quantities
    lam = -alpha*np.sin(th_o)
    eta = (alpha**2 - a**2)*np.cos(th_o)**2 + beta**2

    # output arrays
    r_s = np.empty(alpha.shape)
    Ir = np.empty(alpha.shape)
    Imax = np.empty(alpha.shape)
    Nmax = np.empty(alpha.shape)

    # vortical motion
    vortmask = (eta<=0)
    r_s[vortmask] = -1
    Ir[vortmask] = -1
    Imax[vortmask] = -1
    Nmax[vortmask] = -2

    # regular motion
    if np.any(~vortmask):

        lam_reg = lam[~vortmask]
        eta_reg = eta[~vortmask]
        beta_reg = beta[~vortmask]

        # angular turning points
        (u_plus, u_minus, uratio, a2u_minus) = uplus_uminus(a,th_o,lam_reg,eta_reg)

        # equation 12 for F0
        # checks on xFarg should be handled in uplus_uminus function
        xFarg = np.cos(th_o)/np.sqrt(u_plus)
        F0 = sp.ellipkinc(np.arcsin(xFarg), uratio)

        # equation 17 for K
        K = sp.ellipkinc(0.5*np.pi, uratio)

        # which subring are we in?
        m = mbar + np.heaviside(beta_reg, 0)

        # radial roots
        (r1,r2,r3,r4,rclass) = radial_roots(a,lam_reg,eta_reg)

        # total Mino time
        Imax_reg = mino_total(a, r_o, eta_reg, r1, r2, r3, r4)

        #Equation 81 for the elapsed mino time Ir at the equator
        #and number of maximual crossings Nmax_eq
        #Note that eqn C8 of Gralla, Lupsasca, Marrone has a typo!
        #using sign(0) = 1
        Nmax_reg = np.empty(Imax_reg.shape)
        Ir_reg = np.empty(Imax_reg.shape)

        betamask = (beta_reg<=0)
        if np.any(betamask):
            Nmax_reg[betamask] = (np.floor((Imax_reg*np.sqrt(-u_minus*a**2) - F0) / (2*K)))[betamask]
            Ir_reg[betamask] = ((2*m*K + F0)/np.sqrt(-u_minus*a**2))[betamask]
        if np.any(~betamask):
            Nmax_reg[~betamask] = (np.floor((Imax_reg*np.sqrt(-u_minus*a**2) + F0) / (2*K)) - 1)[~betamask]
            Ir_reg[~betamask] = ((2*m*K - F0)/np.sqrt(-u_minus*a**2))[~betamask]

        # calculate the emission radius given the elapsed mino time
        # TODO -- clean up hacky indexing here
        r_s_reg,_,_,_ = r_integrate(a,r_o,lam_reg,eta_reg,
                                    r1,r2,r3,r4,Ir_reg.reshape(1,len(Ir_reg)),
                                    do_phi_and_t=False)
        r_s_reg = r_s_reg[0]
        r_s_reg[Nmax_reg < mbar] = 0

        # return data
        r_s[~vortmask] = r_s_reg
        Ir[~vortmask] = Ir_reg
        Imax[~vortmask] = Imax_reg
        Nmax[~vortmask] = Nmax_reg

    return (r_s, Ir, Imax, Nmax)

def r_equatorial2(rho, varphi, a, th0, mbar=0):

    # varphi range is [-180,180)

    # image plane coordinates
    alpha = np.cos(varphi)*rho
    beta = np.sin(varphi)*rho

    (r, Ir, Imax, Nmax) = r_equatorial(a, R0, th0, mbar, alpha, beta)


    # fudge to make things continuous
    r = r[0] # TODO 1D
    outside_crit = is_outside_crit(a, th0, alpha, beta)[0]
    if r<=0 and outside_crit==1: #
        r = INF
    elif r<=0:
        r = 1 # this will still be discontinous at the horizon....

    return r

def objfunc(rho, varphi, a, th0, r_target, mbar=0):

    rguess = r_equatorial2(rho, varphi, a, th0, mbar=mbar)
    delta = rguess - r_target

    return delta

def rho_of_req(a, th0, req, mbar=0):
    varphis = np.linspace(-180,179,360)*np.pi/180.

    # bounding ranges eyeballed from Gralla+Lupsasca Fig 6
    if mbar==0:
        rhomin = 0
        rhomax = req + 5
    elif mbar==1:
        rhomin = 0
        rhomax = 100
    else:
        rhomin=1
        rhomax=10

    rhos = np.array([brentq(objfunc, rhomin, rhomax, args=(varphi, a, th0, req, mbar)) for varphi in varphis])
    alphas = np.cos(varphis)*rhos
    betas = np.sin(varphis)*rhos
    return(varphis,rhos,alphas, betas)

def critical_curve(a, th0, n=100000):
    """returns parametrized critical curve (alpha,beta) array with n points
       won't work for a=0 exactly"""

    # this parameterization requires very high accuracy nera beginning/end points to get equatorial slice!

    # beginning and ending points, Eq 40
    rminus = 2*(1 + np.cos(2.*np.arccos(-a)/3.))
    rplus = 2*(1 + np.cos(2.*np.arccos(a)/3.))
    r = np.linspace(rminus,rplus,n)

    # conserveds, Eq 38,39
    #delta = r**2 - 2*r + a**2
    #lam = a + (r/a)*(r - 2*delta/(r-1.))
    #eta = (r**3/a**2)*(4*delta/((r-1.)**2) - r)

    lam = -(r**3 + a**2 + r*(a**2) -3*r**2)/(a*r-a)
    eta = (r**2) * (3*r**2 + a**2 - lam**2)/(r**2-a**2)

    # screen coordinates
    alpha = -lam / np.sin(th0)

    beta = np.sqrt(eta + (a*np.cos(th0))**2 - (lam/np.tan(th0))**2)

    # fails a lot...
    mask = np.isnan(alpha) + np.isnan(beta)
    alpha = alpha[~mask]
    beta = beta[~mask]
    alpha = np.hstack((alpha, np.flip(alpha)))
    beta = np.hstack((beta, -np.flip(beta)))
    return (alpha,beta)

def plot_curves(a, th0, reqs=[3,5,7], ms=[0,1,2]):

    colors=['blue','green','purple']
    lines = ['-','--',':']
    if len(reqs)>len(colors): raise Exception('need more colors!')
    if len(ms)>len(lines): raise Exception('need more lines!')

    # preamble
    fig = plt.figure()
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.set_xticks(range(-8,10,2))
    ax.set_yticks(range(-8,10,2))
    ax.set_xticks(range(-9,10,2),minor=True)
    ax.set_yticks(range(-9,10,2),minor=True)
    ax.set_xlim(-9,9)
    ax.set_ylim(-9,9)
    plt.grid()
    ax.set_aspect('equal')

    # first do the critical curve
    acrit, bcrit = critical_curve(a,th0)
    plt.plot(acrit,bcrit,'r-')

    # next do the filled m=0 horizon
    rh = 1 + np.sqrt(1-a**2)
    varphis,rhos,alphas,betas = rho_of_req(a, th0, rh, mbar=0)
    f_low = interp1d(alphas[varphis<0], betas[varphis<0],kind=3,fill_value='extrapolate')
    plt.fill_between(alphas[varphis>0], f_low(alphas[varphis>0]), betas[varphis>0], ec=None,fc='k',alpha=.3)

    # do the other horizon curves
    alphas0 = alphas
    betas0 = betas
    for j in range(len(ms)):
        print('rh', ms[j])
        if ms[j] == 0:
            alphas = alphas0
            betas=betas0
        else:
            varphis,rhos,alphas,betas = rho_of_req(a, th0, rh, mbar=ms[j])
        plt.plot(alphas,betas,'k-', color='k', ls=lines[j], label=r'$r_{eq}=r_{h}, m=%i$'%(ms[j]))

    # now do the other curves
    for i in range(len(reqs)):
        for j in range(len(ms)):
            print(reqs[i], ms[j])
            varphis,rhos,alphas,betas = rho_of_req(a, th0, reqs[i], mbar=ms[j])
            plt.plot(alphas,betas,'k-', color=colors[i], ls=lines[j], label=r'$r=%.2f, m=%i$'%(reqs[i],ms[j]))

    #plt.legend()

    return fig

def generate_library(which='rh'):
    outpath = './curve_library_' + which

    print(outpath)
    processes=NPROC
    spins = np.linspace(0.01,0.99,99)
    incs = np.linspace(1,89,89)
    mmax = 5

    allincs, allspins = np.meshgrid(incs,spins)
    allincs = allincs.flatten()
    allspins = allspins.flatten()
    ntot = len(allincs)

    counter = parloop.Counter(initval=0, maxval=ntot)
    print("Using Multiprocessing with %d Processes" % processes)
    pool = Pool(processes=processes, initializer=init, initargs=(counter,))

    out = pool.map(make_curves,
                     [[i, which, ntot,
                       allspins[i], allincs[i], mmax, outpath]
                      for i in range(ntot)
                      ])
    pool.close()

def make_curves(args):
    return make_curves2(*args)

def make_curves2(i, which, n, spin, inc, mmax, outpath):
    a = spin
    rh = 1 + np.sqrt(1-a**2)
    th0 = inc*np.pi/180.

    if which=='rh':
        rs = rh
    elif which=='r10':
        rs = 10.
    elif which=='r100':
        rs = 100.
    elif which=='r1000':
        rs = 1000.
    elif which=='r10000':
        rs = 10000.
    else:
        raise Exception()

    if n > 1:
        global counter
        counter.increment()
        obsh.prog_msg(counter.value(), counter.maxval, 'bar', counter.value() - 1)

    # TODO RENAME
    if os.path.isfile('%s/a%02.0f_i%02.0f_%s.txt'%(outpath,100*spin,inc,which)):
        return

    try:
        for mbar in range(mmax+1):
            varphis,rhos,alphas,betas = rho_of_req(a, th0, rs, mbar=mbar)
            if mbar==0:
                out = np.vstack((varphis,rhos)).T
            else:
                out = np.hstack((out, rhos.reshape(-1,1)))

        np.savetxt('%s/a%02.0f_i%02.0f_%s.txt'%(outpath,100*spin,inc,which),out)
    except:
        return



def init(x):
    global counter
    counter = x
