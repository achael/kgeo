# Hacky numeric raytracing from formalism in Gralla+Lupsasa 2019a,b
# 19a: https://arxiv.org/pdf/1910.12881.pdf
# 19b: https://arxiv.org/pdf/1910.12873.pdf

import numpy as np
import scipy.special as sp
import mpmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
from kerr_raytracing_utils import *
import h5py
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

SPIN = 0.94
INC = 20*np.pi/180.
ROUT = 1000 #4.e10 # sgra distance in M
NGEO = 1000
NPIX = 100
EP = 1.e-12
MAXTAUFRAC = (1. - 1.e-10) # NOTE: if we go exactly to tau_tot t and phi diverge on horizon
MINSPIN = 1.e-6 # minimum spin for full formulas to work before taking limits.

pix_1d = np.linspace(-6,0,NPIX)
alpha_default = pix_1d
beta_default = 0*pix_1d + 1.e-1
#alpha_default = np.hstack((pix_1d,0*pix_1d+1.e-2))
#beta_default = np.hstack((0*pix_1d,pix_1d))

def raytrace_num(a=SPIN, 
                 observer_coords = [0,ROUT,INC,0],
                 image_coords = [alpha_default, beta_default],
                 ngeo=NGEO,
                 savedata=False, plotdata=False):

    tstart = time.time()
    
    [_, r_o, th_o, _] = observer_coords # assumes ph_o = 0
    [alpha, beta] = image_coords 
                     
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

    # sign of final angular momentum
    s_o = my_sign(beta)
    
    # spin zero should have no voritical geodesics
    if(a<MINSPIN and np.any(eta<0)):
        eta[eta<0]=EP # TODO ok?
        print("WARNING: there were eta<0 points for spin %f<MINSPIN!"%a)

    # angular turning points
    (u_plus, u_minus, th_plus, th_minus, thclass) = angular_turning(a, th_o, lam, eta)

    # radial roots and radial motion case
    (r1, r2, r3, r4, rclass) = radial_roots(a, lam, eta)

    # total Mino time to infinity
    tau_tot = mino_total(a, r_o, eta, r1, r2, r3, r4)

    # define steps equally spaced in Mino time tau
    # rays have equal numbers of steps -- step size dtau depends on the ray
    # mino time is positive back from screen in GL19b conventions    
    dtau = MAXTAUFRAC*tau_tot / (ngeo - 1)
    tausteps = np.linspace(0, MAXTAUFRAC*tau_tot, ngeo) 

    # numerically integrate
    # TODO this method isn't very precise b/c of hacky pushes at turning points
    print('integrating...')

    sig_s = []
    th_s = []
    ph_s = []
    r_s = []
    t_s = []
    for i in tqdm(range(NPIX)): # TODO parallelize
        tau_num, x_num = integrate_geo_single(a,th_o,r_o,
                                              alpha[i],beta[i],
                                              tau_tot[i],
                                              ngeo=ngeo,verbose=False)
                 
        # interpolate onto regular spaced grid in tau                             
        t_s.append(interp1d(-tau_num,x_num[0],fill_value='extrapolate')(tausteps[:,i]))
        r_s.append(interp1d(-tau_num,x_num[1],fill_value='extrapolate')(tausteps[:,i]))
        th_s.append(interp1d(-tau_num,x_num[2],fill_value='extrapolate')(tausteps[:,i]))
        ph_s.append(interp1d(-tau_num,x_num[3],fill_value='extrapolate')(tausteps[:,i]))
        sig_s.append(interp1d(-tau_num,x_num[4],fill_value='extrapolate')(tausteps[:,i]))

    # create Geodesics object
    affinesteps = np.array(sig_s).T
    geo_coords = [np.array(t_s).T,np.array(r_s).T,np.array(th_s).T,np.array(ph_s).T]
    geos = Geodesics(a, observer_coords, image_coords, tausteps, sig_s, geo_coords)
    
    if savedata and do_phi_and_t:
        print('saving data...')
        try:
            geos.savegeos('./numeric_')
        except:
            print("Error saving to file!")
    if plotdata and do_phi_and_t:
        print('plotting data...')
        try:
            plt.ion()
            geos.plotgeos()
            plt.show()
        except:
            print("Error plotting data!")   
                   
    tstop = time.time()
    print('done!  ', tstop-tstart, ' seconds!')
    return geos

# directly integrate
def dxdtau(tau,x,a,lam,eta,sr,sth):
    t = x[0]
    r = x[1]
    th = x[2]
    ph = x[3]
    sig = x[4]

    Delta = r**2 - 2*r + a**2
    Sigma = r**2 + (a**2) * (np.cos(th)**2)

    R = (r**2 + a**2 -a*lam)**2 - Delta*(eta + (lam-a)**2)
    TH = eta + (a*np.cos(th))**2 - (lam/np.tan(th))**2

    if R<0: R=0.
    if TH<0: TH=0.

    dt = (r**2 + a**2)*(r**2 + a**2 - a*lam)/Delta + a*(lam-a*np.sin(th)**2)
    dr = sr*np.sqrt(R)
    dth = sth*np.sqrt(TH)
    dph = a*(r**2 + a**2 - a*lam)/Delta + lam/(np.sin(th)**2) - a
    dsig = Sigma

    return np.array([dt,dr,dth,dph,dsig])

def jac(tau,x,a,lam,eta,sr,sth):
    t = x[0]
    r = x[1]
    th = x[2]
    ph = x[3]
    sig = x[4]

    Delta = r**2 - 2*r + a**2
    Sigma = r**2 + a**2 * np.cos(th)**2

    R = (r**2 + a**2 -a*lam)**2 - Delta*(eta + (lam-a)**2)
    TH = eta + (a*np.cos(th))**2 - (lam/np.tan(th))**2
    if R<0: R=0.
    if TH<0: TH=0.

    jacout = np.empty((5,5))
    jacout[0,0] = 0.
    jacout[0,1] = (-2*a*lam+4*a**2*r+4*r**3)/Delta - (2*r-2)*(a**4-2*a*lam*r+2*a**2*r**2+r**4)/(Delta**2)
    jacout[0,2] = -2*a**2*np.cos(th)*np.sin(th)
    jacout[0,3] = 0.
    jacout[0,4] = 0.

    jacout[1,0] = 0.
    jacout[1,1] = sr*(-(2*r-2)*(eta+(lam-a)**2) + 4*r*(a**2-a*lam+r**2))/(2*np.sqrt(R))
    jacout[1,2] = 0.
    jacout[1,3] = 0.
    jacout[1,4] = 0.

    jacout[2,0] = 0.
    jacout[2,1] = 0.
    jacout[2,2] = sth*((2*lam**2)/(np.tan(th)*np.sin(th)**2) - 2*a**2*np.cos(th)*np.sin(th))/(2*np.sqrt(TH))
    jacout[2,3] = 0.
    jacout[2,4] = 0.

    jacout[3,0] = 0.
    jacout[3,1] = 2*a*(a**2 + a*lam*(r-1)-r**2)/(Delta**2)
    jacout[3,2] = -2*lam/(np.tan(th)*np.sin(th)**2)
    jacout[3,3] = 0.
    jacout[3,4] = 0.

    jacout[4,0] = 0.
    jacout[4,1] = 2*r
    jacout[4,2] = -2*(a**2)*np.cos(th)*np.sin(th)
    jacout[4,3] = 0.
    jacout[4,4] = 0.

    return jacout

def eventR(t, x, a, lam, eta, sr, sth):
    t = x[0]
    r = x[1]
    th = x[2]
    ph = x[3]

    Delta = r**2 - 2*r + a**2
    R = (r**2 + a**2 -a*lam)**2 - Delta*(eta + (lam-a)**2)

    return R
eventR.terminal = True

def eventTH(t, x, a, lam, eta, sr, sth):
    t = x[0]
    r = x[1]
    th = x[2]
    ph = x[3]

    TH = eta + (a*np.cos(th))**2 - (lam/np.tan(th))**2

    if a<MINSPIN and eta<EP: # this is equatorial motion, TODO ok hack?
        TH=1
    return TH
eventTH.terminal = True

# TODO this method isn't terribly precise, because of kludge in swapping signs
def integrate_geo_single(a,th_o, r_o,aa,bb,taumax,ngeo=NGEO,verbose=False):
    #ll = lam[i]
    #ee = eta[i]
    #tmax = -taumax[i]
    #bb = beta[i]
    sr = 1
    sth = int(np.sign(bb))
    if sth==0: sth=1

    if np.abs(bb)<1.e-6 and th_o!=np.pi/2.: #TODO numeric integration does not work exactly on beta=0.
        bb = sth*1.e-6

    ll = -aa*np.sin(th_o)
    ee = (aa**2 - a**2)*np.cos(th_o)**2 + bb**2

    tmax = -taumax # define tau positive in input, negative back into spacetime
    ts = []
    xs = []
    x0 = np.array([0,r_o,th_o,0,0])
    t0 = 0.
    x = x0
    t = t0
    max_step = np.abs(tmax/ngeo)
    min_step = np.abs(tmax/(ngeo*100))


    nswitch = 0
    while True:
        sol = solve_ivp(dxdtau, (t,tmax), x, 
                        method='DOP853', max_step=max_step,
                        #jac=jac,
                        rtol=1.e-8,atol=1.e-8,
                        args=(a,ll,ee,sr,sth), events=(eventTH,eventR))

        if verbose:
            print('status:', sol.status)

        ts.append(sol.t)
        xs.append(sol.y)

        if nswitch > 10:
            break
        if sol.status == 1:
            t = sol.t[-1]
            x = (sol.y[:,-1].copy())
            tm1 = sol.t[-2]
            xm1 = (sol.y[:,-2].copy())

            if sol.t_events[1].size != 0:
                sr *= -1
                rturn = sol.y_events[1][0][1]
                if verbose:
                    print('changing r sign')
            if sol.t_events[0].size != 0:
                sth *= -1
                thturn = sol.y_events[0][0][2]
                if verbose:
                    print('changing th sign',sth)

            fac = 1.e-8
            dt = fac*(t-tm1)
            dx = fac*(x-xm1)
            t = t-dt
            x = x-dx

            nswitch += 1
        else:
            break


    ts = np.concatenate(ts)
    xs = np.concatenate(xs, axis=1)

    # transform phi to range (-pi/2,pi/2)
    #xs[3] = np.mod(xs[3] - np.pi, 2*np.pi) - np.pi  # put in range (-pi,pi)

    return (ts, xs)
