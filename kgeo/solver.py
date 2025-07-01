#solves where null geodesics cross fieldlines of fixed psi

import sys
import numpy as np
from kgeo.equatorial_images import Iobs
from kgeo.equatorial_lensing import rho_of_req, critical_curve
import ehtim as eh
import matplotlib.pyplot as plt
from kgeo.bfields import Bfield
import kgeo.bfields as kb
from kgeo.velocities import Velocity
from kgeo.emissivities import Emissivity
from scipy.interpolate import interp1d
import numpy as np
import scipy.special as sp
import mpmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
from kgeo.kerr_raytracing_utils import *
from kgeo.equatorial_images import *
from kgeo.kerr_raytracing_ana import *
import h5py
import scipy


MAXTAUFRAC = (1. - 1.e-10) # NOTE: if we go exactly to tau_tot t and phi diverge on horizon

#psi for parabolic BZ solution
def psi_BZ_para(r, theta, a): #theta in radians
    abscth = np.abs(np.cos(theta))
    rp = 1+np.sqrt(1-a**2)
    return r*(1-abscth)+rp*(1+abscth)*(1-np.log(1+abscth))-2*rp*(1-np.log(2))

#psi for monopole BZ solution
def psi_BZ_mono(theta):
    return 1-np.abs(np.cos(theta))

#psi for power law jet approximation
def psi_power(r,theta,pval):
    return r**pval*(1-np.abs(np.cos(theta)))

#general psi
def psifunc(r, theta, a, model='para', pval=1): #defines psi
    if model == 'para':
        return psi_BZ_para(r, theta, a) 
    elif model=='mono':
        return psi_BZ_mono(theta)
    elif model == 'power':
        return psi_power(r,theta,pval)
    else:
        return 0

#counts number of equatorial crossings on way from source to observer
def getneq(a, tau, u_minus, u_plus, th_s, th_o, signptheta, betas): 
    #combine equations 81 and 82 of GL Lensing
    uratio = u_plus/u_minus
    xFarg = np.cos(th_o)/np.sqrt(u_plus)
    F0 = sp.ellipkinc(np.arcsin(xFarg), uratio)

    # equation 17 for K
    K = sp.ellipkinc(0.5*np.pi, uratio)

    #m stuff
    mval = (np.sqrt(-u_minus*a**2)*tau + np.sign(betas)*F0)/(2*K)
    mbarval = mval - np.heaviside(betas, 0)

    return np.ceil(np.nan_to_num(mbarval)) #should always be an integer

#restrict solutions with tau<taumax
def get_maxtau_forwardjet(a, r_o, th_o, alpha, beta, neqmax=1): #maximum minotime before first mth crossing
    lam = -alpha*np.sin(th_o)
    eta = (alpha**2 - a**2)*np.cos(th_o)**2 + beta**2
        
    (r1, r2, r3, r4, rclass) = radial_roots(a, lam, eta)
    tau_tot = mino_total(a, r_o, eta, r1, r2, r3, r4)
    tau_tot2 = tau_tot * MAXTAUFRAC

    if neqmax == None: #no limit to geodesic length
        return tau_tot2

    # angular turning points
    (u_plus, u_minus, uratio, a2u_minus) = uplus_uminus(a,th_o,lam,eta)

    # equation 12 for F0
    # checks on xFarg should be handled in uplus_uminus function
    xFarg = np.cos(th_o)/np.sqrt(u_plus)
    F0 = sp.ellipkinc(np.arcsin(xFarg), uratio)

    # equation 17 for K
    K = sp.ellipkinc(0.5*np.pi, uratio)
    
    #get taumax for non-vortical Geodesics from Eq. 81 of Kerr Lensing
    mbarmax = neqmax-1
    mmax = mbarmax+np.heaviside(beta, 0)   
    taumax = (2*mmax*K-np.sign(beta)*F0)/np.sqrt(-u_minus*a**2)
    
    #now deal with vortical
    taumax[eta<=0] = tau_tot2[eta<=0]
    return np.min(np.array([taumax, tau_tot2]),axis=0) #max tau is either total or mth equatorial crossing

#converts guess array into correct dimensions
def makegoodarray(arr): 
    max_length = max(len(a) for a in arr)
    result = -np.ones((len(arr), max_length))
    for i, a in enumerate(arr):
        result[i, :len(a)] = a
    return np.transpose(np.copy(result))

#returns guesses for the psi of the first equatorial crossing
def getguesses(outgeo, a, rout, inc, alphas, betas, psitarget, ngeo, neqmax=1, model='para', pval=1): 
    tauguesses = []
    
    taumaxes = get_maxtau_forwardjet(a, rout, inc, alphas, betas, neqmax=neqmax)
    psifromgeo = psifunc(outgeo.r_s, outgeo.th_s, a, model=model, pval=pval) - psitarget
    
    for i in range(len(alphas)):
        impactparam = np.sqrt(alphas[i]**2+betas[i]**2)
        if np.abs(np.sin(inc)) < .1 and impactparam > 100: #can use flat-space face-on raytracing for guess
            if model == 'power' and pval > 0:
                rparam = (impactparam**2/(2*psitarget))**(1/(2-pval)) #large r approximations
                tauguesses.append(np.array([1/rparam, 2/impactparam-1/rparam]))
                continue
            elif model == 'para':
                rparam = (impactparam**2+psitarget**2)/(2*psitarget)
                tauguesses.append(np.array([1/rparam, 2/impactparam-1/rparam]))
                continue
            else:
                rparam = 1

        minfunc = np.roll(psifromgeo[:,i], -1)*psifromgeo[:,i] #we need to make sure this function crosses zero
        
        indmax = np.argmin(np.abs(outgeo.mino_times[:,i]-taumaxes[i]))
        minfunc = minfunc[:indmax+1] #+1 for python and conventions
        
        minfunc = minfunc[1:-2] #don't trust left or right endpoint
        mininds = (np.where(minfunc<0)[0])+1 #indices of zero crossings
                        
        if len(mininds) == 0: #no solution
            tauguesses.append(np.array([-1]))
            continue

        #mininds += 1 #necessary since we cut off the endpoints earlier
        #average two points which the zero lives between
        tauguesses.append(np.array([(outgeo.mino_times[indfinal, i]+outgeo.mino_times[indfinal+1, i])/2 for indfinal in mininds])) 
    
    return makegoodarray(tauguesses)



#find crossing using newton's method
def findroot(outgeo, psitarget, alpha, beta, r_o, th_o, a, ngeo, model='para', neqmax=1, tol=1e-8,pval=1): 
    #guesses  
    guesses = getguesses(outgeo, a, r_o, th_o, alpha, beta, psitarget, ngeo, neqmax=neqmax, model=model, pval=pval)
    guesses_shape = guesses.shape
    print(guesses_shape)

    # conserved quantities
    lam = np.tile(-alpha*np.sin(th_o), len(guesses)) #tile them to match shape of guess vector
    eta = np.tile((alpha**2 - a**2)*np.cos(th_o)**2 + beta**2, len(guesses))

    # spin zero should have no vortical geodesics
    if(np.abs(a)<MINSPIN and np.any(eta<0)):
        eta[eta<0]=EP # TODO ok?
        print("WARNING: there were eta<0 points for spin %f<MINSPIN!"%a)

    # sign of final angular momentum
    s_o = np.tile(my_sign(beta), len(guesses))
    betatile = np.tile(beta, len(guesses))

    #now flatten guesses
    guesses = guesses.flatten()
    
    #angular integrals
    (u_plus, u_minus, th_plus, th_minus, thclass) = angular_turning(a, th_o, lam, eta)

    # radial roots and radial motion case
    (r1, r2, r3, r4, rclass) = radial_roots(a, lam, eta)
    tau_tot = mino_total(a, r_o, eta, r1, r2, r3, r4)
    taumax = tau_tot * MAXTAUFRAC
      
    def get_coord_intersect(minotimes):
        #integration in theta
        (th_s, G_ph, G_t) = th_integrate(a,th_o,s_o,lam,eta,u_plus,u_minus,np.reshape(minotimes, (1, len(minotimes))),
                                         do_phi_and_t=True) #AC change do_phi_and_t to False because we don't need G_ph, G_t here?

        #integration in r
        (r_s, I_ph, I_t, I_sig) = r_integrate(a,r_o,lam,eta, r1,r2,r3,r4,np.reshape(minotimes, (1, len(minotimes))),
                                              do_phi_and_t=True) #AC change do_phi_and_t to False because we don't need I_ph, I_t here?
       
        arrhere = psifunc(r_s[0], th_s[0], a, model=model, pval=pval) - psitarget
        arrhere[guesses == -1] = 0 #no intersections

        return arrhere

    perturb = 1e-5
    print('before solve')
    outqty = scipy.optimize.newton(get_coord_intersect, guesses, maxiter=500, tol=tol)
    print('after solve')

    #integration in theta
    (th_s, G_ph, G_t) = th_integrate(a,th_o,s_o,lam,eta,u_plus,u_minus,np.reshape(outqty, (1, len(outqty))),
                                 do_phi_and_t=True) #AC change do_phi_and_t to False because we don't need G_ph, G_t here?
    (th_s_further, G_ph_further, G_t_further) = th_integrate(a,th_o,s_o,lam,eta,u_plus,u_minus,np.reshape(outqty*(1+perturb), (1, len(outqty))),
                                do_phi_and_t=True) #AC change do_phi_and_t to False because we don't need G_ph, G_t here?

    #integration in r
    (r_s, I_ph, I_t, I_sig) = r_integrate(a,r_o,lam,eta, r1,r2,r3,r4,np.reshape(outqty, (1, len(outqty))),
                                  do_phi_and_t=True) #AC change do_phi_and_t to False because we don't need I_ph, I_t here?
    (r_s_further, I_ph_further, I_t_further, I_sig_further) = r_integrate(a,r_o,lam,eta, r1,r2,r3,r4,np.reshape(outqty*(1+perturb), (1, len(outqty))),
                                  do_phi_and_t=True) #AC change do_phi_and_t to False because we don't need I_ph, I_t here?
    
    
    signpr = np.sign(r_s-r_s_further)[0]
    signptheta = np.sign(th_s-th_s_further)[0]
    r_s = np.copy(r_s[0])
    th_s = np.copy(th_s[0])

    neqvals = getneq(a, outqty, u_minus, u_plus, th_s, th_o, signptheta, betatile)

    outqty[guesses == -1] = r_s[guesses==-1] = th_s[guesses==-1] = signpr[guesses==-1] = signptheta[guesses==-1] = 0 #no crossing
        
    return outqty, r_s, th_s, signpr, signptheta, neqvals, guesses_shape
