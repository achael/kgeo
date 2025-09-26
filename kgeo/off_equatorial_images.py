import numpy as np
import scipy.special as sp
import scipy.integrate as scint

from kgeo.bfields import Bfield
from kgeo.velocities import Velocity
from kgeo.emissivities import Emissivity
from kgeo.equatorial_images import calc_redshift, calc_polquantities, calc_evpa
from kgeo.kerr_raytracing_ana import raytrace_ana

from kgeo.off_equatorial_lensing import findroot
from kgeo.geometry import sort_image

import numpy as np

from kgeo.bfields import *
from kgeo.velocities import *


bfield_default = Bfield('rad')
vel_default = Velocity('zamo')
emis_default = Emissivity('bpl')
SPECIND = 1 # default (negative) spectral index

def Iobs_off(a, r_o, r_s, th_o, alpha, beta, kr_sign, kth_sign,
         emissivity=emis_default, velocity=vel_default, bfield=bfield_default,
         polarization=False, specind=SPECIND, th_s=np.pi/2, density=1, retsin = False):

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
                
    if not isinstance(alpha, np.ndarray): alpha = np.array([alpha]).flatten()
    if not isinstance(beta, np.ndarray): beta = np.array([beta]).flatten()
    if not isinstance(r_s, np.ndarray): r_s = np.array([r_s]).flatten()
    if len(alpha) != len(beta):
        raise Exception("alpha, beta are different lengths!")

    # horizon radius
    rh = 1 + np.sqrt(1-a**2)

    # conserved quantities
    lam = -alpha*np.sin(th_o)
    eta = (alpha**2 - a**2)*np.cos(th_o)**2 + beta**2

    # output arrays
    g = np.zeros(alpha.shape)
    sin_thb = np.zeros(alpha.shape)
    Iobs_here = np.zeros(alpha.shape)
    Qobs = np.zeros(alpha.shape)
    Uobs = np.zeros(alpha.shape)
        
    # apply to all pixels - so the mask here always works
    zeromask = np.abs(alpha)<0
        
    if np.any(~zeromask):

        ###############################
        # get velocity and redshift
        ###############################        
        (u0,u1,u2,u3) = velocity.u_lab(a, r_s[~zeromask],th=th_s) 

        gg = calc_redshift(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask], kr_sign, kth_sign, u0, u1, u2, u3, th=th_s)   
        g[~zeromask] = gg

        ###############################
        # get emissivity in local frame
        ###############################
        Iemis = emissivity.jrest(a, r_s[~zeromask])*density

        ###############################
        # get polarization quantities
        # if polarization not used, set sin(theta_b) = 1 everywhere
        ###############################

        if polarization:

            (sinthb, kappa, pathlength, bsq) = calc_polquantities(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask],
                                                 kr_sign, kth_sign, 
                                                 velocity=velocity, 
                                                 bfield=bfield, th=th_s)
            (cos2chi, sin2chi) = calc_evpa(a, th_o, alpha[~zeromask], beta[~zeromask], kappa)

        else:
            sinthb = 1
        
        sin_thb[~zeromask] = sinthb  
        Iemis *= bsq**((1+specind)/2) #for spectral index of 1, we add on factor of B^2 

                             
        ###############################
        # observed emission
        ###############################         

        Iobs_here[~zeromask] = (gg**3) * (gg**specind) * pathlength * Iemis * (sinthb**(1+specind))       


        if polarization:
            Qobs[~zeromask] = cos2chi*Iobs_here[~zeromask]
            Uobs[~zeromask] = sin2chi*Iobs_here[~zeromask]

    else:
        print("masked all pixels in Iobs_off!")

    Iobs_2 = np.copy(Iobs_here)
    Qobs_2 = np.copy(Qobs)
    Uobs_2 = np.copy(Uobs)

    Iobs_2[r_s<=rh] = Qobs_2[r_s<=rh] = Uobs_2[r_s<=rh] = 0 #zero out emission coming from within the horizon


    if not retsin:
        return Iobs_2, Qobs_2, Uobs_2
    return Iobs_2, Qobs_2, Uobs_2, sinthb    





#get stokes parameters for a grid with off-equatorial emission in BZ model
#returns images contained in order of neq
def getstokes(psitarget, alphavals, betavals, r_o, th_o, a, ngeo, 
              do_phi_and_t = True, neqmax=1, outgeo=None, tol=1e-8, 
              model='para', pval=1,   
              nu_parallel = 0,  gammamax=None, retvals = False,
              sigma=2, sumsubring=True, usemono=False, retsin=False): #neqmax is the maximum number of equatorial crossings

    """Get stokes parameters for a grid with off-equatorial emission in BZ jet model
       
       Args:
           psitarget (float): Psi of field line to be imaged in jet model
           alphavals (numpy.array): array of image pixel alpha values 
           betavals (numpy.array): array of image pixel beta values 
           r_o (float): camera radius (should be large)
           th_o (float): camera inclination in radian range (0,pi/2) or (pi/2,pi)
           a (float): bh spin in range [0,1)
           ngeo (int): number of points sampled along geodesic
           
           do_phi_and_t (bool): raytrace in phi_and_t or not (must be true??)
           neqmax (int): maximal number of geodesic equatorial (?) crossings
           outgeo (Geodesics): precomputed geodesics
           tol (float): tolerance for Newton solve of crossing points
           
           model (str): define BZ jet model: 'mono', 'para', or 'power'
           pval (float): index of variable width 'power' jet model

           nu_parallel (float): fractional allowable parallel velocity in range [-1,1]       
           gammamax (float): cutoff lorentz factor in jet model
           
           sigma (float): constant sigma for determining density/emissivity
           
           retvals (bool): whether or not to return data beyond stokes params
           sumsubring (bool): whether or not to sum all crossing data in final return
           usemono (bool): if True, fix Omega_field to monopole rate
           retsin (bool): if True, return sin(theta_B)
            
       Returns:

    """
    ashape = alphavals.shape # store shapes for later
    alphavals = alphavals.flatten() # flatten since we need everything to be a vector for our code to work
    betavals = betavals.flatten()

    if outgeo == None:
        outgeo = raytrace_ana(a=a,
                              observer_coords = [0,r_o,th_o,0],
                              image_coords = [alphavals, betavals], #assumes 1D arrays of alpha and beta
                              ngeo=ngeo,
                              do_phi_and_t=do_phi_and_t,
                              savedata=False, plotdata=False)

    #solve for crossing points and densities there
    tau, rvals, thvals, signpr, signptheta, neqvals, guesses_shape = findroot(outgeo, psitarget, alphavals, betavals, r_o, th_o, a, ngeo, 
                                                                              model=model, neqmax=neqmax, tol=tol, pval=pval)


    print('guesses before ', guesses_shape)
    #reshape coordinates
    alphavals = np.tile(alphavals, guesses_shape[0])
    betavals = np.tile(betavals, guesses_shape[0])


    if model == 'mono' or (model == 'power' and pval == 0):
        dvals = density_mono_all(rvals, thvals, guesses_shape, psitarget, a, nu_parallel, sigma, neqmax=neqmax, gammamax=gammamax)
    elif model == 'para':
        dvals = density_para_all(rvals, thvals, guesses_shape, psitarget, a, nu_parallel, sigma, neqmax=neqmax, gammamax=gammamax)
    elif (model == 'power' and pval > 0):
        dvals = density_power_all(rvals, thvals, guesses_shape, psitarget, a, nu_parallel, sigma, neqmax=neqmax, gammamax=gammamax, pval = pval, usemono=usemono)
    
    #initialize arrays
    if model == 'para':
        bf = Bfield('bz_para')
    elif model == 'mono':
        bf = Bfield('bz_monopole') #initialize bfield
    elif model == 'power':
        bf = Bfield('power', p=pval, usemono=usemono)
    else:
        bf = 0

    #generate intensity data
    outvec = Iobs_off(a, r_o, rvals, th_o, alphavals, betavals, signpr, signptheta,
                      emissivity=Emissivity('constant'), 
                      velocity=Velocity('driftframe', bfield=bf, nu_parallel = nu_parallel, gammamax=gammamax), 
                      bfield=bf,
                      polarization=True, specind=SPECIND, th_s=thvals, density=dvals, retsin=retsin) 


    iobs = np.copy(outvec[0])
    qobs = np.copy(outvec[1])
    uobs = np.copy(outvec[2])

    iobs[rvals == 0] = qobs[rvals==0] = uobs[rvals==0] = 0 #zero out geodesics with no fieldline crossings

    iobs = np.real(np.nan_to_num(np.array(iobs))) #get rid of nans
    qobs = np.real(np.nan_to_num(np.array(qobs)))
    uobs = np.real(np.nan_to_num(np.array(uobs)))

    if not sumsubring:
        #return iobs, qobs, uobs, neqvals, guesses_shape  #just return the raw subrings
        
        return iobs, qobs, uobs, neqvals, np.nan_to_num(rvals), np.nan_to_num(thvals), guesses_shape  #just return the raw subrings #AC edited
        
    #call sorter here
    ivec, qvec, uvec = sort_image(iobs, qobs, uobs, neqvals, guesses_shape, ashape, neqmax)
    evpa = np.nan_to_num(0.5*np.arctan2(uvec, qvec))
    
    if not retvals: #just return stokes
        return ivec, qvec, uvec, evpa
    if not retsin: #return stokes + intersection data
        return ivec, qvec, uvec, evpa, np.nan_to_num(rvals), np.nan_to_num(thvals)
    return ivec, qvec, uvec, evpa, np.nan_to_num(rvals), np.nan_to_num(thvals), np.nan_to_num(outvec[3]) #return stokes+intersection+pitch angle



#returns images contained in order of neq
def sort_image(iobs, qobs, uobs, neqvals, guesses_shape, ashape, neqmax):
    iarr = [] #initialize arrays
    qarr = []
    uarr = []

    for neq in np.arange(0, neqmax, 1):
        ihere = np.copy(iobs)
        qhere = np.copy(qobs)
        uhere = np.copy(uobs)

        ihere[neqvals != neq] = 0 #zero out everything not in this subring
        qhere[neqvals != neq] = 0
        uhere[neqvals != neq] = 0

        iarr.append(np.reshape(np.sum(np.reshape(ihere, guesses_shape), axis=0), ashape)) #sum over all points in the subring
        qarr.append(np.reshape(np.sum(np.reshape(qhere, guesses_shape), axis=0), ashape))
        uarr.append(np.reshape(np.sum(np.reshape(uhere, guesses_shape), axis=0), ashape))

    #add the total image last to the array
    iarr.append(np.reshape(np.sum(np.reshape(iobs, guesses_shape), axis=0), ashape))
    qarr.append(np.reshape(np.sum(np.reshape(qobs, guesses_shape), axis=0), ashape))
    uarr.append(np.reshape(np.sum(np.reshape(uobs, guesses_shape), axis=0), ashape))

    return np.array(iarr), np.array(qarr), np.array(uarr)

###########################################################
# computes density as solution to continuity equation with a Gaussian source of width (in r) sigma
# previously in densityfuncs.py
###########################################################

#r for fixed x and psi for BZ paraboloid
def rparafunc(psi, x, rp):
    num = rp+psi-rp*x-rp*np.log(4)+rp*(1+x)*np.log(1+x)
    return num/(1-x)

#integrand to compute eta(x) for BZ paraboloid
def integrand_para(x, psi, r0, rp, sigma, spin):
    rhere = rparafunc(psi, x, rp)
    source = np.exp(-(rhere-r0)**2/(2*sigma**2))
    num = source*(rhere**2+spin**2*x**2)
    denom = 1-x
    return num/denom

#integrand to compute eta(x) for r^p(1-cos theta)
def integrand_power(r, psi, r0, sigma, spin, p):
    sth = 1#np.sqrt(1-x**2)
    source = np.exp(-(r*sth-r0*sth)**2/(2*sigma**2))
    return source * (r**2+spin**2*(1-psi/r**p)**2)


#compute mass loading for monopole
def eta_mono(rvals, theta, r0, spin, sigma): #sigma is width of Gaussian, normalized so that eta goes is zero at stagnation surface and +1 at infinity
    if sigma == 0: #delta function
        return np.sign(rvals-r0)
    
    # integrate Gaussian
    cth = np.cos(theta)
    numterm1 = 2*sigma*(2*r0-(rvals+r0)*np.exp(-(rvals-r0)**2/(2*sigma**2)))
    numterm2 = np.sqrt(2*np.pi)*sp.erf((rvals-r0)/(np.sqrt(2)*sigma))*(r0**2+sigma**2+spin**2*cth**2)
    denom = 4*r0*sigma+np.sqrt(2*np.pi)*sigma**2+np.sqrt(2*np.pi)*(r0**2+spin**2*cth**2)
    eta = (numterm1+numterm2)/denom
    return eta

#density for mononpole
def density_mono_all(rvals, thvals, guesses_shape, psitarget, spin, nu_parallel, sigma, neqmax=2, gammamax=None):
    #properties of fieldline
    rp = 1+np.sqrt(1-spin**2)

    #initialize
    bfield = Bfield("bz_monopole", C=1)
    rnew = np.reshape(rvals, guesses_shape)
    thnew = np.arccos(np.abs(np.cos(np.reshape(thvals, guesses_shape)))) #only works for above equator, but we can employ reflection symmetry
    rhovals = []
    indstart = np.where(rnew[0]!=0)[0][0]
    omega = bfield.omega_field(spin,rnew[0][indstart],th=thnew[0][indstart])

    #stagnation surface
    r0 = r0min_mono(thnew[0][indstart], omega, spin, 1.0)

    #loop through images
    for ind in range(len(rnew)):
        rdirect = np.reshape(rnew, guesses_shape)[ind] 
        thdirect = np.reshape(thnew, guesses_shape)[ind]

        #magnetic field
        bupper = np.transpose(np.transpose(bfield.bfield_lab(spin, rdirect, th=thdirect)))

        #paraboloid
        velpara = Velocity('driftframe', bfield = bfield, nu_parallel = nu_parallel, gammamax = None)
        (u0,u1,u2,u3) = velpara.u_lab(spin, rdirect, th=thdirect)

        #density
        eta = eta_mono(rdirect, thdirect, r0, spin, sigma)
        rho = eta*bupper[0]/u1
        indstart = np.where(thdirect != 0)[0][0]
        #rho = rho/rho[indstart] #normalize to be 1 at the origin
        rhovals.append(rho)

    return np.nan_to_num(np.array(rhovals).flatten(), nan=0.0)


#compute mass loading for paraboloid
def eta_para(thetavals, theta0, spin, psi, sigma):
    if sigma == 0:
        return np.sign(np.cos(thetavals)-np.cos(theta0))

    xvals = np.cos(thetavals)
    x0 = np.cos(theta0)
    rp = 1+np.sqrt(1-spin**2)
    r0 = rparafunc(psi, x0, rp)

    #define local integrand function
    def inthere(x):
        return integrand_para(x, psi, r0, rp, sigma, spin)
    
    #define normalization factor
    normfac = scint.quad(inthere, x0, 1.0, epsabs = 1e-13, epsrel = 1e-13)[0]

    #if just one theta, only need one integration
    if not hasattr(thetavals, '__len__'):
        return scint.quad(inthere, x0, xvals, epsabs=1e-13, epsrel=1e-13)[0]/normfac

    #loop through and compute integral from each value of x to the next
    indstart = np.where(xvals != 1)[0][0]
    intvals = [0.0 for i in range(indstart)]
    for i in range(indstart,len(xvals),1):
        if xvals[i] == 1:
            intvals.append(0)
            continue
        intcell = scint.quad(inthere, x0, xvals[i], epsabs=1e-13, epsrel=1e-13)[0]
        intvals.append(intcell)

    etavals = np.array(intvals)/normfac
    return etavals


#density for BZ paraboloid
def density_para_all(rvals, thvals, guesses_shape, psitarget, spin, nu_parallel, sigma, neqmax=2, gammamax=None):
    #properties of fieldline
    rp = 1+np.sqrt(1-spin**2)

    #initialize
    bfield = Bfield("bz_para", C=1)
    rnew = np.reshape(rvals, guesses_shape)
    thnew = np.arccos(np.abs(np.cos(np.reshape(thvals, guesses_shape)))) #only works for above equator, but we can employ reflection symmetry
    rhovals = []
    indstart = np.where(np.nan_to_num(rnew)[0]!=0)[0][0]
    omega = bfield.omega_field(spin,rnew[0][indstart],th=thnew[0][indstart])

    #stagnation surface
    r0, theta0 = r0min_para(psiBZpara(rnew[0][indstart], thnew[0][indstart], spin), omega, spin, 1.0)

    #loop through images
    for ind in range(len(rnew)):
        rdirect = np.reshape(rnew, guesses_shape)[ind] 
        thdirect = np.reshape(thnew, guesses_shape)[ind]

        #magnetic field
        bupper = np.transpose(np.transpose(bfield.bfield_lab(spin, rdirect, th=thdirect)))

        #paraboloid
        velpara = Velocity('driftframe', bfield = bfield, nu_parallel = nu_parallel, gammamax = None)
        (u0,u1,u2,u3) = velpara.u_lab(spin, rdirect, th=thdirect)

        #density
        eta = eta_para(thdirect, theta0, spin, psitarget, sigma)
        rho = eta*bupper[0]/u1
        #indstart = np.where(thdirect != 0)[0][0]
        #rho = rho/rho[indstart] #normalize to be 1 at the origin
        rhovals.append(rho)

    return np.nan_to_num(np.array(rhovals).flatten(), nan=0.0)


#compute eta(x) for r^p(1-cos theta)
def eta_power(rvals, r0, spin, pval, psi, sigma):
    if sigma == 0:
        return np.sign(rvals-r0)

    rp = 1+np.sqrt(1-spin**2)

    #define local integrand function
    def inthere(r):
        return integrand_power(r, psi, r0, sigma, spin, pval)
    
    #define normalization factor
    normfac = scint.quad(inthere, r0, np.inf, epsabs = 1e-13, epsrel = 1e-13)[0]

    #if just one theta, only need one integration
    if not hasattr(rvals, '__len__'):
        return scint.quad(inthere, r0, rvals, epsabs=1e-13, epsrel=1e-13)[0]/normfac

    #loop through and compute integral from each value of x to the next
    indstart = np.where(rvals != 1)[0][0]
    intvals = [0.0 for i in range(indstart)]
    for i in range(indstart,len(rvals),1):
        if rvals[i] == 0:
            intvals.append(0)
            continue
        intcell = scint.quad(inthere, r0, rvals[i], epsabs=1e-13, epsrel=1e-13)[0]
        intvals.append(intcell)

    etavals = np.array(intvals)/normfac
    return etavals

#density for r^p(1-cos theta)
def density_power_all(rvals, thvals, guesses_shape, psitarget, spin, nu_parallel, sigma, neqmax=2, gammamax=None, pval = 1.0, usemono=False):
    #properties of fieldline
    rp = 1+np.sqrt(1-spin**2)

    #initialize
    bfield = Bfield("power", p=pval, usemono=usemono)
    rnew = np.reshape(rvals, guesses_shape)
    thnew = np.arccos(np.abs(np.cos(np.reshape(thvals, guesses_shape)))) #only works for above equator, but we can employ reflection symmetry
    rhovals = []
    indstart = np.where(np.nan_to_num(rnew)[0]!=0)[0][0]
    omega = bfield.omega_field(spin,rnew[0][indstart],th=thnew[0][indstart])

    #stagnation surface
    r0, theta0 = r0min_power(psitarget, omega, spin, pval, 1.0, usemono=usemono)

    #loop through images
    for ind in range(len(rnew)):
        rdirect = np.reshape(rnew, guesses_shape)[ind] 
        thdirect = np.reshape(thnew, guesses_shape)[ind]

        #magnetic field
        bupper = np.transpose(np.transpose(bfield.bfield_lab(spin, rdirect, th=thdirect)))

        #paraboloid
        velpara = Velocity('driftframe', bfield = bfield, nu_parallel = nu_parallel, gammamax = None)
        (u0,u1,u2,u3) = velpara.u_lab(spin, rdirect, th=thdirect)

        #density
        eta = eta_power(rdirect, r0, spin, pval, psitarget, sigma)
        rho = eta*bupper[0]/u1
        indstart = np.where(thdirect != 0)[0][0]
        rhovals.append(rho)

    return np.nan_to_num(np.array(rhovals).flatten(), nan=0.0)


