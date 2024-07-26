#computes density as solution to continuity equation with a Gaussian source of width (in r) sigma

import numpy as np
import scipy.special as sp
import scipy.integrate as scint
from kgeo.bfields import *
from kgeo.velocities import *
from kgeo.ff_boost import *

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
#        indstart = np.where(thdirect != 0)[0][0]
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
        if i > 0:
            if intvals[i-1]/normfac == 1.0: #converged to an outflow
                intvals.append(intvals[i-1])
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
