##computes unique parallel boost parameter in force-free electrodynamics

import numpy as np
import scipy.optimize as opt
import kgeo.bfields

#stagnation surface for monopole
def r0min_mono(theta, Omegaf, spin, M): #solves quintic that gets minimum launch point for outflow in monopole
    theta = np.arccos(np.abs(np.cos(theta))) #put in the northern hemisphere by reflection symmetry
    coef0 = -spin**2/2*M*np.cos(theta)**2*(2-spin*Omegaf+spin*Omegaf*np.cos(2*theta))**2
    coef1 = -2*spin**4*Omegaf**2*np.cos(theta)**4*np.sin(theta)**2
    coef2 = M/2*(2-spin*Omegaf+spin*Omegaf*np.cos(2*theta))**2
    coef3 = -spin**2*Omegaf**2*np.sin(2*theta)**2
    coef4 = 0.0
    coef5 = -2*Omegaf**2*np.sin(theta)**2
    poly = np.polynomial.polynomial.Polynomial([coef0, coef1, coef2, coef3, coef4, coef5])
    rroots = poly.roots()
    return np.real(rroots[4])

#stagnation surface for paraboloid
def r0min_para(psi, Omegaf, spin, M): #does numerical root find
    bf = kgeo.bfields.Bfield('bz_para')
    def minfunc(R):
        rsphere = rfromR_para(R, psi, spin)
        theta = np.arcsin(R/rsphere)
        return Nderiv(rsphere, theta, spin, Omegaf, M, bf)
    Rguess = 2*rplusfunc(spin, M) #optimal guess
    Rtrue = opt.newton(minfunc, Rguess)
    rtrue = rfromR_para(Rtrue, psi, spin)
    thetatrue = np.arcsin(Rtrue/rtrue)
    return rtrue, thetatrue

#stagnation surface for r^p(1-costheta)
def r0min_power(psi, Omegaf, spin, p, M, usemono=False): 
    bf = kgeo.bfields.Bfield('power', p=p, usemono=usemono)
    def minfunc(R):
        rsphere = rfromR_power(R, psi, p)
        theta = np.arcsin(R/rsphere)
        Ndval = Nderiv(rsphere, theta, spin, Omegaf, M, bf)
        return Ndval
    Rguess = 4
    Rtrue = opt.newton(minfunc, Rguess)
    rtrue = rfromR_power(Rtrue, psi, p)
    thetatrue = np.arcsin(Rtrue/rtrue)
    return rtrue, thetatrue

def rplusfunc(spin, M): #returns location of outer event horizon
    if M==0 or spin == 0:
        return 0
    return M+np.sqrt(M**2-spin**2)

#psi(r,theta) for BZ paraboloid
def psiBZpara(r, theta, a): #input should be in terms of M in curved space, or in terms of Omegaf in flat space
    rp = rplusfunc(a, 1.0)
    cth = np.abs(np.cos(theta))
    return r*(1-cth)-2*rp*(1-np.log(2))+rp*(1+cth)*(1-np.log(1+cth)) #use rp as the mass scale

#psi(r,theta) for r^p(1-costheta)
def psiBZpower(r, theta, p): #input should be in terms of M in curved space, or in terms of Omegaf in flat space
    return r**p*(1-np.abs(np.cos(theta)))

#invert r(R) for fixed psi
def rfromR_para(R, psi0, a):
    def minfunc(Z):
        return psiBZpara(np.sqrt(R**2+Z**2),np.arctan(R/Z),a)-psi0
    Zguess = 3.0
    try:
        Zout = opt.newton(minfunc, Zguess)
    except: #failing because close to the equator, so guess 0
        Zout = opt.newton(minfunc, 0)
    return np.sqrt(Zout**2+R**2)

#invert r(R) for fixed psi with r^p(1-costheta)
def rfromR_power(R, psi0, p):
    def minfunc(Z):
        return psiBZpower(np.sqrt(R**2+Z**2),np.arctan(R/Z),p)-psi0
    Zguess = 4.0
    try:
        Zout = opt.newton(minfunc, Zguess)
    except: #failing because close to the equator, so guess 0
        Zout = opt.newton(minfunc, 0)
    return np.sqrt(Zout**2+R**2)

#N', where N is the normalization factor for co-moving four-velocity and prime is diff. along fieldline
def Nderiv(r, theta, a, Omegaf, M, bf_here):
    cth = np.cos(theta)
    sth = np.sin(theta)
    Sigma = r**2+a**2*cth**2
    dNdr = a**4*Omegaf**2*r*cth**4*sth**2+r**2*(-M+Omegaf*sth**2*(2*a*M+Omegaf*r**3-a**2*M*Omegaf*sth**2))+a**2*cth**2*(M+Omegaf*sth**2*(-2*a*M+2*Omegaf*r**3+a**2*M*Omegaf*sth**2))
    dNdr /= Sigma**2/2
    
    denomtheta = a**2+2*r**2+a**2*np.cos(2*theta)**2
    dNdtheta = Omegaf**2*(a**2+r*(r-2*M))+8*M*r*(a*(a*Omegaf-1)+Omegaf*r**2)**2/denomtheta**2
    dNdtheta *= np.sin(2*theta)
    
    bvec = bf_here.bfield_lab(a, r, th=theta)
    
    return bvec[0]*dNdr + bvec[1]*dNdtheta #B.nabla(N)


#solve for L as a function of launch point
def getEco(r0, theta, Omegaf, spin, M): #gets Eco=E-OmegaF*L as a function of launch point r0
    g = metric(r0, spin, theta, M)
    ginv = invmetric(r0, spin, theta, M)
    
    gtt = g[0][0]
    gtphi = g[0][3]
    ginvtt = ginv[0][0]
    ginvtphi = ginv[0][3]
    gphiphi = g[3][3]
    ginvphiphi = ginv[3][3]
    gtphifac = gtphi+gphiphi*Omegaf
    gttfac = gtt+gtphi*Omegaf
    
    coef0 = (gtt+Omegaf*(2*gtphi+gphiphi*Omegaf))**2
    coef1 = ginvphiphi*gtphifac**2+2*ginvtphi*gtphifac*gttfac+ginvtt*gttfac**2
    efac2 = -coef0/coef1 #(E-L*Omegaf)^2
    return np.sqrt(efac2)

#define metric components
def metric(r, a, theta, M):
    SigmaK = r**2+a**2*np.cos(theta)**2
    DeltaK = r**2-2*M*r+a**2
    gmunu = np.zeros((4,4,)+r.shape) if hasattr(r, 'shape') else np.zeros((4,4)) #deal with vector vs scalar
    gmunu[0][0] = -(1-2*M*r/SigmaK)
    gmunu[0][3] = gmunu[3][0] = -2*a*M*r/SigmaK*np.sin(theta)**2
    gmunu[1][1] = SigmaK/DeltaK
    gmunu[2][2] = SigmaK
    gmunu[3][3] = (r**2+a**2+2*M*r*a**2/SigmaK*np.sin(theta)**2)*np.sin(theta)**2
    
    return np.swapaxes(gmunu, 0, -1) if hasattr(r, 'shape') else gmunu

#inverse metric
def invmetric(r, a, theta, M):
    SigmaK = r**2+a**2*np.cos(theta)**2
    DeltaK = r**2-2*M*r+a**2
    ginvmunu = np.zeros((4,4,)+r.shape) if hasattr(r, 'shape') else np.zeros((4,4)) #deal with vector vs scalar
    ginvmunu[0][0] = -1/DeltaK*(r**2+a**2+2*M*r*a**2/SigmaK*np.sin(theta)**2)
    ginvmunu[0][3] = ginvmunu[3][0] = -2*M*r*a/(SigmaK*DeltaK)
    ginvmunu[1][1] = DeltaK/SigmaK
    ginvmunu[2][2] = 1/SigmaK
    ginvmunu[3][3] = (DeltaK-a**2*np.sin(theta)**2)/(SigmaK*DeltaK*np.sin(theta)**2)
    
    return np.swapaxes(ginvmunu, 0, -1) if hasattr(r, 'shape') else ginvmunu
