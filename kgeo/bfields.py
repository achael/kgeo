import numpy as np
import scipy.special as sp
from tqdm import tqdm
from kgeo.kerr_raytracing_utils import my_cbrt, radial_roots, mino_total, is_outside_crit, uplus_uminus
from kgeo.equatorial_lensing import r_equatorial, nmax_equatorial, nmax_poloidal
import time
from mpmath import polylog
from scipy.interpolate import UnivariateSpline
import os
import pkg_resources

def f(r):
    """BZ monopole f(r) function"""
    r = r + 0j
    out = (polylog(2,2./r) - np.log(1-2./r)*np.log(r/2.))*r*r*(2*r-3)/8.
    out += (1+3*r-6*r*r)*np.log(r/2.)/12.
    out += 11./72. + 1./(3.*r) + r/2. - r*r/2.
    return float(np.real(out))

# compute f(r) 
RMAXINTERP = 100
NINTERP = 100000
#rsinterp = np.linspace(1,RMAXINTERP,NINTERP)
#fsinterp = np.array([f(r) for r in rsinterp])
    
# get f(r) from pre-saved data
#datafile = os.path.join(os.path.dirname(__file__), 'bz_fr_data.dat')
datafile = pkg_resources.resource_stream(__name__, 'bz_fr_data.dat')
(rsinterp,fsinterp) = np.loadtxt(datafile)

# interpolate f(r) and get its derivative
fi = UnivariateSpline(rsinterp, fsinterp, k=1,s=0, ext=0)
fiprime = fi.derivative()

class Bfield(object):
    """ object for b-field as a function of r, only in equatorial plane for now """
    
    def __init__(self, fieldtype="rad", **kwargs):

        self.fieldtype = fieldtype
        if self.fieldtype in ['rad','vert','tor','simple','simple_rm1','bz_monopole','bz_guess']:
            self.fieldframe = 'lab'
        elif self.fieldtype in ['const_comoving']:
            self.fieldframe = 'comoving'
        else: 
            raise Exception("fieldtype %s not recognized in Bfield!"%self.fieldtype)      
                      
        self.kwargs = kwargs
    
        if self.fieldframe not in ['lab','comoving']:
            raise Exception("Bfield fieldframe must be 'lab' or 'comoving'!")

        if self.fieldtype in ['bz_monopole','bz_guess']:
            self.C = self.kwargs.get('C', 1)
        else:
            if self.fieldtype=='rad':
                self.Cr=1; self.Cvert=0; self.Cph=0
            elif self.fieldtype=='vert':
                self.Cr=0; self.Cvert=1; self.Cph=0
            elif self.fieldtype=='tor':
                self.Cr=0; self.Cvert=0; self.Cph=1
            else:                            
                self.Cr = self.kwargs.get('Cr',0)
                self.Cvert = self.kwargs.get('Cvert',0)
                self.Cph = self.kwargs.get('Cph',0)
                            
            if self.Cr==self.Cvert==self.Cph==0.:
                raise Exception("all field coefficients are 0!")
                                 
    def bfield_lab(self, a, r):
        
        if self.fieldframe!='lab':
            raise Exception("Bfield.bfield_lab only supported for Bfield.fieldtype==lab")
                   
        if self.fieldtype in ['simple', 'rad', 'vert', 'tor']:
            b_components = Bfield_simple(a, r, (self.Cr, self.Cvert, self.Cph))
        elif self.fieldtype=='simple_rm1':
            b_components = Bfield_simple_rm1(a, r, (self.Cr, self.Cvert, self.Cph))
        elif self.fieldtype=='bz_monopole':
            b_components = Bfield_BZmonopole(a, r, self.C)  
        elif self.fieldtype=='bz_guess':
            b_components = Bfield_BZmagic(a, r, self.C)
        else: 
            raise Exception("fieldtype %s not recognized in Bfield.bfield_lab!"%self.fieldtype)
        return b_components
                                
    def bfield_comoving(self, a, r):
        """fluid frame B-field"""
        
        if self.fieldframe!='comoving':
            raise Exception("Bfield.bfield_comoving only supported for Bfield.fieldtype==comoving")
        if self.fieldtype=='const_comoving':                   
            b_components = (self.Cr*np.ones(r.shape),
                            -1*self.Cvert*np.ones(r.shape),
                            self.Cph*np.ones(r.shape))
        else: 
            raise Exception("fieldtype %s not recognized in Bfield.bfield_comoving!"%self.fieldtype)            
            
        return b_components


                        
def Bfield_simple(a, r, coeffs):
    """magnetic field vector in the lab frame in equatorial plane,
       simple configurations"""
    #|Br| falls of as 1/r^2
    #|Bvert| falls of as 1/r
    #|Bph| falls of as 1/r
    
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
        
    # coefficients for each component
    (arad, avert, ator) = coeffs
        
    th = np.pi/2. # TODO equatorial plane only
    Sigma = r**2 + a**2 * np.cos(th)**2    
    gdet = np.abs(np.sin(th))*Sigma #~r^2 in eq plane sin0
    
    # b-field in lab
    Br  = arad*(np.sin(th)/gdet)  + avert*(r*np.cos(th)/gdet)  
    Bth = avert*(-np.sin(th)/gdet) 
    Bph = ator*(1./gdet) 
   
    return (Br, Bth, Bph)

def Bfield_simple_rm1(a, r, coeffs):
    """magnetic field vector in the lab frame in equatorial plane, 
       simple configurations, all fall as 1/r"""
        
    #|Br| falls of as 1/r
    #|Bvert| falls of as 1/r
    #|Bph| falls of as 1/r
    
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    
    # coefficients for each component
    (arad, avert, ator) = coeffs
        
    th = np.pi/2. # TODO equatorial plane only
    Sigma = r**2 + a**2 * np.cos(th)**2    
    gdet = np.abs(np.sin(th))*Sigma #~r^2 in eq plane sin0
    
    # b-field in lab
    Br  = arad*(r*np.sin(th)/gdet)  + avert*(r*np.cos(th)/gdet)  
    Bth = avert*(-np.sin(th)/gdet) 
    Bph = ator*(1./gdet) 
   
    return (Br, Bth, Bph)  
      
def Bfield_BZmonopole(a,r, C=1):
    """perturbative BZ monopole solution.
       C is overall sign of monopole"""

    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")

    th = np.pi/2. # TODO equatorial plane only
    a2 = a**2
    r2 = r**2
    th = np.pi/2. # equatorial
    sth = np.sin(th)
    cth = np.cos(th)
    cth2 = cth**2
    sth2 = sth**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2*cth2
    gdet = sth*Sigma
    
    
    fr = fi(r)
    fr[r>RMAXINTERP] = 1./(4.*r[r>RMAXINTERP])
    frp = fiprime(r)
    frp[r>RMAXINTERP] = -1./(4.*r2[r>RMAXINTERP])
    
    phi = C*(1 - cth) + C*a2*fr*sth2*cth
    
    dphidtheta = C*sth*(1 + 0.5*a2*fr*(1 + 3*np.cos(2*th)))
    dphidr =  C*a2*sth2*cth*frp
    Br = dphidtheta / gdet
    Bth = -dphidr / gdet
        
    # TODO: sign of current? 
    f2 = (6*np.pi*np.pi - 49.)/72.
    omega2 = 1./32. - sth2*(4*f2 - 1)/64.
    I = -1*C*2*np.pi*sth2*(a/8. + (a**3)*(omega2 + 0.25*fr*cth2))
    Bph = I / (2*np.pi*Delta*sth2)
    
    return(Br, Bth, Bph)
 
def Bfield_BZmagic(a, r, C=1):
    """Guess ratio for Bphi/Br from split monopole
       C is overall sign of monopole"""

    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")


    th = np.pi/2. # TODO equatorial plane only
    a2 = a**2
    r2 = r**2
    th = np.pi/2. # equatorial
    sth = np.sin(th)
    cth = np.cos(th)
    cth2 = cth**2
    sth2 = sth**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2*cth2
    gdet = sth*Sigma
    Pi = (r2+a2)**2 - a2*Delta*sth2

    rH = 1 + np.sqrt(1-a2)
    OmegaH = a/(2*rH)
    
    # angular velocity guess from split Monopole
    omega0 = 1./8.
    f2 = (6*np.pi*np.pi - 49.)/72.
    omega2 = 1./32. - (4*f2 - 1)*sth2/64.
    Omega = a*omega0 + (a**3)*omega2

    # Magnetic field ratio Bphi/Br
    # TODO sign seems right, now consistent with BZ monopole
    Bratioguess = -1 * (OmegaH - Omega)*np.sqrt(Pi)/Delta
    
    Br = C/r2
    Bph = Bratioguess*Br
    Bth = 0*Br
    
    return(Br, Bth, Bph)
    
