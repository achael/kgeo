import numpy as np
import scipy.special as sp
from mpmath import polylog
from scipy.interpolate import UnivariateSpline
import pkg_resources


def f(r):
    """BZ monopole f(r) function"""
    r = r + 0j
    out = (polylog(2,2./r) - np.log(1-2./r)*np.log(r/2.))*r*r*(2*r-3)/8.
    out += (1+3*r-6*r*r)*np.log(r/2.)/12.
    out += 11./72. + 1./(3.*r) + r/2. - r*r/2.
    return float(np.real(out))

# compute f(r) 
#RMAXINTERP = 100
#NINTERP = 100000
#rsinterp = np.linspace(1,RMAXINTERP,NINTERP)
#fsinterp = np.array([f(r) for r in rsinterp])
#np.savetxt('./bz_fr_data.dat', np.vstack((rsinterp,fsinterp)).T)

# get f(r) from pre-saved data
datafile = pkg_resources.resource_stream(__name__, 'bz_fr_data.dat')
(rsinterp,fsinterp) = np.loadtxt(datafile)
NINTERP = len(rsinterp)
RMAXINTERP = np.max(rsinterp)

# interpolate f(r) and get its derivative
fi = UnivariateSpline(rsinterp, fsinterp, k=1,s=0, ext=0)
fiprime = fi.derivative()


# used in testing
_allowed_bfield_models = [
    'rad', 'vert', 'tor', 'simple', 'simple_rm1', 'bz_monopole', 
    'bz_guess', 'bz_para', 'const_comoving'
]


class Bfield(object):
    """
    Parent class definition for velocity object with member functions
    to access (TODO) some combination of magnetic field components, 
    electric field components, Faraday/Maxwell tensor components, and/or
    fieldline rotation.

    Parameters
    ----------
    fieldtype (str) : Bfield model; see above descriptions (default == rad)

    **kwargs (dict) : keyword arguments passed to Bfield model on eval

    TODOs
    -----
    - currently only works for equatorial plane
    """
    
    def __init__(self, fieldtype="rad", **kwargs):

        # this check ensures that _allowed_bfield_models is updated when
        # new bfield models are added, which is important for testing
        if fieldtype not in _allowed_bfield_models:
            print("Allowed BField models are:")
            print(_allowed_bfield_models)
            raise Exception(f"fieldtype {fieldtype} not recognized in Bfield!")

        self.fieldtype = fieldtype
        if self.fieldtype in ['rad','vert','tor','simple','simple_rm1','bz_monopole','bz_guess','bz_para']:
            self.fieldframe = 'lab'
        elif self.fieldtype in ['const_comoving']:
            self.fieldframe = 'comoving'
        else: 
            raise Exception("fieldtype %s not recognized in Bfield!"%self.fieldtype)      
                      
        self.kwargs = kwargs
    
        if self.fieldframe not in ['lab','comoving']:
            raise Exception("Bfield fieldframe must be 'lab' or 'comoving'!")

        if self.fieldtype in ['bz_monopole','bz_guess','bz_para']:
            self.secondorder_only = self.kwargs.get('secondorder_only', False)
            self.C = self.kwargs.get('C', 1)
        elif self.fieldtype=='rad':
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
                                     
    def bfield_lab(self, a, r, th=np.pi/2):
        """lab frame b field starF^i0"""
         
        if self.fieldframe!='lab':
            raise Exception("Bfield.bfield_lab only supported for Bfield.fieldtype==lab")
                   
        if self.fieldtype in ['simple', 'rad', 'vert', 'tor']:
            b_components = Bfield_simple(a, r, (self.Cr, self.Cvert, self.Cph))
        elif self.fieldtype=='simple_rm1':
            b_components = Bfield_simple_rm1(a, r, (self.Cr, self.Cvert, self.Cph))
        elif self.fieldtype=='bz_monopole':
            (B1,B2,B3,omega) = Bfield_BZmonopole(a, r, th, self.C, secondorder_only=self.secondorder_only)  
            b_components = (B1,B2,B3)
        elif self.fieldtype=='bz_para':
            (B1,B2,B3,omega) = Bfield_BZpara(a, r, th, self.C)  
            b_components = (B1,B2,B3)
        elif self.fieldtype=='bz_guess':
            (B1,B2,B3,omega) = Bfield_BZmagic(a, r, th, self.C)
            b_components = (B1,B2,B3)            
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

    # TODO ANDREW FIX THESE WITH BETTER DATA STRUCTURES
    def omega_field(self, a, r, th=np.pi/2):
        """fieldline angular speed"""    
        if self.fieldtype=='bz_monopole':
            (B1,B2,B3,omega) = Bfield_BZmonopole(a, r, th, self.C,secondorder_only=self.secondorder_only)
        elif self.fieldtype=='bz_guess':
            (B1,B2,B3,omega) = Bfield_BZmagic(a, r, th, self.C)
        elif self.fieldtype=='bz_para':
            (B1,B2,B3,omega) = Bfield_BZpara(a, r, th, self.C) 
        else: 
            raise Exception("self.efield_lab currently only works for self.fieldtype='bz_monopole' or 'bz_guess'!")       
        return omega
        
    def efield_lab(self, a, r, th=np.pi/2):
        """lab frame electric field F^{0i} in BL coordinates. 
           below defn is for stationary, axisymmetric fields"""

        if self.fieldtype=='bz_monopole' and self.secondorder_only:
            e_components = Efield_BZmonopole(a,r,th, self.C)
        elif self.fieldtype in ['bz_monopole','bz_guess','bz_para']:
            if self.fieldtype=='bz_monopole':
                (B1,B2,B3,omega) = Bfield_BZmonopole(a, r, th, self.C,secondorder_only=self.secondorder_only)
            elif self.fieldtype=='bz_guess':
                (B1,B2,B3,omega) = Bfield_BZmagic(a, r, th, self.C)
            elif self.fieldtype=='bz_para':
                (B1,B2,B3,omega) = Bfield_BZpara(a, r, th, self.C)
            a2 = a**2
            r2 = r**2
            cth2 = np.cos(th)**2
            sth2 = np.sin(th)**2
            Delta = r2 - 2*r + a2
            Sigma = r2 + a2 * cth2  
            Pi = (r2+a2)**2 - a2*Delta*sth2
            omegaz=2*a*r/Pi
            E1 = (omega-omegaz)*Pi*np.sin(th)*B2/Sigma
            E2 = -(omega-omegaz)*Pi*np.sin(th)*B1/(Sigma*Delta)
            E3 = np.zeros_like(E2) if hasattr(E2, '__len__') else 0
            e_components = (E1, E2, E3)                               
#            (F01, F02, F03, F12, F13, F23) = self.faraday(a,r)
#            e_components = (F01, F02, F03)
        else: 
            raise Exception("self.efield_lab currently only works for self.fieldtype='bz_monopole' or 'bz_guess'!")                        
            
        return e_components
        
    def maxwell(self, a, r, th=np.pi/2):
        """Maxwell tensor starF^{\\mu\\nu} in BL coordinates. 
           below defn is for stationary, axisymmetric fields"""

        if self.fieldtype in ['bz_monopole','bz_guess','bz_para']:
            if self.fieldtype=='bz_monopole':
                (B1,B2,B3,OmegaF) = Bfield_BZmonopole(a, r, th, self.C)  
            elif self.fieldtype=='bz_guess':
                (B1,B2,B3,OmegaF) = Bfield_BZmagic(a, r, th, self.C)
            elif self.fieldtype=='bz_para':
                (B1,B2,B3,OmegaF) = Bfield_BZpara(a, r, th, self.C)
            sF01 = -B1
            sF02 = -B2
            sF03 = -B3
            sF12 = 0*B3
            sF13 = OmegaF*B1
            sF23 = OmegaF*B2
                        
            sF_out = (sF01, sF02, sF03, sF12, sF13, sF23)
            
        else: 
            raise Exception("self.maxwell currently only works for self.fieldtype='bz_monopole' or 'bz_guess'!")                        
            
        return sF_out
         
    def faraday(self, a, r, th=np.pi/2):
        """Faraday tensor F^{\\mu\\nu} in BL coordinates. 
           below defn is for stationary, axisymmetric fields"""

        if self.fieldtype in ['bz_monopole','bz_guess','bz_para']:
            if self.fieldtype=='bz_monopole':
                (B1,B2,B3,OmegaF) = Bfield_BZmonopole(a, r, th, self.C)  
            elif self.fieldtype=='bz_guess':
                (B1,B2,B3,OmegaF) = Bfield_BZmagic(a, r, th, self.C)   
            elif self.fieldtype=='bz_para':
                (B1,B2,B3,OmegaF) = Bfield_BZpara(a, r, th, self.C)   
                         
            # Metric in BL
            a2 = a**2
            r2 = r**2
            cth2 = np.cos(th)**2
            sth2 = np.sin(th)**2
            Delta = r2 - 2*r + a2
            Sigma = r2 + a2 * cth2
            
            gdet = Sigma*np.sin(th)
            
            g00_up = -(r2 + a2 + 2*r*a2*sth2/Sigma) / Delta
            g11_up = Delta/Sigma
            g22_up = 1./Sigma
            g33_up = (Delta - a2*sth2)/(Sigma*Delta*sth2)
            g03_up = -(2*a*r)/(Sigma*Delta)
            
            # F_{\mu\nu}
            F_01 = -gdet*OmegaF*B2
            F_02 =  gdet*OmegaF*B1
            F_03 =  0*B3
            F_12 =  gdet*B3
            F_13 = -gdet*B2
            F_23 =  gdet*B1
            
            # raise indices
            F01 = g00_up*g11_up*F_01 + g03_up*g11_up*(-F_13)
            F02 = g00_up*g22_up*F_02 + g03_up*g22_up*(-F_23)
            F03 = 0*B3
            F12 = g11_up*g22_up*F_12
            F13 = g11_up*g33_up*F_13 + g11_up*g03_up*(-F_01)
            F23 = g22_up*g33_up*F_23 + g22_up*g03_up*(-F_02)
            
            F_out = (F01, F02, F03, F12, F13, F23)
            
        else: 
            raise Exception("self.faraday currently only works for self.fieldtype='bz_monopole' or 'bz_guess'!")                        
            
        return F_out
         
                        
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
       
def Bfield_BZmagic(a, r, th, C=1):
    """Guess ratio for Bphi/Br from split monopole
       C is overall sign of monopole"""

    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")


   # th = np.pi/2. # TODO equatorial plane only
    a2 = a**2
    r2 = r**2
#    th = np.pi/2. # equatorial
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
    # TODO is this right for negative spin? 
    spin = np.abs(a)
    omega0 = 1./8.
    f2 = (6*np.pi*np.pi - 49.)/72.
    omega2 = 1./32. - (4*f2 - 1)*sth2/64.
    OmegaBZ = spin*omega0 + (spin**3)*omega2
    if a<0: OmegaBZ*=-1 #  TODO: is this right for a<0? 
    
    # Magnetic field ratio Bphi/Br
    # TODO sign seems right, now consistent with BZ monopole
    Bratioguess = -1 * (OmegaH - OmegaBZ)*np.sqrt(Pi)/Delta
    
    Br = C/r2
    Bph = Bratioguess*Br
    Bth = 0*Br
    
    return(Br, Bth, Bph, OmegaBZ)

def Bfield_BZmonopole(a, r, th, C=1, secondorder_only=False):
    """perturbative BZ monopole solution.
       C is overall sign of monopole"""

    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")

#    th = np.pi/2. # TODO equatorial plane only
    a2 = a**2
    r2 = r**2
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
       
    # sign of current corresponds to energy outflow 
    f2 = (6*np.pi*np.pi - 49.)/72.
    omega2 = 1./32. - sth2*(4*f2 - 1)/64.
    I = -1*C*2*np.pi*sth2*(a/8. + (a**3)*(omega2 + 0.25*fr*cth2))
    
    # field components
    Br = dphidtheta / gdet
    Bth = -dphidr / gdet   
    Bph = I / (2*np.pi*Delta*sth2)
    
    # second (or third?) order only
    if secondorder_only: # divergence happens at 2M here, not horizon
        Br = C/r2 + C*a2*(-cth2/(r2*r2) + fr*(0.5 + 1.5*np.cos(2*th))/r2)
        Bth = -C*a2*frp*cth*sth/r2
        Bph = -C*0.125*a/(r2-2*r) + C*(a**3)*(0.125/((r2-2*r)**2) - (omega2 + 0.25*cth2*fr)/((r2-2*r)))
    
    # angular frequency
    spin = np.abs(a)    
    OmegaBZ = spin/8. + spin**3 * omega2
    if a<0: OmegaBZ*=-1 #  TODO: is this right for a<0? 

    return(Br, Bth, Bph, OmegaBZ)
 
def Efield_BZmonopole(a, r, th, C=1):
    """perturbative BZ monopole solution for electric field
       only up to 2nd order
       C is overall sign of monopole"""

    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")

#    th = np.pi/2. # TODO equatorial plane only
    Eth = a*np.sin(th)*(-1./8. + 2./r**3)/(r**2 - 2*r)
                
    return(0, Eth, 0)   

def omega_BZpara(th, psi, a):
    rp = 1+np.sqrt(1-a**2) #outer radius
    abscth = np.abs(np.cos(th))
    wfunc = sp.lambertw(-psi/rp+np.log(4), k=0)
    xfunc = np.exp(wfunc)
    yfunc = 1+psi/(1+wfunc)/(xfunc*(2-xfunc))
    return a/4/yfunc

def Bfield_BZpara(a, r, th, C=1):
    """perturbative BZ paraboloid solution.
       C is overall sign"""

    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")

#    th = np.pi/2. # TODO equatorial plane only
    a2 = a**2
    r2 = r**2
    rp = 1+np.sqrt(1-a2) #outer horizon
    sth = np.sin(th)
    cth = np.cos(th)
    abscth = np.abs(cth)
    cth2 = cth**2
    sth2 = sth**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2*cth2
    gdet = sth*Sigma
    
    #vector potential
    psi = r*(1-abscth)+rp*(1+abscth)*(1-np.log(1+abscth))-2*rp*(1-np.log(2))
    dpsidtheta = np.sign(cth) * sth * (r+rp*np.log(1+abscth))
    dpsidr = 1-abscth

    argvals = -psi/rp + np.log(4) #valid solution up until this equals zero

    if not hasattr(psi, '__len__'): #allows for psi to be an array or a scalar
        if argvals > 0:
            OmegaBZ = omega_BZpara(th, psi, a)
        else:
            OmegaBZ = 0

    else:
        OmegaBZ = C*omega_BZpara(th, psi, a)*((argvals>=-1/np.e).astype('int'))
    
    #current
    I = -4*np.pi*psi*OmegaBZ * np.sign(cth)

    # # field components
    Br = C*dpsidtheta / gdet
    Bth = -C*dpsidr / gdet   
    Bph = I / (2*np.pi*Delta*sth2) 

    return(Br, Bth, Bph, OmegaBZ)


