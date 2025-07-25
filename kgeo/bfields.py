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


class Bfield(object):
    """ object for b-field as a function of r, only in equatorial plane for now """

    def __init__(self, fieldtype="rad", **kwargs):

        self.fieldtype = fieldtype

        if self.fieldtype in ['rad', 'vert', 'tor', 'simple', 'simple_rm1',
                              'monopoleA', 'bz_monopole', 'bz_guess',
                              'bz_para', 'power', 'fromfile']:
            self.fieldframe = 'lab'
        elif self.fieldtype in ['const_comoving']:
            self.fieldframe = 'comoving'
        else:
            raise Exception("fieldtype %s not recognized in Bfield!"%self.fieldtype)

        self.kwargs = kwargs

        if self.fieldframe not in ['lab', 'comoving']:
            raise Exception("Bfield fieldframe must be 'lab' or 'comoving'!")

        if self.fieldtype in ['bz_monopole', 'bz_guess', 'bz_para']:
            self.secondorder_only = self.kwargs.get('secondorder_only', False)
            self.C = self.kwargs.get('C', 1)
        elif self.fieldtype in ['monopoleA']:
            self.omegafac = self.kwargs.get('omegafac', 0.5)
            self.C = self.kwargs.get('C', 1)
        elif self.fieldtype == 'power':
            self.secondorder_only = self.kwargs.get('secondorder_only', False)
            self.C = self.kwargs.get('C', 1)
            self.pval = self.kwargs.get('p', 1)
            self.usemono = self.kwargs.get('usemono', False)
        elif self.fieldtype == 'rad':
            self.Cr = 1
            self.Cvert = 0
            self.Cph = 0
        elif self.fieldtype == 'vert':
            self.Cr = 0
            self.Cvert = 1
            self.Cph = 0
        elif self.fieldtype == 'tor':
            self.Cr = 0
            self.Cvert = 0
            self.Cph = 1
        elif self.fieldtype == 'fromfile':
            self.filename = self.kwargs.get('file', None)
            self.cached_data = load_cache_from_file(self.filename)
        else:
            self.Cr = self.kwargs.get('Cr',0)
            self.Cvert = self.kwargs.get('Cvert',0)
            self.Cph = self.kwargs.get('Cph',0)

            if self.Cr == self.Cvert == self.Cph == 0.:
                raise Exception("all field coefficients are 0!")

    def bfield_lab(self, a, r, th=np.pi/2):

        """lab frame b field starF^i0"""
        if not isinstance(r, np.ndarray):
            r = np.array([r]).flatten()

        if self.fieldframe != 'lab':
            raise Exception("Bfield.bfield_lab only supported for Bfield.fieldtype==lab")

        if self.fieldtype in ['simple', 'rad', 'vert', 'tor']:
            b_components = Bfield_simple(a, r, (self.Cr, self.Cvert, self.Cph))
        elif self.fieldtype == 'simple_rm1':
            b_components = Bfield_simple_rm1(a, r, (self.Cr, self.Cvert, self.Cph))
        elif self.fieldtype == 'bz_monopole':
            (B1,B2,B3,omega) = Bfield_BZmonopole(a, r, th, self.C, secondorder_only=self.secondorder_only)
            b_components = (B1,B2,B3)
        elif self.fieldtype == 'monopoleA':
            (B1,B2,B3,omega) = Bfield_monopoleA(a, r, th, self.C, omegafac=self.omegafac)
            b_components = (B1,B2,B3)
        elif self.fieldtype == 'bz_para':
            (B1,B2,B3,omega) = Bfield_BZpara(a, r, th, self.C)
            b_components = (B1,B2,B3)
        elif self.fieldtype == 'bz_guess':
            (B1,B2,B3,omega) = Bfield_BZmagic(a, r, th, self.C)
            b_components = (B1,B2,B3)
        elif self.fieldtype == 'power':
            (B1,B2,B3,omega) = Bfield_power(a, r, th, self.pval, C=self.C, usemono=self.usemono)
            b_components = (B1,B2,B3)
        elif self.fieldtype=='fromfile':
            B1, B2, B3 = Bfield_from_cache(a, r, self.cached_data)
            b_components = B1, B2, B3
        else:
            raise Exception("fieldtype %s not recognized in Bfield.bfield_lab!"%self.fieldtype)

        return b_components

    def bfield_comoving(self, a, r):
        """fluid frame B-field"""
        if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

        if self.fieldframe != 'comoving':
            raise Exception("Bfield.bfield_comoving only supported for Bfield.fieldtype==comoving")
        if self.fieldtype == 'const_comoving':
            b_components = (self.Cr*np.ones(r.shape),
                            -1*self.Cvert*np.ones(r.shape),
                            self.Cph*np.ones(r.shape))
        else:
            raise Exception("fieldtype %s not recognized in Bfield.bfield_comoving!"%self.fieldtype)

        return b_components

    # TODO ANDREW FIX THESE WITH BETTER DATA STRUCTURES
    def omega_field(self, a, r, th=np.pi/2):
        """fieldline angular speed"""
        if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

        if self.fieldtype=='bz_monopole':
            (B1,B2,B3,omega) = Bfield_BZmonopole(a, r, th, self.C,secondorder_only=self.secondorder_only)
        elif self.fieldtype=='monopoleA':
            (B1,B2,B3,omega) = Bfield_monopoleA(a, r, th, self.C, omegafac=self.omegafac)
        elif self.fieldtype=='bz_guess':
            (B1,B2,B3,omega) = Bfield_BZmagic(a, r, th, self.C)
        elif self.fieldtype=='bz_para':
            (B1,B2,B3,omega) = Bfield_BZpara(a, r, th, self.C)
        elif self.fieldtype=='power':
            (B1,B2,B3,omega) = Bfield_power(a, r, th, self.pval, C=self.C, usemono=self.usemono)

        else:
            raise Exception("self.omega_field not implemented for fieldtype %s'!"%self.fieldtype)
        return omega


    def efield_lab(self, a, r, th=np.pi/2):
        """lab frame electric field F^{0i} in BL coordinates.
           below defn is for stationary, axisymmetric fields"""
        if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

        if self.fieldtype=='bz_monopole' and self.secondorder_only:
            e_components = Efield_BZmonopole(a,r,th, self.C)
        elif self.fieldtype in ['bz_monopole','monopoleA','bz_guess','bz_para','power']:
            if self.fieldtype=='bz_monopole':
                (B1,B2,B3,omega) = Bfield_BZmonopole(a, r, th, self.C,secondorder_only=self.secondorder_only)
            elif self.fieldtype=='monopoleA':
                (B1,B2,B3,omega) = Bfield_monopoleA(a, r, th, self.C, omegafac=self.omegafac)
            elif self.fieldtype=='bz_guess':
                (B1,B2,B3,omega) = Bfield_BZmagic(a, r, th, self.C)
            elif self.fieldtype=='bz_para':
                (B1,B2,B3,omega) = Bfield_BZpara(a, r, th, self.C)
            elif self.fieldtype=='power':
                (B1,B2,B3,omega) = Bfield_power(a, r, th, self.pval, C=self.C, usemono=self.usemono)
            a2 = a**2
            r2 = r**2
            cth2 = np.cos(th)**2
            sth2 = np.sin(th)**2
            Delta = r2 - 2*r + a2
            Sigma = r2 + a2 * cth2
            Pi = (r2+a2)**2 - a2*Delta*sth2
            omegaz = 2*a*r/Pi
            E1 = (omega-omegaz)*Pi*np.sin(th)*B2/Sigma
            E2 = -(omega-omegaz)*Pi*np.sin(th)*B1/(Sigma*Delta)
            E3 = np.zeros_like(E2) if hasattr(E2, '__len__') else 0
            e_components = (E1, E2, E3)
        else:
            raise Exception("self.efield_lab not implemented for fieldtype %s'!"%self.fieldtype)

        return e_components

    def maxwell(self, a, r, th=np.pi/2):
        """Maxwell tensor starF^{\mu\nu} in BL coordinates.
           below defn is for stationary, axisymmetric fields"""

        if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

        if self.fieldtype in ['bz_monopole','monopoleA','bz_guess','bz_para','power']:
            if self.fieldtype=='bz_monopole':
                (B1,B2,B3,OmegaF) = Bfield_BZmonopole(a, r, th, self.C)
            elif self.fieldtype=='monopoleA':
                (B1,B2,B3,omega) = Bfield_monopoleA(a, r, th, self.C, omegafac=self.omegafac)
            elif self.fieldtype=='bz_guess':
                (B1,B2,B3,OmegaF) = Bfield_BZmagic(a, r, th, self.C)
            elif self.fieldtype=='bz_para':
                (B1,B2,B3,OmegaF) = Bfield_BZpara(a, r, th, self.C)
            elif self.fieldtype=='power':
                (B1,B2,B3,omega) = Bfield_power(a, r, th, self.pval, C=self.C, usemono=self.usemono)
            sF01 = -B1
            sF02 = -B2
            sF03 = -B3
            sF12 = 0*B3
            sF13 = OmegaF*B1
            sF23 = OmegaF*B2

            sF_out = (sF01, sF02, sF03, sF12, sF13, sF23)

        else:
            raise Exception("self.maxwell not implemented for fieldtype %s'!"%self.fieldtype)

        return sF_out

    def faraday(self, a, r, th=np.pi/2):
        """Faraday tensor F^{\mu\nu} in BL coordinates.
           below defn is for stationary, axisymmetric fields"""
        if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

        if self.fieldtype in ['bz_monopole','monopoleA','bz_guess','bz_para','power']:
            if self.fieldtype=='bz_monopole':
                (B1,B2,B3,OmegaF) = Bfield_BZmonopole(a, r, th, self.C)
            elif self.fieldtype=='monopoleA':
                (B1,B2,B3,omega) = Bfield_monopoleA(a, r, th, self.C, omegafac=self.omegafac)
            elif self.fieldtype=='bz_guess':
                (B1,B2,B3,OmegaF) = Bfield_BZmagic(a, r, th, self.C)
            elif self.fieldtype=='bz_para':
                (B1,B2,B3,OmegaF) = Bfield_BZpara(a, r, th, self.C)
            elif self.fieldtype=='power' :
                (B1,B2,B3,omega) = Bfield_power(a, r, th, self.pval, C=self.C, usemono=self.usemono)

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
            F_02 = gdet*OmegaF*B1
            F_03 = 0*B3
            F_12 = gdet*B3
            F_13 = -gdet*B2
            F_23 = gdet*B1

            # raise indices
            F01 = g00_up*g11_up*F_01 + g03_up*g11_up*(-F_13)
            F02 = g00_up*g22_up*F_02 + g03_up*g22_up*(-F_23)
            F03 = 0*B3
            F12 = g11_up*g22_up*F_12
            F13 = g11_up*g33_up*F_13 + g11_up*g03_up*(-F_01)
            F23 = g22_up*g33_up*F_23 + g22_up*g03_up*(-F_02)

            F_out = (F01, F02, F03, F12, F13, F23)

        else:
            raise Exception("self.faraday not implemented for fieldtype %s'!"%self.fieldtype)

        return F_out

    def bfield_fluid(self, a, r, velocity, th=np.pi/2.):
        """returns b^\mu in frame u^\mu, making ideal MHD assumption, e^\mu=0"""
        if not isinstance(r, np.ndarray):
            r = np.array([r]).flatten()

        # Metric
        a2 = a**2
        r2 = r**2
        cth2 = np.cos(th)**2
        sth2 = np.sin(th)**2
        Delta = r2 - 2*r + a2
        Sigma = r2 + a2 * cth2

        g00 = -(1 - 2*r/Sigma)
        g11 = Sigma/Delta
        g22 = Sigma
        g33 = (r2 + a2 + 2*r*a2*sth2 / Sigma) * sth2
        g03 = -2*r*a*sth2 / Sigma

        # bfield components
        (B1, B2, B3) = self.bfield_lab(a, r, th=th)

        # velocity components
        (u0, u1, u2, u3) = velocity.u_lab(a, r, th=th)
        (u0_l, u1_l, u2_l, u3_l) = velocity.u_lab_cov(a, r, th=th)

        # here, we assume the field is degenerate and e^\mu = u_\nu F^{\mu\nu} = 0
        # (standard GRMHD assumption)
        b0 = B1*u1_l + B2*u2_l + B3*u3_l
        b1 = (B1 + b0*u1)/u0
        b2 = (B2 + b0*u2)/u0
        b3 = (B3 + b0*u3)/u0

        return (b0, b1, b2, b3)

    def bsq(self, a, r, velocity, th=np.pi/2.):
        """returns b^2 in frame u^\mu, making ideal MHD assumption, e^\mu=0"""
        if not isinstance(r, np.ndarray):
            r = np.array([r]).flatten()

        # Metric
        a2 = a**2
        r2 = r**2
        cth2 = np.cos(th)**2
        sth2 = np.sin(th)**2
        Delta = r2 - 2*r + a2
        Sigma = r2 + a2 * cth2

        g00 = -(1 - 2*r/Sigma)
        g11 = Sigma/Delta
        g22 = Sigma
        g33 = (r2 + a2 + 2*r*a2*sth2 / Sigma) * sth2
        g03 = -2*r*a*sth2 / Sigma

        # contravarient components
        (b0, b1, b2, b3) = self.bfield_fluid(a, r, velocity, th=th)

        # covarient components
        b0_l = g00*b0 + g03*b3
        b1_l = g11*b1
        b2_l = g22*b2
        b3_l = g33*b3 + g03*b0

        # field strength squared
        bsq = b0*b0_l + b1*b1_l + b2*b2_l + b3*b3_l

        return bsq


def Bfield_simple(a, r, coeffs):
    """magnetic field vector in the lab frame in equatorial plane,
       simple configurations"""
    #|Br| falls of as 1/r^2
    #|Bvert| falls of as 1/r
    #|Bph| falls of as 1/r

    if not (isinstance(a, float) and (0 <= np.abs(a) < 1)):
        raise Exception("|a| should be a float in range [0,1)")

    # coefficients for each component
    (arad, avert, ator) = coeffs

    th = np.pi/2.  # TODO equatorial plane only
    Sigma = r**2 + a**2 * np.cos(th)**2
    gdet = np.abs(np.sin(th))*Sigma  # ~r^2 in eq plane sin0

    # b-field in lab
    Br = arad*(np.sin(th)/gdet) + avert*(r*np.cos(th)/gdet)
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

    th = np.pi/2.  # TODO equatorial plane only
    Sigma = r**2 + a**2 * np.cos(th)**2
    gdet = np.abs(np.sin(th))*Sigma  # ~r^2 in eq plane sin0

    # b-field in lab
    Br = arad*(r*np.sin(th)/gdet) + avert*(r*np.cos(th)/gdet)
    Bth = avert*(-np.sin(th)/gdet)
    Bph = ator*(1./gdet)

    return (Br, Bth, Bph)


def Bfield_monopoleA(a, r, th, C=1, omegafac=0.5):
    """Simplified monopole enforcing Znajek condition at horizon
       will not have physical drift velocity at all radii"""

    rh = 1 + np.sqrt(1-a**2)  # horizon radius
    omegah = a/(2*rh)         # horizon angular speed
    phih = 4*np.pi*C    # horizon flux

    a2 = a**2
    r2 = r**2
    cth = np.cos(th)
    cth2 = cth**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2*cth2
    omegaf = omegafac*omegah  # fieldline angular speed

    I = phih*(np.sin(th)**2)*rh*(omegaf-omegah)/(rh**2 + a**2*np.cos(th)**2)  # current
    B1 = (phih/(4*np.pi))/Sigma
    B2 = 0
    B3 = I / (2*np.pi*Delta*(np.sin(th)**2))

    return (B1,B2,B3,omegaf)


def Bfield_BZmagic(a, r, th, C=1):
    """Guess ratio for Bphi/Br from split monopole
       C is overall sign of monopole"""

    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")

    a2 = a**2
    r2 = r**2
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

    return (Br, Bth, Bph, OmegaBZ)


def Bfield_BZmonopole(a, r, th, C=1, secondorder_only=False):
    """perturbative BZ monopole solution.
       C is overall sign of monopole"""

    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")

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

    return (Br, Bth, Bph, OmegaBZ)


def Efield_BZmonopole(a, r, th, C=1):
    """perturbative BZ monopole solution for electric field
       only up to 2nd order
       C is overall sign of monopole"""

    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")


    Eth = a*np.sin(th)*(-1./8. + 2./r**3)/(r**2 - 2*r)

    return (0, Eth, 0)


def omega_BZpara(th, psi, a):
    rp = 1+np.sqrt(1-a**2)  # outer radius
    abscth = np.abs(np.cos(th))
    wfunc = sp.lambertw(-psi/rp+np.log(4), k=0)
    xfunc = np.exp(wfunc)
    yfunc = 1+psi/(1+wfunc)/(xfunc*(2-xfunc))
    return a/4/yfunc


def omega_BZpower(th, psi, a, p, usemono=False):
    if usemono or p == 0:  # just return the monopole rate
        return a/8
    rp = 1+np.sqrt(1-a**2)
    cthhorizon = 1-psi*rp**(-p)
    denomfac = 8/(1+cthhorizon)
    return a/(4+denomfac)  # equal to a/(4(1+sec^2(theta/2))) via trig identities


def Bfield_BZpara(a, r, th, C=1):
    """perturbative BZ paraboloid solution.
       C is overall sign"""

    if not (isinstance(a, float) and (0 <= np.abs(a) < 1)):
        raise Exception("|a| should be a float in range [0,1)")

#    th = np.pi/2. # TODO equatorial plane only
    a2 = a**2
    r2 = r**2
    rp = 1+np.sqrt(1-a2)  # outer horizon
    sth = np.sin(th)
    cth = np.cos(th)
    abscth = np.abs(cth)
    cth2 = cth**2
    sth2 = sth**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2*cth2
    gdet = sth*Sigma

    # vector potential
    psi = r*(1-abscth)+rp*(1+abscth)*(1-np.log(1+abscth))-2*rp*(1-np.log(2))
    dpsidtheta = np.sign(cth) * sth * (r+rp*np.log(1+abscth))
    dpsidr = 1-abscth

    argvals = -psi/rp + np.log(4)  # valid solution up until this equals zero

    if not hasattr(psi, '__len__'):  # allows for psi to be an array or a scalar
        if argvals >= 0:

            OmegaBZ = omega_BZpara(th, psi, a)
        else:
            OmegaBZ = 0

    else:
        OmegaBZ = C*omega_BZpara(th, psi, a)*((argvals>=-1/np.e).astype('int'))

    # current
    I = -4*np.pi*psi*OmegaBZ * np.sign(cth)

    # # field components
    Br = C*dpsidtheta / gdet
    Bth = -C*dpsidr / gdet
    Bph = I / (2*np.pi*Delta*sth2)

    return (Br, Bth, Bph, OmegaBZ)


# power law fields
def Bfield_power(a, r, th, p, C=1, usemono=False):
    """stream function of the form psi=r^p(1-costheta) with same Bphi as paraboloid (I should already match to leading order)"""

    if not (isinstance(a, float) and (0 <= np.abs(a) < 1)):
        raise Exception("|a| should be a float in range [0,1)")

    a2 = a**2
    r2 = r**2
    rp = 1+np.sqrt(1-a2)  # outer horizon
    sth = np.sin(th)
    cth = np.cos(th)
    abscth = np.abs(cth)
    cth2 = cth**2
    sth2 = sth**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2*cth2
    gdet = sth*Sigma

    # vector potential
    psi = r**p*(1-abscth)
    dpsidtheta = np.sign(cth) * sth * (r**p)
    dpsidr = p*psi/r

    OmegaBZ = omega_BZpower(th, psi, a, p, usemono=usemono)

    # current
    I = -4*np.pi*psi*OmegaBZ * np.sign(cth) if p > 0 else -2*np.pi*psi*(2-psi)*OmegaBZ * np.sign(cth) #different if monopole or not

    # # field components
    Br = C*dpsidtheta / gdet
    Bth = -C*dpsidr / gdet
    Bph = I / (2*np.pi*Delta*sth2)

    return (Br, Bth, Bph, OmegaBZ)


def load_cache_from_file(filename):
    """
    Load Br,Btheta,Bphi primitive (lab frame) magnetic field components from file
    """
    # load header from file
    with open(filename, 'r') as f:
        header = f.readline().strip()
        if header[0] != '#':
            header = None
            raise Exception("file %s does not have a header!" % filename)
        else:
            header = [x.strip() for x in header[1:].split(',')]

    # load header from file
    with open(filename, 'r') as f:
        header = f.readline().strip()
        if header[0] != '#':
            header = None
            raise Exception("file %s does not have a header!" % filename)
        else:
            header = [x.strip() for x in header[1:].split(',')]

    # load data from file
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    n_samples, _ = data.shape
    radii = data[:, header.index('r')]

    Br = np.zeros_like(radii)
    Bth = np.zeros_like(radii)
    Bph = np.zeros_like(radii)

    if 'Br' in header:
        Br = data[:, header.index('Br')]
    if 'Btheta' in header:
        Bth = data[:, header.index('Btheta')]
    if 'Bphi' in header:
        Bph = data[:, header.index('Bphi')]

    return dict(radii=radii, Br=Br, Bth=Bth, Bph=Bph)


def Bfield_from_cache(a, r, rb123_cache):
    """
    Four-velocity as linearly interpolated from input file. By
    convention, one file is for a single spin, which means that
    this function will fail *silently* for other cases.
    """
    # TODO add check for spin?

    # cast input to numpy array
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

    # get values from cache
    Br = np.interp(r, rb123_cache['radii'], rb123_cache['Br'])
    Bth = np.interp(r, rb123_cache['radii'], rb123_cache['Bth'])
    Bph = np.interp(r, rb123_cache['radii'], rb123_cache['Bph'])

    return (Br, Bth, Bph)
