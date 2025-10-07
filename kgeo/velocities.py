import numpy as np
import scipy.optimize as opt
from kgeo.bfields import Bfield

# simulation fit factors
ELLISCO =1.; VRISCO = 2;
P1=6.; P2=2.; DD=0.2;

# gelles model parameters
BETA = 0.3
CHI = -150*np.pi/180.

# default b field for drift frame
BFIELD_DEFAULT = Bfield('bz_para')

# used in testing
_allowed_velocity_models = [
    'zamo', 'infall', 'kep', 'cunningham', 'subkep', 'cunningham_subkep',
    'general', 'gelles', 'simfit', 'fromfile', 'driftframe'
]


class Velocity(object):
    """ object for lab frame velocity as a function of r, only in equatorial plane for now """

    def __init__(self, veltype="kep", **kwargs):

        self.veltype = veltype
        self.kwargs = kwargs

        if self.veltype=='zamo' or self.veltype=='infall':
            pass

        elif self.veltype=='kep' or self.veltype=='cunningham':
            self.retrograde = self.kwargs.get('retrograde', False)

        elif self.veltype=='subkep' or self.veltype=='cunningham_subkep':
            self.retrograde = self.kwargs.get('retrograde', False)
            self.fac_subkep = self.kwargs.get('fac_subkep', 1)

        elif self.veltype=='general':
            self.retrograde = self.kwargs.get('retrograde', False)
            self.fac_subkep = self.kwargs.get('fac_subkep', 1)
            self.beta_phi = self.kwargs.get('beta_phi', 1)
            self.beta_r = self.kwargs.get('beta_r',1)

        elif self.veltype=='gelles':
            self.gelles_beta = self.kwargs.get('gelles_beta', BETA)
            self.gelles_chi = self.kwargs.get('gelles_chi', CHI)

        elif self.veltype=='simfit':
            self.ell_isco = self.kwargs.get('ell_isco', ELLISCO)
            self.vr_isco = self.kwargs.get('vr_isco', VRISCO)
            self.p1 = self.kwargs.get('p1', P1)
            self.p2 = self.kwargs.get('p2', P2)
            self.dd = self.kwargs.get('dd', DD)

        elif self.veltype=='fromfile':
            self.filename = self.kwargs.get('file', None)
            self.cached_data = load_cache_from_file(self.filename)

        elif self.veltype=='driftframe':
            self.bfield = self.kwargs.get('bfield', BFIELD_DEFAULT)
            self.nu_parallel = self.kwargs.get('nu_parallel', 0)
            self.gammamax = self.kwargs.get('gammamax', None)
            
        elif self.veltype=='MHD':
            self.bfield = self.kwargs.get('bfield', BFIELD_DEFAULT)
            self.gammamax = self.kwargs.get('gammamax', 100.0)
                           
        else: 

            raise Exception("veltype %s not recognized in Velocity!"%self.veltype)

    def u_lab(self, a, r, th=np.pi/2., retqty=False):
        """Return lab frame contravarient velocity vector"""
        if self.veltype=='zamo':
            ucon = u_zamo(a, r)
        elif self.veltype=='infall':
            ucon = u_infall(a, r)
        elif self.veltype=='kep' or self.veltype=='cunningham':
            ucon = u_kep(a, r, retrograde=self.retrograde)
        elif self.veltype=='subkep' or self.veltype=='cunningham_subkep':
            ucon = u_subkep(a, r, retrograde=self.retrograde, fac_subkep=self.fac_subkep)
        elif self.veltype=='general':
            ucon = u_general(a, r, retrograde=self.retrograde, fac_subkep=self.fac_subkep,
                             beta_phi=self.beta_phi, beta_r=self.beta_r)
        elif self.veltype=='gelles':
            ucon = u_gelles(a, r, beta=self.gelles_beta, chi=self.gelles_chi)
        elif self.veltype=='simfit':
            ucon = u_grmhd_fit(a,r, ell_isco=self.ell_isco, vr_isco=self.vr_isco, p1=self.p1, p2=self.p2, dd=self.dd)
        elif self.veltype=='fromfile':
            ucon = u_from_u123(a, r, self.cached_data)
        elif self.veltype=='driftframe':
            ucon = u_driftframe(a, r, bfield=self.bfield, nu_parallel=self.nu_parallel, th=th, gammamax = self.gammamax, retqty=retqty)
        elif self.veltype=='MHD':
            ucon = u_MHD(a, r, bfield=self.bfield, th=th, gammamax = self.gammamax)

        else:
            raise Exception("veltype %s not recognized in Velocity.u_lab!"%self.veltype)

        return ucon

    def u_lab_cov(self, a, r, th=np.pi/2.):
        """Return lab frame covarient velocity vector"""
        (u0, u1, u2, u3) = self.u_lab(a, r, th=th)

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

        # covariant velocity
        u0_l = g00*u0 + g03*u3
        u1_l = g11*u1
        u2_l = g22*u2
        u3_l = g33*u3 + g03*u0

        return (u0_l, u1_l, u2_l, u3_l)

    def tetrades(self, a, r, th=np.pi/2):
        """Return tetrads for transformation to orthonormal frame"""

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

        # velocity components
        (u0, u1, u2, u3) = self.u_lab(a, r, th=th)
        (u0_l, u1_l, u2_l, u3_l) = self.u_lab_cov(a, r, th=th)

        # define tetrads to comoving frame
        Nr = np.sqrt(-g11*(u0_l*u0 + u3_l*u3)*(1 + u2_l*u2))
        Nth = np.sqrt(g22*(1 + u2_l*u2))
        Nph = np.sqrt(-Delta*sth2*(u0_l*u0 + u3_l*u3))

        e0_t = -u0
        e1_t = -u1
        e2_t = -u2
        e3_t = -u3

        e0_x = u1_l*u0/Nr
        e1_x = -(u0_l*u0 + u3_l*u3)/Nr
        e2_x = 0
        e3_x = u1_l*u3/Nr

        e0_y = u2_l*u0/Nth
        e1_y = u2_l*u1/Nth
        e2_y = (1+u2_l*u2)/Nth
        e3_y = u2_l*u3/Nth

        e0_z = u3_l/Nph
        e1_z = 0
        e2_z = 0
        e3_z = -u0_l/Nph

        # output 4 tetrades
        tetrades = ((e0_t, e1_t, e2_t, e3_t),
                    (e0_x, e1_x, e2_x, e3_x),
                    (e0_y, e1_y, e2_y, e3_y),
                    (e0_z, e1_z, e2_z, e3_z))

        return tetrades


def u_zamo(a, r):
    """velocity for zero angular momentum frame"""
    # checks
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

    # Metric
    th = np.pi/2.  # TODO equatorial only
    Delta = r**2 - 2*r + a**2
    Sigma = r**2 + a**2 * np.cos(th)**2
    g00 = -(1-2*r/Sigma)
    g11 = Sigma/Delta
    g22 = Sigma
    g33 = (r**2 + a**2 + 2*r*(a*np.sin(th))**2 / Sigma) * np.sin(th)**2
    g03 = -2*r*a*np.sin(th)**2 / Sigma

    # zamo angular velocity
    v3 = -g03/g33

    # Compute u0
    aa = g00
    bb = 2*g03*v3
    cc = g33*v3*v3
    u0 = np.sqrt(-1./(aa + bb + cc))

    # Compute the 4-velocity (contravariant)
    u3 = u0*v3

    return (u0, 0, 0, u3)


def u_infall(a, r):
    """ velocity for geodesic equatorial infall from infinity"""

    # checks
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

    Delta = r**2 + a**2 - 2*r
    Xi = (r**2 + a**2)**2 - Delta*a**2

    u0 = Xi/(r**2 * Delta)
    u1 = -np.sqrt(2*r*(r**2 + a**2))/(r**2)
    u3 = 2*a/(r*Delta)

    return (u0, u1, 0, u3)


def u_kep(a, r, retrograde=False):
    """Cunningham velocity for material on keplerian orbits and infalling inside isco"""
    # checks
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

    if retrograde:
        s = -1
    else:
        s = 1

    # isco radius
    z1 = 1 + np.cbrt(1-a**2)*(np.cbrt(1+a) + np.cbrt(1-a))
    z2 = np.sqrt(3*a**2 + z1**2)
    ri = 3 + z2 - s*np.sqrt((3-z1)*(3+z1+2*z2))
    #print("r_isco: ",ri)

    u0 = np.zeros(r.shape)
    u1 = np.zeros(r.shape)
    u3 = np.zeros(r.shape)

    iscomask = (r >= ri)
    spin = np.abs(a)
    asign = np.sign(a)
    # outside isco
    if np.any(iscomask):
        rr = r[iscomask]

        Omega = asign*s / (rr**1.5 + s*spin)
        u0[iscomask] = (rr**1.5 + s*spin) / np.sqrt(rr**3 - 3*rr**2 + 2*s*spin*rr**1.5)
        u3[iscomask] = Omega*u0[iscomask]

    # inside isco
    if np.any(~iscomask):
        rr = r[~iscomask]

        # preliminaries
        Delta = (rr**2 - 2*rr + a**2)

        # isco conserved quantities
        ell_i = s*asign*(ri**2 + a**2 - s*2*spin*np.sqrt(ri))/(ri**1.5 - 2*np.sqrt(ri) + s*spin)
        gam_i = np.sqrt(1 - 2./(3.*ri))  # nice expression only for isco, prograde or retrograde

        # contravarient vel
        H = (2*rr - a*ell_i)/Delta
        chi = 1/(1 + (2/rr)*(1+H))

        u0[~iscomask] = gam_i*(1 + (2/rr)*(1 + H))
        u1[~iscomask] = -np.sqrt(2./(3*ri))*(ri/rr - 1)**1.5
        u3[~iscomask] = gam_i*(ell_i + a*H)/(rr**2)

    return (u0, u1, 0, u3)


def u_subkep(a, r, fac_subkep=1, retrograde=False):
    """(sub) keplerian velocty and infalling inside isco"""
    # checks
    if not (isinstance(a,float) and (0 <= np.abs(a) < 1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not (0 <= fac_subkep <= 1):
        raise Exception("fac_subkep should be in the range [0,1]")
    if not isinstance(r, np.ndarray):
        r = np.array([r]).flatten()

    if retrograde:
        s = -1
    else:
        s = 1

    # isco radius
    z1 = 1 + np.cbrt(1-a**2)*(np.cbrt(1+a) + np.cbrt(1-a))
    z2 = np.sqrt(3*a**2 + z1**2)
    ri = 3 + z2 - s*np.sqrt((3-z1)*(3+z1+2*z2))
    #print("r_isco:", ri)

    u0 = np.zeros(r.shape)
    u1 = np.zeros(r.shape)
    u3 = np.zeros(r.shape)

    iscomask = (r >= ri)
    spin = np.abs(a)
    asign = np.sign(a)
    # outside isco)
    if np.any(iscomask):
        rr = r[iscomask]

        # preliminaries
        Delta = (rr**2 - 2*rr + a**2)
        Xi = (rr**2 + a**2)**2 - Delta*a**2

        # conserved quantities
        ell = asign*s * (rr**2 + spin**2 - s*2*spin*np.sqrt(rr))/(rr**1.5 - 2*np.sqrt(rr) + s*spin)
        ell *= fac_subkep
        gam = np.sqrt(Delta/(Xi/rr**2 - 4*a*ell/rr - (1-2/rr)*ell**2))

        # contravarient vel
        H = (2*rr - a*ell)/Delta
        chi = 1 / (1 + (2/rr)*(1+H))
        Omega = (chi/rr**2)*(ell + a*H)

        u0[iscomask] = gam/chi
        u3[iscomask] = (gam/chi)*Omega

    # inside isco
    if np.any(~iscomask):
        rr = r[~iscomask]

        # preliminaries
        Delta = (rr**2 - 2*rr + a**2)
        Xi = (rr**2 + a**2)**2 - Delta*a**2

        # isco conserved quantities
        Delta_i = (ri**2 - 2*ri + a**2)
        Xi_i = (ri**2 + a**2)**2 - Delta_i*a**2

        ell_i = asign*s * (ri**2 + spin**2 - s*2*spin*np.sqrt(ri))/(ri**1.5 - 2*np.sqrt(ri) + s*spin)
        ell_i *= fac_subkep
        gam_i = np.sqrt(Delta_i/(Xi_i/ri**2 - 4*a*ell_i/ri - (1-2/ri)*ell_i**2))

        # contravarient vel
        H = (2*rr - a*ell_i)/Delta
        chi = 1 / (1 + (2/rr)*(1+H))
        Omega = (chi/rr**2)*(ell_i + a*H)
        nu = (rr/Delta)*np.sqrt(Xi/rr**2 - 4*a*ell_i/rr - (1-2/rr)*ell_i**2 - Delta/gam_i**2)

        u0[~iscomask] = gam_i/chi
        u1[~iscomask] = -gam_i*(Delta/rr**2)*nu
        u3[~iscomask] = (gam_i/chi)*Omega

    return (u0, u1, 0, u3)


def u_gelles(a, r, beta=0.3, chi=-150*(np.pi/180.)):
    """velocity prescription from Gelles+2021, Eq A4"""
    # checks
    if not (isinstance(a, float) and (0 <= np.abs(a) < 1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not isinstance(r, np.ndarray):
        r = np.array([r]).flatten()

    # Metric
    a2 = a**2
    r2 = r**2
    th = np.pi/2. # equatorial only
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * cth2
    Xi = (r2 + a2)**2 - Delta*a2*sth2
    omegaz = 2*a*r/Xi

    gamma = 1/np.sqrt(1-beta**2)
    coschi = np.cos(chi)
    sinchi = np.sin(chi)

    u0 = (gamma/r)*np.sqrt(Xi/Delta)
    u1 = (beta*gamma*coschi/r)*np.sqrt(Delta)
    u3 = (gamma*omegaz/r)*np.sqrt(Xi/Delta) + (r*beta*gamma*sinchi)/np.sqrt(Xi)

    return (u0, u1, 0, u3)


def u_grmhd_fit(a, r, ell_isco=ELLISCO, vr_isco=VRISCO, p1=P1, p2=P2, dd=DD):
    """velocity for power laws fit to grmhd ell, conserved inside isco
       should be timelike throughout equatorial plane
       might not work for all spins
    """
    # checks
    if not (isinstance(a, float) and (0 <= np.abs(a) < 1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

    # isco radius
    rh = 1 + np.sqrt(1-a**2)
    z1 = 1 + np.cbrt(1-a**2)*(np.cbrt(1+a) + np.cbrt(1-a))
    z2 = np.sqrt(3*a**2 + z1**2)
    r_isco = 3 + z2 - np.sqrt((3-z1)*(3+z1+2*z2))

    # Metric
    a2 = a**2
    r2 = r**2
    th = np.pi/2.  # TODO equatorial only
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * cth2

    g00_up = -(r2 + a2 + 2*r*a2*sth2/Sigma) / Delta
    g11_up = Delta/Sigma
    g22_up = 1./Sigma
    g33_up = (Delta - a2*sth2)/(Sigma*Delta*sth2)
    g03_up = -(2*r*a)/(Sigma*Delta)

    # Fitting function should work down to the horizon
    # u_phi/u_t fitting function
    ell = ell_isco*(r/r_isco)**.5  # defined positive
    vr = -vr_isco*((r/r_isco)**(-p1)) * (0.5*(1+(r/r_isco)**(1/dd)))**((p1-p2)*dd)
    gam = np.sqrt(-1./(g00_up + g11_up*vr*vr + g33_up*ell*ell - 2*g03_up*ell))

    # compute u_t
    u_0 = -gam
    u_1 = vr*gam
    u_3 = ell*gam

    # raise indices
    u0 = g00_up*u_0 + g03_up*u_3
    u1 = g11_up*u_1
    u3 = g33_up*u_3 + g03_up*u_0

    return (u0, u1, 0, u3)


def u_general(a, r, fac_subkep=1, beta_phi=1, beta_r=1, retrograde=False):
    """general velocity model from AART paper, keplerian by default"""
    # checks
    if not (isinstance(a, float) and (0 <= np.abs(a) < 1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not (0 <= fac_subkep <= 1):
        raise Exception("fac_subkep should be in the range [0,1]")
    if not (0 <= beta_phi <= 1):
        raise Exception("beta_phi should be in the range [0,1]")
    if not (0 <= beta_r <= 1):
        raise Exception("beta_r should be in the range [0,1]")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

    Delta = r**2 + a**2 - 2*r

    (u0_infall, u1_infall, _, u3_infall) = u_infall(a,r)
    Omega_infall = u3_infall/u0_infall  # 2ar/Xi
    (u0_subkep, u1_subkep, _, u3_subkep) = u_subkep(a,r,retrograde=retrograde,fac_subkep=fac_subkep)
    Omega_subkep = u3_subkep/u0_subkep

    u1 = u1_subkep + (1-beta_r)*(u1_infall - u1_subkep)
    Omega = Omega_subkep + (1-beta_phi)*(Omega_infall - Omega_subkep)

    u0 = np.sqrt(1 + (r**2) * (u1**2) / Delta)  #???
    u0 /= np.sqrt(1 - (r**2 + a**2)*Omega**2 - (2/r)*(1 - a*Omega)**2)

    u3 = u0*Omega

    return (u0, u1, 0, u3)


# get boost parameter that conserves energy in co-rotating frame
def getnu_cons(bf_here, r, theta, r0, theta0, Omegaf, spin, M):
    Aconst = getEco(r0, theta0, Omegaf, spin, M)  #E-L*Omegaf
    ghere = metric(r, spin, theta, M)  #metric (as a matrix)
    (alpha, vphiupper, gammap, Bhatphi) = u_driftframe(spin, r, bfield=bf_here, nu_parallel=0, th=theta, retbunit=True) #get quantities from nu=0 case
    ffunc = gammap*(alpha-(ghere[:,0,3]+ghere[:,3,3]*Omegaf)*vphiupper)  #random function (useful for xi computation)
    bred = Bhatphi*(ghere[:,0,3]+ghere[:,3,3]*Omegaf)  #reduced Bphiunit (useful for xi computation)
    nunum = ffunc*bred+np.sign(np.cos(theta))*np.sign(r-r0)*Aconst*np.sqrt(Aconst**2-ffunc**2+bred**2)
    nudenom = Aconst**2+bred**2
    nutot = nunum/nudenom
    return np.real(nutot)


# #full GRMHD solution to wind equation
# def u_MHD(a, r, bfield=BFIELD_DEFAULT, th=np.pi/2, gammamax = 10.0):
# gammamax = none means that we're in pure FF, nu_parallel = 'FF' means conserve energy in Force-Free case
def u_driftframe(a,r, bfield=BFIELD_DEFAULT, nu_parallel=0, th=np.pi/2, gammamax=None, retbunit = False, retqty = False, eps = -1):
    """drift frame velocity for a given EM field in BL"""

    # get boost from conservation of energy if requested
    if nu_parallel == 'FF':
        ind = np.where(np.nan_to_num(r) != 0)[0][0]  #important for indirect images, which are filled with zeros when there's no crossing
        omega = bfield.omega_field(a,r[ind],th=th[ind])  #single fieldline so single omega

        # stagnation surface for monopole
        if bfield.fieldtype == 'bz_monopole' or (bfield.fieldtype == 'power' and bfield.pval == 0):
            theta0 = th[ind]
            r0 = r0min_mono(theta0, omega, a, 1.0) #1.0 is just the mass scale

        # stagnation surface for paraboloid
        elif bfield.fieldtype == 'bz_para':
            psihere = psiBZpara(r[ind], th[ind], a, shift=bfield.shift) #compute psi of the fieldline chosen
            try:
                r0, theta0 = r0min_para(psihere, omega, a, 1.0, shift=bfield.shift)
            except:
                r0, theta0 = r0min_para(.999999*psihere, omega, a, 1.0, shift=bfield.shift)
        
        elif bfield.fieldtype == 'power':
            psihere = psiBZpower(r[ind], th[ind], bfield.pval) #compute psi of the fieldline chosen
            try:
                r0, theta0 = r0min_power(psihere, omega, a, bfield.pval, 1.0, usemono=bfield.usemono)
            except:
                r0, theta0 = r0min_power(.999999*psihere, omega, a, bfield.pval, 1.0, usemono=bfield.usemono)

        # get the parallel boost
        nu_parallel = getnu_cons(bfield, r, th, r0, theta0, omega, a, 1.0)

    # checks
    nu_parallel = nu_parallel*np.ones_like(r)  #make sure that nu_parallel is an appropriately sized
    if not (isinstance(a,float) and (0 <= np.abs(a) < 1)):
        raise Exception("|a| should be a float in range [0,1)")
    if np.any(np.logical_or(nu_parallel > 1, nu_parallel < -1)):
        raise Exception("nu_parallel should be in the range (-1,1)")
    if not isinstance(r, np.ndarray): r = np.array([r]).flatten()

    # metric
    a2 = a**2
    r2 = r**2
    cth2 = np.cos(th)**2
    sth2 = np.sin(th)**2

    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * cth2
    Xi = (r2 + a2)**2 - Delta*a2*sth2
    omegaz = 2*a*r/Xi
    gdet = Sigma*np.sin(th)

    g00 = -(1-2*r/Sigma)
    g11 = Sigma/Delta
    g22 = Sigma
    g33 = Xi*sth2/Sigma
    g03 = -2*r*a*np.sin(th)**2 / Sigma

    # lapse and shift
    alpha2 = Delta*Sigma/Xi
    alpha = np.sqrt(alpha2)  # lapse
    eta1 = 0
    eta2 = 0
    eta3 = 2*a*r/np.sqrt(Delta*Sigma*Xi)

    # e and b field
    omegaf = bfield.omega_field(a, r, th=th)
    B1, B2, B3 = bfield.bfield_lab(a, r, th=th)
    # (E1,E2,E3) = bfield.efield_lab(a,r,th=th) #unnecessary

    E1 = (omegaf-omegaz)*Xi*np.sin(th)*B2/Sigma
    E2 = -(omegaf-omegaz)*Xi*np.sin(th)*B1/(Sigma*Delta)
    E3 = 0

    Bsq = g11*B1*B1 + g22*B2*B2 + g33*B3*B3
    Esq = g11*E1*E1 + g22*E2*E2 + g33*E3*E3

    B1_cov = g11*B1
    B2_cov = g22*B2
    B3_cov = g33*B3

    E1_cov = g11*E1
    E2_cov = g22*E2
    E3_cov = g33*E3

    # perp velocity in the lnrf, vtilde_perp
    vperp1 = (alpha/(Bsq*gdet))*(E2_cov*B3_cov - B2_cov*E3_cov)
    vperp2 = (alpha/(Bsq*gdet))*(E3_cov*B1_cov - B3_cov*E1_cov)
    vperp3 = (alpha/(Bsq*gdet))*(E1_cov*B2_cov - B1_cov*E2_cov)

    # parallel velocity in the lnrf, vtilde_perp
    vpar_max = np.sqrt(1 - Esq/Bsq)
    vpar1 = nu_parallel*vpar_max*B1/np.sqrt(Bsq)
    vpar2 = nu_parallel*vpar_max*B2/np.sqrt(Bsq)
    vpar3 = nu_parallel*vpar_max*B3/np.sqrt(Bsq)

    # convert to four-velocity
    v1 = vperp1 + vpar1
    v2 = vperp2 + vpar2
    v3 = vperp3 + vpar3

    if retbunit:  #returns gammaperp and raised unit vector along B (useful for FF computations)
        return (alpha, v3, 1/vpar_max, B3/np.sqrt(Bsq))

    vsq = g11*v1*v1 + g22*v2*v2 + g33*v3*v3
    gamma = 1./np.sqrt(1-vsq)

    #approximate MHD gamma by summing gamma_FF and gamma_max in series
    if gammamax:  
        pval0 = 2.0
        gammamax = gammamax*np.ones_like(gamma)
        gammaeff = (1/gammamax**pval0+1/gamma**pval0)**(-1/pval0)
        #argdiv = np.argmin(np.abs(np.nan_to_num(gammaeff, nan=np.inf)))
        if eps >= 0:
            gammaeff0 = (1+eps)*gammaeff#gammaeff*gamma[argdiv]/gammaeff[argdiv] #ensure gamma>1 always

        else:
            argdiv = np.argmin(np.abs(np.nan_to_num(gammaeff, nan=np.inf)))
            gammaeff0 = gammaeff*gamma[argdiv]/gammaeff[argdiv]  #ensure gamma>1 always

        vsqeff = 1-1/gammaeff0**2  # convert
        v1new = v1*np.sqrt(vsqeff/vsq)
        v2new = v2*np.sqrt(vsqeff/vsq)
        v3new = v3*np.sqrt(vsqeff/vsq)

        v1 = v1new
        v2 = v2new
        v3 = v3new
        gamma = np.real(gammaeff0)

    # convert to 4-velocity
    u0 = gamma/alpha
    u1 = gamma*(v1 + eta1)
    u2 = gamma*(v2 + eta2)
    u3 = gamma*(v3 + eta3)

    if retqty:
        Bdotv = g11*v1*B1 + g22*v2*B2 + g33*v3*B3
        v1perp = v1 - B1*Bdotv/Bsq #subtract off vpar to get vperp
        v2perp = v2 - B2*Bdotv/Bsq
        v3perp = v3 - B3*Bdotv/Bsq
        vperpsq = g11*v1perp*v1perp + g22*v2perp*v2perp + g33*v3perp*v3perp
        return (np.sqrt(vperpsq), v1perp, v2perp, v3perp) #returns magnitude of vperp
    
    return (u0, u1, u2, u3)

#now MHD helper functions
def getL(E, sigma, r0, theta, Omegaf, spin, M): #gets L as a function of launch point r0
    if spin==0 and M==0:
        L = (E-np.sqrt(1-r0**2*np.sin(theta)**2*Omegaf**2))/Omegaf
        return L
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
    
    if efac2<0:
        print('Requested launch point is outside of light cylinder!')
        return 0
    
    L = (E-np.sqrt(efac2))/Omegaf
    return L

#solve for conserved quantities as function of more physical parameters
def allcons(gammamax, r0, Omegaf, spin, M, pval, theta0=np.pi/2, useexact = False): #returns quantities for minimum-energy trans-fast wind launched from equator at r0 (in units of M)
    R0 = r0*Omegaf*M
    upmax = np.sqrt(gammamax**2-1) 
    sigmahere = upmax**3/(np.sin(theta0)**2) if pval == 0 else upmax**3/2/psiBZpower(R0,theta0,pval)
    Ehere = gammamax**3
    Lhere = getL(Ehere, sigmahere, r0, theta0, Omegaf, spin, M)
    return sigmahere, Ehere, Lhere

#solves the quartic equation for the alfven point
def alfven(theta, Omegaf, spin, M, L, E): 
    if spin==0 and M==0:
        return np.sqrt(L/E/Omegaf)/np.sin(theta) # in units of 1/Omegaf, the 1/sin(theta) switches from cylindrical to spherical
    coef0 = spin**2*Omegaf*np.cos(theta)**2*(L-spin**2*E*Omegaf*np.sin(theta)**2)
    coef1 = -M/2*Omegaf*(-E*spin+2*L+spin*E*np.cos(2*theta))*(2-spin*Omegaf+spin*Omegaf*np.cos(2*theta))
    coef2 = Omegaf/2*(2*L-spin**2*E*Omegaf*(3+np.cos(2*theta))*np.sin(theta)**2)
    coef3 = 0.0
    coef4 = -E*Omegaf**2*np.sin(theta)**2
    poly = np.polynomial.polynomial.Polynomial([coef0, coef1, coef2, coef3, coef4])
    rroots = poly.roots()
    return rroots[3] #the correct root

#get coefficients of quartic wind equation for u^r
def wind_quartic(E, L, sigma, r, theta, Omegaf, spin, M, p): #r, spin, and Omegaf in units of M
    g = metric(r, spin, theta, M)
    ginv = invmetric(r, spin, theta, M)
    
    gtt = g[0][0]
    gtphi = g[0][3]
    ginvtt = ginv[0][0]
    ginvtphi = ginv[0][3]
    grr = g[1][1]
    gthetatheta = g[2][2]
    gphiphi = g[3][3]
    ginvphiphi = ginv[3][3]

    alpha = gthetatheta/(r**p*Omegaf**(p-2)*sigma) #4pi*eta/B^r
    brat = p/r*(1-np.cos(theta))/np.sin(theta) #btheta/bphi
    beta = ginvtphi*gtt+ginvphiphi*gphiphi*Omegaf
    gamma = gtt+2*gtphi*Omegaf+gphiphi*Omegaf**2
    delta=E-L*Omegaf
    epsilon = gtphi+Omegaf*gphiphi
    zeta = gtt+Omegaf*gtphi
    gfunc = (grr+gthetatheta*brat**2)
    gfunc2 = (gtt+Omegaf*(epsilon+gtphi))

    coef4 = alpha**2*gfunc
    coef3 = 2*alpha*gfunc*gfunc2

    coef2 = gfunc*gfunc2**2+alpha**2*(1+ginvtt*E**2-2*E*L*ginvtphi+ginvphiphi*L**2)
    coef1 = 2*alpha*(gtt+E**2*(ginvtphi*epsilon+ginvtt*zeta)-E*L*(ginvphiphi*epsilon+ginvtt*Omegaf*zeta+ginvtphi*gamma)+Omegaf*(gphiphi*Omegaf+L**2*beta+gtphi*(2+L**2*(ginvphiphi+ginvtphi*Omegaf))))
    coef0 = ginvphiphi*epsilon**2*delta**2+2*ginvtphi*epsilon*zeta*delta**2+ginvtt*zeta**2*delta**2+gfunc2**2
    
    return np.polynomial.polynomial.Polynomial([coef0, coef1, coef2, coef3, coef4])

#solve quartic wind equation for ur
def ursolve(E, L, sigma, r, theta, Omegaf, spin, M, phere, rA = None):
    if rA == None:
        rA = alfven(theta, Omegaf, spin, M, L, E) #Alfven point
    poly = wind_quartic(E, L, sigma, r, theta, Omegaf, spin, M, phere)
    root = poly.roots()[1] if r<rA else poly.roots()[2]
    if np.imag(root) > 1e-10:
        return 0
    return np.real(root)

#get all components of GRMHD four-velocity
def umuMHD(E, L, sigma, r, theta, Omegaf, spin, M, phere, rA = None):
    ur = ursolve(E, L, sigma, r, theta, Omegaf, spin, M, phere, rA = rA)
    eta = np.sign(ur)
    g = metric(r, spin, theta, M) #metric (as a matrix)
    ginv = invmetric(r, spin, theta, M) #inverse metric (as a matrix)
    bthetarat = -phere/r*(1-np.cos(theta))/np.sin(theta)
    utheta = -ur*bthetarat
    xifdotxif = g[0][0]+2*g[0][3]*Omegaf+g[3][3]*Omegaf**2
    num = g[0][0]*L+g[3][3]*Omegaf*E+g[0][3]*(E+Omegaf*L)
    Br = sigma*Omegaf**(pval-2)*r**pval/g[2][2]
    Mp2 = ur/Br
    denom = xifdotxif+Mp2
    Bbar = -num/denom
    ulowert = -(E+Omegaf*Bbar/eta)
    ulowerphi = L+Bbar/eta
    ut = ginv[0][0]*ulowert+ginv[0][3]*ulowerphi 
    uphi = ginv[3][3]*ulowerphi+ginv[0][3]*ulowert 
    normhere = (ulowert*ut+ulowerphi*uphi+ur**2*g[1][1]+utheta**2*g[2][2])
    if np.abs(normhere+1)>1e-8:
        print('problem!', normhere)
    return (ut, ur, utheta, uphi)

def load_cache_from_file(filename):
    """
    Load U123 primitive velocity data from file into cache.
    """
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

    U1 = np.zeros_like(radii)
    U2 = np.zeros_like(radii)
    U3 = np.zeros_like(radii)

    if 'U1' in header:
        U1 = data[:, header.index('U1')]
    if 'U2' in header:
        U2 = data[:, header.index('U2')]
    if 'U3' in header:
        U3 = data[:, header.index('U3')]

    return dict(radii=radii, U1=U1, U2=U2, U3=U3)


def u_from_u123(a, r, ru123_cache):
    """
    Four-velocity as linearly interpolated from input file. By
    convention, one file is for a single spin, which means that
    this function will fail *silently* for other cases.
    """
    # TODO add check for spin?

    # cast input to numpy array
    if not isinstance(r, np.ndarray):
        r = np.array([r]).flatten()

    # get primitives
    u1 = np.interp(r, ru123_cache['radii'], ru123_cache['U1'])
    u2 = np.interp(r, ru123_cache['radii'], ru123_cache['U2'])
    u3 = np.interp(r, ru123_cache['radii'], ru123_cache['U3'])

    # metric (gcov)
    r2 = r*r
    a2 = a*a
    th = np.pi/2.  # TODO equatorial only
    Delta = r2 - 2*r + a2
    Sigma = r2 + a2 * np.cos(th)**2
    g00 = -(1-2*r/Sigma)
    g11 = Sigma/Delta
    g22 = Sigma
    g33 = (r2 + a2 + 2*r*(a*np.sin(th))**2 / Sigma) * np.sin(th)**2
    g03 = -2*r*a*np.sin(th)**2 / Sigma
    # inverse metric (gcon)
    gcon00 = - (r2 + a2 + 2*r*a2 / Sigma * np.sin(th)**2) / Delta
    gcon03 = -2*r*a / Sigma / Delta

    alpha = 1. / np.sqrt(-gcon00)
    gamma = np.sqrt(1. + g11*u1*u1 + g22*u2*u2 + g33*u3*u3)

    u0 = gamma / alpha
    u1 = u1
    u2 = u2
    u3 = u3 - gamma * alpha * gcon03

    return (u0, u1, u2, u3)
    

########################################################################
# computes unique parallel boost parameter in force-free electrodynamics
# previously in ff_boost.py
########################################################################

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
def r0min_para(psi, Omegaf, spin, M, shift=0): #does numerical root find
    bf = Bfield('bz_para', shift=shift)
    def minfunc(R):
        rsphere = rfromR_para(R, psi, spin, shift=shift)
        theta = np.arcsin(R/rsphere)
        return Nderiv(rsphere, theta, spin, Omegaf, M, bf)
    Rguess = 2*rplusfunc(spin, M) #optimal guess
    Rtrue = opt.newton(minfunc, Rguess)
    rtrue = rfromR_para(Rtrue, psi, spin, shift=shift)
    thetatrue = np.arcsin(Rtrue/rtrue)
    return rtrue, thetatrue

#stagnation surface for r^p(1-costheta)
def r0min_power(psi, Omegaf, spin, p, M, usemono=False): 
    bf = Bfield('power', p=p, usemono=usemono)
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


#returns location of outer event horizon
def rplusfunc(spin, M):
    if M==0 or spin == 0:
        return 0
    return M+np.sqrt(M**2-spin**2)

#psi(r,theta) for BZ paraboloid
def psiBZpara(r, theta, a, shift=0): #input should be in terms of M in curved space, or in terms of Omegaf in flat space
    rp = rplusfunc(a, 1.0)
    cth = np.abs(np.cos(theta))
    return (r+shift)*(1-cth)-2*rp*(1-np.log(2))+rp*(1+cth)*(1-np.log(1+cth)) #use rp as the mass scale


#psi(r,theta) for r^p(1-costheta)
def psiBZpower(r, theta, p): #input should be in terms of M in curved space, or in terms of Omegaf in flat space
    return r**p*(1-np.abs(np.cos(theta)))

#invert r(R) for fixed psi
def rfromR_para(R, psi0, a, shift=0):
    def minfunc(Z):
        return psiBZpara(np.sqrt(R**2+Z**2),np.arctan(R/Z),a,shift=shift)-psi0
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
    Zguess = 1.0
    try:
        Zout = opt.newton(minfunc, Zguess)
    except: #failing because close to the equator, so guess 0
        Zguess = (R**2/(2*psi0))**(1/(2-p))
        Zout = opt.newton(minfunc, 0, tol=1e-6)
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
#gets Eco=E-OmegaF*L as a function of launch point r0
def getEco(r0, theta, Omegaf, spin, M): 
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

