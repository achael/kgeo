import numpy as np
from kgeo.bfields import Bfield
from kgeo.ff_boost import getEco,metric,psiBZpara,psiBZpower,r0min_mono,r0min_para,r0min_power

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
    'general', 'gelles', 'simfit', 'fromfile', 'fromdict', 'driftframe'
]


class Velocity(object):
    """ object for lab frame velocity as a function of r, only in equatorial plane for now """

    def __init__(self, veltype="kep", **kwargs):

        self.veltype = veltype
        self.kwargs = kwargs

        if self.veltype in ['zamo', 'infall']:
            pass
        elif self.veltype in ['kep', 'cunningham']:
            self.retrograde = self.kwargs.get('retrograde', False)
        elif self.veltype in ['subkep', 'cunningham_subkep']:
            self.retrograde = self.kwargs.get('retrograde', False)
            self.fac_subkep = self.kwargs.get('fac_subkep', 1)
        elif self.veltype == 'general':
            self.retrograde = self.kwargs.get('retrograde', False)
            self.fac_subkep = self.kwargs.get('fac_subkep', 1)
            self.beta_phi = self.kwargs.get('beta_phi', 1)
            self.beta_r = self.kwargs.get('beta_r',1)
        elif self.veltype == 'gelles':
            self.gelles_beta = self.kwargs.get('gelles_beta', BETA)
            self.gelles_chi = self.kwargs.get('gelles_chi', CHI)
        elif self.veltype == 'simfit':
            self.ell_isco = self.kwargs.get('ell_isco', ELLISCO)
            self.vr_isco = self.kwargs.get('vr_isco', VRISCO)
            self.p1 = self.kwargs.get('p1', P1)
            self.p2 = self.kwargs.get('p2', P2)
            self.dd = self.kwargs.get('dd', DD)
        elif self.veltype == 'fromfile':
            self.filename = self.kwargs.get('file', None)
            self.cached_data = load_cache_from_file(self.filename)
        elif self.veltype == 'fromdict':
            self.cached_data = self.kwargs.get('data', None)
        elif self.veltype == 'driftframe':
            self.bfield = self.kwargs.get('bfield', BFIELD_DEFAULT)
            self.nu_parallel = self.kwargs.get('nu_parallel', 0)
            self.gammamax = self.kwargs.get('gammamax', None)
        else:
            raise Exception("veltype %s not recognized in Velocity!"%self.veltype)

    def u_lab(self, a, r, th=np.pi/2., retqty=False):
        """Return lab frame contravarient velocity vector"""
        if self.veltype == 'zamo':
            ucon = u_zamo(a, r)
        elif self.veltype == 'infall':
            ucon = u_infall(a, r)
        elif self.veltype in ['kep', 'cunningham']:
            ucon = u_kep(a, r, retrograde=self.retrograde)
        elif self.veltype in ['subkep', 'cunningham_subkep']:
            ucon = u_subkep(a, r, retrograde=self.retrograde, fac_subkep=self.fac_subkep)
        elif self.veltype == 'general':
            ucon = u_general(a, r, retrograde=self.retrograde, fac_subkep=self.fac_subkep,
                             beta_phi=self.beta_phi, beta_r=self.beta_r)
        elif self.veltype == 'gelles':
            ucon = u_gelles(a, r, beta=self.gelles_beta, chi=self.gelles_chi)
        elif self.veltype == 'simfit':
            ucon = u_grmhd_fit(a, r, ell_isco=self.ell_isco, vr_isco=self.vr_isco, p1=self.p1, p2=self.p2, dd=self.dd)
        elif self.veltype in ['fromfile', 'fromdict']:
            ucon = u_from_u123(a, r, self.cached_data)
        elif self.veltype == 'driftframe':
            ucon = u_driftframe(a, r, bfield=self.bfield, nu_parallel=self.nu_parallel, th=th,
                                gammamax=self.gammamax, retqty=retqty)
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
            psihere = psiBZpara(r[ind], th[ind], a) #compute psi of the fieldline chosen
            try:
                r0, theta0 = r0min_para(psihere, omega, a, 1.0)
            except:
                r0, theta0 = r0min_para(.999999*psihere, omega, a, 1.0)

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

    if gammamax:  #approximate MHD gamma by summing gamma_FF and gamma_max in series
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

    u0 = gamma/alpha
    u1 = gamma*(v1 + eta1)
    u2 = gamma*(v2 + eta2)
    u3 = gamma*(v3 + eta3)

    if retqty:
        return (gamma, g11*v1, g22*v2, g33*v3)

    return (u0, u1, u2, u3)


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
