
import numpy as np

# Loader for synchrotron power-law fit data, used by a function that is
# not yet re-implemented. Uncomment when that function is restored. The
# __file__-relative path resolves after install regardless of cwd.
# import os
# _synchpl_gxfit_file = os.path.join(os.path.dirname(__file__), 'data', 'synchpl_gxfit.csv')
# synchpl_gxfit = np.loadtxt(_synchpl_gxfit_file, delimiter=',')

# Broken Power Law parameters
P1E_230=-2.0; P2E_230=-0.5 # for  230 GHz
P1E_86=0; P2E_86=-.75  # for 86 GHz

# GLM model parameters
GAMMAOFF = -1.5
SIGMA_GLM = 0.5
R_RING = 4.5
SIGMA_RING = 0.3

# Thermal model parameters
e  = 4.80320425e-10  # statcoul
me = 9.10938356e-28  # g
c  = 2.99792458e10   # cm/s
kB = 1.380649e-16    # erg/K
h  = 6.62607015e-27  # erg*s

# default observation frequency and spectral index
OBSFREQ = 230.e9
SPECIND = 0.0

class Emissivity:
    """ object for rest frame emissivity as a function of r, only in equatorial plane for now (theta=np.pi/2) """
    def __init__(self, emistype="bpl", **kwargs):
        #print("Emissivity emistype =", repr(emistype))

        self.emistype = emistype
        self.kwargs = kwargs

        self.emiscut_in = self.kwargs.get('emiscut_in', 0)
        self.emiscut_out = self.kwargs.get('emiscut_out', 1.e10)

        if self.emistype=='thermal':
            self.Rb = float(self.kwargs.get('Rb', 5.0))
            self.ne0 = float(self.kwargs['ne0'])
            self.Te0 = float(self.kwargs['Te0'])
            self.B0 = float(self.kwargs['B0'])
            self.alpha_n = float(self.kwargs.get('alpha_n', 0.7))
            self.alpha_T = float(self.kwargs.get('alpha_T', 1.0))
            self.alpha_B = float(self.kwargs.get('alpha_B', 1.5))
            self.bfield_model = bool(self.kwargs.get('bfield_model', False))

        # thermal synchrotron with tabulated radial profiles n_e(r), T_e(r)
        elif self.emistype=='thermal_file':
            self.filename = self.kwargs.get('file', None)
            if self.filename is None:
                raise Exception("thermal_file Emissivity needs a 'file' keyword argument!")
            # temperature column units: 'K' (Kelvin, default) or 'dimensionless' (Theta_e = kT/me c^2)
            self.temp_kind = self.kwargs.get('temp_kind', 'K')
            # multiplicative factor applied to the tabulated temperature
            # (e.g. Te_scale=0.1 for an electron temperature 10x below tabulated T)
            self.Te_scale = float(self.kwargs.get('Te_scale', 1.0))
            self.cached_data = load_thermal_profile_file(self.filename)
            self.cached_data['_surface_metric'] = bool(self.kwargs.get('surface_metric', False))

        elif self.emistype=='bpl':
            self.p1 = self.kwargs.get('p1', P1E_230)
            self.p2 = self.kwargs.get('p2', P2E_230)
        elif self.emistype =='ring':
            self.mu_ring = self.kwargs.get('r_ring', R_RING)
            self.gamma_off = self.kwargs.get('gamma_off', 0)
            self.sigma = self.kwargs.get('sigma', SIGMA_RING)
        elif self.emistype=='glm':
            self.mu_ring = False
            self.gamma_off = self.kwargs.get('gamma_off', GAMMAOFF)
            self.sigma = self.kwargs.get('sigma', SIGMA_GLM)
        elif self.emistype == 'constant':
            pass
        else:
            raise Exception(f"emistype {self.emistype} not recognized in Emissivity!")


    # power law disk profiles to define thermal variables
    def profiles_plaw(self, r):
        x = np.asarray(r, dtype=float)/self.Rb
        ne = self.ne0 * np.power(x, -self.alpha_n) # cm^-3
        Te = self.Te0 * np.power(x, -self.alpha_T) # K
        B  = self.B0  * np.power(x, -self.alpha_B) # Gauss
        return ne, Te, B

    def jrest(self, a, r, g=None, sinthetab=None, Bmag=None, nu_obs=OBSFREQ, specind=SPECIND):

        ## Physical models
        if self.emistype=='thermal':
            nu_em = np.asarray(nu_obs) / np.asarray(g)     # emitted frequency
            ne, Te, B = self.profiles_plaw(r)              # local plasma properties

            # option to overwrite default power law field strength with actual |B| from model
            if self.bfield_model:
                B = np.asarray(Bmag, dtype=float)
                # TODO: change to handle normalization inside emissivities
                #B = (Bmag / self.Rb) * self.B0

            j = j_nu_thermal(ne, B, Te, nu_em, sinthetab)

        elif self.emistype=='thermal_file':
            nu_em = np.asarray(nu_obs) / np.asarray(g)     # emitted frequency in fluid frame
            cache = self.cached_data
            ne = np.interp(r, cache['radii'], cache['ne'], left=0., right=0.)  # cm^-3
            Te = np.interp(r, cache['radii'], cache['Te'], left=0., right=0.)
            Te = Te * self.Te_scale
            if self.temp_kind == 'dimensionless':
                Te = Te * (me*c*c/kB)                      # Theta_e -> Kelvin
            # B in Gauss: computed directly from the tabulated fluid-frame b^mu,
            # |b| = sqrt(b_mu b^mu), evaluated with the BL metric ON THE DATA GRID
            # (interpolation-free), then interpolated in radius. Falls back to the
            # pipeline-provided Bmag if the file has no b^mu columns.
            if cache.get('b0', None) is not None:
                if cache.get('_Bmag_grid', None) is None:
                    rr = cache['radii']
                    # linear order in H/r: midplane metric (theta corrections O((H/r)^2))
                    if cache.get('hor', None) is not None and cache.get('_surface_metric', False):
                        cthg = np.clip(cache['hor'], 0., 1.); sth2g = 1. - cthg**2
                    else:
                        cthg = np.zeros_like(rr); sth2g = np.ones_like(rr)
                    a2g, r2g = a**2, rr**2
                    Deltag = r2g - 2.*rr + a2g
                    Sigmag = r2g + a2g*cthg**2
                    g00g = -(1. - 2.*rr/Sigmag)
                    g11g = Sigmag/Deltag
                    g33g = (r2g + a2g + 2.*rr*a2g*sth2g/Sigmag)*sth2g
                    g03g = -2.*rr*a*sth2g/Sigmag
                    bb0, bbr, bbp = cache['b0'], cache['br'], cache['bph']
                    b_t  = g00g*bb0 + g03g*bbp
                    b_r  = g11g*bbr
                    b_ph = g33g*bbp + g03g*bb0
                    bsqg = b_t*bb0 + b_r*bbr + b_ph*bbp     # Gauss^2, spacelike > 0
                    cache['_Bmag_grid'] = np.sqrt(np.clip(bsqg, 0., None))
                B = np.interp(r, cache['radii'], cache['_Bmag_grid'], left=0., right=0.)
            else:
                B = np.asarray(Bmag, dtype=float)
            # guard: no emission where tabulated values are nonphysical (e.g. Te<=0)
            good = (ne > 0) & (Te > 0) & np.isfinite(B) & (B > 0)
            j = np.zeros_like(np.asarray(r, dtype=float))
            if np.any(good):
                j[good] = j_nu_thermal(ne[good], B[good], Te[good],
                                       np.broadcast_to(nu_em, j.shape)[good],
                                       np.broadcast_to(sinthetab, j.shape)[good])

        ## Phenomenological models
        elif self.emistype=='constant':
            j = np.ones(r.shape)

        elif self.emistype=='bpl':
            j = emisBPL(a, r, p1=self.p1, p2=self.p2)

        elif self.emistype=='glm' or self.emistype=='ring':
            j = emisGLM(a, r, gamma_off=self.gamma_off, sigma=self.sigma, mu_ring=self.mu_ring)

        else:
            raise Exception(f"emistype {self.emistype} not recognized in Emissivity.emis!")

        # add spectral behavior for non-physical emissivities
        if self.emistype not in ('thermal', 'thermal_file'):
            if g is not None:
                j *=  (g**specind)
            if sinthetab is not None:
                j *= (sinthetab**(1+specind))

        return j


def emisBPL(a, r, p1=P1E_230, p2=P2E_230):
    """emissivity at radius r - broken power law model fit to GRMHD"""

    rh = 1 + np.sqrt(1-a**2)
    emis = np.exp(p1*np.log(r/rh) + p2*np.log(r/rh)**2)

    return emis


def emisGLM(a, r, gamma_off=GAMMAOFF, sigma=SIGMA_GLM, mu_ring=False):
    """emissivity at radius r from GLM paper"""

    if mu_ring:
        mu = mu_ring
    else:
        mu = 1 - np.sqrt(1-a**2)
    emis = np.exp(-0.5*(gamma_off+np.arcsinh((r-mu)/sigma))**2) / np.sqrt((r-mu)**2 + sigma**2)
    return emis

# Thermal model function definitions
def II_fit(x):
    return 2.5651 * (1.0 + 1.92*x**(-1.0/3.0) + 0.9977*x**(-2.0/3.0)) * np.exp(-1.8899 * x**(1.0/3.0))

# critical freqneucy (in fluid frame)
def nu_c_fcn(B, Theta_e, sin_thetaB):
    nu_B = e*B/(2.0*np.pi*me*c)
    nu_c = 1.5 * nu_B * Theta_e**2 * sin_thetaB
    return nu_c
    #return (3.0/(4.0*np.pi)) * (e*B*Theta_e**2)/(me*c) * sin_thetaB

# Emissivity j_nu (fluid frame, per unit volume, freq, steradians)
# cgs units: erg s^-1 cm^-3 Hz^-1 sr^-1
def j_nu_thermal(ne, B, Te, nu, sin_thetaB):
    Theta_e = kB*Te/(me*c*c)       # strictly valid only in Theta_e > 3 limit
    nu_c = nu_c_fcn(B, Theta_e, sin_thetaB)
    x = nu/np.maximum(nu_c, 1e-40) # strictly valid only in nu > nu_c limit
    return (ne * e**2 * nu)/(2 * np.sqrt(3) * c * (Theta_e**2)) * II_fit(x)


def load_thermal_profile_file(filename):
    """
    Load tabulated radial plasma profiles for the thermal_file emissivity from
    a plain text file (whitespace- or comma-separated, no header needed).

    Expected columns, either:
        3 columns:   r, n_e, T_e
        9+ columns:  r, b^0, b^r, b^phi, u^0, u^r, u^phi, n_e, T_e
                     (combined model file; columns 8-9 are used; the same file
                     can be passed to Bfield/Velocity 'fluid_file')
        r   : Boyer-Lindquist radius in units of M
        n_e : electron number density in cm^-3 (e.g. rho/(mu_e*m_p))
        T_e : electron temperature, in K by default
              (pass temp_kind='dimensionless' if the column is Theta_e = kT/me c^2)
    """
    try:
        data = np.loadtxt(filename)
    except ValueError:
        data = np.loadtxt(filename, delimiter=',')

    if data.ndim != 2 or data.shape[1] < 3:
        raise Exception(f"thermal profile file {filename} must have >=3 columns: r, n_e, T_e")

    radii = data[:, 0]
    if np.any(np.diff(radii) <= 0):
        idx = np.argsort(radii)
        data = data[idx]
        radii = data[:, 0]

    if data.shape[1] >= 9:
        # combined model file: also keep b^mu to compute |b| = sqrt(b_mu b^mu)
        # directly from the data (used for the synchrotron field strength)
        out = dict(radii=radii, ne=data[:, 7], Te=data[:, 8],
                   b0=data[:, 1], br=data[:, 2], bph=data[:, 3])
        if data.shape[1] >= 10:
            out['hor'] = data[:, 9]        # scale height H/r
        if data.shape[1] >= 11:
            out['beta'] = data[:, 10]      # plasma beta (loaded, available)
        return out
    else:
        return dict(radii=radii, ne=data[:, 1], Te=data[:, 2])
