# Generate a black hole image from a tabulated semi-analytic model.
#
# Model file: ONE plain-text file (tab/space/comma separated, no header) with
# 10 columns:
#   r, b^0[G], b^r[G], b^phi[G], u^0, u^r, u^phi, n_e[cm^-3], T_e[K], H/r
# (9-column files without H/r are also accepted; a constant opening angle is
#  then used for the emitting column)
# where r is the Boyer-Lindquist radius in units of M, b^mu is the fluid-frame
# magnetic field four-vector (b^theta = 0), u^mu is the contravariant BL
# four-velocity (u^theta = 0, normalized u_mu u^mu = -1, G=c=1).
#
# Requires the 'fluid_file' Bfield/Velocity types and the 'thermal_file'
# Emissivity type added to kgeo-samik.

import numpy as np
import matplotlib.pyplot as plt
from kgeo.equatorial_images import make_image
from kgeo.bfields import Bfield
from kgeo.velocities import Velocity
from kgeo.emissivities import Emissivity

# ============================ USER PARAMETERS ============================
datafile = '../../check_case_shockfree.dat'      # <-- path to YOUR 9-column model file
spin     = 0.94                  # BH spin; MUST match the model file
th_o_deg = 0.1#163.0 #30.0                   # observer inclination in degrees (Sgr A*: low)
nu_obs   = 230.e9                # observing frequency [Hz]

MBH_msun = 6.5e9 #4.2e6                 # Sgr A* mass in solar masses (your value)
D_kpc    = 16.8e3  #8.127                 # distance to Sgr A* [kpc] (GRAVITY 2018)

npix     = 1024                   # image pixels per side (512+ for production)
amax     = 15.                   # half field of view in units of M
nmax     = 2                     # include photon subrings up to n=2
Te_scale = 3.4 #3.675(shock)                   # electron temperature = Te_scale * tabulated T
outfile  = 'Midplane_M87_check_case_No_shock_Te=Ti_3'   # output name stem
# =========================================================================

# ---- physical scales set by the BH mass ----
G_cgs  = 6.67430e-8
c_cgs  = 2.99792458e10
Msun_g = 1.98892e33
kpc_cm = 3.0857e21

rg_cm    = G_cgs * MBH_msun * Msun_g / c_cgs**2        # gravitational radius [cm]
D_cm     = D_kpc * kpc_cm
thetag   = rg_cm / D_cm                                # angular size of 1 M [rad]
thetag_uas = thetag * 180./np.pi * 3600. * 1.e6
print(f"r_g = {rg_cm:.3e} cm   |  theta_g = {thetag_uas:.3f} micro-arcsec")
print(f"field of view = {2*amax*thetag_uas:.1f} micro-arcsec")

# ---- model objects: everything from your one file ----
th_o     = th_o_deg * np.pi/180.
if th_o_deg == 90.:
    raise Exception("exactly edge-on (90 deg) is not supported; use e.g. 89.")

# ---- data hygiene: drop tabulated points too close to the horizon, where
# ---- u^0 diverges and finite-precision data cannot stay normalized ----
rh = 1. + np.sqrt(1. - spin**2)
_d = np.loadtxt(datafile)
_keep = _d[:, 0] > rh + 5.e-3          # keep r > r_horizon + 0.005 M
if (~_keep).sum() > 0:
    print(f"data hygiene: dropping {(~_keep).sum()} point(s) within 0.005 M of the horizon")
    datafile_clean = datafile + '.clean'
    np.savetxt(datafile_clean, _d[_keep])
else:
    datafile_clean = datafile

# linear order in H/r: vectors and metric at the midplane; the tabulated
# scale height enters only the emitting column (pathlength), where it is linear
if _d.shape[1] >= 10:
    _rgrid, _hor = _d[_keep][:, 0], _d[_keep][:, 9]
    diskangle = lambda r: np.interp(r, _rgrid, 2.*_hor, left=2.*_hor[0], right=2.*_hor[-1])
    print(f"using tabulated H/r for the emitting column: 2H/r in [{2*_hor.min():.3f}, {2*_hor.max():.3f}]")
else:
    diskangle = 0.1   # kgeo default full opening angle

velocity   = Velocity('fluid_file', file=datafile_clean)
bfield     = Bfield('fluid_file', file=datafile_clean, velocity=velocity)
emissivity = Emissivity('thermal_file', file=datafile_clean, Te_scale=Te_scale)   # n_e [cm^-3], T_e [K]

# ---- raytrace and build the image ----
psize = 2.*amax/npix
out = make_image(spin, np.inf, th_o, nmax,
                 -amax, amax, -amax, amax, psize,
                 emissivity=emissivity, velocity=velocity, bfield=bfield,
                 polarization=True, pathlength=True,
                 specind=1, nu_obs=nu_obs, diskangle=diskangle)

n = int(np.sqrt(out[0].shape[0]))
I = np.nan_to_num(out[0]).sum(axis=1).reshape(n, n)   # erg s^-1 cm^-3 Hz^-1 sr^-1 * (path in M)
Q = np.nan_to_num(out[1]).sum(axis=1).reshape(n, n)
U = np.nan_to_num(out[2]).sum(axis=1).reshape(n, n)

# ---- convert to physical units using the BH mass ----
# pathlength from kgeo is in units of M -> multiply by rg_cm for cgs intensity
I_cgs = I * rg_cm                                     # erg s^-1 cm^-2 Hz^-1 sr^-1
Q_cgs = Q * rg_cm
U_cgs = U * rg_cm

pix_sr   = (psize*thetag)**2                          # pixel solid angle [sr]
Ftot_Jy  = I_cgs.sum() * pix_sr * 1.e23               # total flux density [Jy]
#print(f"total flux at {nu_obs/1e9:.0f} GHz: {Ftot_Jy:.4f} Jy")
#print(f"(Sgr A* observed ~2.4 Jy at 230 GHz -- use this to calibrate n_e normalization)")

print(f"total flux at {nu_obs/1e9:.0f} GHz: {Ftot_Jy:.6e} Jy")
#ax[0].set_title(f'Stokes I, {nu_obs/1e9:.0f} GHz  ({Ftot_Jy:.3e} Jy)')

# ---- plots ----
ext_uas = amax*thetag_uas
extent  = [-ext_uas, ext_uas, -ext_uas, ext_uas]

fig, ax = plt.subplots(1, 3, figsize=(15.5, 5))
ax[0].imshow(I_cgs, origin='lower', extent=extent, cmap='afmhot')
ax[0].set_title(f'Stokes I, {nu_obs/1e9:.0f} GHz  ({Ftot_Jy:.3e} Jy)')
ax[0].set_xlabel(r'$\alpha$ [$\mu$as]'); ax[0].set_ylabel(r'$\beta$ [$\mu$as]')

P = np.sqrt(Q_cgs**2 + U_cgs**2)
ax[1].imshow(P, origin='lower', extent=extent, cmap='viridis')
ax[1].set_title('linear polarization $|P|$')
ax[1].set_xlabel(r'$\alpha$ [$\mu$as]')

# EVPA ticks over intensity
ax[2].imshow(I_cgs, origin='lower', extent=extent, cmap='gray')
chi = 0.5*np.arctan2(U_cgs, Q_cgs)
xs  = np.linspace(-ext_uas, ext_uas, n); X, Y = np.meshgrid(xs, xs)
s   = max(n//22, 1)
m   = np.where(P > 0.05*P.max(), 1., np.nan)
ax[2].quiver(X[::s,::s], Y[::s,::s],
             (-np.sin(chi)*P/P.max()*m)[::s,::s],
             ( np.cos(chi)*P/P.max()*m)[::s,::s],
             color='cyan', scale=18, headwidth=1, headlength=0,
             headaxislength=0, pivot='mid')
ax[2].set_title('EVPA')
ax[2].set_xlabel(r'$\alpha$ [$\mu$as]')

plt.tight_layout()
plt.savefig(f'{outfile}.png', dpi=150, bbox_inches='tight')
print(f"saved {outfile}.png")
#plt.show()
