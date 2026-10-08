"""Degenerate parameter values: a = 0, 1; th_o = 0, pi/2, pi; lambda = 0; eta = 0; beta = 0.

Known failures are strict xfails tagged with the plan step that should fix them,
so each fix must flip its tests to passing (an unexpected pass fails the suite).
"""

import numpy as np
import pytest

from geodesic_ode import conserved, integrate_geodesic
from helpers import trace, wrap, xfail_until

R_O = 1000.


def _assert_matches_ode(a, th_o, alpha, beta, ngeo=40, angle_tol=1e-10, rtol=1e-10,
                        phi_mod_2pi=False, check_t=False):
    """Analytic r, theta, phi of one ray agree with the ODE oracle, clear of the horizon.
    Generic rays agree to ~1e-11 (see test_geodesic_invariants.py).
    phi_mod_2pi: compare phi modulo 2 pi (for lambda = 0, where +-pi pole jumps are
    equivalent and their sign is a convention).
    check_t: also compare t (relative to |t| + 1), with the same rtol."""
    geo = trace(a, th_o, [alpha], [beta], ngeo=ngeo, r_o=R_O)
    ode = integrate_geodesic(a, R_O, th_o, alpha, beta, geo['tau'][:, 0])
    ok = geo['r'][:, 0] > 1 + np.sqrt(1 - a**2) + 0.2
    np.testing.assert_allclose(geo['r'][ok, 0], ode['r'][ok], rtol=rtol, err_msg="r")
    np.testing.assert_allclose(geo['th'][ok, 0], ode['th'][ok], rtol=0, atol=angle_tol, err_msg="theta")
    dph = geo['ph'][ok, 0] - ode['ph'][ok]
    if phi_mod_2pi:
        dph = wrap(dph)
    np.testing.assert_allclose(dph, 0, rtol=0, atol=angle_tol, err_msg="phi")
    if check_t:
        dt = (geo['t'][ok, 0] - ode['t'][ok])/(np.abs(ode['t'][ok]) + 1)
        np.testing.assert_allclose(dt, 0, rtol=0, atol=rtol, err_msg="t")


@xfail_until("B4", "arcsin(cos th_o/sqrt(u_+)) at argument 1 - O(eps) loses half the digits "
                   "(theta, phi errors 1e-9 to 1e-7)")
@pytest.mark.parametrize("a, th_o, alpha", [
    (0.0, 1.2, 5.0),
    (0.3, 1.2, 5.0),
    (0.94, 0.3, 5.0),
    (0.94, 1.2, -4.0),    # class I
    (0.94, 2.2, 6.0),
])
def test_beta_zero_turning_point(a, th_o, alpha):
    """beta = 0: the observer sits exactly at a theta turning point (the beta = 0 row
    of every image grid)."""
    _assert_matches_ode(a, th_o, alpha, 0.0)


# vortical-boundary ray with eta == 0 exactly in floating point (beta != 0)
_A_VB, _TH_VB, _ALPHA_VB = 0.5, 0.3, 0.2
_BETA_VB = np.sqrt(_A_VB**2 - _ALPHA_VB**2)*np.cos(_TH_VB)


@xfail_until("B3", "eta = 0: u_minus bumped to -EP, theta off by 0.3-1.2 rad, phi by 0.5-0.8 rad")
@pytest.mark.parametrize("a, th_o, alpha, beta, angle_tol", [
    # alpha = +-a, beta = 0. 1e-6 tolerance: these also carry the beta = 0 error (B4)
    (0.5, 0.3, 0.5, 0.0, 1e-6),
    (0.5, 0.3, -0.5, 0.0, 1e-6),
    (0.94, 1.2, 0.94, 0.0, 1e-6),
    # on the boundary of the vortical region, beta != 0
    (_A_VB, _TH_VB, _ALPHA_VB, _BETA_VB, 1e-10),
])
def test_eta_zero(a, th_o, alpha, beta, angle_tol):
    """eta = 0: u_- = 0 is a double root of the theta potential and the ray
    asymptotes to the equator (boundary between ordinary and vortical motion)."""
    lam, eta = conserved(a, th_o, alpha, beta)
    assert eta == 0
    _assert_matches_ode(a, th_o, alpha, beta, angle_tol=angle_tol)


@xfail_until("B2/B3", "u_minus = Delta - sqrt(Delta^2 + eta/a^2) cancels: error ~ eps/|eta| "
                      "(1e-7 at |eta| = 1e-9, up to 2 rad at 1e-12)")
@pytest.mark.parametrize("deta", [1e-9, 1e-12, -1e-12, -1e-9])
@pytest.mark.parametrize("a, th_o, alpha", [(0.5, 0.3, 0.2), (0.94, 2.2, 0.3)])
def test_eta_near_zero(a, th_o, alpha, deta):
    """Rays just inside/outside the vortical region (beta != 0), eta ~ deta."""
    beta = np.sqrt((a**2 - alpha**2)*np.cos(th_o)**2 + deta)
    _assert_matches_ode(a, th_o, alpha, beta)


@xfail_until("B2/B3", "u_minus cancellation near eta = 0 (phi errors up to 2 rad at |eta| = 1e-12)")
@pytest.mark.parametrize("deta", [1e-12, -1e-12])
@pytest.mark.parametrize("a, th_o, sign", [(0.5, 0.3, 1), (0.94, 1.2, 1), (0.94, 2.2, -1)])
def test_eta_near_zero_beta_zero(a, th_o, sign, deta):
    """Approach eta = 0 along the beta = 0 row, through alpha = +-a (the ends of the
    vortical region's boundary). 1e-6 tolerance: these rays also carry the beta = 0
    error (~1e-7, B4); at |eta| = 1e-9 the eta error is below that, so only 1e-12 here.
    r (independent of u_-) is ~1e-10 off near the horizon here, so rtol = 1e-9."""
    alpha = sign*a*np.sqrt(1 + deta/(a*np.cos(th_o))**2)
    _assert_matches_ode(a, th_o, alpha, 0.0, angle_tol=1e-6, rtol=1e-9)


@xfail_until("B4", "lam*G_phi = 0*inf: ellip_pi_arr returns inf for n = u_+ = 1, |phi| > pi/2")
@pytest.mark.parametrize("a, th_o, beta", [
    (0.5, 0.3, 8.0),     # escapes, two pole passages
    (0.5, 0.3, 4.0),     # captured, one pole passage
    (0.0, 1.0, 6.0),     # Schwarzschild
    (0.94, 2.0, -7.0),   # th_o > pi/2
    # vortical: oscillates between theta_- and the pole, crossing it once here
    (0.94, 0.1, 0.3),
    (0.99, 0.1, 0.3),    # no NaN, but phi silently off by pi after the pole
])
def test_lambda_zero(a, th_o, beta):
    """alpha = 0 (lam = 0): the ray stays in the meridional plane and phi jumps by
    pi at each pole passage. Compared with the folded ODE oracle, phi mod 2 pi."""
    _assert_matches_ode(a, th_o, 0.0, beta, phi_mod_2pi=True)


@xfail_until("B4", "lam * Pi(u_+ -> 1; ...) cancels: phi error 4e-7 at alpha = 1e-3, "
                   "3e-3 at 1e-4, 0.13 rad at 1e-5")
@pytest.mark.parametrize("a, th_o, alpha, beta", [
    (0.5, 0.3, 1e-3, 8.0),
    (0.5, 0.3, 1e-4, 8.0),
    (0.5, 0.3, -1e-4, 8.0),
    (0.5, 0.3, 1e-5, 8.0),
    (0.0, 1.0, 1e-4, 6.0),
    (0.94, 2.0, 1e-4, -7.0),
    (0.94, 0.1, 1e-4, 0.3),   # vortical
])
def test_lambda_near_zero(a, th_o, alpha, beta):
    """Small lam != 0: a fast but smooth phi swing of ~pi at each near-pole passage.
    1e-7 tolerance: the ODE's near-pole stiffness limits it to ~1e-8 at alpha = 1e-5."""
    _assert_matches_ode(a, th_o, alpha, beta, angle_tol=1e-7)


@xfail_until("B3", "lam = eta = 0 at a = 0: radial_roots NaN, th_integrate raises")
@pytest.mark.parametrize("th_o", [0.3, 1.0, 2.2])
def test_schwarzschild_radial_ray(th_o):
    """a = 0, screen center: lam = eta = 0, a purely radial ray with R = r^4, Theta = 0.
    dr/dtau = -r^2 gives r = 1/(1/r_o + tau) and tau_tot = 1/2 - 1/r_o; theta = th_o,
    phi = 0; t = -(r*_o - r*) with r* = r + 2 ln(r/2 - 1); sigma = -(r_o - r)."""
    geo = trace(0.0, th_o, [0.], [0.], ngeo=40, r_o=R_O)
    tau, r = geo['tau'][:, 0], geo['r'][:, 0]

    def rstar(x):
        return x + 2*np.log(x/2 - 1)

    np.testing.assert_allclose(geo['tautot'], 1/2 - 1/R_O, rtol=1e-12)
    np.testing.assert_allclose(r, 1/(1/R_O + tau), rtol=1e-12)
    np.testing.assert_allclose(geo['th'][:, 0], th_o, rtol=0, atol=1e-12)
    np.testing.assert_allclose(geo['ph'][:, 0], 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(geo['t'][:, 0], -(rstar(R_O) - rstar(r)), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(geo['sig'][:, 0], -(R_O - r), rtol=1e-10, atol=1e-10)


@xfail_until("B0/B6", "th_o = pi/2 rejected by the input checks")
@pytest.mark.parametrize("a", [0.0, 0.5, 0.94])
@pytest.mark.parametrize("alpha, beta", [(6.0, 2.0), (-5.0, -4.0), (2.0, 3.0), (-1.26, 0.1)])
def test_edge_on_observer(a, alpha, beta):
    """th_o = pi/2 (beta != 0): eta = beta^2 > 0, so no vortical motion, and the
    observer starts on the equator (theta amplitude 0)."""
    _assert_matches_ode(a, np.pi/2, alpha, beta)


@xfail_until("B0/B3/B6", "th_o = pi/2 rejected; with exact trig, beta = 0 gives eta = 0 exactly")
@pytest.mark.parametrize("a", [0.0, 0.5, 0.94])
@pytest.mark.parametrize("alpha", [5.0, -4.0, 0.3])
def test_equatorial_ray(a, alpha):
    """th_o = pi/2, beta = 0: eta = 0 and Theta(pi/2) = Theta'(pi/2) = 0, so the ray
    stays in the equatorial plane (for |lam| < a the equator is an unstable
    equilibrium of the theta motion, but still exact)."""
    geo = trace(a, np.pi/2, [alpha], [0.0], ngeo=40, r_o=R_O)
    np.testing.assert_allclose(geo['th'][:, 0], np.pi/2, rtol=0, atol=1e-14)
    _assert_matches_ode(a, np.pi/2, alpha, 0.0)


def _face_on_expected_phi(alpha, beta, ode):
    """phi(tau) for an observer exactly on the axis, defined as the th_o -> 0+ limit.

    Near the pole, kgeo's backward tau gives tangent-plane velocity (dtheta, theta dphi)
    = (-beta, +alpha), so the ray leaves the pole along azimuth atan2(alpha, -beta).
    The folded ODE (started at theta = 0) supplies frame dragging and the pi jumps of
    later pole passages; for beta >= 0 its initial dtheta/dtau < 0 folds immediately,
    which is not a pole passage and is removed. (Checked against th_o = 1e-3, 1e-4.)"""
    return np.arctan2(alpha, -beta) + ode['ph'] - np.pi*(beta >= 0)


@xfail_until("B5", "th_o = 0 rejected by the input checks (lam = 0 for every pixel)")
@pytest.mark.parametrize("a", [0.0, 0.5, 0.94])
@pytest.mark.parametrize("alpha, beta", [
    (3.0, 4.0), (-2.0, 5.0), (4.0, -6.0), (-5.0, -1.0), (6.0, 0.0),
    (0.5, 0.2),   # vortical at a = 0.94 (rho < a)
])
def test_face_on_observer(a, alpha, beta):
    """th_o = 0: every ray has lam = 0 and starts on the axis. r, theta vs the folded
    ODE; phi vs the th_o -> 0+ limit (mod 2 pi)."""
    geo = trace(a, 0.0, [alpha], [beta], ngeo=40, r_o=R_O)
    ode = integrate_geodesic(a, R_O, 0.0, alpha, beta, geo['tau'][:, 0])
    ok = geo['r'][:, 0] > 1 + np.sqrt(1 - a**2) + 0.2
    np.testing.assert_allclose(geo['r'][ok, 0], ode['r'][ok], rtol=1e-10, err_msg="r")
    np.testing.assert_allclose(geo['th'][ok, 0], ode['th'][ok], rtol=0, atol=1e-10, err_msg="theta")
    dph = wrap(geo['ph'][ok, 0] - _face_on_expected_phi(alpha, beta, ode)[ok])
    np.testing.assert_allclose(dph, 0, rtol=0, atol=1e-10, err_msg="phi")


@xfail_until("B5", "th_o = pi rejected by the input checks")
@pytest.mark.parametrize("a", [0.0, 0.5, 0.94])
@pytest.mark.parametrize("alpha, beta", [(3.0, 4.0), (4.0, -6.0), (6.0, 0.0), (0.5, 0.2)])
def test_south_pole_observer(a, alpha, beta):
    """th_o = pi: r, theta vs the ODE started at the south pole (folded through pi);
    phi from equatorial reflection (pi, alpha, beta) <-> (0, alpha, -beta)."""
    geo = trace(a, np.pi, [alpha], [beta], ngeo=40, r_o=R_O)
    ode = integrate_geodesic(a, R_O, np.pi, alpha, beta, geo['tau'][:, 0])
    ode_north = integrate_geodesic(a, R_O, 0.0, alpha, -beta, geo['tau'][:, 0])
    ok = geo['r'][:, 0] > 1 + np.sqrt(1 - a**2) + 0.2
    np.testing.assert_allclose(geo['r'][ok, 0], ode['r'][ok], rtol=1e-10, err_msg="r")
    np.testing.assert_allclose(geo['th'][ok, 0], ode['th'][ok], rtol=0, atol=1e-10, err_msg="theta")
    np.testing.assert_allclose(np.pi - ode_north['th'][ok], ode['th'][ok], rtol=0, atol=1e-10,
                               err_msg="oracle reflection")
    dph = wrap(geo['ph'][ok, 0] - _face_on_expected_phi(alpha, -beta, ode_north)[ok])
    np.testing.assert_allclose(dph, 0, rtol=0, atol=1e-10, err_msg="phi")


# generic rays for the near-extremal / extremal tests (none on the NHEK line lam = 2)
_RAYS_EXTREMAL = [(6.0, 2.0), (-5.0, -4.0), (2.0, 3.0), (0.4, -1.5), (0.1, 0.1)]


@pytest.mark.parametrize("th_o", [0.3, 1.2])
@pytest.mark.parametrize("alpha, beta", _RAYS_EXTREMAL)
def test_near_extremal(th_o, alpha, beta):
    """a = 1 - 1e-8 (r_+ - r_- ~ 3e-4): currently accurate to ~1e-9 incl. t, phi.
    Regression guard for the a = 1 work (B7)."""
    _assert_matches_ode(1 - 1e-8, th_o, alpha, beta, angle_tol=1e-8, rtol=1e-8, check_t=True)


@xfail_until("B7", "I_phi, I_t divided difference [F(r_+) - F(r_-)]/(r_+ - r_-) cancels "
                   "as r_+ -> r_-: phi, t off by 5e-8..2e-6 (r, theta fine)")
@pytest.mark.parametrize("th_o, alpha, beta", [(0.3, 2.0, 3.0), (1.2, 0.4, -1.5), (0.3, 0.1, 0.1)])
def test_extremal_limit_cancellation(th_o, alpha, beta):
    """a = 1 - 1e-14 (r_+ - r_- ~ 3e-7), captured rays, whose phi and t are dominated
    by the near-horizon I_+- integrals."""
    _assert_matches_ode(1 - 1e-14, th_o, alpha, beta, angle_tol=1e-8, rtol=1e-8, check_t=True)


@xfail_until("B7", "a = 1 rejected by the input checks; I_phi, I_t need the r_+ = r_- limit")
@pytest.mark.parametrize("th_o", [0.3, 1.2])
@pytest.mark.parametrize("alpha, beta", _RAYS_EXTREMAL)
def test_extremal(th_o, alpha, beta):
    """a = 1: r_+ = r_- = 1 and Delta = (r - 1)^2 has a double root."""
    _assert_matches_ode(1.0, th_o, alpha, beta, angle_tol=1e-8, rtol=1e-8, check_t=True)
