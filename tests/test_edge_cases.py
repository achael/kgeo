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
                        phi_mod_2pi=False):
    """Analytic r, theta, phi of one ray agree with the ODE oracle, clear of the horizon.
    Generic rays agree to ~1e-11 (see test_geodesic_invariants.py).
    phi_mod_2pi: compare phi modulo 2 pi (for lambda = 0, where +-pi pole jumps are
    equivalent and their sign is a convention)."""
    geo = trace(a, th_o, [alpha], [beta], ngeo=ngeo, r_o=R_O)
    ode = integrate_geodesic(a, R_O, th_o, alpha, beta, geo['tau'][:, 0])
    ok = geo['r'][:, 0] > 1 + np.sqrt(1 - a**2) + 0.2
    np.testing.assert_allclose(geo['r'][ok, 0], ode['r'][ok], rtol=rtol, err_msg="r")
    np.testing.assert_allclose(geo['th'][ok, 0], ode['th'][ok], rtol=0, atol=angle_tol, err_msg="theta")
    dph = geo['ph'][ok, 0] - ode['ph'][ok]
    if phi_mod_2pi:
        dph = wrap(dph)
    np.testing.assert_allclose(dph, 0, rtol=0, atol=angle_tol, err_msg="phi")


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
