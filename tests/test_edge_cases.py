"""Degenerate parameter values: a = 0, 1; th_o = 0, pi/2, pi; lambda = 0; eta = 0; beta = 0.

Known failures are strict xfails tagged with the plan step that should fix them,
so each fix must flip its tests to passing (an unexpected pass fails the suite).
"""

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad

from geodesic_ode import conserved, integrate_geodesic, radial_potential
from helpers import trace, wrap, xfail_until

R_O = 1000.

# Tolerances, with the worst error measured in [brackets] (2026-10-10, after B4 1/3);
# [target] where only strict xfails use them
ODE_RTOL = 1e-10              # r vs the ODE oracle                                        [4e-12]
ODE_ANGLE_ATOL = 1e-10        # theta, phi vs the ODE oracle                               [7e-12]
NEAR_EXTREMAL_TOL = 1e-8      # a = 1 - 1e-8: r, theta, phi, t (rel. to |t| + 1)           [1e-9]
                              # (a = 1, a = 1 - 1e-14: [target, B7])
SMALL_LAM_ANGLE_ATOL = 1e-7   # alpha ~ 1e-5: the ODE's near-pole stiffness gives ~1e-8    [target, B4]
RADIAL_RAY_TOL = 1e-12        # a = 0 radial ray vs closed form: tau_tot, r, theta, phi    [0]
RADIAL_RAY_T_TOL = 1e-10      # a = 0 radial ray: t, sigma (rtol and atol)                 [1e-13]
CONST_TH_ANGLE_ATOL = 1e-7    # constant-theta rays: u_+, u_- differ by ~1e-8 roundoff (B9) [5e-8]
CONST_TH_RTOL = 1e-11         # constant-theta rays: tau(r), t vs quadrature               [5e-12]
EQUATORIAL_RAY_ATOL = 1e-14   # th_o = pi/2, beta = 0: theta = pi/2                        [target, B6]
DISTANT_T_ATOL = 1e-6         # r_o = 1e7: t, sigma vs 30-digit quadrature                 [target, B8]


def _assert_matches_ode(a, th_o, alpha, beta, ngeo=40, angle_tol=ODE_ANGLE_ATOL, rtol=ODE_RTOL,
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


@pytest.mark.parametrize("a, th_o, alpha", [
    (0.0, 1.2, 5.0),
    (0.3, 1.2, 5.0),
    (0.94, 0.3, 5.0),
    (0.94, 1.2, -4.0),    # class I
    (0.94, 2.2, 6.0),
    # vortical: at u_- for |alpha| < a sin(th_o), at u_+ for |alpha| > a sin(th_o)
    (0.94, 0.3, 0.1),
    (0.94, 0.3, -0.6),
    (0.94, 2.2, 0.3),
    (0.94, 2.2, 0.8),
])
def test_beta_zero_turning_point(a, th_o, alpha):
    """beta = 0: the observer sits exactly at a theta turning point (the beta = 0 row
    of every image grid). On this row the roots of the theta potential in u = cos^2(theta)
    are cos^2(th_o) and 1 - alpha^2/a^2, so vortical rays start at u_- or u_+.
    (Before B4, arcsin(1 - O(eps)) in the observer amplitude gave theta, phi errors ~1e-8.)"""
    _assert_matches_ode(a, th_o, alpha, 0.0)


# vortical-boundary ray with eta == 0 exactly in floating point (beta != 0)
_A_VB, _TH_VB, _ALPHA_VB = 0.5, 0.3, 0.2
_BETA_VB = np.sqrt(_A_VB**2 - _ALPHA_VB**2)*np.cos(_TH_VB)


@pytest.mark.parametrize("a, th_o, alpha, beta", [
    # alpha = +-a, beta = 0
    (0.5, 0.3, 0.5, 0.0),
    (0.5, 0.3, -0.5, 0.0),
    (0.94, 1.2, 0.94, 0.0),
    # on the boundary of the vortical region, beta != 0; both signs of beta, and the southern reflection
    (_A_VB, _TH_VB, _ALPHA_VB, _BETA_VB),
    (_A_VB, _TH_VB, _ALPHA_VB, -_BETA_VB),
    (_A_VB, np.pi - _TH_VB, _ALPHA_VB, -_BETA_VB),
])
def test_eta_zero(a, th_o, alpha, beta):
    """eta = 0: u_- = 0 is a double root of the theta potential and the ray
    asymptotes to the equator (boundary between ordinary and vortical motion)."""
    lam, eta = conserved(a, th_o, alpha, beta)
    assert eta == 0
    _assert_matches_ode(a, th_o, alpha, beta)


@pytest.mark.parametrize("deta", [1e-9, 1e-12, 1e-14, -1e-14, -1e-12, -1e-9])
@pytest.mark.parametrize("a, th_o, alpha", [(0.5, 0.3, 0.2), (0.94, 2.2, 0.3)])
def test_eta_near_zero(a, th_o, alpha, deta):
    """Rays just inside/outside the vortical region (beta != 0), eta ~ deta."""
    beta = np.sqrt((a**2 - alpha**2)*np.cos(th_o)**2 + deta)
    _assert_matches_ode(a, th_o, alpha, beta)


@pytest.mark.parametrize("deta", [1e-12, -1e-12])
@pytest.mark.parametrize("a, th_o, sign", [(0.5, 0.3, 1), (0.94, 1.2, 1), (0.94, 2.2, -1)])
def test_eta_near_zero_beta_zero(a, th_o, sign, deta):
    """Approach eta = 0 along the beta = 0 row, through alpha = +-a (the ends of the
    vortical region's boundary)."""
    alpha = sign*a*np.sqrt(1 + deta/(a*np.cos(th_o))**2)
    _assert_matches_ode(a, th_o, alpha, 0.0)


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
    """Small lam != 0: a fast but smooth phi swing of ~pi at each near-pole passage."""
    _assert_matches_ode(a, th_o, alpha, beta, angle_tol=SMALL_LAM_ANGLE_ATOL)


@pytest.mark.parametrize("th_o", [0.3, 1.0, 2.2])
def test_schwarzschild_radial_ray(th_o):
    """a = 0, screen center: lam = eta = 0, a purely radial ray with R = r^4, Theta = 0.
    dr/dtau = -r^2 gives r = 1/(1/r_o + tau) and tau_tot = 1/2 - 1/r_o; theta = th_o,
    phi = 0; t = -(r*_o - r*) with r* = r + 2 ln(r/2 - 1); sigma = -(r_o - r)."""
    geo = trace(0.0, th_o, [0.], [0.], ngeo=40, r_o=R_O)
    tau, r = geo['tau'][:, 0], geo['r'][:, 0]

    def rstar(x):
        return x + 2*np.log(x/2 - 1)

    np.testing.assert_allclose(geo['tautot'], 1/2 - 1/R_O, rtol=RADIAL_RAY_TOL)
    np.testing.assert_allclose(r, 1/(1/R_O + tau), rtol=RADIAL_RAY_TOL)
    np.testing.assert_allclose(geo['th'][:, 0], th_o, rtol=0, atol=RADIAL_RAY_TOL)
    np.testing.assert_allclose(geo['ph'][:, 0], 0, rtol=0, atol=RADIAL_RAY_TOL)
    np.testing.assert_allclose(geo['t'][:, 0], -(rstar(R_O) - rstar(r)), rtol=RADIAL_RAY_T_TOL, atol=RADIAL_RAY_T_TOL)
    np.testing.assert_allclose(geo['sig'][:, 0], -(R_O - r), rtol=RADIAL_RAY_T_TOL, atol=RADIAL_RAY_T_TOL)


def _assert_constant_theta_matches_quad(a, th_o, sign, angle_tol=CONST_TH_ANGLE_ATOL, rtol=CONST_TH_RTOL):
    """beta = 0, alpha = -sign a sin(th_o): u_+ = u_- = cos^2(th_o), so theta = th_o (GL19a 24).
    Compared with quadrature in r of tau, I_phi, I_t (GL19a 13), with G_phi = -tau/sin^2(th_o)
    and G_t = -tau cos^2(th_o) (kgeo's tau runs backward from the observer)."""
    s, c = np.sin(th_o), np.cos(th_o)
    geo = trace(a, th_o, [-sign*a*s], [0.], ngeo=12, r_o=R_O)
    lam, eta = geo['lam'][0], geo['eta'][0]

    def q(f, r):
        return quad(f, r, R_O, epsabs=0, epsrel=1e-13, limit=200)[0]

    def R(r):
        return radial_potential(r, a, lam, eta)

    def D(r):
        return r*r - 2*r + a*a

    np.testing.assert_allclose(geo['th'][:, 0], th_o, rtol=0, atol=angle_tol, err_msg="theta")
    for rs, tau, ph, t in zip(geo['r'][1:, 0], geo['tau'][1:, 0], geo['ph'][1:, 0], geo['t'][1:, 0]):
        ph_ex = -q(lambda r: a*(2*r - a*lam)/(D(r)*np.sqrt(R(r))), rs) - lam*tau/s**2
        t_ex = -q(lambda r: (r*r*D(r) + 2*r*(r*r + a*a - a*lam))/(D(r)*np.sqrt(R(r))), rs) - a*a*c*c*tau
        np.testing.assert_allclose(q(lambda r: 1/np.sqrt(R(r)), rs), tau, rtol=rtol, err_msg="tau(r)")
        np.testing.assert_allclose(ph, ph_ex, rtol=0, atol=angle_tol, err_msg="phi")
        np.testing.assert_allclose(t, t_ex, rtol=rtol, err_msg="t")


@pytest.mark.parametrize("a, th_o, sign", [
    # principal null congruence with zeta = 0 exactly in floating point (so z = 0 in radial_roots)
    (0.1, 2.2, 1), (0.5, 0.3, 1), (0.7, 2.2, 1), (0.94, 1.0, 1),
    # counter-rotating constant-theta ray, lam = -a sin^2(th_o)
    (0.1, 0.3, -1), (0.5, 1.0, -1), (0.94, 2.2, -1), (0.99, 2.8, -1),
])
def test_constant_theta_rays(a, th_o, sign):
    """The two double roots of the theta potential on the beta = 0 row, both NaN before B3.
    sign = 1: principal null congruence, zeta = 0 and sqrt(R) = r^2 + a^2 cos^2(th_o)
    (GL19a 75, 76; its a -> 0 limit is the radial ray). sign = -1: the counter-rotating ray."""
    if sign == 1:
        lam, eta = conserved(a, th_o, -a*np.sin(th_o), 0.)
        assert eta + (lam - a)**2 == 0
    _assert_constant_theta_matches_quad(a, th_o, sign)


@xfail_until("B9", "near zeta = 0, z ~ sqrt(eps) from roundoff and the case-4 I_p, I_m lose ~eps/z^2: "
                   "phi off by 5e-6 to 4e-5 at the PNC pixel when zeta is not exactly 0")
@pytest.mark.parametrize("a, th_o", [(0.94, 2.2), (0.99, 2.2)])
def test_principal_null_congruence_near_degenerate(a, th_o):
    """The PNC pixel when zeta rounds to a nonzero ~1e-17, so the case-4 formulas are used."""
    _assert_constant_theta_matches_quad(a, th_o, 1)


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
    np.testing.assert_allclose(geo['th'][:, 0], np.pi/2, rtol=0, atol=EQUATORIAL_RAY_ATOL)
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
    np.testing.assert_allclose(geo['r'][ok, 0], ode['r'][ok], rtol=ODE_RTOL, err_msg="r")
    np.testing.assert_allclose(geo['th'][ok, 0], ode['th'][ok], rtol=0, atol=ODE_ANGLE_ATOL, err_msg="theta")
    dph = wrap(geo['ph'][ok, 0] - _face_on_expected_phi(alpha, beta, ode)[ok])
    np.testing.assert_allclose(dph, 0, rtol=0, atol=ODE_ANGLE_ATOL, err_msg="phi")


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
    np.testing.assert_allclose(geo['r'][ok, 0], ode['r'][ok], rtol=ODE_RTOL, err_msg="r")
    np.testing.assert_allclose(geo['th'][ok, 0], ode['th'][ok], rtol=0, atol=ODE_ANGLE_ATOL, err_msg="theta")
    np.testing.assert_allclose(np.pi - ode_north['th'][ok], ode['th'][ok], rtol=0, atol=ODE_ANGLE_ATOL,
                               err_msg="oracle reflection")
    dph = wrap(geo['ph'][ok, 0] - _face_on_expected_phi(alpha, -beta, ode_north)[ok])
    np.testing.assert_allclose(dph, 0, rtol=0, atol=ODE_ANGLE_ATOL, err_msg="phi")


# generic rays for the near-extremal / extremal tests (none on the NHEK line lam = 2)
_RAYS_EXTREMAL = [(6.0, 2.0), (-5.0, -4.0), (2.0, 3.0), (0.4, -1.5), (0.1, 0.1)]


@pytest.mark.parametrize("th_o", [0.3, 1.2])
@pytest.mark.parametrize("alpha, beta", _RAYS_EXTREMAL)
def test_near_extremal(th_o, alpha, beta):
    """a = 1 - 1e-8 (r_+ - r_- ~ 3e-4): currently accurate to ~1e-9 incl. t, phi.
    Regression guard for the a = 1 work (B7)."""
    _assert_matches_ode(1 - 1e-8, th_o, alpha, beta, angle_tol=NEAR_EXTREMAL_TOL, rtol=NEAR_EXTREMAL_TOL,
                        check_t=True)


@xfail_until("B7", "I_phi, I_t divided difference [F(r_+) - F(r_-)]/(r_+ - r_-) cancels "
                   "as r_+ -> r_-: phi, t off by 5e-8..2e-6 (r, theta fine)")
@pytest.mark.parametrize("th_o, alpha, beta", [(0.3, 2.0, 3.0), (1.2, 0.4, -1.5), (0.3, 0.1, 0.1)])
def test_extremal_limit_cancellation(th_o, alpha, beta):
    """a = 1 - 1e-14 (r_+ - r_- ~ 3e-7), captured rays, whose phi and t are dominated
    by the near-horizon I_+- integrals."""
    _assert_matches_ode(1 - 1e-14, th_o, alpha, beta, angle_tol=NEAR_EXTREMAL_TOL, rtol=NEAR_EXTREMAL_TOL,
                        check_t=True)


@xfail_until("B7", "a = 1 rejected by the input checks; I_phi, I_t need the r_+ = r_- limit")
@pytest.mark.parametrize("th_o", [0.3, 1.2])
@pytest.mark.parametrize("alpha, beta", _RAYS_EXTREMAL)
def test_extremal(th_o, alpha, beta):
    """a = 1: r_+ = r_- = 1 and Delta = (r - 1)^2 has a double root."""
    _assert_matches_ode(1.0, th_o, alpha, beta, angle_tol=NEAR_EXTREMAL_TOL, rtol=NEAR_EXTREMAL_TOL,
                        check_t=True)


@xfail_until("B8", "the observer enters only via the last digits of O(1) elliptic arguments: "
                   "t, sigma carry a constant offset ~eps r_o^2 (7.7e-3 at r_o = 1e7)")
def test_distant_observer_time():
    """r_o = 1e7, a = 0, captured ray (no radial turning point): exact t and sigma along
    the ray from 30-digit quadrature, t(r) = -int_r^r_o r^3/((r-2) sqrt(R)) dr and
    sigma(r) = -int_r^r_o r^2/sqrt(R) dr (the theta parts vanish at a = 0)."""
    r_o, th_o, alpha, beta = 1e7, 1.0, 2.0, 3.0
    geo = trace(0.0, th_o, [alpha], [beta], ngeo=12, r_o=r_o)
    assert geo['rclass'][0] == 3
    lam, eta = conserved(0.0, th_o, alpha, beta)

    def integrand_t(y):
        return y**3/((y - 2)*mp.sqrt(radial_potential(y, 0.0, lam, eta)))

    def integrand_sig(y):
        return y**2/mp.sqrt(radial_potential(y, 0.0, lam, eta))

    r = geo['r'][1:, 0]                  # tau = 0 (r = r_o) is trivially t = sigma = 0
    with mp.workdps(30):
        t_exact = [-float(mp.quad(integrand_t, [mp.mpf(x), r_o])) for x in r]
        sig_exact = [-float(mp.quad(integrand_sig, [mp.mpf(x), r_o])) for x in r]
    np.testing.assert_allclose(geo['t'][1:, 0], t_exact, rtol=0, atol=DISTANT_T_ATOL, err_msg="t")
    np.testing.assert_allclose(geo['sig'][1:, 0], sig_exact, rtol=0, atol=DISTANT_T_ATOL, err_msg="sigma")
