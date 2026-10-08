import numpy as np
import pytest
from scipy.integrate import quad

from kgeo.kerr_raytracing_utils import is_outside_crit, mino_total, radial_roots, uplus_uminus

from geodesic_ode import conserved, radial_potential
from helpers import xfail_until

# screen points covering radial classes I, III, IV, plus three near the origin
# (vortical for a > 0) and two in the small class II patch that appears near
# edge-on at high spin (alpha ~ -1.3, beta ~ 0 for a = 0.94, th_o = 1.5)
_ALPHA, _BETA = np.meshgrid(np.linspace(-7.5, 7.5, 7), np.linspace(-7.3, 7.3, 6))
ALPHA = np.concatenate([_ALPHA.ravel(), [0.1, -0.2, 0.3, -1.26, -1.4]])
BETA = np.concatenate([_BETA.ravel(), [0.1, 0.05, -0.1, 0.1, -0.3]])

SPINS = [0.0, 0.5, 0.94]
INCS = [0.3, 1.0, 1.5, 2.2]


def _vieta_residual(a, lam, eta):
    """Largest scaled mismatch between the roots' elementary symmetric polynomials
    and the coefficients of R(r) = r^4 + A r^2 + B r + C (GL19a eq 95), per pixel."""
    roots = np.array(radial_roots(a, lam, eta)[:4])
    A = a**2 - eta - lam**2
    B = 2*(eta + (lam - a)**2)
    C = -a**2*eta

    e1 = roots.sum(0)
    e2 = sum(roots[i]*roots[j] for i in range(4) for j in range(i + 1, 4))
    e3 = sum(roots[i]*roots[j]*roots[k]
             for i in range(4) for j in range(i + 1, 4) for k in range(j + 1, 4))
    e4 = roots.prod(0)

    # e_k ~ r^k, so scale each relation to be dimensionless
    scale = 1 + np.abs(roots).max(0)
    return np.max([np.abs(e1)/scale, np.abs(e2 - A)/scale**2,
                   np.abs(e3 + B)/scale**3, np.abs(e4 - C)/scale**4], axis=0)


@pytest.mark.parametrize("a", SPINS)
@pytest.mark.parametrize("th_o", INCS)
def test_radial_roots_vieta(a, th_o):
    """r1..r4 are the roots of R(r): sum r_i = 0, sum r_i r_j = A, etc."""
    lam, eta = conserved(a, th_o, ALPHA, BETA)
    # 1e-9, not machine precision: near the P = 0 curve the Cardano step loses
    # up to ~7 digits (see test_radial_roots_resolvent_cancellation); tighten after B1
    assert np.all(_vieta_residual(a, lam, eta) < 1e-9)


@xfail_until("B1", "Cardano cancellation: omega_minus^3 = -Q/2 - sqrt(...) loses all digits "
                   "when P ~ 0; use omega_plus*omega_minus = -P/3, preserving root ordering")
@pytest.mark.parametrize("a, th_o, alpha, beta", [
    (0.94, 1.0, 0.72, -0.36),    # P ~ 1e-5, roots off by ~1e-6
    (0.5, 0.05, -1.24, -1.48),
    (0.5, 1.5, 0.3, -0.1),
])
def test_radial_roots_resolvent_cancellation(a, th_o, alpha, beta):
    """Screen points near the curve where the resolvent cubic's P vanishes."""
    lam, eta = conserved(a, th_o, np.array([alpha]), np.array([beta]))
    assert _vieta_residual(a, lam, eta)[0] < 1e-13


@xfail_until("B3", "z = 0 gives B/(4z) = 0/0 in radial_roots")
@pytest.mark.parametrize("th_o", [0.3, 1.0])
def test_radial_roots_schwarzschild_radial_ray(th_o):
    """a = 0, screen center: lam = eta = 0, so R(r) = r^4 has a quadruple root at r = 0."""
    zero = np.array([0.])
    roots = np.array(radial_roots(0.0, *conserved(0.0, th_o, zero, zero))[:4])
    assert np.all(np.isfinite(roots))
    assert np.all(np.abs(roots) < 1e-12)


def _mino_total_quad(a, lam, eta, rclass, r4_guess, r_o):
    """Total Mino time int dr/sqrt(R), from the polynomial R (independent of kgeo's roots)."""
    def R(r):
        return radial_potential(r, a, lam, eta)

    if rclass != 1:
        # captured: from the observer straight down to the horizon
        rplus = 1 + np.sqrt(1 - a**2)
        return quad(lambda r: 1/np.sqrt(R(r)), rplus, r_o, epsabs=0, epsrel=1e-12, limit=200)[0]

    # class I: in to the turning point r4 and back out. Newton-refine r4 on R itself,
    # then substitute r = r4 + s^2 so the integrand 2s/sqrt(R) stays finite at s = 0
    r4 = r4_guess
    for _ in range(5):
        dR = 4*r4*(r4**2 + a**2 - a*lam) - (2*r4 - 2)*(eta + (lam - a)**2)
        r4 -= R(r4)/dR
    leg = quad(lambda s: 2*s/np.sqrt(R(r4 + s*s)), 0, np.sqrt(r_o - r4),
               epsabs=0, epsrel=1e-12, limit=200)[0]
    return 2*leg


@pytest.mark.parametrize("a", SPINS)
@pytest.mark.parametrize("th_o", [0.3, 1.0, 1.5])
def test_mino_total_matches_quadrature(a, th_o):
    r_o = 1000.
    lam, eta = conserved(a, th_o, ALPHA, BETA)
    r1, r2, r3, r4, rclass = radial_roots(a, lam, eta)
    tautot = mino_total(a, r_o, eta, r1, r2, r3, r4)
    expected = [_mino_total_quad(a, lam[i], eta[i], rclass[i], np.real(r4[i]), r_o)
                for i in range(len(lam))]
    np.testing.assert_allclose(tautot, expected, rtol=1e-9)


@pytest.mark.parametrize("a", SPINS)
@pytest.mark.parametrize("th_o", INCS)
def test_radial_class_matches_potential(a, th_o):
    """Class I (turning point outside the horizon) <=> R(r) < 0 somewhere on (r_+, inf);
    classes II-IV have R > 0 there and fall in. Checked by evaluating R directly,
    independently of the root finder, and against is_outside_crit."""
    lam, eta = conserved(a, th_o, ALPHA, BETA)
    rclass = radial_roots(a, lam, eta)[4]

    rplus = 1 + np.sqrt(1 - a**2)
    r = np.geomspace(rplus, 1000., 4000)[:, None]
    has_turning_point = np.any(radial_potential(r, a, lam, eta) < 0, axis=0)

    assert set(np.unique(rclass)) <= {1, 2, 3, 4}
    assert np.array_equal(rclass == 1, has_turning_point)
    assert np.array_equal(is_outside_crit(a, th_o, ALPHA, BETA).astype(bool), has_turning_point)


@pytest.mark.parametrize("a", [0.5, 0.94])
@pytest.mark.parametrize("th_o", INCS)
def test_angular_roots(a, th_o):
    """u_pm = cos^2 of the theta turning points: with u = cos^2(theta),
    (1-u) Theta = eta + (a^2 - eta - lam^2) u - a^2 u^2, so u_pm are its roots."""
    lam, eta = conserved(a, th_o, ALPHA, BETA)
    up, um, uratio, a2um = uplus_uminus(a, th_o, lam, eta)

    def poly(u):
        return eta + (a**2 - eta - lam**2)*u - a**2*u**2

    scale = 1 + np.abs(eta) + lam**2 + a**2
    assert np.all(np.abs(poly(up))/scale < 1e-13)
    assert np.all(np.abs(poly(um))/scale < 1e-13)
    np.testing.assert_allclose(up*um, -eta/a**2, rtol=1e-12)
    np.testing.assert_allclose(uratio, up/um, rtol=1e-15)
    np.testing.assert_allclose(a2um, a**2*um, rtol=1e-15)

    # the observer lies in the allowed band: cos^2(th_o) <= u_+ always,
    # and u_- <= cos^2(th_o) for vortical rays (eta < 0)
    u_o = np.cos(th_o)**2
    assert np.all(u_o <= up)
    assert np.all(um[eta < 0] <= u_o)


@pytest.mark.parametrize("th_o", INCS)
def test_angular_roots_schwarzschild(th_o):
    """a = 0: Theta = eta - lam^2 cot^2(theta) vanishes at u_+ = eta/(eta + lam^2);
    a^2 u_- = -eta/u_+ -> -(eta + lam^2) and u_+/u_- -> 0 in the a -> 0 limit."""
    lam, eta = conserved(0.0, th_o, ALPHA, BETA)
    up, um, uratio, a2um = uplus_uminus(0.0, th_o, lam, eta)
    np.testing.assert_allclose(up, eta/(eta + lam**2), rtol=1e-15)
    np.testing.assert_allclose(a2um, -(eta + lam**2), rtol=1e-15)
    assert np.all(uratio == 0)
