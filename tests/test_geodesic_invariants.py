import numpy as np
import pytest

from geodesic_ode import integrate_geodesic
from helpers import trace

R_O = 1000.

# kgeo's t and sigma carry a constant roundoff offset ~ eps r_o^2 along the whole ray
# (the observer enters only via the last digits of O(1) elliptic arguments; B8), which
# varies with numpy/scipy versions. Compare them with an absolute floor of 50 eps r_o^2.
T_ATOL = 50*np.finfo(float).eps*R_O**2

# generic screen points: two escaping (class I), two captured, one near the origin
# (vortical for a > 0), and one in the class II patch (a = 0.94, th_o = 1.5).
# No alpha ~ 0 or eta ~ 0: those are in test_edge_cases.py
ALPHA = np.array([6.0, -5.0, 2.0, 0.4, 0.1, -1.26])
BETA = np.array([2.0, -4.0, 3.0, -1.5, 0.1, 0.1])


@pytest.mark.parametrize("a", [0.0, 0.3, 0.94])
@pytest.mark.parametrize("th_o", [0.3, 1.2, 1.5, 2.2])
def test_matches_ode_oracle(a, th_o):
    """Analytic r, theta, phi, t, sigma agree with direct integration of the
    Mino-time equations of motion at the same Mino times."""
    geo = trace(a, th_o, ALPHA, BETA, ngeo=40, r_o=R_O, taufrac=0.97)
    rplus = 1 + np.sqrt(1 - a**2)

    for i in range(len(ALPHA)):
        ode = integrate_geodesic(a, R_O, th_o, ALPHA[i], BETA[i], geo['tau'][:, i])
        # stay clear of the horizon, where t and phi diverge logarithmically
        ok = geo['r'][:, i] > rplus + 0.2
        msg = f"ray (alpha, beta) = ({ALPHA[i]}, {BETA[i]})"
        np.testing.assert_allclose(geo['r'][ok, i], ode['r'][ok], rtol=1e-8, err_msg=msg)
        np.testing.assert_allclose(geo['th'][ok, i], ode['th'][ok], rtol=0, atol=1e-8, err_msg=msg)
        np.testing.assert_allclose(geo['ph'][ok, i], ode['ph'][ok], rtol=0, atol=1e-8, err_msg=msg)
        np.testing.assert_allclose(geo['t'][ok, i], ode['t'][ok], rtol=1e-8, atol=1e-8, err_msg=msg)
        np.testing.assert_allclose(geo['sig'][ok, i], ode['sig'][ok], rtol=1e-8, atol=1e-8, err_msg=msg)


@pytest.mark.parametrize("a", [0.0, 0.3, 0.94])
@pytest.mark.parametrize("th_o", [0.3, 1.2, 1.5])
def test_equatorial_reflection(a, th_o):
    """Kerr is symmetric under theta -> pi - theta: reflecting the observer to
    pi - th_o and flipping beta (p_theta) gives the same r, phi, t, sigma with
    theta -> pi - theta. Checks the th_o > pi/2 paths against th_o < pi/2.
    (beta = 0, observer at a turning point, is in test_edge_cases.py)"""
    north = trace(a, th_o, ALPHA, BETA, ngeo=40, r_o=R_O)
    south = trace(a, np.pi - th_o, ALPHA, -BETA, ngeo=40, r_o=R_O)

    np.testing.assert_allclose(south['tau'], north['tau'], rtol=1e-12)
    ok = north['r'] > 1 + np.sqrt(1 - a**2) + 0.2
    np.testing.assert_allclose(south['r'][ok], north['r'][ok], rtol=1e-9)
    np.testing.assert_allclose(np.pi - south['th'][ok], north['th'][ok], rtol=0, atol=1e-9)
    np.testing.assert_allclose(south['ph'][ok], north['ph'][ok], rtol=0, atol=1e-9)
    np.testing.assert_allclose(south['t'][ok], north['t'][ok], rtol=1e-9, atol=T_ATOL)
    np.testing.assert_allclose(south['sig'][ok], north['sig'][ok], rtol=1e-9, atol=T_ATOL)


@pytest.mark.parametrize("a", [0.3, 0.94])
@pytest.mark.parametrize("th_o", [0.3, 1.2, 2.2])
def test_spin_flip(a, th_o):
    """Kerr is invariant under (a, phi) -> (-a, -phi), which maps lam -> -lam
    (alpha -> -alpha) with eta unchanged: same r, theta, t, sigma and phi -> -phi.
    Checks kgeo's a < 0 paths against a > 0."""
    pos = trace(a, th_o, ALPHA, BETA, ngeo=40, r_o=R_O)
    neg = trace(-a, th_o, -ALPHA, BETA, ngeo=40, r_o=R_O)

    np.testing.assert_allclose(neg['tau'], pos['tau'], rtol=1e-12)
    ok = pos['r'] > 1 + np.sqrt(1 - a**2) + 0.2
    np.testing.assert_allclose(neg['r'][ok], pos['r'][ok], rtol=1e-9)
    np.testing.assert_allclose(neg['th'][ok], pos['th'][ok], rtol=0, atol=1e-9)
    np.testing.assert_allclose(-neg['ph'][ok], pos['ph'][ok], rtol=0, atol=1e-9)
    np.testing.assert_allclose(neg['t'][ok], pos['t'][ok], rtol=1e-9, atol=T_ATOL)
    np.testing.assert_allclose(neg['sig'][ok], pos['sig'][ok], rtol=1e-9, atol=T_ATOL)


@pytest.mark.parametrize("th_o", [0.3, 1.2, 1.5, 2.2])
def test_schwarzschild_orbital_plane(th_o):
    """a = 0: the angular momentum vector is conserved, so each geodesic lies in
    a plane through the origin. Checks r, theta, phi jointly."""
    geo = trace(0.0, th_o, ALPHA, BETA, ngeo=60, r_o=R_O)
    xhat = np.array([np.sin(geo['th'])*np.cos(geo['ph']),
                     np.sin(geo['th'])*np.sin(geo['ph']),
                     np.cos(geo['th'])])                      # (3, ngeo, npix)

    for i in range(len(ALPHA)):
        x = xhat[:, :, i].T                                  # (ngeo, 3)
        cross = np.cross(x[0], x)
        normal = cross[np.argmax(np.linalg.norm(cross, axis=1))]
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(x @ normal, 0, atol=1e-9,
                                   err_msg=f"ray (alpha, beta) = ({ALPHA[i]}, {BETA[i]})")
