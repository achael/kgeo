import numpy as np
import pytest

from geodesic_ode import integrate_geodesic
from helpers import trace

R_O = 1000.

# Tolerances, with the worst error measured in [brackets] (2026-10-10, after B4 1/3)
ODE_RTOL = 1e-10               # generic rays vs the ODE oracle: r, t, sigma                [2e-12]
ODE_ANGLE_ATOL = 1e-10         # generic rays vs the ODE oracle: theta, phi                 [1e-11]
SYMMETRY_TAU_RTOL = 1e-13      # reflection, spin flip: Mino time grids                     [1e-15]
SYMMETRY_RTOL = 1e-11          # reflection, spin flip: r, t, sigma                         [6e-13]
SYMMETRY_ANGLE_ATOL = 1e-12    # reflection, spin flip: theta, phi                          [6e-14]
# kgeo's t and sigma carry a constant roundoff offset ~ eps r_o^2 along the whole ray
# (the observer enters only via the last digits of O(1) elliptic arguments; B8), which
# varies with numpy/scipy versions. Compare them with an absolute floor of 50 eps r_o^2.
T_ATOL = 50*np.finfo(float).eps*R_O**2  # = 1.1e-8                                         [2e-9]
SMALL_SPIN_RTOL = 1e-10        # 1e-9 <= a <= 1e-3 vs the ODE oracle: r                     [1e-12]
SMALL_SPIN_ANGLE_ATOL = 1e-10  # 1e-9 <= a <= 1e-3 vs the ODE oracle: theta, phi            [1e-11]
ORBITAL_PLANE_ATOL = 1e-13     # a = 0: distance of x-hat from the orbital plane            [2e-15]

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
        np.testing.assert_allclose(geo['r'][ok, i], ode['r'][ok], rtol=ODE_RTOL, err_msg=msg)
        np.testing.assert_allclose(geo['th'][ok, i], ode['th'][ok], rtol=0, atol=ODE_ANGLE_ATOL, err_msg=msg)
        np.testing.assert_allclose(geo['ph'][ok, i], ode['ph'][ok], rtol=0, atol=ODE_ANGLE_ATOL, err_msg=msg)
        np.testing.assert_allclose(geo['t'][ok, i], ode['t'][ok], rtol=ODE_RTOL, atol=T_ATOL, err_msg=msg)
        np.testing.assert_allclose(geo['sig'][ok, i], ode['sig'][ok], rtol=ODE_RTOL, atol=T_ATOL, err_msg=msg)


@pytest.mark.parametrize("a", [0.0, 0.3, 0.94])
@pytest.mark.parametrize("th_o", [0.3, 1.2, 1.5])
def test_equatorial_reflection(a, th_o):
    """Kerr is symmetric under theta -> pi - theta: reflecting the observer to
    pi - th_o and flipping beta (p_theta) gives the same r, phi, t, sigma with
    theta -> pi - theta. Checks the th_o > pi/2 paths against th_o < pi/2.
    (beta = 0, observer at a turning point, is in test_edge_cases.py)"""
    north = trace(a, th_o, ALPHA, BETA, ngeo=40, r_o=R_O)
    south = trace(a, np.pi - th_o, ALPHA, -BETA, ngeo=40, r_o=R_O)

    np.testing.assert_allclose(south['tau'], north['tau'], rtol=SYMMETRY_TAU_RTOL)
    ok = north['r'] > 1 + np.sqrt(1 - a**2) + 0.2
    np.testing.assert_allclose(south['r'][ok], north['r'][ok], rtol=SYMMETRY_RTOL)
    np.testing.assert_allclose(np.pi - south['th'][ok], north['th'][ok], rtol=0, atol=SYMMETRY_ANGLE_ATOL)
    np.testing.assert_allclose(south['ph'][ok], north['ph'][ok], rtol=0, atol=SYMMETRY_ANGLE_ATOL)
    np.testing.assert_allclose(south['t'][ok], north['t'][ok], rtol=SYMMETRY_RTOL, atol=T_ATOL)
    np.testing.assert_allclose(south['sig'][ok], north['sig'][ok], rtol=SYMMETRY_RTOL, atol=T_ATOL)


@pytest.mark.parametrize("a", [0.3, 0.94])
@pytest.mark.parametrize("th_o", [0.3, 1.2, 2.2])
def test_spin_flip(a, th_o):
    """Kerr is invariant under (a, phi) -> (-a, -phi), which maps lam -> -lam
    (alpha -> -alpha) with eta unchanged: same r, theta, t, sigma and phi -> -phi.
    Checks kgeo's a < 0 paths against a > 0."""
    pos = trace(a, th_o, ALPHA, BETA, ngeo=40, r_o=R_O)
    neg = trace(-a, th_o, -ALPHA, BETA, ngeo=40, r_o=R_O)

    np.testing.assert_allclose(neg['tau'], pos['tau'], rtol=SYMMETRY_TAU_RTOL)
    ok = pos['r'] > 1 + np.sqrt(1 - a**2) + 0.2
    np.testing.assert_allclose(neg['r'][ok], pos['r'][ok], rtol=SYMMETRY_RTOL)
    np.testing.assert_allclose(neg['th'][ok], pos['th'][ok], rtol=0, atol=SYMMETRY_ANGLE_ATOL)
    np.testing.assert_allclose(-neg['ph'][ok], pos['ph'][ok], rtol=0, atol=SYMMETRY_ANGLE_ATOL)
    np.testing.assert_allclose(neg['t'][ok], pos['t'][ok], rtol=SYMMETRY_RTOL, atol=T_ATOL)
    np.testing.assert_allclose(neg['sig'][ok], pos['sig'][ok], rtol=SYMMETRY_RTOL, atol=T_ATOL)


@pytest.mark.parametrize("a", [1e-9, 1e-7, 0.99e-6, 1.01e-6, 1e-5, 0.99e-4, 1.01e-4, 1e-3])
def test_small_spin_matches_ode_oracle(a):
    """No accuracy valley at small spin: before the cancellation-free u_+- (B2), the full
    theta formulas lost digits like eps/a^2 just above the old MINSPIN = 1e-4 (phi off by
    2e-6 at a = 1.01e-4), and the a = 0 limit branch below it dropped O(a^2) terms."""
    for th_o in [0.3, 1.2, 2.2]:
        geo = trace(a, th_o, ALPHA[:4], BETA[:4], ngeo=40, r_o=R_O)
        for i in range(4):
            ode = integrate_geodesic(a, R_O, th_o, ALPHA[i], BETA[i], geo['tau'][:, i])
            ok = geo['r'][:, i] > 1 + np.sqrt(1 - a**2) + 0.2
            msg = f"th_o = {th_o}, ray ({ALPHA[i]}, {BETA[i]})"
            np.testing.assert_allclose(geo['r'][ok, i], ode['r'][ok], rtol=SMALL_SPIN_RTOL, err_msg=msg)
            np.testing.assert_allclose(geo['th'][ok, i], ode['th'][ok], rtol=0, atol=SMALL_SPIN_ANGLE_ATOL, err_msg=msg)
            np.testing.assert_allclose(geo['ph'][ok, i], ode['ph'][ok], rtol=0, atol=SMALL_SPIN_ANGLE_ATOL, err_msg=msg)


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
        np.testing.assert_allclose(x @ normal, 0, atol=ORBITAL_PLANE_ATOL,
                                   err_msg=f"ray (alpha, beta) = ({ALPHA[i]}, {BETA[i]})")
