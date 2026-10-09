import numpy as np
import pytest

from kgeo.kerr_raytracing_ana import coords_at_tau
from kgeo.kerr_raytracing_utils import n_angular_turnings, n_equatorial_crossings, n_poloidal_orbits

from geodesic_ode import conserved
from helpers import quiet, xfail_until

# (a, th_o, alpha, beta): ordinary and vortical rays, both hemispheres, both signs of beta
TURNING_CASES = [
    (0.5, 1.0, 4.0, 3.0), (0.5, 1.0, 4.0, -3.0),                    # ordinary, north
    (0.5, np.pi - 1.0, 4.0, 3.0), (0.5, np.pi - 1.0, 4.0, -3.0),    # ordinary, south
    (0.94, 0.3, 0.1, 0.2), (0.94, 0.3, 0.1, -0.2),                  # vortical, north
    (0.94, np.pi - 0.3, 0.1, 0.2), (0.94, np.pi - 0.3, 0.1, -0.2),  # vortical, south
]


def _theta(a, th_o, alpha, beta, tau):
    """kgeo's theta(tau) for one ray (the theta solution is valid beyond tau_tot)"""
    with quiet():
        _, (t, r, th, ph) = coords_at_tau(a, [0, 1000., th_o, 0], [np.array([alpha]), np.array([beta])],
                                          tau.reshape(-1, 1), do_phi_and_t=False)
    return th[:, 0]


def _turnings(a, th_o, alpha, beta, nhalf=3.5, npts=6000):
    """n_angular_turnings over ~nhalf half-librations, and the turning points counted
    independently from sign changes of dtheta/dtau along kgeo's theta(tau)"""
    A, B = np.array([alpha]), np.array([beta])
    slope = n_angular_turnings(a, th_o, A, B, np.array([[1.0]]))[0, 0] - n_angular_turnings(a, th_o, A, B, np.array([[0.0]]))[0, 0]
    tau = np.linspace(0, nhalf/slope, npts)
    n = n_angular_turnings(a, th_o, A, B, tau.reshape(-1, 1))[:, 0]
    dth = np.sign(np.diff(_theta(a, th_o, alpha, beta, tau)))
    counted = np.concatenate([[0], np.cumsum(dth[1:] != dth[:-1])])
    counted = np.append(counted, counted[-1])
    return n, counted


@pytest.mark.parametrize("a, th_o, alpha, beta", TURNING_CASES)
def test_n_angular_turnings_counts_turning_points(a, th_o, alpha, beta):
    """floor(n) is the number of polar turning points passed, counted independently
    from the sign changes of dtheta/dtau (skipping samples next to a turning point)."""
    n, counted = _turnings(a, th_o, alpha, beta)
    step = n[1] - n[0]
    clear = np.abs(n - np.round(n)) > 2*step
    assert counted[-1] >= 3
    np.testing.assert_array_equal(np.floor(n)[clear], counted[clear])


@pytest.mark.parametrize("a, th_o, alpha", [
    (0.5, 1.0, 4.0), (0.5, np.pi - 1.0, 4.0),        # ordinary: observer at theta_-/theta_+
    (0.94, 0.3, 0.05), (0.94, np.pi - 0.3, 0.05),    # vortical: observer at the equator-side turning point
    (0.94, 0.3, 0.6), (0.94, np.pi - 0.3, 0.6),      # vortical: observer at the pole-side turning point
])
def test_n_angular_turnings_beta_zero(a, th_o, alpha):
    """beta = 0: the observer itself sits on a turning point, which is not counted, so
    n(0) = 0; later turning points are counted as usual. (1e-6: at the theta_-/theta_+
    and pole-side turning points F0 comes from arcsin(1 - O(eps)), off by ~1e-8; B4)"""
    n, counted = _turnings(a, th_o, alpha, 0.0)
    assert abs(n[0]) < 1e-6
    step = n[1] - n[0]
    clear = np.abs(n - np.round(n)) > 2*step
    np.testing.assert_array_equal(np.floor(n)[clear], counted[clear])


@xfail_until("B4", "beta = 0: arcsin(1 - O(eps)) in F0 loses half the digits, n(0) ~ 1e-8 "
                   "(some cases pass by roundoff since B2)", strict=False)
@pytest.mark.parametrize("a, th_o, alpha", [
    (0.5, 1.0, 4.0), (0.5, np.pi - 1.0, 4.0), (0.94, 0.3, 0.6), (0.94, np.pi - 0.3, 0.6),
])
def test_n_angular_turnings_beta_zero_exact(a, th_o, alpha):
    A, B = np.array([alpha]), np.array([0.0])
    assert abs(n_angular_turnings(a, th_o, A, B, np.array([[0.0]]))[0, 0]) < 1e-12


def test_all_vortical_input_returns_flag():
    """n_poloidal_orbits and n_equatorial_crossings are only defined for ordinary
    geodesics; an input of only vortical rays gets the flag -2 (used to raise
    UnboundLocalError because the ordinary branch never ran)."""
    a, th_o = 0.94, 0.3
    alpha, beta = np.array([0.1, -0.2, 0.3]), np.array([0.2, -0.1, 0.0])
    assert np.all(conserved(a, th_o, alpha, beta)[1] < 0)
    tau = np.outer(np.linspace(0, 0.5, 5), np.ones(3))
    assert np.all(n_poloidal_orbits(a, th_o, alpha, beta, tau) == -2)
    assert np.all(n_equatorial_crossings(a, th_o, alpha, beta, tau) == -2)
