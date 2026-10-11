import numpy as np
import pytest

from kgeo.bfields import Bfield
from kgeo.equatorial_images import Iobs
from kgeo.kerr_raytracing_utils import check_inclination, check_spin, obs_trig, radial_roots
from kgeo.velocities import Velocity

from helpers import quiet

# Tolerances, with the worst error measured in [brackets] (2026-10-11, CI)
SCALAR_TYPE_RTOL = 1e-6   # Iobs with int / numpy-scalar a, th_o vs python floats   [0; 7e-8 with numpy >= 2,
                          # where a float32 spin makes the arithmetic float32 (NEP 50)]


@pytest.mark.parametrize("a", [0, 0.0, 0.5, -0.5, 0.999, np.float64(0.3), np.float32(0.3), np.int64(0)])
def test_check_spin_accepts(a):
    check_spin(a)


@pytest.mark.parametrize("a", [1.0, 1, -1.0, 1.5, True, "0.5", None, np.array([0.5]), 0.5 + 0j, np.nan])
def test_check_spin_rejects(a):
    with pytest.raises(Exception, match=r"\|a\| should be a float"):
        check_spin(a)


@pytest.mark.parametrize("th_o", [1, 0.3, 2.2, np.float64(1.0), np.float32(1.0), np.int64(3), 0, 0.0, np.pi])
def test_check_inclination_accepts(th_o):
    check_inclination(th_o)


@pytest.mark.parametrize("th_o", [np.pi/2, 4.0, -0.1, True, "1", None, np.array([1.0])])
def test_check_inclination_rejects(th_o):
    with pytest.raises(Exception, match="th_o should be a float"):
        check_inclination(th_o)


def test_obs_trig():
    assert obs_trig(0) == (0., 1.)
    assert obs_trig(0.0) == (0., 1.)
    assert obs_trig(np.pi/2) == (1., 0.)
    assert obs_trig(np.pi) == (0., -1.)
    for th in [0.3, 1.2, 2.2, 1e-8, np.pi/2 - 1e-12]:
        assert obs_trig(th) == (np.sin(th), np.cos(th))


def test_public_functions_accept_int_and_numpy_scalars():
    """Spins and inclinations given as ints / numpy scalars give the same results as
    python floats (they used to be rejected by isinstance(a, float))."""
    lam, eta = np.array([-2.0, 3.5]), np.array([10., 20.])
    for a in [0, np.int64(0)]:
        np.testing.assert_array_equal(radial_roots(a, lam, eta)[:4], radial_roots(0.0, lam, eta)[:4])

    alpha, beta = np.array([3.0, -4.0]), np.array([2.0, 5.0])
    with quiet():
        expected = Iobs(0.5, 1000., 1.0, 0, alpha, beta)[0]
        for a, th_o in [(np.float64(0.5), 1), (np.float32(0.5), np.int64(1))]:
            np.testing.assert_allclose(Iobs(a, 1000., th_o, 0, alpha, beta)[0], expected, rtol=SCALAR_TYPE_RTOL)

    r = np.array([4.0, 8.0])
    # u_lab returns u^theta as a scalar 0 next to arrays, so compare per component
    for u_np, u_float in zip(Velocity('kep').u_lab(np.float64(0.5), r), Velocity('kep').u_lab(0.5, r)):
        np.testing.assert_array_equal(u_np, u_float)
    np.testing.assert_array_equal(Bfield('bz_monopole').bfield_lab(0, r), Bfield('bz_monopole').bfield_lab(0.0, r))
