import numpy as np
import pytest

from kgeo.kerr_raytracing_utils import check_inclination, check_spin, obs_trig


@pytest.mark.parametrize("a", [0, 0.0, 0.5, -0.5, 0.999, np.float64(0.3), np.float32(0.3), np.int64(0)])
def test_check_spin_accepts(a):
    check_spin(a)


@pytest.mark.parametrize("a", [1.0, 1, -1.0, 1.5, True, "0.5", None, np.array([0.5]), 0.5 + 0j, np.nan])
def test_check_spin_rejects(a):
    with pytest.raises(Exception, match=r"\|a\| should be a float"):
        check_spin(a)


@pytest.mark.parametrize("th_o", [1, 0.3, 2.2, np.float64(1.0), np.float32(1.0), np.int64(3)])
def test_check_inclination_accepts(th_o):
    check_inclination(th_o)


@pytest.mark.parametrize("th_o", [0, 0.0, np.pi/2, np.pi, 4.0, -0.1, True, "1", None, np.array([1.0])])
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
