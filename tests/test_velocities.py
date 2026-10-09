import numpy as np
import pytest

from kgeo.velocities import Velocity, _allowed_velocity_models, rplusfunc

from helpers import DATA_DIR, xfail_until

_MIDPLANE = np.pi/2.


def _gcov_bl(a, r, th=np.pi/2.):
    """
    Return 4x4 covariant metric tensor in Boyer-Lindquist coordinates for
    bhspin == a and radius == r. Optionally supports theta == th (default
    is theta = pi/2).
    """
    Sigma = r*r + a*a*np.cos(th)**2.
    Delta = r*r - 2*r + a*a
    gcov = np.zeros((4, 4))
    gcov[0, 0] = -(1 - 2*r/Sigma)
    gcov[0, 3] = -2*a*r*np.sin(th)**2 / Sigma
    gcov[1, 1] = Sigma / Delta
    gcov[2, 2] = Sigma
    gcov[3, 0] = gcov[0, 3]
    gcov[3, 3] = (r*r + a*a + 2*a*a*r*np.sin(th)**2 / Sigma) * np.sin(th)**2
    return gcov


def test_velocity_usq():
    """
    Iterate over known velocity models and fuzz check that the components
    are in BL coordinates / that u.u == -1.
    """
    np.random.seed(42)
    for bhspin in np.random.random(3):
        reh = 1. + np.sqrt(1 - bhspin**2)
        for radius in np.linspace(1.1, 15.):
            radius = np.array(radius)
            if radius <= reh:
                continue
            gcov = _gcov_bl(bhspin, radius)
            for veltype in _allowed_velocity_models:
                # ignore certain "special case" models in this "randomized" test
                if veltype in ['fromfile', 'fromdict']:
                    continue
                if veltype == 'simfit':   # TODO, it appears that this model is broken?
                    continue
                if veltype == 'driftframe':  # TODO, also appears this has restrictions
                    continue
                print(f'Testing u.u == -1 for {veltype} with a={bhspin} at r={radius}...')
                v = Velocity(veltype)
                ucon = np.array(v.u_lab(bhspin, radius))
                ucov = np.einsum('ij,i...->j...', gcov, ucon)
                usq = np.einsum('i...,i...->...', ucon, ucov)
                assert np.isclose(usq, -1., rtol=1.e-5), \
                    f'Failed for {veltype} with a={bhspin} at r={radius}: u.u = {usq}'


def test_velocity_fromfile():
    """
    Check that the 'fromfile' velocity model produces expected four-velocities.
    """
    v = Velocity('fromfile', file=str(DATA_DIR / 'a0p6_inflow.csv'))
    tests = {
        # radius, theta -> ucon0, ucon1, ucon2, ucon3
        (2.15708, _MIDPLANE): [5.0056, -0.54416, 0.0, 0.75899],
        (2.5, _MIDPLANE): [3.0016, -0.45323831, 0.0, 0.41926807],
        (3.58635, _MIDPLANE): [1.7755, -0.22549, 0.0, 0.19684],
        (4.90566, _MIDPLANE): [1.4859, -0.021582, 0.0, 0.12754],
    }

    for (radius, theta), ucon in tests.items():
        bhspin = np.array(0.6)
        radius = np.array(radius)
        ucon_out = np.squeeze(v.u_lab(bhspin, radius, th=theta))
        assert np.allclose(ucon_out, ucon, rtol=1.e-4), \
            f'Failed for radius={radius}, theta={theta}: expected {ucon}, got {ucon_out}'


def test_velocity_general():
    """
    Check whether a few known inputs produce expected output for the 'general' velocity
    model. The tests that appear below were shown to agree with output from ipole.
    """
    tests = {
        # retrograde, fac_subkep, beta_phi, beta_r, bhspin, radius, theta -> ucon0, ucon1, ucon2, ucon3
        (0, 1, 0.7, 0.7, 0.3, 4.0, _MIDPLANE): [1.65511, -0.243725, 0, 0.141355],
        (0, 0.3, 0.1, 0.4, 0.9375, 1.6, _MIDPLANE): [14.0203, -0.934042, 0, 3.63511],
        (0, 0.1, 0.36, 0.72, 0.1707, 3.4, _MIDPLANE): [2.04774, -0.549666, 0, 0.0269644],
        (1, 0.24, 0.42, 0.24, 0.42, 10., _MIDPLANE): [1.19678, -0.340182, 0, -0.00297456],
        (1, 0.24, 0.42, 0.24, 0.42, 3.5, _MIDPLANE): [2.23166, -0.71236, 0, 0.0121293],
    }

    for (retro, fac_subkep, beta_phi, beta_r, bhspin, radius, theta), ucon in tests.items():
        kwargs = dict(
            retrograde=True if retro > 0 else False,
            fac_subkep=fac_subkep,
            beta_phi=beta_phi,
            beta_r=beta_r
        )
        v = Velocity('general', **kwargs)
        u0, u1, u2, u3 = v.u_lab(bhspin, radius, th=theta)
        ucon_out = np.array([np.squeeze(u0), np.squeeze(u1), np.squeeze(u2), np.squeeze(u3)])
        assert np.allclose(ucon_out, ucon, rtol=1.e-5), \
            f'Failed for retro={retro}, fac_subkep={fac_subkep}, beta_phi={beta_phi}, ' \
            f'beta_r={beta_r}, bhspin={bhspin}, radius={radius}, theta={theta}: ' \
            f'expected {ucon}, got {ucon_out}'


# analytic models with default kwargs (as in test_velocity_usq)
_MODELS = ['zamo', 'infall', 'kep', 'cunningham', 'subkep', 'cunningham_subkep', 'general', 'gelles']
_ASIGN = "asign = np.sign(a) = 0 at a = 0"


def _ucon(veltype, a, r):
    return np.array(Velocity(veltype).u_lab(a, np.array(r)), dtype=float)


def _usq(veltype, a, r):
    u = _ucon(veltype, a, r)
    return u @ _gcov_bl(a, r) @ u


@pytest.mark.parametrize("veltype", [
    pytest.param(m, marks=xfail_until("B2", _ASIGN + ": u^phi = 0 but u^t Keplerian"))
    if m in ('kep', 'cunningham') else m
    for m in _MODELS
])
def test_velocity_normalized_schwarzschild(veltype):
    """a = 0 (outside r_+ = 2, incl. inside the ISCO at r = 6): u.u = -1."""
    for r in np.linspace(2.1, 15., 30):
        assert np.isclose(_usq(veltype, 0.0, r), -1., rtol=1e-10), f"r = {r}"


@pytest.mark.parametrize("veltype", [
    pytest.param(m, marks=xfail_until("B2", _ASIGN + ": disk stops rotating at exactly a = 0"))
    if m in ('kep', 'cunningham', 'subkep', 'cunningham_subkep', 'general') else m
    for m in _MODELS
])
def test_velocity_continuous_at_zero_spin(veltype):
    """Nothing is singular at a = 0, so u^mu(a = 0) = u^mu(a = 1e-8) to O(1e-8).
    Catches models that stay normalized but silently stop rotating at a = 0."""
    for r in np.linspace(2.1, 15., 30):
        np.testing.assert_allclose(_ucon(veltype, 0.0, r), _ucon(veltype, 1e-8, r),
                                   rtol=1e-6, atol=1e-6, err_msg=f"r = {r}")


@xfail_until("B7", "a = 1 rejected by the input checks; ISCO = horizon = 1 makes the "
                   "Keplerian ell at the ISCO 0/0")
@pytest.mark.parametrize("veltype", _MODELS)
def test_velocity_normalized_extremal(veltype):
    """a = 1, outside the horizon r_+ = 1: u.u = -1."""
    for r in np.linspace(1.1, 15., 30):
        assert np.isclose(_usq(veltype, 1.0, r), -1., rtol=1e-10), f"r = {r}"


@pytest.mark.parametrize("spin, M", [
    pytest.param(0.0, 1.0, marks=xfail_until("B2", "rplusfunc returns 0 when spin == 0")),
    (0.5, 1.0), (1.0, 1.0), (1.0, 2.0),
])
def test_rplusfunc_horizon(spin, M):
    """Outer horizon r_+ = M + sqrt(M^2 - a^2) (used by the BZ paraboloid helpers)."""
    assert np.isclose(rplusfunc(spin, M), M + np.sqrt(M**2 - spin**2), rtol=1e-15)
