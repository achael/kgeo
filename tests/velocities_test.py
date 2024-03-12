import numpy as np
from kgeo.velocities import Velocity, _allowed_velocity_models


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
            if radius <= reh:
                continue
            gcov = _gcov_bl(bhspin, radius)
            for veltype in _allowed_velocity_models:
                if veltype == 'simfit':   # TODO, it appears that this model is broken?
                    continue
                print(f'Testing u.u == -1 for {veltype} with a={bhspin} at r={radius}...')
                v = Velocity(veltype)
                ucon = np.array(v.u_lab(bhspin, radius))
                ucov = np.einsum('ij,i...->j...', gcov, ucon)
                usq = np.einsum('i...,i...->...', ucon, ucov)
                assert np.isclose(usq, -1., rtol=1.e-5)


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
        ucon_out = np.squeeze(v.u_lab(bhspin, radius, th=theta))
        print(ucon_out, ucon)
        assert np.allclose(ucon_out, ucon, rtol=1.e-5)
