import numpy as np
from kgeo.velocities import Velocity


_known_velocity_models = [
    'zamo', 'infall', 'kep', 'cunningham', 'subkep', 'cunningham_subkep',
    'general', 'gelles', 'simfit', 'driftframe'
]


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
            for veltype in _known_velocity_models:
                if veltype == 'simfit':   # TODO, it appears that this model is broken?
                    continue
                print(f'Testing u.u == -1 for {veltype} with a={bhspin} at r={radius}...')
                v = Velocity(veltype)
                ucon = np.array(v.u_lab(bhspin, radius))
                ucov = np.einsum('ij,i...->j...', gcov, ucon)
                usq = np.einsum('i...,i...->...', ucon, ucov)
                assert np.isclose(usq, -1., rtol=1.e-5)
