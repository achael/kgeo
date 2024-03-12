import numpy as np
import kgeo


def _interp_signed(x0, x, z):
    """
    Return value of z where x == x0.
    """
    if x[0] > x[1]:
        x, z = x[::-1], z[::-1]
    return np.interp(x0, x, z)


def _get_txyz_coords(bhspin, inc, alpha, beta):
    """
    Return t, x, y, z coordinate arrays for geodesic.
    """
    kwargs = dict(a=bhspin,
                  observer_coords=[0, 1000., inc/180.*np.pi, 0],
                  ngeo=10000,
                  image_coords=[alpha, beta],
                  plotdata=False)

    kgeo_g = kgeo.kgeo.kerr_raytracing_ana.raytrace_ana(**kwargs)
    kgeo_g = np.array(kgeo_g.geo_coords)[:, :, 0]

    kgeo_t = kgeo_g[0]
    kgeo_r = kgeo_g[1]
    kgeo_h = kgeo_g[2]
    kgeo_p = kgeo_g[3]

    kgeo_x = kgeo_r * np.sin(kgeo_h) * np.cos(kgeo_p)
    kgeo_y = kgeo_r * np.sin(kgeo_h) * np.sin(kgeo_p)
    kgeo_z = kgeo_r * np.cos(kgeo_h)

    return kgeo_t, kgeo_x, kgeo_y, kgeo_z


def _get_midplane_crossing(bhspin, inc, alpha, beta):
    """
    Return x, y coordinates of the first midplane crossing.
    """
    t, x, y, z = _get_txyz_coords(bhspin, inc, alpha, beta)
    return _interp_signed(0, z, x), _interp_signed(0, z, y)


def test_midplane_intersections():
    """
    Check whether the geodesic crosses the midplane where it should.

    The numbers included in the tests below agree with ipole's output
    and the output of kgeo as of the writing of this test.
    """
    tests = {
        (0.42, 60., -17., 0.): (-0.0522336, -16.0342),
        (0.42, 60., -9., 0.): (-0.103693, -8.09784),
        (0.42, 60., -4., 0.): (-0.308159, -3.20445),
        (0.42, 60., 1., 0.): (-0.327624, 1.67648),
        (0.42, 60., 6., 0.): (877.845, -13.6213),
        (0.42, 60., 8., 0.): (0.129720, 6.94829),
        (0.9, 10., -12., 0.): (-0.164094, -11.0082),
        (0.9, 10., -4., 0.): (-0.655582, -2.99558),
        (0.9, 10., 1., 0.): (-0.518830, -0.632352),
        (0.9, 10., 2., 0.): (-0.691096, -1.15767),
        (0.73, 35., 0.0001, -10): (11.8258, -0.0858775),
        (0.73, 35., 0.0001, -1.): (1.02201, 1.13478),
        (0.73, 35., 0.0001, 1.): (0.0148225, 0.0165869),
        (0.73, 35., 0.0001, 3.): (-1.20034, -1.08604),
        (0.73, 35., 0.0001, 4.): (-2.11221, 1.02322),
    }

    for key, value in tests.items():
        bhspin, inc, alpha, beta = key
        x0, y0 = _get_midplane_crossing(bhspin, inc, alpha, beta)
        assert np.allclose([x0, y0], value, atol=1e-5)
