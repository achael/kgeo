import numpy as np
from kgeo.bfields import Bfield
from kgeo.velocities import Velocity
from kgeo.emissivities import Emissivity
from kgeo.equatorial_images import make_image


def _get_evpa_kgeo(Q, U):
    """
    Get EVPA from Q and U using the kgeo convention.
    """
    evpa = 180./np.pi * 0.5 * np.arctan2(U, Q)
    evpa += 90.
    evpa[evpa > 90] -= 180.
    evpa[evpa > 90] -= 180.
    return evpa


def _get_kgeo_image(bhspin, inc, fac_subkep, beta_r, beta_phi, nx):
    """
    Get EVPA for kgeo images for given set of viewing and model parameters.
    """
    emissivity_model = Emissivity('ring')
    velocity_model = Velocity('general', fac_subkep=fac_subkep, beta_phi=beta_phi, beta_r=beta_r)
    bfield = Bfield('bz_guess', C=1.)

    half_fov_in_M = 10.
    pixel_size_in_M = 2 * half_fov_in_M / nx
    r_observer = np.inf
    theta_observer = inc * np.pi / 180.
    n_subrings = 2
    spectral_index = 1.

    kwargs = dict(
        nmax_only=False,
        emissivity=emissivity_model,
        velocity=velocity_model,
        bfield=bfield,
        polarization=True,
        specind=spectral_index
    )

    ploc = np.linspace(-half_fov_in_M, half_fov_in_M, nx+1)
    ploc = 0.5 * (ploc[1:] + ploc[:-1])

    image_data = make_image(bhspin, r_observer, theta_observer, n_subrings,
                      -half_fov_in_M, half_fov_in_M, -half_fov_in_M, half_fov_in_M,
                      pixel_size_in_M,
                      **kwargs)

    _, outarr_Q, outarr_U, _, _, _, _, _, _ = image_data

    evpas = []
    for i in range(2):
        Q = outarr_Q[:, i].reshape((nx, nx))
        U = outarr_U[:, i].reshape((nx, nx))
        evpas.append(_get_evpa_kgeo(Q, U))

    return ploc, *evpas


def test_lowspin_keplerian():
    """
    Check whether the EVPA for a low-spin, Keplerian disk is similar to
    output from ipole for n=0 and n=1.
    """
    n0_limits = [-10, -3, 3, 10, -10, -3, 3, 6]
    n1_limits = [-6.05, -5.25, 5.25, 6, -6, -5.1, 5.1, 6]
    ipole_ploc, n0_a, n0_b, n1_a, n1_b = np.loadtxt('tests/data/lowspin_keplerian.csv', delimiter=',', unpack=True)
    _, evpa_n0, evpa_n1 = _get_kgeo_image(0.01, 0.1, 1.0, 1.0, 1.0, 800)

    mask_n0_a = (ipole_ploc > n0_limits[0]) & (ipole_ploc < n0_limits[1])
    mask_n0_b = (ipole_ploc > n0_limits[2]) & (ipole_ploc < n0_limits[3])
    mask_n1_a = (ipole_ploc > n1_limits[0]) & (ipole_ploc < n1_limits[1])
    mask_n1_b = (ipole_ploc > n1_limits[2]) & (ipole_ploc < n1_limits[3])

    assert np.allclose(n0_a[mask_n0_a], evpa_n0[400, mask_n0_a], rtol=1.e-2)
    assert np.allclose(n0_b[mask_n0_b], evpa_n0[mask_n0_b, 400], rtol=4.e-2)
    assert np.allclose(n1_a[mask_n1_a], evpa_n1[400, mask_n1_a], rtol=1.e-2)
    assert np.allclose(n1_b[mask_n1_b], evpa_n1[mask_n1_b, 400], rtol=1.e-2)


def test_midspin_inclined_keplerian():
    """
    Check whether the EVPA for an intermediate spin, inclined Keplerian disk
    is similar to output from ipole for n=0 and n=1.
    """
    n0_limits = [-10, -4.3125, 5., 10., -10, -1.2, 4.2, 10]
    n1_limits = [-5.3, -4.4, 5.7, 6.8, -10, -5, 5.1, 5.4]
    ipole_ploc, n0_a, n0_b, n1_a, n1_b = np.loadtxt('tests/data/midspin_inclined_keplerian.csv', delimiter=',', unpack=True)
    _, evpa_n0, evpa_n1 = _get_kgeo_image(0.42, 60., 1.0, 1.0, 1.0, 800)

    mask_n0_a = (ipole_ploc > n0_limits[0]) & (ipole_ploc < n0_limits[1])
    mask_n0_b = (ipole_ploc > n0_limits[2]) & (ipole_ploc < n0_limits[3])
    mask_n1_a = (ipole_ploc > n1_limits[0]) & (ipole_ploc < n1_limits[1])
    mask_n1_b = (ipole_ploc > n1_limits[2]) & (ipole_ploc < n1_limits[3])

    assert np.allclose(n0_a[mask_n0_a], evpa_n0[400, mask_n0_a], rtol=1.e-2)
    assert np.allclose(n0_b[mask_n0_b], evpa_n0[mask_n0_b, 400], rtol=3.e-2)
    assert np.allclose(n1_a[mask_n1_a], evpa_n1[400, mask_n1_a], rtol=1.e-2)
    assert np.allclose(n1_b[mask_n1_b], evpa_n1[mask_n1_b, 400], rtol=1.e-2)


def test_midspin_inclined_subkep():
    """
    Check whether the EVPA for an intermediate spin, inclined disk with
    non-trivial sub-Keplerian motion is similar to output from ipole for
    n=0 and n=1.
    """
    n0_limits = [-10, -2.7, 4, 10, -10, -1.8, 3.8, 10]
    n1_limits = [-5., -4, 5.8, 7., -8, -4.7, 5., 5.3]
    ipole_ploc, n0_a, n0_b, n1_a, n1_b = np.loadtxt('tests/data/midspin_inclined_subkep.csv', delimiter=',', unpack=True)
    _, evpa_n0, evpa_n1 = _get_kgeo_image(0.7, 45., 0.6, 0.7, 0.8, 800)

    mask_n0_a = (ipole_ploc > n0_limits[0]) & (ipole_ploc < n0_limits[1])
    mask_n0_b = (ipole_ploc > n0_limits[2]) & (ipole_ploc < n0_limits[3])
    mask_n1_a = (ipole_ploc > n1_limits[0]) & (ipole_ploc < n1_limits[1])
    mask_n1_b = (ipole_ploc > n1_limits[2]) & (ipole_ploc < n1_limits[3])

    assert np.allclose(n0_a[mask_n0_a], evpa_n0[400, mask_n0_a], rtol=1.e-2)
    assert np.allclose(n0_b[mask_n0_b], evpa_n0[mask_n0_b, 400], rtol=2.e-2)
    assert np.allclose(n1_a[mask_n1_a], evpa_n1[400, mask_n1_a], rtol=1.e-2)
    assert np.allclose(n1_b[mask_n1_b], evpa_n1[mask_n1_b, 400], rtol=1.e-2)


if __name__ == "__main__":

    test_lowspin_keplerian()
    test_midspin_inclined_keplerian()
    test_midspin_inclined_subkep()
