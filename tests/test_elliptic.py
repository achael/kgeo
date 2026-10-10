import itertools

import mpmath as mp
import numpy as np
import pytest
import scipy.special as sp

from kgeo.scipy_ellip_binding import ellip_pi, ellip_pi_arr, ellipj_m1

# |phi| > pi/2 and > pi exercise the periodicity reductions; tiny phi the phi->0 limit
_PHIS = [-215., -14.3, -7.7, -4.1, -np.pi, -2.37, -np.pi/2, -0.1, 1e-11,
         0.026, 1.2, np.pi/2, 2.37, np.pi, 4.1, 6., 14.3, 215.]

# Tolerances, with the worst error measured in [brackets] (2026-10-10, after B4 1/3)
PI_RTOL = 1e-12            # ellip_pi_arr vs mpmath                                       [1e-14]
PI_SCALAR_RTOL = 1e-14     # ellip_pi (scalar) vs ellip_pi_arr                            [0]
PI_SPECIAL_RTOL = 1e-13    # Pi(0; phi|m) = F(phi|m), Pi(0; phi|0) = phi                  [4e-16]
ELLIPJ_ATOL = 1e-13        # ellipj_m1 vs scipy away from m = 1                           [5e-14]
ELLIPJ_M1_TOL = 1e-13      # ellipj_m1 vs mpmath near m = 1: sn, dn relative; cn, am abs  [8e-15]
GUDERMANNIAN_ATOL = 1e-14  # m = 1: am = arctan(sinh u)                                   [2e-16]


def _mp_ellippi(n, phi, m):
    return float(mp.re(mp.ellippi(n, phi, m)))


def test_ellip_pi_matches_mpmath():
    """Pi(n; phi|m) for n < 1, m < 1 (the real domain for any phi) against mpmath."""
    ns = [-50., -2., -0.5, 0., 0.3, 0.8, 0.95, 0.999]
    ms = [-1e6, -5., -0.33, 0., 0.5, 0.9, 0.999]
    args = np.array(list(itertools.product(ns, _PHIS, ms)))
    n, phi, m = args.T
    expected = np.array([_mp_ellippi(*arg) for arg in args])
    np.testing.assert_allclose(ellip_pi_arr(n, phi, m), expected, rtol=PI_RTOL, atol=0)


def test_ellip_pi_n_above_one_is_principal_value():
    """For n sin^2(phi) > 1 (rho < 0 branch) the result is the Cauchy principal value,
    i.e. the real part of the analytic continuation."""
    args = [(1.5, 1.2, 0.5), (4., 0.8, -3.), (2., 1.5, 0.3), (1.2, 1.4, -0.33), (3., 0.7, 0.9)]
    n, phi, m = np.array(args).T
    assert np.all(n*np.sin(phi)**2 > 1)
    expected = np.array([_mp_ellippi(*arg) for arg in args])
    np.testing.assert_allclose(ellip_pi_arr(n, phi, m), expected, rtol=PI_RTOL, atol=0)


def test_ellip_pi_broadcasting():
    """1-D parameters n, m broadcast against a 2-D phi along the last axis,
    as in kgeo's (ntau, npix) calls."""
    n = np.array([0.3, -2., 0.8])
    m = np.array([0.5, -5., 0.])
    phi = np.array([[0.1, 2.37, -7.7], [4.1, -0.026, 14.3]])
    out = ellip_pi_arr(n, phi, m)
    assert out.shape == phi.shape
    expected = [[_mp_ellippi(n[j], phi[i, j], m[j]) for j in range(3)] for i in range(2)]
    np.testing.assert_allclose(out, expected, rtol=PI_RTOL, atol=0)


def test_ellip_pi_scalar_matches_array():
    for n, phi, m in [(0.8, 4.1, -0.33), (-2., -14.3, 0.5), (0.3, 1.2, 0.9), (0.95, -2.37, -5.)]:
        assert np.isclose(ellip_pi(n, phi, m), ellip_pi_arr(n, phi, m)[0], rtol=PI_SCALAR_RTOL, atol=0)


def test_ellip_pi_special_values():
    phi = np.array([-7.7, -1.2, 0.3, 2.37, 14.3])
    m = np.array([-5., 0.5, -0.33, 0.9, 0.])
    # Pi(0; phi|m) = F(phi|m),  Pi(0; phi|0) = phi,  Pi(n; 0|m) = 0
    np.testing.assert_allclose(ellip_pi_arr(np.zeros(5), phi, m), sp.ellipkinc(phi, m), rtol=PI_SPECIAL_RTOL)
    np.testing.assert_allclose(ellip_pi_arr(np.zeros(5), phi, np.zeros(5)), phi, rtol=PI_SPECIAL_RTOL)
    assert np.all(ellip_pi_arr(np.array([0.3, -2., 0.8]), np.zeros(3), np.array([0.5, 0., -5.])) == 0)


@pytest.mark.parametrize("m", [0., 0.3, 0.9, 0.999])
def test_ellipj_m1_matches_scipy(m):
    """Away from m = 1, where scipy.special.ellipj is accurate: same sn, cn, dn, am,
    including u = 0, negative u and many periods."""
    u = np.concatenate([np.linspace(-40, 40, 2001), [0.]])
    for new, ref in zip(ellipj_m1(u, 1 - m), sp.ellipj(u, m)):
        np.testing.assert_allclose(new, ref, rtol=0, atol=ELLIPJ_ATOL)


@pytest.mark.parametrize("m1", [1e-6, 1e-9, 1e-13, 1e-16])
def test_ellipj_m1_near_m_one(m1):
    """Near m = 1, where scipy.special.ellipj loses dn at |u| ~ K: sn, dn (relative) and
    cn (absolute; it vanishes at K) against mpmath; am via (sin am, cos am) = (sn, cn).
    (F(am|m) = u would be ill-conditioned: dF/dphi = 1/dn ~ 1/sqrt(m1) near K.)"""
    with mp.workdps(40):
        m = mp.mpf(1) - mp.mpf(m1)
        K = float(mp.ellipk(m))
        u = np.array([0.3, 0.5*K, 0.9*K, K, 1.7*K, 3.3*K, -2.2*K])
        sn, cn, dn, am = ellipj_m1(u, m1)
        for i, ui in enumerate(u):
            sn_ref, cn_ref = float(mp.ellipfun('sn', ui, m=m)), float(mp.ellipfun('cn', ui, m=m))
            assert abs(sn[i]/sn_ref - 1) < ELLIPJ_M1_TOL
            assert abs(cn[i] - cn_ref) < ELLIPJ_M1_TOL
            assert abs(dn[i]/float(mp.ellipfun('dn', ui, m=m)) - 1) < ELLIPJ_M1_TOL
            assert abs(np.sin(am[i]) - sn_ref) < ELLIPJ_M1_TOL and abs(np.cos(am[i]) - cn_ref) < ELLIPJ_M1_TOL


@pytest.mark.parametrize("m1", [0.5, 1e-6, 1e-13, 0.])
def test_ellipj_m1_amplitude_branch(m1):
    """am is continuous and non-decreasing (dam/du = dn > 0; it saturates at +-pi/2 for
    m = 1) across many periods, so the 2 pi branch is right (a wrong branch is a 2 pi
    jump); at m = 1 (m1 = 0), am = gd(u) = arctan(sinh u)."""
    u = np.linspace(-60, 60, 200001)
    am = ellipj_m1(u, m1)[3]
    assert np.all(np.diff(am) >= 0)
    assert np.all(np.diff(am) < 1)
    if m1 == 0:
        np.testing.assert_allclose(am, np.arctan(np.sinh(u)), rtol=0, atol=GUDERMANNIAN_ATOL)
