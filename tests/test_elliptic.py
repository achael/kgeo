import itertools

import mpmath as mp
import numpy as np
import pytest
import scipy.special as sp

from kgeo.kerr_raytracing_ana import R1_diff
from kgeo.scipy_ellip_binding import ellip_pi_arr, ellip_pi_scaled, ellipj_m1

# |phi| > pi/2 and > pi exercise the periodicity reductions; tiny phi the phi->0 limit
_PHIS = [-215., -14.3, -7.7, -4.1, -np.pi, -2.37, -np.pi/2, -0.1, 1e-11,
         0.026, 1.2, np.pi/2, 2.37, np.pi, 4.1, 6., 14.3, 215.]

# Tolerances, with the worst error measured in [brackets] (2026-10-10, after B4 1/3)
PI_RTOL = 1e-12            # ellip_pi_arr vs mpmath                                       [1e-14]
PI_SPECIAL_RTOL = 1e-13    # Pi(0; phi|m) = F(phi|m), Pi(0; phi|0) = phi                  [4e-16]
PI_SCALED_ATOL = 1e-12     # ellip_pi_scaled = C Pi(n; phi|m) vs mpmath, |phi| <= 20      [1.4e-13]
R1_DIFF_TOL = 1e-13        # R1_diff vs quadrature, relative to max(|R1 diff|, 1)          [6e-15]
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


def test_ellip_pi_tiny_phi_huge_m():
    """Where ellip_pi_arr used to cut off (|phi| <= 1e-10 -> phi, m < -1e14 -> 0; the true value ~ log|m|/sqrt|m|,
    reached near eta = 0, where m = u_+/u_- ~ -a^2 u_+^2/eta), the Carlson form is accurate; phi = 0 and
    m = -inf are the exact limits 0."""
    cases = [(0.5, 1.2, -1.01e14), (0.5, 1.2, -1e16), (0.5, -1.2, -1e30), (-3e13, 1.2, -1e16), (-2e15, 0.3, -2e15),
             (0.5, 4.1, -1e16), (-1e14, 1e-11, -1e13), (0.9, 1e-12, 0.9), (-50., -1e-11, -1e6)]
    n, phi, m = (np.array(x) for x in zip(*cases))
    expected = np.array([_mp_ellippi(*c) for c in cases])
    np.testing.assert_allclose(ellip_pi_arr(n, phi, m), expected, rtol=PI_RTOL, atol=0)
    assert np.all(ellip_pi_arr(np.array([0.5, -1e14]), np.zeros(2), np.array([-1e16, 0.3])) == 0)
    assert np.all(ellip_pi_arr(np.array([0.5, -3.]), np.array([1.2, 4.1]), np.full(2, -np.inf)) == 0)


def test_ellip_pi_special_values():
    phi = np.array([-7.7, -1.2, 0.3, 2.37, 14.3])
    m = np.array([-5., 0.5, -0.33, 0.9, 0.])
    # Pi(0; phi|m) = F(phi|m),  Pi(0; phi|0) = phi,  Pi(n; 0|m) = 0
    np.testing.assert_allclose(ellip_pi_arr(np.zeros(5), phi, m), sp.ellipkinc(phi, m), rtol=PI_SPECIAL_RTOL)
    np.testing.assert_allclose(ellip_pi_arr(np.zeros(5), phi, np.zeros(5)), phi, rtol=PI_SPECIAL_RTOL)
    assert np.all(ellip_pi_arr(np.array([0.3, -2., 0.8]), np.zeros(3), np.array([0.5, 0., -5.])) == 0)


@pytest.mark.parametrize("one_minus_n", [1e-1, 1e-3, 1e-6, 1e-9, 1e-12])
def test_ellip_pi_scaled_matches_mpmath(one_minus_n):
    """c Pi(n; phi|m) for c = +-C, C = sqrt((1-n)(n-m)/n), in the circular case m < 0 < n < 1,
    as n -> 1 (Pi itself grows like 1/C across each phi = pi/2 + j pi), any phi, both signs of c."""
    rng = np.random.default_rng(int(-np.log10(one_minus_n)))
    n = 1 - one_minus_n*rng.uniform(1, 9, 20)
    m = -rng.uniform(0, 50, 20)
    phi = rng.uniform(-20, 20, 20)
    sgn = rng.choice([-1., 1.], 20)
    with mp.workdps(40):
        C = [mp.sqrt((1 - mp.mpf(ni))*(mp.mpf(ni) - mp.mpf(mi))/mp.mpf(ni)) for ni, mi in zip(n, m)]
        expected = [float(s*Ci*mp.re(mp.ellippi(mp.mpf(ni), mp.mpf(p), mp.mpf(mi))))
                    for s, Ci, ni, p, mi in zip(sgn, C, n, phi, m)]
    c = sgn*np.array([float(Ci) for Ci in C])
    np.testing.assert_allclose(ellip_pi_scaled(c, m/n, phi, m), expected, rtol=0, atol=PI_SCALED_ATOL)


@pytest.mark.parametrize("al", [-1.5, -1 - 1e-8, -1 - 1e-13, -1., -1 + 1e-13, -1 + 1e-8, -0.5,
                                0.5, 1 - 1e-10, 1 + 1e-10, 1.5, -40., 40.])
def test_R1_diff_matches_quadrature(al):
    """R1(al; phi_a|j) - R1(al; phi_b|j) = int_{phi_b}^{phi_a} dt/((1 + al cos t) sqrt(1 - j sin^2 t)) (GL19a B61),
    through al = -1 (where each R1 diverges, and B62 cancels) and al^2 < 1 (where B62 is NaN), endpoints on one
    side of the pole cos t = -1/al; 2-D phi_a with 1-D phi_b, al, j as in r_integrate."""
    rng = np.random.default_rng(int(1e3*abs(al)) % 2**32)
    j = rng.uniform(0.05, 0.95, 3)
    lo = np.arccos(1/abs(al)) + 0.05 if al < -1 else 0.05
    hi = np.pi - np.arccos(1/al) - 0.05 if al > 1 else np.pi - 0.05
    phi_b = rng.uniform(lo, hi, 3)
    phi_a = rng.uniform(lo, hi, (4, 3))
    with mp.workdps(30):
        expected = np.array([[float(mp.quad(lambda t: 1/((1 + al*mp.cos(t))*mp.sqrt(1 - j[k]*mp.sin(t)**2)),
                                            [phi_b[k], phi_a[i, k]])) for k in range(3)] for i in range(4)])
    out = R1_diff(np.full(3, al), phi_a, phi_b, j)
    np.testing.assert_allclose((out - expected)/np.maximum(np.abs(expected), 1), 0, rtol=0, atol=R1_DIFF_TOL)


def test_ellip_pi_scaled_c_zero():
    """c = 0 (n = 1): only the pi steps at phi = pi/2 + j pi remain"""
    phi = np.array([0.3, 1.6, -1.6, 4.0, 7.9])
    out = ellip_pi_scaled(np.zeros(5), np.full(5, -3.), phi, np.full(5, -3.))
    np.testing.assert_array_equal(out, np.pi*np.array([0., 1., -1., 1., 3.]))


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
