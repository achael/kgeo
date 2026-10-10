import numpy as np
import pytest

from kgeo.equatorial_images import Iobs, make_image, radial_momentum_sign, theta_momentum_sign
from kgeo.equatorial_lensing import critical_curve, r_equatorial
from kgeo.kerr_raytracing_ana import coords_at_tau
from kgeo.kerr_raytracing_utils import is_outside_crit

from helpers import quiet, xfail_until

R_O = 1000.

# Tolerances, with the worst error measured in [brackets] (2026-10-10, after B4 1/3);
# [target] where only strict xfails use them
ON_EQUATOR_ATOL = 1e-13        # cos(theta) at the crossing time Ir                       [9e-15]
CROSSING_R_RTOL = 1e-12        # r at Ir vs r_equatorial's r_s                            [0]
MIRROR_RTOL = 1e-12            # a = 0 image: I(alpha, beta) = I(-alpha, beta)            [0]
IMAGE_SYMMETRY_RTOL = 1e-10    # equatorial reflection of the image (incl. beta = 0 row)  [2e-12]
                               # (th_o = 0 axisymmetry: [target, B5])
EDGE_ON_LIMIT_RTOL = 1e-7      # th_o = pi/2 image vs th_o = pi/2 - 1e-11                 [target, B6]

# a coarse grid plus a dense horizontal cut, which crosses the thin mbar = 2 band
# near the critical curve. alpha includes 0 (lam = 0 is harmless here: no G_phi
# needed); beta avoids 0, which is tested separately (observer at a turning point)
_ALPHA, _BETA = np.meshgrid(np.linspace(-8, 8, 9), np.linspace(-8, 8, 8))
_CUT = np.linspace(-8, 8, 4001)
ALPHA = np.concatenate([_ALPHA.ravel(), _CUT])
BETA = np.concatenate([_BETA.ravel(), np.full_like(_CUT, 0.7)])


@pytest.mark.parametrize("a", [0.3, 0.94])
@pytest.mark.parametrize("th_o", [0.3, 1.2, 2.2])
@pytest.mark.parametrize("mbar", [0, 1, 2])
def test_r_equatorial_lands_on_equator(a, th_o, mbar):
    """At the Mino time Ir of the mbar-th equatorial crossing (GL19b eq 81),
    the full geodesic is on the equator, at the radius r_s that r_equatorial returns."""
    _assert_crossings_on_equator(a, th_o, mbar, ALPHA, BETA)


@pytest.mark.parametrize("a", [0.3, 0.94])
@pytest.mark.parametrize("th_o", [0.3, 1.2])
def test_r_equatorial_beta_zero_row(a, th_o):
    """The same on the beta = 0 row, where F0 = +-K exactly (observer at a turning point;
    before B4 the crossings missed the equator by 1e-8..3e-7)."""
    alpha = np.linspace(-8, 8, 9)
    _assert_crossings_on_equator(a, th_o, 0, alpha, np.zeros_like(alpha))


def _assert_crossings_on_equator(a, th_o, mbar, alpha, beta):
    r_s, Ir, Imax, Nmax = r_equatorial(a, R_O, th_o, mbar, alpha, beta)
    hit = (Nmax >= mbar) & (r_s > 1 + np.sqrt(1 - a**2))
    assert hit.sum() > 3   # the test isn't vacuous

    with quiet():
        _, (t, r, th, ph) = coords_at_tau(a, [0, R_O, th_o, 0], [alpha[hit], beta[hit]],
                                          Ir[hit].reshape(1, -1), do_phi_and_t=False)
    np.testing.assert_allclose(np.cos(th[0]), 0, atol=ON_EQUATOR_ATOL)
    np.testing.assert_allclose(r[0], r_s[hit], rtol=CROSSING_R_RTOL)


@pytest.mark.parametrize("a", [0.3, 0.94])
@pytest.mark.parametrize("th_o", [0.3, 1.2, 2.2])
def test_nmax_counts_crossings(a, th_o):
    """Crossings m = 0..Nmax exist (Nmax = -1: none), so Nmax + 1 equals the number
    of sign changes of cos(theta) along the geodesic out to the horizon or infinity."""
    alpha = np.concatenate([ALPHA[:72], _CUT[::20]])
    beta = np.concatenate([BETA[:72], np.full_like(_CUT[::20], 0.7)])
    r_s, Ir, Imax, Nmax = r_equatorial(a, R_O, th_o, 0, alpha, beta)
    regular = Nmax >= -1          # Nmax = -2 marks vortical rays
    assert np.any(Nmax[regular] >= 2)

    tau = np.outer(np.linspace(0, 1 - 1e-10, 2000), Imax[regular])
    with quiet():
        _, (t, r, th, ph) = coords_at_tau(a, [0, R_O, th_o, 0], [alpha[regular], beta[regular]],
                                          tau, do_phi_and_t=False)
    crossings = np.sum(np.diff(np.sign(np.cos(th)), axis=0) != 0, axis=0)
    np.testing.assert_array_equal(crossings, Nmax[regular] + 1)


@pytest.mark.parametrize("a", [0.3, 0.94])
@pytest.mark.parametrize("th_o", [
    0.3, 1.2, 2.2,
    pytest.param(np.pi/2, marks=xfail_until(
        "B6", "th_o = pi/2 rejected; theta_momentum_sign has no branch for pi/2")),
])
@pytest.mark.parametrize("mbar", [0, 1, 2])
def test_momentum_signs(a, th_o, mbar):
    """Forward-time signs of k^r and k_theta at the mbar-th crossing match the geodesic:
    kgeo's tau runs backward, so sign(k^r) = -sign(dr/dtau), sign(k_theta) = -sign(dtheta/dtau)."""
    r_s, Ir, Imax, Nmax = r_equatorial(a, R_O, th_o, mbar, ALPHA, BETA)
    hit = (Nmax >= mbar) & (r_s > 1 + np.sqrt(1 - a**2))
    alpha, beta, Ir, Imax = ALPHA[hit], BETA[hit], Ir[hit], Imax[hit]

    h = 1e-7*Imax
    tau = np.array([Ir - h, Ir + h])
    with quiet():
        _, (t, r, th, ph) = coords_at_tau(a, [0, R_O, th_o, 0], [alpha, beta], tau, do_phi_and_t=False)
    dr, dth = r[1] - r[0], th[1] - th[0]
    clear = np.abs(dr) > 1e-12*r_s[hit]      # not at a radial turning point

    kr_sign = radial_momentum_sign(a, th_o, alpha, beta, Ir, Imax)
    kth_sign = theta_momentum_sign(th_o, mbar)
    np.testing.assert_array_equal(kr_sign[clear], -np.sign(dr[clear]))
    np.testing.assert_array_equal(np.full(len(dth), kth_sign), -np.sign(dth))


_EMPTY_CURVE = "critical_curve divides by a and sin(th_o): silently returns an empty curve"


@pytest.mark.parametrize("a, th_o", [
    (0.3, 0.3), (0.3, 1.2), (0.3, 2.2), (0.94, 0.3), (0.94, 1.2), (0.94, 2.2),
    (0.94, np.pi/2),
    pytest.param(0.0, 0.3, marks=xfail_until("B5", _EMPTY_CURVE)),
    pytest.param(0.0, 1.2, marks=xfail_until("B5", _EMPTY_CURVE)),
    pytest.param(0.5, 0.0, marks=xfail_until("B5", _EMPTY_CURVE)),
    pytest.param(0.5, np.pi, marks=xfail_until("B5", _EMPTY_CURVE)),
    pytest.param(1.0, 1.2, marks=xfail_until("B7", "a = 1 rejected by is_outside_crit")),
])
def test_critical_curve_separates_inside_outside(a, th_o):
    """Points just inside/outside the critical curve (scaled about its centroid by
    1 -+ 1e-3) are classified accordingly by is_outside_crit (radial roots)."""
    alpha, beta = critical_curve(a, th_o, n=2000)
    keep = np.abs(beta) > 1e-2      # the parametrization degrades near beta = 0
    alpha, beta = alpha[keep], beta[keep]
    assert len(alpha) > 100
    ca, cb = alpha.mean(), beta.mean()
    for scale, outside in [(1 - 1e-3, 0), (1 + 1e-3, 1)]:
        flags = is_outside_crit(a, th_o, ca + scale*(alpha - ca), cb + scale*(beta - cb))
        assert np.all(flags == outside)


@pytest.mark.parametrize("a, th_o", [
    (0.5, 1.0),
    (0.0, 1.0),
    pytest.param(1.0, 1.0, marks=xfail_until("B7", "a = 1 rejected by the input checks")),
    pytest.param(0.5, 0.0, marks=xfail_until("B5", "th_o = 0 rejected by the input checks")),
    pytest.param(0.5, np.pi/2, marks=xfail_until("B6", "th_o = pi/2 rejected by the input checks")),
    pytest.param(0.5, np.pi, marks=xfail_until("B5", "th_o = pi rejected by the input checks")),
])
def test_make_image_smoke(a, th_o):
    """A small unpolarized image (default bpl emissivity, ZAMO, radial field) is
    finite and non-negative, with some emission."""
    with quiet():
        out = make_image(a, R_O, th_o, 1, -8., 8., -8., 8., 0.8)
    intensity = out[0]                                   # (npix, mbar_max + 1)
    assert intensity.shape == (400, 2)
    assert np.all(np.isfinite(intensity))
    assert np.all(intensity >= 0)
    assert np.any(intensity[:, 0] > 0)


# symmetric pixel grid (make_image's own grid is not symmetric about 0)
_SYM = np.linspace(-7, 7, 15)
SYM_ALPHA, SYM_BETA = (x.ravel() for x in np.meshgrid(_SYM, _SYM))


def _intensity(a, th_o, mbar, alpha=SYM_ALPHA, beta=SYM_BETA):
    """Unpolarized Iobs with the defaults (bpl emissivity, ZAMO, radial field)"""
    with quiet():
        return Iobs(a, R_O, th_o, mbar, alpha, beta)[0]


@pytest.mark.parametrize("th_o", [0.3, 1.0, 2.2])
@pytest.mark.parametrize("mbar", [0, 1])   # the mbar = 2 ring is too thin for this grid
def test_image_schwarzschild_left_right(th_o, mbar):
    """a = 0, static (ZAMO) fluid: reflection through the plane of the line of sight
    and the axis (lam -> -lam) gives I(alpha, beta) = I(-alpha, beta)."""
    image = _intensity(0.0, th_o, mbar)
    mirror = _intensity(0.0, th_o, mbar, -SYM_ALPHA, SYM_BETA)
    assert np.any(image > 0)
    np.testing.assert_allclose(image, mirror, rtol=MIRROR_RTOL, atol=0)


@pytest.mark.parametrize("a", [0.5, 0.94])
@pytest.mark.parametrize("th_o", [0.3, 1.0])
@pytest.mark.parametrize("mbar", [0, 1])
def test_image_equatorial_reflection(a, th_o, mbar):
    """Disk, ZAMO fluid and unpolarized emission are symmetric under theta -> pi - theta,
    so I(alpha, beta; th_o) = I(alpha, -beta; pi - th_o). Checks Iobs's th_o > pi/2
    branches (r_equatorial, theta_momentum_sign, redshift). The beta = 0 row is
    tested separately."""
    off_row = SYM_BETA != 0
    alpha, beta = SYM_ALPHA[off_row], SYM_BETA[off_row]
    north = _intensity(a, th_o, mbar, alpha, beta)
    south = _intensity(a, np.pi - th_o, mbar, alpha, -beta)
    assert np.any(north > 0)
    np.testing.assert_allclose(south, north, rtol=IMAGE_SYMMETRY_RTOL, atol=0)


@xfail_until("B5", "th_o = 0 rejected by the input checks")
@pytest.mark.parametrize("a", [0.0, 0.5, 0.94])
@pytest.mark.parametrize("mbar", [0, 1])
def test_image_face_on_axisymmetric(a, mbar):
    """th_o = 0: disk, fluid (incl. frame dragging) and observer are axisymmetric,
    so rotating the screen by 90 degrees leaves the image unchanged."""
    image = _intensity(a, 0.0, mbar)
    rotated = _intensity(a, 0.0, mbar, -SYM_BETA, SYM_ALPHA)
    assert np.any(image > 0)
    np.testing.assert_allclose(rotated, image, rtol=IMAGE_SYMMETRY_RTOL, atol=0)


@xfail_until("B6", "th_o = pi/2 rejected by the input checks")
@pytest.mark.parametrize("a", [0.0, 0.5, 0.94])
def test_image_edge_on_limit(a):
    """th_o = pi/2, mbar = 0, defined as the th_o -> pi/2- limit with the tau = 0
    crossing (the observer itself) excluded. Below pi/2, beta > 0 rays cross first
    near the hole (label 0), while beta < 0 rays first cross at the observer (label 0,
    far away, no emission) and near the hole at label 1. So at pi/2:
    I(beta > 0, 0) = lim I(beta > 0, 0),  I(beta < 0, 0) = lim I(beta < 0, 1),
    and the beta = 0 row (rays in the disk plane) is dark. The limit is linear in
    eps (~300 eps relative); eps = 1e-11 serves as the reference."""
    below = np.pi/2 - 1e-11
    up, down = SYM_BETA > 0, SYM_BETA < 0
    alpha_up, beta_up = SYM_ALPHA[up], SYM_BETA[up]
    alpha_dn, beta_dn = SYM_ALPHA[down], SYM_BETA[down]

    edge_up = _intensity(a, np.pi/2, 0, alpha_up, beta_up)
    edge_dn = _intensity(a, np.pi/2, 0, alpha_dn, beta_dn)
    assert np.any(edge_up > 0)
    np.testing.assert_allclose(edge_up, _intensity(a, below, 0, alpha_up, beta_up), rtol=EDGE_ON_LIMIT_RTOL, atol=0)
    np.testing.assert_allclose(edge_dn, _intensity(a, below, 1, alpha_dn, beta_dn), rtol=EDGE_ON_LIMIT_RTOL, atol=0)

    row = SYM_BETA == 0
    assert np.all(_intensity(a, np.pi/2, 0, SYM_ALPHA[row], SYM_BETA[row]) == 0)


@pytest.mark.parametrize("a", [0.5, 0.94])
@pytest.mark.parametrize("mbar", [0, 1])
def test_image_equatorial_reflection_beta_zero_row(a, mbar):
    th_o = 1.0
    alpha = _SYM
    beta = np.zeros_like(alpha)
    north = _intensity(a, th_o, mbar, alpha, beta)
    south = _intensity(a, np.pi - th_o, mbar, alpha, -beta)
    assert np.any(north > 0)
    np.testing.assert_allclose(south, north, rtol=IMAGE_SYMMETRY_RTOL, atol=0)
