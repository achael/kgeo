import numpy as np

from kgeo.bfields import Bfield

from helpers import DATA_DIR

_MIDPLANE = np.pi/2.

# Tolerances, with the worst error measured in [brackets] (2026-10-10, after B4 1/3)
NP_DEFAULT_ATOL = 1e-8     # np.allclose's default atol, made explicit below
BZ_GUESS_RTOL = 1e-5       # 'bz_guess' Bphi/Br vs ipole                [2e-6]
FROMFILE_RTOL = 1e-4       # 'fromfile' B vs tabulated values           [2e-5]
BZ_MONOPOLE_RTOL = 1e-3    # 'bz_monopole' B vs ipole / GNW's script    [4e-4]
BZ_MONOPOLE_ATOL = 1e-15   #   for the B_theta = 0 components


def test_bfield_bz_guess():
    """
    Check whether a few known inputs produce expected output for the 'bz_guess' Bfield
    models. The tests that appear below were shown to agree with output from ipole.
    """
    tests = {
        # C, bhspin, radius, theta -> Bphi/Br
        (1, 0.5, 2.9, _MIDPLANE): -0.201073,
        (1, 0.5, 3.7, _MIDPLANE): -0.141622,
        (1, 0.5, 10., _MIDPLANE): -0.0832697,
        (1, 0.9375, 1.35, _MIDPLANE): -382.704,
        (1, 0.9375, 2.8, _MIDPLANE): -0.546836,
        (-1, 0.9375, 2.8, _MIDPLANE): -0.546836,
    }

    for (C, bhspin, radius, theta), Bfield_expected in tests.items():
        b = Bfield('bz_guess', C=C)
        Bfield_out = np.squeeze(b.bfield_lab(bhspin, radius, th=theta))
        assert np.allclose(Bfield_expected, Bfield_out[2] / Bfield_out[0], rtol=BZ_GUESS_RTOL, atol=NP_DEFAULT_ATOL)


def test_bfield_fromfile():
    """
    Check that the 'fromfile' Bfield model produces expected lab frame magnetic fields.
    """
    b = Bfield('fromfile', file=str(DATA_DIR / 'a0p6_inflow.csv'))
    tests = {
        # radius, theta -> Br, Bth, Bph
        (2.15708, _MIDPLANE): [0.84877, 0.0, -0.53419],
        (2.5, _MIDPLANE): [0.63206, 0.0, -0.23666],
        (3.58635, _MIDPLANE): [0.30706, 0.0, -0.066873],
        (4.90566, _MIDPLANE): [0.16411, 0.0, -0.029673],
    }

    for (radius, theta), Brhp in tests.items():
        Bfield_out = np.squeeze(b.bfield_lab(0.6, radius, th=theta))
        assert np.allclose(Bfield_out, Brhp, rtol=FROMFILE_RTOL, atol=NP_DEFAULT_ATOL)


def test_bfield_bz_monopole():
    """
    Check whether a few known inputs produce expected output for the 'bz_monopole'
    Bfield models. The tests that appear below were shown to agree with output from
    both ipole and GNW's standalone test script.
    """
    tests = {
        # C, bhspin, radius, theta -> Br, Btheta, Bphi
        (1, 0.1, 10., _MIDPLANE): [0.009993184914347399, 0.0, -0.0001567054809727936],
        (1, 0.1, 2.8, _MIDPLANE): [0.12737094452501346, 0.0, -0.005572446903392542],
        (1, 0.75, 2.35, _MIDPLANE): [0.1688274613110802, 0.0, -0.07926612568926913],
        (-1, 0.75, 2.35, _MIDPLANE): [-0.1688274613110802, 0.0, 0.07926612568926913],
        (1, 0.9, 1.45, _MIDPLANE): [0.3990916726028977, 0.0, -11.216482663168948]
    }

    for (C, bhspin, radius, theta), Bfield_expected in tests.items():
        b = Bfield('bz_monopole', C=C)
        Bfield_out = np.squeeze(b.bfield_lab(bhspin, radius, th=theta))
        assert np.allclose(Bfield_expected, Bfield_out, rtol=BZ_MONOPOLE_RTOL, atol=BZ_MONOPOLE_ATOL)
