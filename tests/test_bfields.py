import numpy as np
from kgeo.bfields import Bfield


_MIDPLANE = np.pi/2.


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
        assert np.allclose(Bfield_expected, Bfield_out[2] / Bfield_out[0], rtol=1.e-5)


def test_bfield_fromfile():
    """
    Check that the 'fromfile' Bfield model produces expected lab frame magnetic fields.
    """
    b = Bfield('fromfile', file='tests/data/a0p6_inflow.csv')
    tests = {
        # radius, theta -> Br, Bth, Bph
        (2.15708, _MIDPLANE): [0.84877, 0.0, -0.53419],
        (2.5, _MIDPLANE): [0.63206, 0.0, -0.23666],
        (3.58635, _MIDPLANE): [0.30706, 0.0, -0.066873],
        (4.90566, _MIDPLANE): [0.16411, 0.0, -0.029673],
    }

    for (radius, theta), Brhp in tests.items():
        Bfield_out = np.squeeze(b.bfield_lab(0.6, radius, th=theta))
        assert np.allclose(Bfield_out, Brhp, rtol=1.e-4)


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
        assert np.allclose(Bfield_expected, Bfield_out, rtol=1.e-3, atol=1.e-15)


if __name__ == "__main__":

    test_bfield_bz_guess()
    test_bfield_fromfile()
    test_bfield_bz_monopole()
