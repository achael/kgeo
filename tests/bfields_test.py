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
