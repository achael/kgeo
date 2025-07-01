import numpy as np
from kgeo.emissivities import Emissivity


def test_emissivity():
    """
    Verify that the elements of the Emisivity class are working as expected.

    There's nothing particularly deep going on here but we can at least
    use this test to loudly complain if something changes when we don't
    expect it to.
    """
    glm_kwargs = dict(gamma_off=-2, sigma=1.)
    ring_kwargs = dict(r_ring=4, gamma_off=0.1, sigma=0.6)

    tests = {
        ('constant', 0, 1): (dict(), 1),
        ('bpl', 0.3, 8): (dict(), 0.022089),
        ('bpl', 0.3, 25): (dict(), 0.00023715),
        ('bpl', 0.4, 3): (dict(), 0.36913),
        ('bpl', 0.5, 10): (dict(), 0.0085085),
        ('bpl', 0.9, 3.141593): (dict(), 0.15376),
        ('bpl', 0.3679, 3.141593): (dict(), 0.33512),
        ('glm', 0.3679, 3.141593): (dict(), 0.19198),
        ('glm', 0.5, 5): (dict(), 0.069270),
        ('glm', 0.3, 5): (dict(), 0.066282),
        ('glm', 0.83, 9): (dict(), 0.014741),
        ('glm', 0.3679, 3.1415927): (glm_kwargs, 0.30569),
        ('glm', 0.4, 5): (glm_kwargs, 0.19078),
        ('ring', 0.3679, 3.141593): (dict(), 0.061782),
        ('ring', 0.4, 5): (dict(), 0.75227),
        ('ring', 0.83, 9): (dict(), 0.00067951),
        ('ring', 0.5, 2.4): (ring_kwargs, 0.16079),
        ('ring', 0.318, 7): (ring_kwargs, 0.017807),
        ('ring', 0.1, 5.): (ring_kwargs, 0.32917),
    }

    for key, value in tests.items():
        model, bhspin, radius = key
        kwargs, answer = value
        emissivity = Emissivity(model, **kwargs)
        radius = np.array(radius)
        assert np.isclose(emissivity.jrest(bhspin, radius), answer, rtol=1e-4)


if __name__ == "__main__":

    test_emissivity()
