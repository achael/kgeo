import numpy as np

# fitting function parameters for emissivity

# for 230 GHz
P1E_230 = -2.0
P2E_230 = -0.5
# for 86 GHz
P1E_86 = 0
P2E_86 = -.75

# default parameters for other emissivity models
GAMMAOFF = -1.5
SIGMA_GLM = 0.5
R_RING = 4.5
SIGMA_RING = 0.3


class Emissivity(object):
    """
    Parent class definition for emissivity object with member function
    jrest(a, r), which should return rest-frame emissivity as a function
    of radius.

    Parameters
    ----------
    emistype (str) : Emissivity model to use. Options are:
            'constant' - constant emissivity vs. radius
            'bpl'      - broken power law fit to GRMHD with parameters p1, p2
            'glm'      - GLM ring model with parameters sigma and gamma_off
            'ring'     - alias for 'glm' model with extra input radius r_ring

    **kwargs (dict) : keyword arguments passed to emissivity model on eval

    TODOs
    -----
    - currently only works for equatorial plane
    """
    def __init__(self, emistype="bpl", **kwargs):

        self.emistype = emistype
        self.kwargs = kwargs

        self.emiscut_in = self.kwargs.get('emiscut_in', 0)
        self.emiscut_out = self.kwargs.get('emiscut_out', 1.e10)

        if self.emistype == 'constant':
            pass
        elif self.emistype == 'bpl':
            self.p1 = self.kwargs.get('p1', P1E_230)
            self.p2 = self.kwargs.get('p2', P2E_230)
        elif self.emistype == 'ring':
            self.mu_ring = self.kwargs.get('r_ring', R_RING)
            self.gamma_off = self.kwargs.get('gamma_off', 0)
            self.sigma = self.kwargs.get('sigma', SIGMA_RING)
        elif self.emistype == 'glm':
            self.mu_ring = False
            self.gamma_off = self.kwargs.get('gamma_off', GAMMAOFF)
            self.sigma = self.kwargs.get('sigma', SIGMA_GLM)
        else:
            raise Exception(f"emistype {self.emistype} not recognized in Emissivity!")

    def jrest(self, a, r):
        if self.emistype == 'constant':
            if type(r) is np.ndarray:
                j = np.ones(r.shape)
            elif type(r) in [float, int]:
                j = 1
            elif type(r) is list:
                j = [1 for i in r]
        elif self.emistype == 'bpl':
            j = emisBPL(a, r, p1=self.p1, p2=self.p2)
        elif self.emistype == 'glm' or self.emistype == 'ring':
            j = emisGLM(a, r, gamma_off=self.gamma_off, sigma=self.sigma, mu_ring=self.mu_ring)
        else:
            raise Exception(f"emistype {self.emistype} not recognized in Emissivity.emis!")
        return j


def emisBPL(a, r, p1=P1E_230, p2=P2E_230):
    """
    Compute emissivity at radius r using broken power law model fit to GRMHD.
    """
    rh = 1 + np.sqrt(1-a**2)
    emis = np.exp(p1*np.log(r/rh) + p2*np.log(r/rh)**2)

    return emis


def emisGLM(a, r, gamma_off=GAMMAOFF, sigma=SIGMA_GLM, mu_ring=False):
    """
    Compute emissivity at radius r using GLM ring model.
    """
    if mu_ring:
        mu = mu_ring
    else:
        mu = 1 - np.sqrt(1-a**2)
    emis = np.exp(-0.5*(gamma_off+np.arcsinh((r-mu)/sigma))**2) / np.sqrt((r-mu)**2 + sigma**2)

    return emis
