import numpy as np

from kgeo.solver import *


source = 'M87'
MoD = 3.77883459  # this was used for M/D in uas for the M87 simulations
ra = 12.51373
dec = 12.39112
flux230 = 0.6     # total flux in Jy
rotation = 90 * eh.DEGREE  # rotation angle, for m87 prograde=90, retrograde=-90 (used in display only)


def makeim(ivals, qvals, uvals, agrid, saveim=None):
    npix = len(agrid)**2        # number of pixels
    amax = np.max(agrid)         # maximum alpha,beta in R
    psize = 2.*amax/len(agrid)
    psize_rad = psize*MoD*eh.RADPERUAS

    ivals_im = np.real(np.flipud(ivals))
    qvals_im = np.real(np.flipud(qvals))
    uvals_im = np.real(np.flipud(uvals))
    fluxscale = flux230/np.sum(ivals_im)

    im = eh.image.Image(ivals_im*fluxscale, psize_rad, ra, dec)
    im.add_qu(qvals_im*fluxscale, uvals_im*fluxscale)
    im.source = source

    if saveim is not None:
        im.save_fits(saveim)

    return im
