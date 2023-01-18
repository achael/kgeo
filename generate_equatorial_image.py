# Generate m87-esque image from equatorial emision model
import numpy as np
from kgeo.equatorial_images import make_image
import ehtim as eh

MoD = 3.77883459  # this is what was used for M/D in uas for the M87 simulations
ra = 12.51373 
dec = 12.39112 
flux230 = 0.6     # total flux
nmax = 4          # maximum subring number
npix = 1024       # number of pixels
amax = 10         # maximum alpha,beta in Rg
fudge1 = 1 #2./3. # scaling factor for first subring flux
fudge = 1  #2./3. # scaling factor for all subsequent subring fluxes
whichvel = 'simfit' # 'simfit' or 'cunningham',ss or 'zamo'. 
                        # Note 'simfit' may give unphysical velocities at low spin resulting in nans
TH0 = 20*np.pi/180.  # inclination angle, does not work for th0=0 exactly!
A = 0.99             # black hole spin, does not work for a=0 or a=1 exactly!
R0 = np.inf          # outer radius

label='i20a99'
source = 'M87'

# generate the equatorial model image arrays
psize=2.*amax/npix
(outarr_I, outarr_r, outarr_t, outarr_g, outarr_n, outarr_np) = make_image(A,R0, TH0, nmax, -amax, amax, -amax, amax, psize,nmax_only=False,whichvel=whichvel)

nanmask = np.isnan(outarr_I)
print("NaNs: ", np.sum(nanmask))
outarr_I[nanmask] = 0

# add up the subrings
imarr0 = np.flipud(outarr_I[:,0].reshape(npix,npix))
imarr1 = np.flipud(outarr_I[:,1].reshape(npix,npix))
imarr2 = np.flipud(outarr_I[:,2].reshape(npix,npix))
imarrRings = np.flipud(np.sum(outarr_I[:,2:],axis=1).reshape(npix,npix))

imarr = imarr0 + fudge1*imarr1 + fudge*imarrRings
s
# make an Image and normalize and save
psize_rad = psize*MoD*eh.RADPERUAS
im = eh.image.Image(imarr, psize_rad, ra, dec)
im.imvec *= flux230/im.total_flux()
im.source = source
im.save_fits('./m87_model_%s.fits'%label)

# make a image of the subring number and save
narr = np.flipud(outarr_n.reshape(npix,npix))   # number of equatorial crossings
nparr = np.flipud(outarr_np.reshape(npix,npix)) # fractional number of poloidal orbits
imn = eh.image.Image(narr, psize_rad, ra, dec)
imn.source = source
imn.save_fits('./m87_model_%s_n.fits'%label)

