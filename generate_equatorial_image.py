# Generate m87-esque image from equatorial emision model
import numpy as np
from kgeo.equatorial_images import make_image
import ehtim as eh

MoD = 3.77883459  # this is what was used for M/D in uas for the M87 simulations
ra = 12.51373 
dec = 12.39112 
flux230 = 0.6     # total flux
nmax = 3          # maximum subring number
npix = 512        # number of pixels
amax = 12         # maximum alpha,beta in Rg
fudge1 = 1 #2./3. # scaling factor for first subring flux
fudge = 1  #2./3. # scaling factor for all subsequent subring fluxes
whichvel = 'zamo'  # 'simfit' or 'cunningham', or 'cunningham_subkep' or 'zamo'. 
                    # Note 'simfit' may give unphysical velocities at low spin resulting in nans
                    
whichb = 'rad' # 'bz_monopole' or 'bz_guess' or 'vert' or 'rad' or 'tor'
polarization = False

th_o = 160*np.pi/180.  # inclination angle, does not work for th0=0 exactly!
spin = 0.001             # black hole spin, does not work for a=0 or a=1 exactly!
r_o = np.inf          # outer radius

label='test'
source = 'M87'



# generate the equatorial model image arrays
psize=2.*amax/npix
imagedat = make_image(spin,r_o, th_o, nmax, -amax, amax, -amax, amax, psize,nmax_only=False, 
                      whichvel=whichvel,whichb=whichb,polarization=polarization)
(outarr_I, outarr_Q, outarr_U, outarr_r, outarr_t, outarr_g, outarr_n, outarr_np) = imagedat

nanmask = np.isnan(outarr_I)
print("NaNs: ", np.sum(nanmask))
outarr_I[nanmask] = 0

# add up the subrings

imarr0 = np.flipud(outarr_I[:,0].reshape(npix,npix))
imarr1 = np.flipud(outarr_I[:,1].reshape(npix,npix))
imarrRings = np.flipud(np.sum(outarr_I[:,2:],axis=1).reshape(npix,npix))

imarr = imarr0 + fudge1*imarr1 + fudge*imarrRings

if polarization:
    imarr0_Q = np.flipud(outarr_Q[:,0].reshape(npix,npix))
    imarr1_Q = np.flipud(outarr_Q[:,1].reshape(npix,npix))
    imarrRings_Q = np.flipud(np.sum(outarr_Q[:,2:],axis=1).reshape(npix,npix))
    imarr_Q = imarr0_Q + fudge1*imarr1_Q + fudge*imarrRings_Q

    imarr0_U = np.flipud(outarr_U[:,0].reshape(npix,npix))
    imarr1_U = np.flipud(outarr_U[:,1].reshape(npix,npix))
    imarrRings_U = np.flipud(np.sum(outarr_U[:,2:],axis=1).reshape(npix,npix))
    imarr_U = imarr0_U + fudge1*imarr1_U + fudge*imarrRings_U
    
# make an Image and normalize and save
psize_rad = psize*MoD*eh.RADPERUAS
fluxscale = flux230/np.sum(imarr)
im = eh.image.Image(imarr*fluxscale, psize_rad, ra, dec)
im.source = source
if polarization:
    im.add_qu(imarr_Q*fluxscale, imarr_U*fluxscale)
    
im.save_fits('./m87_model_%s.fits'%label)

# make a image of the subring number and save
narr = np.flipud(outarr_n.reshape(npix,npix))   # number of equatorial crossings
nparr = np.flipud(outarr_np.reshape(npix,npix)) # fractional number of poloidal orbits
imn = eh.image.Image(narr, psize_rad, ra, dec)
imn.source = source
imn.save_fits('./m87_model_%s_n.fits'%label)

