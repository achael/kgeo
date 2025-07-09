# Generate m87-esque image from Zack's jet emision model (in progress)
import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt
from kgeo.off_eq import getstokes
from kgeo.bfields import Bfield
from kgeo.geometry import sort_image

# file label
label='jettest'
save_image = False
display_image = False

# source and image parameters
source = 'M87'
MBH = 6.5e9       # solar masses
rg = 147700*MBH   # cm
MoD = 3.82 #3.77883459  # this is what was used for M/D in uas for the M87 simulations
ra = 12.51373 
dec = 12.39112 
flux230 = 0.6     # total flux
npix = 128        # number of pixels
amax = 30         # maximum alpha,beta in R
f0 = 1            # scaling factor for n=0 flux
f1 = 1            # scaling factor for n=1 flux
f2 = 1            # scaling factor for n>=2 flux
nmax = 3          # maximum subring number
rotation = 0      #-90*eh.DEGREE  # rotation angle, for m87 prograde=90,retrograde=-90 (used in display only)
polarization = True      # make polarized image or not
specind = 1              # spectral index
nu_obs = 230.e9          # frequency

ngeo=500                 # number of geodesic points

# bh and observer parameters
r_o = np.inf           # outer radius
th_o = 17*np.pi/180.  # inclination angle, does not work for th0=0 exactly!
spin = 0.9             # black hole spin, does not work for a=0 or a=1 exactly!
rh = 1 + np.sqrt(1-spin*spin) # horizon radius
                 
# bfield model
bmodel = 'power'
pval = 0.75
bfield = Bfield(bmodel,p=pval, C=1)
psitarget =  rh**pval  # target field line

# velocity model (drift frame is hardcoded)
nu_parallel = 0
gammamax = None

# emissivity model (constant sigma is hardcoded)
sigma = 2
                        
################################################################################################################
# generate the equatorial model image arrays
psize = 2.*amax/npix
alphas = np.linspace(-amax, -amax+npix*psize, npix)
betas = np.linspace(-amax, -amax+npix*psize, npix)
alpha_arr, beta_arr = np.meshgrid(alphas, betas)
imshape = alpha_arr.shape

sumsubring=False
imagedat =  getstokes(psitarget, alpha_arr, beta_arr, r_o, th_o, spin, ngeo, 
                      do_phi_and_t = True, neqmax=nmax, outgeo=None, tol=1e-8, 
                      model=bmodel, pval=pval,   
                      nu_parallel=nu_parallel,  gammamax=gammamax, 
                      sigma=sigma, usemono=False, 
                      sumsubring=sumsubring, retvals=True, retsin=False)


(ivals, qvals, uvals, neqvals, rvals, thvals, guesses_shape) = imagedat

# TODO is sort_image ignoring last subring? what exactly is this doing? 
iarr, qarr, uarr = sort_image(ivals, qvals, uvals, neqvals, guesses_shape, imshape, nmax)    
    
#try to make a contour plt...
rhovals = rvals*np.sin(thvals)
zvals = rvals*np.cos(thvals)

zlevelstop = np.arange(0,200,20)
zlevelsbot = np.arange(-200,0,20)
plt.contour(alpha_arr, beta_arr, np.reshape(zvals,(guesses_shape[0],npix,npix))[0], zlevelstop, colors='k', linestyles='solid')
plt.contour(alpha_arr, beta_arr, np.reshape(zvals,(guesses_shape[0],npix,npix))[0], zlevelsbot, colors='r', linestyles='solid')

plt.contour(alpha_arr, beta_arr, np.reshape(zvals,(guesses_shape[0],npix,npix))[1], zlevelstop, colors='k', linestyles='dashed')
plt.contour(alpha_arr, beta_arr, np.reshape(zvals,(guesses_shape[0],npix,npix))[1], zlevelsbot, colors='r', linestyles='dashed')

#plt.contour(alpha_arr, beta_arr, np.reshape(zvals,(guesses_shape[0],npix,npix))[2], zlevelstop, colors='k', linestyles='solid')
#plt.contour(alpha_arr, beta_arr, np.reshape(zvals,(guesses_shape[0],npix,npix))[2], zlevelsbot, colors='r', linestyles='solid')
#plt.contour(alpha_arr, beta_arr, np.reshape(zvals,(guesses_shape[0],npix,npix))[3], zlevelstop, colors='k', linestyles='dashed')
#plt.contour(alpha_arr, beta_arr, np.reshape(zvals,(guesses_shape[0],npix,npix))[3], zlevelsbot, colors='r', linestyles='dashed')


    
   

    
    
    
    



















