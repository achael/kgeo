# Make images of subring order for M87 type models:
# equatorial crossings (nmax_equatorial), poloidal orbits (nmax_poloidal) and polar turning points (n_angular_turnings)
import os

import ehtim as eh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from kgeo.equatorial_lensing import nmax_equatorial, nmax_poloidal
from kgeo.kerr_raytracing_utils import mino_total, n_angular_turnings, obs_trig, radial_roots

##general preamble
matplotlib.rcdefaults()
matplotlib.rc('font',**{'family':'serif','size':16})
os.environ['PATH'] = os.environ['PATH'] + ':/opt/local/bin'
os.environ['PATH']
plt.close('all')
plt.rc('text', usetex=True)

# file label
outfile = './m87_nrings.pdf'
label='nring'
save_pdf = False      # save the nmax_equatorial plot
save_fits = False     # save the nmax_equatorial image
display_image = True

# source and image parameters
source = 'M87'
MoD = 3.77883459  # this is what was used for M/D in uas for the M87 simulations
ra = 12.51373
dec = 12.39112
npix = 1024        # number of pixels
amax = 15         # maximum alpha,beta in R

# bh and observer parameters
th_o = 163*np.pi/180.  # inclination angle
spin = 0.5           # black hole spin
r_o = np.inf          # outer radius

# pixel grid, as in make_image
psize = 2.*amax/npix
psize_rad = psize*MoD*eh.RADPERUAS
n_alpha = int(np.floor(2*amax)/psize)
alphas = np.linspace(-amax, -amax+n_alpha*psize, n_alpha)
alpha_arr, beta_arr = np.meshgrid(alphas, alphas)
alpha_arr = alpha_arr.flatten()
beta_arr = beta_arr.flatten()

# maximum number of equatorial crossings (integer index, -1 if none, -2 vortical)
outarr_n = nmax_equatorial(spin, r_o, th_o, alpha_arr, beta_arr)

# fractional number of poloidal orbits (-2 vortical)
outarr_np = nmax_poloidal(spin, r_o, th_o, alpha_arr, beta_arr)

# fractional number of polar turning points, at the total Mino time
sin_o, cos_o = obs_trig(th_o)
lam = -alpha_arr*sin_o
eta = (alpha_arr**2 - spin**2)*cos_o**2 + beta_arr**2
(r1,r2,r3,r4,rclass) = radial_roots(spin,lam,eta)
Imax = mino_total(spin, r_o, eta, r1, r2, r3, r4)
outarr_nt = n_angular_turnings(spin, th_o, alpha_arr, beta_arr, Imax)

# make a image of the subring number and save
narr = np.flipud(outarr_n.reshape(npix,npix))   # number of equatorial crossings
imn = eh.image.Image(narr, psize_rad, ra, dec)
imn.source = source
if save_fits: imn.save_fits('./m87_model_%s_n.fits'%label)

#define custom colorbar
cmap = matplotlib.colors.ListedColormap(['dimgrey','blue','limegreen','orange','red'])
bounds = [-0.5,0.5,1.5,2.5,3.5,4.5]
norm = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

# continuous colorbar
cmap_cont = matplotlib.colormaps['viridis'].with_extremes(under='dimgrey')
norm_cont = matplotlib.colors.Normalize(vmin=0, vmax=4)

fs = 30
fs1 = 36

# Scale
fov = imn.fovx()/eh.RADPERUAS/MoD
pdim = imn.psize/eh.RADPERUAS/MoD
npix = imn.xdim #assume square for now
minpix = -fov/2 + pdim/2
maxpix = fov/2 - pdim/2
xs = np.linspace(minpix, maxpix, npix)

X,Y = np.meshgrid(xs,xs)

def plot_n(imarr, cblabel, discrete=True, fignum=1):
    fig=plt.figure(fignum,figsize=(10,10))

    # linear scale
    ax1=plt.gca()
    if discrete:
        imS = plt.pcolormesh(X,Y,imarr,cmap=cmap,norm=norm,shading='gouraud',rasterized=True)
    else:
        imS = plt.pcolormesh(X,Y,imarr,cmap=cmap_cont,norm=norm_cont,shading='gouraud',rasterized=True)

    ax1.set_aspect('equal')
    ax1.set_xlim(-10.,10.)
    ax1.set_ylim(-10.,10.) # flip up down for 163 deg (invert alpha, since we are rotated 90)

    ax1.set_xticks([-10,-5,0,5,10])  # coordinates: TODO make less confusing!
    ax1.set_xticklabels([r"$-10M$",r"$-5M$",r"$0$",r"$5M$",r"$10M$"])  # coordinates: TODO make less confusing!
    ax1.set_yticks([-10,-5,0,5,10])  # coordinates: TODO make less confusing!
    ax1.set_yticklabels([r"$-10M$",r"$-5M$",r"$0$",r"$5M$",r"$10M$"])  # coordinates: TODO make less confusing!

    ax1.tick_params(axis='both',direction='in',color='w',labelcolor='k',labelsize=fs,length=20,width=2,pad=15)

    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.set_ylabel(r'$\beta$',fontsize=fs1)
    ax1.set_xlabel(r'$\alpha$',fontsize=fs1)

    cax = ax1.inset_axes([1.01, 0.0, 0.05, 1.0])
    if discrete:
        cb = matplotlib.colorbar.Colorbar(cax,imS,orientation='vertical',boundaries=bounds,ticks=[0,1,2,3])
    else:
        cb = matplotlib.colorbar.Colorbar(cax,imS,orientation='vertical')

    cb.set_label(label=cblabel, size=fs1)
    cb.ax.tick_params(labelsize=fs)
    cb.set_ticks([0,1,2,3,4])

    cax.yaxis.set_ticks_position('right')

    return fig

plt.close('all')

# equatorial crossings: 1 + mbar index = number of crossings
imarr = 1+imn.imarr()
imarr[imarr < 1.e-10*np.max(imarr)] = 1.e-10*np.max(imarr) ## FLOOR ??
plot_n(imarr, r'$N_{\rm max}$', fignum=1)
if save_pdf: plt.savefig(outfile,bbox_inches='tight')

# poloidal orbits and turning points, same orientation as imn.imarr()
nparr = np.flipud(outarr_np.reshape(npix,npix))
ntarr = np.flipud(outarr_nt.reshape(npix,npix))
plot_n(np.floor(nparr), r'$\lfloor n_{\rm pol}\rfloor$', fignum=2)
plot_n(nparr, r'$n_{\rm pol}$', discrete=False, fignum=3)
plot_n(np.floor(ntarr), r'$\lfloor n_{\rm turn}\rfloor$', fignum=4)
plot_n(ntarr, r'$n_{\rm turn}$', discrete=False, fignum=5)

if display_image: plt.show()
