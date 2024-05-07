# Make an image of subring order for M87 type models
import numpy as np
import ehtim as eh
import matplotlib.pyplot as plt
from kgeo.equatorial_images import make_image
from kgeo.bfields import Bfield
from kgeo.velocities import Velocity
from kgeo.emissivities import Emissivity
import ehtplot
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines
import os

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
save_image = False
display_image = False

# source and image parameters
source = 'M87'
MoD = 3.77883459  # this is what was used for M/D in uas for the M87 simulations
ra = 12.51373 
dec = 12.39112 
flux230 = 0.6     # total flux
npix = 1024        # number of pixels
amax = 15         # maximum alpha,beta in R
f0 = 1            # scaling factor for n=0 flux
f1 = 1            # scaling factor for n=1 flux
f2 = 1            # scaling factor for n>=2 flux
nmax = 4          # maximum subring number
rotation = 90*eh.DEGREE  # rotation angle, for m87 prograde=90,retrograde=-90 (used in display only)

# bh and observer parameters
th_o = 163*np.pi/180.  # inclination angle, does not work for th0=0 exactly!
spin = 0.5           # black hole spin, does not work for a=0 or a=1 exactly!
r_o = np.inf          # outer radius
rh = 1+np.sqrt(1-spin**2)

# emissivity
#emissivity = Emissivity("ring", r_ring=4, sigma=0.3, emiscut_in=3.5, emiscut_out=4.5)
#emissivity = Emissivity("ring", r_ring=6, sigma=0.3, emiscut_in=5.5, emiscut_out=6.5)
#emissivity = Emissivity("glm", sigma=0.5, gamma_off=-1)
emissivity = Emissivity("bpl", p1=-2.0, p2=-0.5)
specind = 1

# velocity
#velocity = Velocity('simfit')
#velocity = Velocity('gelles', gelles_beta=0.3, gelles_chi=-120*np.pi/180.)
#velocity = Velocity('subkep', retrograde=True, fac_subkep=0.7)
velocity = Velocity('general', retrograde=False, fac_subkep=0.7, beta_phi=0.7, beta_r=0.7)

# bfield 
polarization = False
#bfield = Bfield("simple", Cr=0.87, Cvert=0, Cph=0.5)
#bfield = Bfield("simple_rm1", Cr=0.87, Cvert=0, Cph=0.5) 
#bfield = Bfield("const_comoving", Cr=0.5, Cvert=0, Cph=0.87) 
#bfield = Bfield("bz_monopole",C=1)
bfield = Bfield("bz_guess",C=1)

# generate the equatorial model image arrays
psize = 2.*amax/npix
psize_rad = psize*MoD*eh.RADPERUAS
imagedat = make_image(spin,r_o, th_o, nmax, -amax, amax, -amax, amax, psize,
                      nmax_only=False,
                      emissivity=emissivity,velocity=velocity, bfield=bfield,
                      polarization=polarization, specind=specind)
                      
(outarr_I, outarr_Q, outarr_U, outarr_r, outarr_t, outarr_g, outarr_sinthb, outarr_n, outarr_np) = imagedat

# make a image of the subring number and save
narr = np.flipud(outarr_n.reshape(npix,npix))   # number of equatorial crossings
imn = eh.image.Image(narr, psize_rad, ra, dec)
imn.source = source
if save_image: imn.save_fits('./m87_model_%s_n.fits'%label)

#define custom colorbar
cmap = matplotlib.colors.ListedColormap(['dimgrey','blue','limegreen','orange','red'])
bounds = [-0.5,0.5,1.5,2.5,3.5,4.5]
norm = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

fs = 30
fs1 = 36

plt.close('all')
fig=plt.figure(1,figsize=(10,10))

# Scale
fov = imn.fovx()/eh.RADPERUAS/MoD
pdim = imn.psize/eh.RADPERUAS/MoD
npix = imn.xdim #assume square for now
minpix = -fov/2 + pdim/2
maxpix = fov/2 - pdim/2
xs = np.linspace(minpix, maxpix, npix)

scalefac = 1
imarr = 1+imn.imarr()#*scalefac#/1.e10
imarr[imarr < 1.e-10*np.max(imarr)] = 1.e-10*np.max(imarr) ## FLOOR ?? 

X,Y = np.meshgrid(xs,xs)


# linear scale

ax1=plt.gca()
imS = plt.pcolormesh(X,Y,imarr,cmap=cmap,norm=norm,shading='gouraud',rasterized=True)

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
cb = matplotlib.colorbar.Colorbar(cax,imS,orientation='vertical',boundaries=bounds,ticks=[0,1,2,3])


cb.set_label(label=r'$N_{\rm max}$', size=fs1)
cb.ax.tick_params(labelsize=fs)
cb.set_ticks([0,1,2,3,4])


cax.yaxis.set_ticks_position('right')

plt.savefig(outfile,bbox_inches='tight')
if display_image: plt.show()
