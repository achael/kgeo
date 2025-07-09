import numpy as np
from kgeo.kerr_raytracing_ana import raytrace_ana
from kgeo.equatorial_lensing import rho_of_req, nmax_equatorial

spin = 0.99
inc = 17*np.pi/180.
rout = 4.e10 # sgra distance in M
ngeo = 2 # this can be small bc we are only interested in final points
maxtaufrac = (1. - 1.e-14) # NOTE: if we go exactly to tau_tot t and phi diverge on horizon

alpha_max=beta_max=10
alpha_min=beta_min=-10
psize=0.01

rh= 1+np.sqrt(1-spin**2)

n_alpha = int(np.floor(alpha_max - alpha_min)/psize)
alphas = np.linspace(alpha_min, alpha_min+n_alpha*psize, n_alpha)
n_beta = int(np.floor(beta_max - beta_min)/psize)
betas = np.linspace(beta_min, beta_min+n_beta*psize, n_beta)
imsize = (len(alphas), len(betas))

alpha_arr, beta_arr = np.meshgrid(alphas, betas)
alpha_arr = alpha_arr.flatten()
beta_arr = beta_arr.flatten()

# get latitudes of horizon/celestial sphere
geos = raytrace_ana(a=spin, observer_coords = [0,rout,inc,0],
                    image_coords = [alpha_arr, beta_arr],
                    ngeo=ngeo, maxtaufrac=maxtaufrac,
                    do_phi_and_t=False, savedata=False, plotdata=False)

latfinals = geos.geo_coords[2][-1].reshape(imsize)
rfinals = geos.geo_coords[1][-1].reshape(imsize)

# get inner shadow alpha, betas

varphis = np.linspace(-180,179,100)*np.pi/180.
(_, rhos_is, alphas_is, betas_is) = rho_of_req(spin, inc, rh, mbar=0, varphis=varphis)

# get critical curve alpha,betas
varphis = np.linspace(-180,179,100)*np.pi/180.
(_, rhos_c, alphas_c, betas_c) = rho_of_req(spin, inc, rh, mbar=100, varphis=varphis) 

# get number of equatorial crossings
nmax = nmax_equatorial(spin, rout, inc, alpha_arr, beta_arr).reshape(imsize)

# plot
plt.close('all')
plt.figure(1)
#pc=plt.pcolormesh(alphas,betas,latfinals/np.pi, cmap='jet',rasterized=True,shading='gouraud',vmin=0,vmax=1)
pc=plt.imshow(np.flipud(latfinals/np.pi), cmap='RdYlGn',rasterized=True,vmin=0,vmax=1,extent=(alpha_min,alpha_max,beta_min,beta_max))
plt.contour(alphas,betas,latfinals/np.pi,levels=[0.5],colors='k')
clb=plt.colorbar(pc)
clb.ax.set_title(r'$\theta_s/\pi$')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')

plt.plot(alphas_c,betas_c,color='m',ls='-')
plt.plot(alphas_is,betas_is,color='c',ls='-')
plt.title(r'$a=%.2f, \theta_o=%.2f\pi$'%(spin,inc/np.pi))

plt.figure(2)
#pc=plt.pcolormesh(alphas,betas,np.log(rfinals/rh),rasterized=True,shading='gouraud',cmap='Spectral')
pc=plt.imshow(np.flipud(np.log(rfinals/rh)),rasterized=True,cmap='Spectral',extent=(alpha_min,alpha_max,beta_min,beta_max))
plt.contour(alphas,betas,latfinals/np.pi,levels=[0.5],colors='k')
clb=plt.colorbar(pc)
clb.ax.set_title(r'$\log(r_s/r_+)$')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')

plt.plot(alphas_c,betas_c,color='m',ls='-')
plt.plot(alphas_is,betas_is,color='c',ls='-')
plt.title(r'$a=%.2f, \theta_o=%.2f\pi$'%(spin,inc/np.pi))

plt.figure(3)
#pc=plt.pcolormesh(alphas,betas,np.log(rfinals/rh),rasterized=True,shading='gouraud',cmap='Spectral')
nmax[nmax==-2]=-1
pc=plt.imshow(np.flipud(nmax),rasterized=True,cmap='Set2',extent=(alpha_min,alpha_max,beta_min,beta_max))
plt.contour(alphas,betas,latfinals/np.pi,levels=[0.5],colors='k')
plt.contour(alphas,betas,latfinals/np.pi,levels=[0.25],colors='w',linestyles='--')
clb=plt.colorbar(pc)
clb.ax.set_title(r'$N_{\rm max}$')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')

plt.plot(alphas_c,betas_c,color='m',ls='-')
plt.plot(alphas_is,betas_is,color='c',ls='-')
plt.title(r'$a=%.2f, \theta_o=%.2f\pi$'%(spin,inc/np.pi))


plt.show()


# get contour data for horizon n=0,n=1
