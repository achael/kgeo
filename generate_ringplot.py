# Generate m87-esque image from equatorial emision model
import numpy as np
from kgeo.equatorial_images import Iobs
from kgeo.equatorial_lensing import rho_of_req, critical_curve
import ehtim as eh
import matplotlib.pyplot as plt
from kgeo.bfields import Bfield
from kgeo.velocities import Velocity
from kgeo.emissivities import Emissivity
from scipy.interpolate import interp1d

plt.close('all')
plt.ion()
plt.show()

##################################
# definitions
##################################
# bh and observer parameters
th_o = 50*np.pi/180.  # inclination angle, does not work for th0=0 exactly!
spin = -0.99          # black hole spin, does not work for a=0 or a=1 exactly!
r_o = np.inf          # outer radius
ring_radius = 6.      # equatorial emission ring radius
mbar = 0
specind = 1
polarization = True
rh = 1 + np.sqrt(1-spin**2)
npoints = 180
nvec = 25
color = 'b'

# emissivity
emissivity = Emissivity('constant')

# velocity
#velocity = Velocity('gelles', gelles_beta=0., gelles_chi=0.) # should be zamo, appropriate for zack fig 1,2
velocity = Velocity('gelles', gelles_beta=0.3, gelles_chi=-120*np.pi/180.) # appropriate for zack fig 3,4
#velocity = Velocity('kep', retrograde=False) # appropriate for QU loops, zack fig 7,8

# bfield 
#bfield = Bfield("simple", Cr=0.87, Cph=0.5 Cvert=0)
bfield = Bfield("simple_rm1", Cr=0.5, Cph=0.87, Cvert=0) # should be close to what's in zack's paper, left col fig 3,4
#bfield = Bfield("simple_rm1", Cr=0.71, Cph=0.71, Cvert=0) # should be close to whats in zack's paper, fig 7
#bfield = Bfield("simple_rm1", Cr=0.0, Cph=0.0, Cvert=1) # should be close to whats in zack's paper, fig 8
#bfield = Bfield("const_comoving", Cr=0.5, Cph=0.87, Cvert=0) # should be exactly what's in zack's paper, not 100% sure about comoving convention still

##################################
# calculations
##################################
# get critical curve alpha,betas
varphis = np.linspace(-180,179,npoints)*np.pi/180.
#(alphas_c, betas_c) = critical_curve(spin, th_o, n=500) # numerically solving for high m works better
(_, _, alphas_c, betas_c) = rho_of_req(spin, th_o, rh, mbar=5, varphis=varphis) 

# get inner shadow alpha, betas
(_, _, alphas_is, betas_is) = rho_of_req(spin, th_o, rh, mbar=0, varphis=varphis)

# get ring alpha,betas
(_, _, alphas, betas) = rho_of_req(spin, th_o, ring_radius, mbar=mbar, varphis=varphis)


(Ivals, Qvals, Uvals, g, r_s, sinthb, Ir, Imax, Nmax) = Iobs(spin, r_o, th_o, mbar, 
                                                             alphas, betas, 
                                                             emissivity=emissivity,
                                                             velocity=velocity,
                                                             bfield=bfield,
                                                             polarization=polarization,
                                                             specind=specind)
##################################
# plots
##################################
#plt.close('all')

# preamble
fig = plt.figure()
ax = plt.gca()
#ax.spines['left'].set_position('zero')
#ax.spines['right'].set_color('none')
##ax.spines['bottom'].set_position('zero')
#ax.spines['top'].set_color('none')
ax.axvline(0, -10,10, color='k')
ax.axhline(0,-10,10,color='k')
ax.set_xticks(range(-8,10,2))
ax.set_yticks(range(-8,10,2))
ax.set_xticks(range(-9,10,2),minor=True)
ax.set_yticks(range(-9,10,2),minor=True)
ax.set_xlim(-9,9)
ax.set_ylim(-9,9)
#plt.grid()
ax.set_aspect('equal')

# first do the critical curve
plt.plot(alphas_c,betas_c,ls='--',color=color,lw=1)

# next do the filled inner shadow
f_low = interp1d(alphas_is[varphis<0], betas_is[varphis<0],kind=3,fill_value='extrapolate')
plt.fill_between(alphas_is[varphis>0], f_low(alphas_is[varphis>0]), betas_is[varphis>0], ec=None,fc='k',alpha=.3)

# next the ring curve
plt.plot(alphas,betas,color=color, ls='-', lw=0.5)
    
# next the quiver tick plots

x = -np.sin(np.angle(Qvals + 1j * Uvals) / 2)
x *= np.sqrt(Qvals**2 + Uvals**2)             
y =  np.cos(np.angle(Qvals + 1j * Uvals) / 2)
y *= np.sqrt(Qvals**2 + Uvals**2)             


thin = np.arange(0,len(varphis), len(varphis)//nvec)            
plt.quiver(alphas[thin], betas[thin], x[thin], y[thin],pivot='mid',color=color,angles='uv',
           units='width', width=0.005, headwidth=1, headlength=0.01, minlength=0, minshaft=1)

plt.title(r'$a=%0.2f$, $\theta_o=%0.0f$ deg, $r_{eq}=%0.1f$, $m=%i$'%(spin, th_o*180/np.pi, ring_radius,mbar))         

# plot QU loop 
plt.figure()
plt.plot(Qvals,Uvals,ls='-',color=color)
plt.plot(0,0,'k+')
plt.title(r'$a=%0.2f$, $\theta_o=%0.0f$ deg, $r_{eq}=%0.1f$, $m=%i$'%(spin, th_o*180/np.pi, ring_radius,mbar))         
plt.gca().set_aspect('equal')

# display
plt.show()
plt.pause(0.01)

                           
