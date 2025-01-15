# Generate lensed images of a ring of constant equatorial radius
import numpy as np
from kgeo.equatorial_lensing import rho_of_req, critical_curve
import ehtim as eh
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.integrate as integrate

plt.close('all')
plt.ion()
plt.show()


##################################
# definitions
##################################
# bh and observer parameters
th_o = 30*np.pi/180.  # inclination angle, does not work for th0=0 exactly!
spin = 0.99          # black hole spin, does not work for a=0 or a=1 exactly!
r_o = np.inf          # outer radius
ring_radius = 5      # equatorial emission ring radius
mbars = [0,1,2,3]      # subimages to generate
rh = 1 + np.sqrt(1-spin**2)
npoints = 180
color = 'b'
linestyles = ['-','--','-.',':'] # same length as mbars
outfileRing = './ring_a%0.0f_i%0.0f_r%0.1f.pdf'%(spin*100,th_o*180/np.pi,ring_radius)

##################################
# function to characterize curves (Chael+2021 appendix B)
##################################
def calc_imagemoments(rhos, varphis):
    """calculate image moments from a closed convex curve rho(varphi)"""

    # zeroth moment: area
    A = 0.5*integrate.trapezoid(rhos**2,varphis)
    
    # first moment: centroid
    mua = integrate.trapezoid((rhos**3)*np.cos(varphis),varphis)/(3*A)
    mub = integrate.trapezoid((rhos**3)*np.sin(varphis),varphis)/(3*A)
       
    # second moment: principal axes and orientation angle
    Sigma_aa =  integrate.trapezoid((rhos**4)*(np.cos(varphis)**2),varphis)/(4*A) - mua**2
    Sigma_bb =  integrate.trapezoid((rhos**4)*(np.sin(varphis)**2),varphis)/(4*A) - mub**2
    Sigma_ab =  integrate.trapezoid((rhos**4)*(np.cos(varphis)*np.sin(varphis)),varphis)/(4*A) - mua*mub
    D = np.sqrt((Sigma_aa-Sigma_bb)**2 + 4*Sigma_ab**2)
    
    rmaj = np.sqrt(2*(Sigma_aa + Sigma_bb + D))
    rmin = np.sqrt(2*(Sigma_aa + Sigma_bb - D))
    chi = 0.5*np.arcsin(2*Sigma_ab/D)
    
    return np.array((A,mua,mub,rmaj,rmin,chi))
            
##################################
# calculate and plot
##################################
# get critical curve alpha,betas
varphis = np.linspace(-180,179,npoints)*np.pi/180.
(_, rhos_c, alphas_c, betas_c) = rho_of_req(spin, th_o, rh, mbar=100, varphis=varphis) 

# get inner shadow alpha, betas
(_, rhos_is, alphas_is, betas_is) = rho_of_req(spin, th_o, rh, mbar=0, varphis=varphis)

# plotting preamble
fig = plt.figure()
ax = plt.gca()
ax.axvline(0, -10,10, color='k')
ax.axhline(0,-10,10,color='k')
ax.set_xticks(range(-8,10,2))
ax.set_yticks(range(-8,10,2))
ax.set_xticks(range(-9,10,2),minor=True)
ax.set_yticks(range(-9,10,2),minor=True)
ax.set_xlim(-9,9)
ax.set_ylim(-9,9)
ax.set_aspect('equal')

# first do the critical curve
plt.plot(alphas_c,betas_c,ls='--',color='r',lw=1)

# next do the filled inner shadow
f_low = interp1d(alphas_is[varphis<0], betas_is[varphis<0],kind=3,fill_value='extrapolate')
plt.fill_between(alphas_is[varphis>0], f_low(alphas_is[varphis>0]), betas_is[varphis>0], ec=None,fc='k',alpha=.3)

print("a=%0.0f%%, i=%0.0f deg, r=%0.2f r_g"%(spin*100,th_o*180/np.pi,ring_radius))
for kk in range(len(mbars)):
    mbar = mbars[kk]
    ls = linestyles[kk]
    
    # get ring alpha,betas
    (_, rhos, alphas, betas) = rho_of_req(spin, th_o, ring_radius, mbar=mbar, varphis=varphis)

    # plot the ring curve
    plt.plot(alphas,betas,color=color, ls=ls, lw=0.5,label=r'$\bar{m}=%i$'%mbar)
    
    # get the image moments
    A,mua,mub,rmaj,rmin,chi = calc_imagemoments(rhos, varphis)
    
    print("m=%0i | r_maj = %0.2f , r_min = %0.2f"%(mbar,rmaj,rmin))
    
plt.title(r'$a=%0.2f$, $\theta_o=%0.0f$ deg, $r_{eq}=%0.1f$'%(spin, th_o*180/np.pi, ring_radius))         
plt.legend()
plt.show()
#plt.savefig(outfileRing,bbox_inches='tight')


