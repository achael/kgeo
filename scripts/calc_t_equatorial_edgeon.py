# stitch together n=0 and n=1 images of time delay from equatorial plane for nearly edge on images
# TODO this is 2x as slow as it needs to be because we are double computing entire image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from kgeo.equatorial_lensing import r_equatorial
from kgeo.kerr_raytracing_ana import raytrace_ana, coords_at_tau

# black hole and observer parameters (TODO: replace with command line arguments)
SPIN = 0.99
INC = 89.999*np.pi/180. # keep this very close to 90 deg
INC2 = 90.001*np.pi/180. # keep this very close to 90 deg

# TODO ROUT shouldn't matter, but it seems to when it is very large. Some issue in Mino steps? 
ROUT = 1.e6 #5.4e10 is M87 distance in M. Unfortunately subraction becomes an issue here...

# image parameters (TODO: replace with command line arguments)
FOV_ALPHA = 14
FOV_BETA = 14
PSIZE = 0.1 #0.1 #(use 0.01 for good resolution of n=2)

fs_cb = 14
fs_ax = 14
fs_title= 14


def main(spin=SPIN, rout=ROUT, fov_alpha=FOV_ALPHA, fov_beta=FOV_BETA, psize=PSIZE):
    rh = 1+np.sqrt(1-spin*spin)

    # determine pixel grid
    alpha_max = 0.5*fov_alpha; alpha_min = -alpha_max
    beta_max = 0.5*fov_beta; beta_min = -beta_max
    n_alpha = int(np.floor(alpha_max - alpha_min)/psize)
    alphas = np.linspace(alpha_min, alpha_min+n_alpha*psize, n_alpha)
    n_beta = int(np.floor(beta_max - beta_min)/psize)
    betas = np.linspace(beta_min, beta_min+n_beta*psize, n_beta)

    alpha_arr, beta_arr = np.meshgrid(alphas, betas)
    alpha_arr = alpha_arr.flatten()
    beta_arr = beta_arr.flatten()

    # useful for plotting
    extent = (np.min(alphas), np.max(alphas), np.min(betas), np.max(betas))
    imshape = (n_beta, n_alpha)
        
    ##############upper half plane
    print('upper half plane...')
    # this gets us the source radius and mino time for each equatorial crossing
    # NOTE: Ir gives nonsense values outside given mbar ring
    (r_s, tau, taumax, Nmax) = r_equatorial(spin, rout, INC, 0, alpha_arr, beta_arr)

    # look at points in upper half plane (bent over BH toward equatorial observer)
    ring_mask_up = (Nmax>=0) * (beta_arr>=0)

    # this gets us the full source coordinate x^mu for output mino times
    sig_s, geo_coords = coords_at_tau(spin, [0,rout,INC,0], [alpha_arr,beta_arr], tau, do_phi_and_t=True)

    t_crossings_up = np.ma.masked_array(geo_coords[0][0], ~ring_mask_up)
    r_crossings_up = np.ma.masked_array(geo_coords[1][0], ~ring_mask_up)
    theta_crossings_up = np.ma.masked_array(geo_coords[2][0], ~ring_mask_up)
    phi_crossings_up = np.ma.masked_array(np.angle(np.exp(1j*geo_coords[3][0])), ~ring_mask_up) 

    # check that raytrace_ana and r_equatorial get same tau, theta, r
    # TODO: what should these tolerances be? 
    if np.max(np.abs((1-theta_crossings_up/(0.5*np.pi))[ring_mask_up])) > 1.e-14:
        print(np.max(np.abs((1-theta_crossings_up/(0.5*np.pi))[ring_mask_up])))
        raise Exception("raytrace_ana not finding eqauatorial crossings for mbar=%i!"%mbar)        
    if np.max(np.abs((1-r_crossings_up/r_s)[ring_mask_up])) > 1.e-12:
        print(np.max(np.abs((1-r_crossings_up/r_s)[ring_mask_up])))
        raise Exception("raytrace_ana not finding consistent r_s for mbar=%i!"%mbar)        

    ##############lower half plane
    print('lower half plane...')
    # TODO:switching inc->180+inc should be pretty similar to taking mbar->1
    # this gets us the source radius and mino time for each equatorial crossing
    # NOTE: Ir gives nonsense values outside given mbar ring
    (r_s, tau, taumax, Nmax) = r_equatorial(spin, rout, INC2, 0, alpha_arr, beta_arr)

    # look at points in lower half plane (bent under BH toward equatorial observer)
    ring_mask_down = (Nmax>=0) * (beta_arr<0)

    # this gets us the full source coordinate x^mu for output mino times
    sig_s, geo_coords = coords_at_tau(spin, [0,rout,INC2,0], [alpha_arr,beta_arr], tau, do_phi_and_t=True)

    t_crossings_down = np.ma.masked_array(geo_coords[0][0], ~ring_mask_down)
    r_crossings_down = np.ma.masked_array(geo_coords[1][0], ~ring_mask_down)
    theta_crossings_down = np.ma.masked_array(geo_coords[2][0], ~ring_mask_down)
    phi_crossings_down = np.ma.masked_array(np.angle(np.exp(1j*geo_coords[3][0])), ~ring_mask_down) 

    # check that raytrace_ana and r_equatorial get same tau, theta, r
    # TODO: what should these tolerances be? 
    if np.max(np.abs((1-theta_crossings_up/(0.5*np.pi))[ring_mask_down])) > 1.e-14:
        print(np.max(np.abs((1-theta_crossings_up/(0.5*np.pi))[ring_mask_down])))
        raise Exception("raytrace_ana not finding eqauatorial crossings for mbar=%i!"%mbar)        
    if np.max(np.abs((1-r_crossings_up/r_s)[ring_mask_down])) > 1.e-12:
        print(np.max(np.abs((1-r_crossings_up/r_s)[ring_mask_down])))
        raise Exception("raytrace_ana not finding consistent r_s for mbar=%i!"%mbar)        

    # stitch these together
    Nmax2 = Nmax.copy()
    Nmax2[beta_arr>=0] -= 1  #careful, needs to be 2nd nmax here
    ring_mask = Nmax2>=0

    t_crossings = np.zeros(alpha_arr.shape)
    t_crossings[ring_mask_up] = t_crossings_up[ring_mask_up]; 
    t_crossings[ring_mask_down] = t_crossings_down[ring_mask_down]
    t_crossings = np.ma.masked_array(t_crossings,~ring_mask)

    r_crossings = np.zeros(alpha_arr.shape)
    r_crossings[ring_mask_up] = r_crossings_up[ring_mask_up]; 
    r_crossings[ring_mask_down] = r_crossings_down[ring_mask_down]
    r_crossings = np.ma.masked_array(r_crossings,~ring_mask)

    phi_crossings = np.zeros(alpha_arr.shape)
    phi_crossings[ring_mask_up] = phi_crossings_up[ring_mask_up]; 
    phi_crossings[ring_mask_down] = phi_crossings_down[ring_mask_down]
    phi_crossings = np.ma.masked_array(phi_crossings,~ring_mask)

    #########################################################
    # make some plots
    #########################################################

    # source radius
    plt.figure(1)
    ax_rs = plt.gca()
    pc_rs = ax_rs.imshow(np.flipud(r_crossings.reshape(imshape)),
                        cmap='cool', rasterized=False, vmin=rh,
                        extent=extent, interpolation='none')
    clevels = np.linspace(0,50,25)
    cset = ax_rs.contour(alphas, betas,
                        r_crossings.reshape(imshape),
                        levels=clevels, colors='k', linestyles='-',linewidths=0.5)
    _divider_rs = make_axes_locatable(ax_rs)
    _cax_rs = _divider_rs.append_axes("right", size="5%", pad=0.05)
    clb_rs = plt.colorbar(pc_rs, cax=_cax_rs)
    clb_rs.ax.set_title(r'$r_s$', fontsize=fs_cb)
    ax_rs.set_xlabel(r'$\alpha\;[M]$', fontsize=fs_ax)
    plt.suptitle(r'Source $r$')
    ax_rs.set_title(r'$a=%.2f$'%(spin), fontsize=fs_title)
    ax_rs.tick_params(labelleft=False)    

    # source BL azimuth
    plt.figure(2)
    ax_phis = plt.gca()
    pc_phis = ax_phis.imshow(np.flipud(phi_crossings.reshape(imshape)),
                        cmap='hsv', rasterized=False, vmin=-np.pi, vmax=np.pi,
                        extent=extent, interpolation='none')
    clevels=[-7*np.pi/8.,-5*np.pi/8.,-3*np.pi/8.,-np.pi/8,np.pi/8,3*np.pi/8.,5*np.pi/8, 7*np.pi/8]
    cset = ax_phis.contour(alphas, betas,
                        phi_crossings.reshape(imshape),
                        levels=clevels, 
                        colors='k', linestyles='-',linewidths=0.5)
    _divider_phis = make_axes_locatable(ax_phis)
    _cax_phis = _divider_phis.append_axes("right", size="5%", pad=0.05)
    clb_phis = plt.colorbar(pc_phis, cax=_cax_phis)
    clb_phis.ax.set_title(r'$\phi_s$', fontsize=fs_cb)
    ax_phis.set_xlabel(r'$\alpha\;[M]$', fontsize=fs_ax)
    plt.suptitle(r'Source $\phi$')
    ax_phis.set_title(r'$a=%.2f$'%(spin), fontsize=fs_title)
    ax_phis.tick_params(labelleft=False)    

    # source coordinate time shifted by r_camera/c
    plt.figure(3)
    tplot = -t_crossings - rout
    tmax = 50 #np.max(np.abs(tplot))
    ax_ts = plt.gca()
    pc_ts = ax_ts.imshow(np.flipud(tplot.reshape(imshape)),
                        cmap='plasma', rasterized=False, vmin=0, vmax=tmax,
                        extent=extent, interpolation='none')
    
    clevels = np.linspace(0,tmax,100)
    cstyles = np.array(['-' for level in clevels])
    cstyles[clevels==0] = '--'
    cstyles[clevels<0] = ':'
    cset = ax_ts.contour(alphas, betas,
                        tplot.reshape(imshape),
                        levels=clevels,
                        linestyles=cstyles,
                        colors='k', linewidths=0.5)
    _divider_ts = make_axes_locatable(ax_ts)
    _cax_ts = _divider_ts.append_axes("right", size="5%", pad=0.05)
    clb_ts = plt.colorbar(pc_ts, cax=_cax_ts)
    clb_ts.ax.set_title(r'$t_s$', fontsize=fs_cb)
    ax_ts.set_xlabel(r'$\alpha\;[M]$', fontsize=fs_ax)
    plt.suptitle(r'Source $-t - r_o$')
    ax_ts.set_title(r'$a=%.2f$'%(spin), fontsize=fs_title)
    ax_ts.tick_params(labelleft=False)    

    plt.show()

if __name__ == '__main__':
    main()