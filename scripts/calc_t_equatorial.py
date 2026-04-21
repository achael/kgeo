# Calculate crossing times for different images of the equatorial plane

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from kgeo.equatorial_lensing import r_equatorial
from kgeo.kerr_raytracing_ana import raytrace_ana, coords_at_tau

# black hole and observer parameters (TODO: replace with command line arguments)
SPIN = 0.99
INC = 60*np.pi/180.

# TODO ROUT shouldn't matter, but it seems to when it is very large. Some issue in Mino steps? 
ROUT = 1.e6 #5.4e10 is M87 distance in M. Unfortunately subraction becomes an issue here...
MBAR_MAX=1 # direct image only for now

# image parameters (TODO: replace with command line arguments)
FOV_ALPHA = 14
FOV_BETA = 14
PSIZE = 0.01 #0.1 #(use 0.01 for good resolution of n=2)

fs_cb = 14
fs_ax = 14
fs_title= 14


def main(spin=SPIN, inc=INC, rout=ROUT, mbar_max=MBAR_MAX, fov_alpha=FOV_ALPHA, fov_beta=FOV_BETA, psize=PSIZE):
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
    for mbar in range(mbar_max+1):
        print('mbar=',mbar)
        # this gets us the source radius and mino time for each equatorial crossing
        # NOTE: Ir gives nonsense values outside given mbar ring
        (r_s, tau, taumax, Nmax) = r_equatorial(spin, rout, inc, mbar, alpha_arr, beta_arr)

        # what points in the image cross the equatorial plane mbar+1 times
        ring_mask = (Nmax>=mbar)

        # this gets us the full source coordinate x^mu for output mino times
        sig_s, geo_coords = coords_at_tau(spin, [0,rout,inc,0], [alpha_arr,beta_arr], tau, do_phi_and_t=True)

        t_crossings = np.ma.masked_array(geo_coords[0][0], ~ring_mask)
        r_crossings = np.ma.masked_array(geo_coords[1][0], ~ring_mask)
        theta_crossings = np.ma.masked_array(geo_coords[2][0], ~ring_mask)
        # TODO: this domain restrction to (-pi,pi) should have been done already
        phi_crossings = np.ma.masked_array(np.angle(np.exp(1j*geo_coords[3][0])), ~ring_mask) 

        # check that raytrace_ana and r_equatorial get same tau, theta, r
        # TODO: what should these tolerances be? 
        if np.max(np.abs((1-theta_crossings/(0.5*np.pi))[ring_mask])) > 1.e-14:
            print(np.max(np.abs((1-theta_crossings/(0.5*np.pi))[ring_mask])))
            raise Exception("raytrace_ana not finding eqauatorial crossings for mbar=%i!"%mbar)        
        if np.max(np.abs((1-r_crossings/r_s)[ring_mask])) > 1.e-12:
            print(np.max(np.abs((1-r_crossings/r_s)[ring_mask])))
            raise Exception("raytrace_ana not finding consistent r_s for mbar=%i!"%mbar)        

        #########################################################
        # make some plots
        #########################################################
        # max equatorial crossings
        plt.figure(100*mbar + 1)
        ax_nmax = plt.gca()
        cmap_nmax = matplotlib.colors.ListedColormap(['lightgray',
          '#0072B2','#F0E442','#D55E00'] 
        )
        pc_nmax = ax_nmax.imshow(np.flipud(Nmax.reshape(imshape)),
                                cmap=cmap_nmax, rasterized=False, vmin=-0.5, vmax=3.5,
                                extent=extent, interpolation='none')
        _divider_nmax = make_axes_locatable(ax_nmax)
        _cax_nmax = _divider_nmax.append_axes("right", size="5%", pad=0.05)
        clb_nmax = plt.colorbar(pc_nmax, cax=_cax_nmax)
        clb_nmax.set_ticks([0, 1, 2, 3])
        clb_nmax.set_ticklabels([r'$0$', r'$1$', r'$2$', r'$3$'])
        clb_nmax.ax.set_title(r'$n_{\rm max}$', fontsize=fs_cb)
        ax_nmax.set_xlabel(r'$\alpha\;[M]$', fontsize=fs_ax)
        ax_nmax.set_title(r'Equatorial Crossings', fontsize=fs_title)
        ax_nmax.tick_params(labelleft=False)    

        # source radius
        plt.figure(100*mbar + 2)
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
        ax_rs.set_title(r'$a=%.2f$, $i=%.1f$ deg, $m=%i$'%(spin, inc*180/np.pi, mbar), fontsize=fs_title)
        ax_rs.tick_params(labelleft=False)    

        # source BL azimuth
        plt.figure(100*mbar + 3)
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
        ax_phis.set_title(r'$a=%.2f, i=%.1f$ deg, $m=%i$'%(spin, inc*180/np.pi, mbar), fontsize=fs_title)
        ax_phis.tick_params(labelleft=False)    

        # source coordinate time shifted by r_camera/c
        plt.figure(100*mbar + 4)
        tplot = -t_crossings - rout
        tmax = 200#np.max(np.abs(tplot))
        ax_ts = plt.gca()
        pc_ts = ax_ts.imshow(np.flipud(tplot.reshape(imshape)),
                            cmap='bwr', rasterized=False, vmin=-tmax, vmax=tmax,
                            extent=extent, interpolation='none')
        
        clevels = np.linspace(-tmax,tmax,100)
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
        ax_ts.set_title(r'$a=%.2f, i=%.1f$ deg, $m=%i$'%(spin, inc*180/np.pi, mbar), fontsize=fs_title)
        ax_ts.tick_params(labelleft=False)    

        plt.show()

if __name__ == '__main__':
    main()