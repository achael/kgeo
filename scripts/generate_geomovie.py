# make a movie of image null geodesics w/r/t coordinate time
# make sure you have ffmpeg installed!
# TODO: this is slow. parallelize.

from kgeo.kerr_raytracing_ana import *
from scipy.interpolate import interp1d
import matplotlib.animation as animation
import subprocess
from pylab import *
import numpy as np

# bh params
spin = 0.94            # black hole spin, does not work for a=0 or a=1 exactly!
npix = 251             # number of pixels in each dimension
amax = 7               # maximum alpha,beta in R
th_o = 20*np.pi/180.   # inclination angle, does not work for th0=0 exactly!
r_o = 100000.          # outer radius
ngeo = 250             # number of points along geodesic
rplus  = 1 + np.sqrt(1-spin**2) # horizon radius

# movie params
xlim = 10  # max image radius 
rmax = 10  # max radius around BH where we start to plot 
nframes = 100  # number of frames
fps = 10      # frames per second in output
dpi = 200     # image quality ipython
ringcolors = ['darkgrey','b','g','orange','r'] # colors of the different photon rings
ringalphas=[0.1,0.1,0.2,0.5,0.75]       
ringsizes = [8,8,8,10,12]
rings_to_plot = [1,2,3] # nring+1, so n=0 ring is 1. Max is 4. 
plot_lines = True # if True, plot thin lines under moving geodesic points
forward_time = True # if True, plot forward in time instead of backwards
outfile = './geomovie_demo'


####################################################################################################################
# determine pixel grid
psize = 2.*amax/npix
alpha_max = amax
alpha_min = -amax
beta_max = amax
beta_min = -amax
n_alpha = int(np.floor(alpha_max - alpha_min)/psize)
alphas = np.linspace(alpha_min, alpha_min+n_alpha*psize, n_alpha)
n_beta = int(np.floor(beta_max - beta_min)/psize)
betas = np.linspace(beta_min, beta_min+n_beta*psize, n_beta)

alpha_arr, beta_arr = np.meshgrid(alphas, betas)
alpha_arr = alpha_arr.flatten()
beta_arr = beta_arr.flatten()

# generate geodesics
print("generating geodesics")
geos = raytrace_ana(a=spin,
                 observer_coords = [0,r_o,th_o,0],
                 image_coords = [alpha_arr, beta_arr],
                 ngeo=ngeo,
                 do_phi_and_t=True,
                 savedata=False, plotdata=False)

# interpolate solution
print("interpolating geodesics....")
r_funs = [interp1d(geos.t_s[:,i],geos.r_s[:,i],bounds_error=False,fill_value=0) for i in range(len(geos.alpha))]
th_funs = [interp1d(geos.t_s[:,i],geos.th_s[:,i],bounds_error=False,fill_value=0) for i in range(len(geos.alpha))]
ph_funs = [interp1d(geos.t_s[:,i],geos.ph_s[:,i],bounds_error=False,fill_value=0) for i in range(len(geos.alpha))]

####################################################################################################################
# make movie
plt.close('all')
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(-xlim,xlim)
ax.set_ylim(-xlim,xlim)
ax.set_zlim(-xlim,xlim)
ax.auto_scale_xyz([-xlim, xlim], [-xlim, xlim], [-xlim, xlim])
ax.set_axis_off()      
fig.tight_layout()

# plot horizon 
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
ax.plot_surface(rplus*np.cos(u) * np.sin(v),  rplus*np.sin(u) * np.sin(v),  rplus*np.cos(v), color='black',zorder=1000)

# plot equatorial plane
xxeq, yyeq = np.mgrid[-xlim:xlim, -xlim:xlim:100j]
zzeq = np.zeros(xxeq.shape)
ax.plot_surface(xxeq, yyeq, zzeq, alpha=0.3,linewidth=0,color='lightblue')

# photon ring order masks
nmax_eq = geos.nmax_eq
maskIS = (nmax_eq==-2) + (nmax_eq==-1)
mask0 = nmax_eq==0
mask1 = nmax_eq==1
mask2 = nmax_eq==2
mask3 = nmax_eq==3 
masks = [maskIS,mask0,mask1,mask2,mask3]

# plot geos as background thin lines
if plot_lines:
    r_s = geos.r_s
    th_s = geos.th_s
    ph_s = geos.ph_s
    tausteps = geos.tausteps


    x_s = r_s * np.cos(ph_s) * np.sin(th_s)
    y_s = r_s * np.sin(ph_s) * np.sin(th_s)
    z_s = r_s * np.cos(th_s)

    for jj in rings_to_plot:
        mask = masks[jj]
        color = ringcolors[jj]
        alpha=ringalphas[jj]
        xs = x_s[:,mask];ys = y_s[:,mask];zs = z_s[:,mask];
        rs = r_s[:,mask];tau = tausteps[:,mask]
        plotgeos = range(0,xs.shape[-1],100)

        for i in plotgeos:
            x = xs[:,i]; y=ys[:,i]; z=zs[:,i]
            mask = ((rs[:,i] < rmax) + (tau[:,i] < .5*tau[-1,i]))
            mask *= rs[:,i] < 3*rmax
            x = x[mask]; y = y[mask]; z = z[mask]
            ax.plot3D(x,y,z,color,alpha=alpha,linewidth=1) 
                  
# get points in time  
def get_pts(t):
    r_t = [r_funs[i](t) for i in range(len(geos.alpha))] 
    th_t = [th_funs[i](t) for i in range(len(geos.alpha))] 
    ph_t = [ph_funs[i](t) for i in range(len(geos.alpha))] 

    mask = r_t>rplus

    xx = (r_t*np.sin(th_t)*np.cos(ph_t))
    yy = (r_t*np.sin(th_t)*np.sin(ph_t))
    zz = (r_t*np.cos(th_t))
    return xx,yy,zz,mask
    
# times to plot   
times = np.linspace(-geos.r_o+15,-geos.r_o-100,nframes)
if forward_time: times = np.flip(times)

# make initial plot of each photon ring
xx,yy,zz,maskh=get_pts(times[0])
pts = []
for jj in range(5):
    mask = masks[jj]
    alpha = ringalphas[jj]
    color = ringcolors[jj]
    ms = ringsizes[jj]
    if jj in rings_to_plot:
        pts.append(ax.scatter3D(xx[mask*maskh],yy[mask*maskh],zz[mask*maskh],alpha=alpha,marker=".",s=ms,color=color,linewidths=0,zorder=(jj+1)))
    else:
        pts.append(None)

# plot update function   
def update_plot(kk):

    t = times[kk]
    xx,yy,zz,maskh=get_pts(t)

    for jj in rings_to_plot:
        mask = masks[jj]
        pts[jj]._offsets3d=(xx[mask*maskh],yy[mask*maskh],zz[mask*maskh])

    plt.draw()   
               
# make and save animation
print("generating movie...")
ani = animation.FuncAnimation(fig,update_plot,nframes,interval=30)
writer = animation.writers['ffmpeg'](fps=fps)
ani.save(outfile+'.mp4',writer=writer,dpi=dpi,progress_callback=lambda i, nframes: print('frame', i+1, 'of', nframes))
subprocess.run(['ffmpeg','-y','-i',outfile+'.mp4', outfile+'.gif'])
plt.close('all')
      
