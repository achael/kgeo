# make a movie of geodesics moving backward w/r/t coordinate time
from kgeo.kerr_raytracing_ana import *
from scipy.interpolate import interp1d
import matplotlib.animation as animation
from pylab import *
import numpy as np

# bh params
spin = 0.94            # black hole spin, does not work for a=0 or a=1 exactly!
npix = 255             # number of pixels
amax = 7               # maximum alpha,beta in R
th_o = 17*np.pi/180.   # inclination angle, does not work for th0=0 exactly!
r_o = 100000.          # outer radius
ngeo = 250             # number of points along geodesic
rplus  = 1 + np.sqrt(1-spin**2) # horizon radius

# movie params
xlim = 10  # max image radius 
rmax = 10  # max radius around BH where we start to plot 
nframes = 50  # number of frames
fps = 10      # frames per second in output
dpi = 500     # image quality ipython
ringcolors = ['darkgrey','b','g','orange','r'] # colors of the different photon rings
ringalphas=[0.2,0.2,0.2,0.4,0.5]       
rings_to_plot = [1] # nring+1, so n=0 ring is 1. Max is 4. 
plot_lines = True # if True, plot thin lines under moving geodesic points
outfile = './demo.mp4'

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

def get_pts(t):
    r_t = [r_funs[i](t) for i in range(len(geos.alpha))] 
    th_t = [th_funs[i](t) for i in range(len(geos.alpha))] 
    ph_t = [ph_funs[i](t) for i in range(len(geos.alpha))] 

    mask = r_t>rplus

    xx = (r_t*np.sin(th_t)*np.cos(ph_t))
    yy = (r_t*np.sin(th_t)*np.sin(ph_t))
    zz = (r_t*np.cos(th_t))
    return xx,yy,zz,mask
    
# make movie
plt.close('all')
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(projection='3d')
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
ax.plot_surface(rplus*np.cos(u) * np.sin(v),  rplus*np.sin(u) * np.sin(v),  rplus*np.cos(v), color='black',zorder=1000)

ax.set_xlim(-xlim,xlim)
ax.set_ylim(-xlim,xlim)
ax.set_zlim(-xlim,xlim)
ax.auto_scale_xyz([-xlim, xlim], [-xlim, xlim], [-xlim, xlim])
ax.set_axis_off()      

xxeq, yyeq = np.mgrid[-xlim:xlim, -xlim:xlim:100j]
zzeq = np.zeros(xxeq.shape)
ax.plot_surface(xxeq, yyeq, zzeq, alpha=0.3,linewidth=0,color='lightblue')

fig.tight_layout()

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
        
        xs = x_s[:,mask];ys = y_s[:,mask];zs = z_s[:,mask];
        rs = r_s[:,mask];tau = tausteps[:,mask]
        plotgeos = range(0,xs.shape[-1],100)

        for i in plotgeos:
            x = xs[:,i]; y=ys[:,i]; z=zs[:,i]
            mask = ((rs[:,i] < rmax) + (tau[:,i] < .5*tau[-1,i]))
            mask *= rs[:,i] < 3*rmax
            x = x[mask]; y = y[mask]; z = z[mask]
            ax.plot3D(x,y,z,color,alpha=.1,linewidth=1) 
                  
# get points in time        

xx,yy,zz,maskh=get_pts(-geos.r_o+10)
pts = []
#for jj in range(5):
#    mask = masks[jj]
#    alpha = ringalphas[jj]
#    color = ringcolors[jj]
#    if jj in rings_to_plot:
#        pts.append(ax.scatter3D(xx[mask*maskh],yy[mask*maskh],zz[mask*maskh],alpha=alpha,marker=".",color=color,s=10,linewidths=0))
#    else:
#        pts.append(None)


#ptsIS = ax.scatter3D(xx[maskIS*maskh],yy[maskIS*maskh],zz[maskIS*maskh],alpha=.2,marker=".",color=ringcolors[0],s=10,linewidths=0) 
pts0 = ax.scatter3D(xx[mask0*maskh],yy[mask0*maskh],zz[mask0*maskh],alpha=.2,marker=".",color=ringcolors[1],s=10,linewidths=0)      
#pts1 = ax.scatter3D(xx[mask1*maskh],yy[mask1*maskh],zz[mask1*maskh],alpha=.2,marker=".",color=ringcolors[2],s=10,linewidths=0)      
#pts2 = ax.scatter3D(xx[mask2*maskh],yy[mask2*maskh],zz[mask2*maskh],alpha=.4,marker=".",color=ringcolors[3],s=10,linewidths=0)                                   
#pts3 = ax.scatter3D(xx[mask3*maskh],yy[mask3*maskh],zz[mask3*maskh],alpha=.5,marker=".",color=ringcolors[4],s=10,linewidths=0)                                   
 
# plot update function   
def update_plot(kk):
    times = np.linspace(-geos.r_o+15,-geos.r_o-100,nframes)
    t = times[kk]
    xx,yy,zz,maskh=get_pts(t)

    #for jj in rings_to_plot:
    #    mask = masks[jj]
    #    pts[jj]._offsets3d=(xx[mask*maskh],yy[mask*maskh],zz[mask*maskh])
    #plt.draw()
    
    #ptsIS._offsets3d=(xx[maskIS*maskh],yy[maskIS*maskh],zz[maskIS*maskh])
    pts0._offsets3d=(xx[mask0*maskh],yy[mask0*maskh],zz[mask0*maskh])
    #pts1._offsets3d=(xx[mask1*maskh],yy[mask1*maskh],zz[mask1*maskh])
    #pts2._offsets3d=(xx[mask2*maskh],yy[mask2*maskh],zz[mask2*maskh])                
    #pts3._offsets3d=(xx[mask3*maskh],yy[mask3*maskh],zz[mask3*maskh])                
    plt.draw()   
               
# make and save animation
print("generating movie...")
ani = animation.FuncAnimation(fig,update_plot,nframes,interval=30)
writer = animation.writers['ffmpeg'](fps=fps)
ani.save(outfile,writer=writer,dpi=dpi,progress_callback=lambda i, nframes: print('frame', i+1, 'of', nframes))

plt.close('all')
        
