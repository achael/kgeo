
from scipy.interpolate import interp1d
import matplotlib.animation as animation
from pylab import *
import numpy as np




self=geos
rplus  = 1 + np.sqrt(1-self.a**2)
    
r_funs = [interp1d(self.t_s[:,i],self.r_s[:,i],bounds_error=False,fill_value=0) for i in range(len(self.alpha))]
th_funs = [interp1d(self.t_s[:,i],self.th_s[:,i],bounds_error=False,fill_value=0) for i in range(len(self.alpha))]
ph_funs = [interp1d(self.t_s[:,i],self.ph_s[:,i],bounds_error=False,fill_value=0) for i in range(len(self.alpha))]

def get_pts(t):
    r_t = [r_funs[i](t) for i in range(len(self.alpha))] 
    th_t = [th_funs[i](t) for i in range(len(self.alpha))] 
    ph_t = [ph_funs[i](t) for i in range(len(self.alpha))] 

    #i = 1
    #r_t = self.r_s[i]
    #th_t = self.th_s[i]
    #ph_t = self.ph_s[i]


    mask = r_t>rplus

    xx = (r_t*np.sin(th_t)*np.cos(ph_t))#[mask]
    yy = (r_t*np.sin(th_t)*np.sin(ph_t))#[mask]
    zz = (r_t*np.cos(th_t))#[mask]
    return xx,yy,zz,mask


xlim = 10
nframes = 10
plt.close('all')
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(projection='3d')
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
ax.plot_surface(rplus*np.cos(u) * np.sin(v),  rplus*np.sin(u) * np.sin(v),  rplus*np.cos(v), color='black',zorder=1000)

#rmax2 = 3*self.r_o
#x_o = rmax2 * np.cos(self.ph_o) * np.sin(self.th_o)
#y_o = rmax2 * np.sin(self.ph_o) * np.sin(self.th_o)
#z_o = rmax2 * np.cos(self.th_o)
#ax.plot3D([0,x_o],[0,y_o],[0,z_o],'black',ls='dashed')

ax.set_xlim(-xlim,xlim)
ax.set_ylim(-xlim,xlim)
ax.set_zlim(-xlim,xlim)
ax.auto_scale_xyz([-xlim, xlim], [-xlim, xlim], [-xlim, xlim])
ax.set_axis_off()      

xxeq, yyeq = np.mgrid[-xlim:xlim, -xlim:xlim:100j]
zzeq = np.zeros(xxeq.shape)
ax.plot_surface(xxeq, yyeq, zzeq, alpha=0.3,linewidth=0,color='lightblue')

fig.tight_layout()

        
nmax_eq = geos.nmax_eq
maskIS = (nmax_eq==-2) + (nmax_eq==-1)
mask0 = nmax_eq==0
mask1 = nmax_eq==1
mask2 = nmax_eq==2
mask3 = nmax_eq==3 

xx,yy,zz,maskh=get_pts(-self.r_o+10)
ptsIS = ax.scatter3D(xx[maskIS*maskh],yy[maskIS*maskh],zz[maskIS*maskh],alpha=.2,marker=".",color='darkgrey',s=10,linewidths=0) 
pts0 = ax.scatter3D(xx[mask0*maskh],yy[mask0*maskh],zz[mask0*maskh],alpha=.2,marker=".",color='b',s=10,linewidths=0)      
pts1 = ax.scatter3D(xx[mask1*maskh],yy[mask1*maskh],zz[mask1*maskh],alpha=.2,marker=".",color='g',s=10,linewidths=0)      
pts2 = ax.scatter3D(xx[mask2*maskh],yy[mask2*maskh],zz[mask2*maskh],alpha=.4,marker=".",color='orange',s=10,linewidths=0)                                   
pts3 = ax.scatter3D(xx[mask3*maskh],yy[mask3*maskh],zz[mask3*maskh],alpha=.5,marker=".",color='r',s=10,linewidths=0)                                   
        
def update_plot(kk):
    times = np.linspace(-self.r_o+15,-self.r_o-100,nframes)
    t = times[kk]
    xx,yy,zz,maskh=get_pts(t)

    ptsIS._offsets3d=(xx[maskIS*maskh],yy[maskIS*maskh],zz[maskIS*maskh])
    pts0._offsets3d=(xx[mask0*maskh],yy[mask0*maskh],zz[mask0*maskh])
    pts1._offsets3d=(xx[mask1*maskh],yy[mask1*maskh],zz[mask1*maskh])
    pts2._offsets3d=(xx[mask2*maskh],yy[mask2*maskh],zz[mask2*maskh])                
    pts3._offsets3d=(xx[mask3*maskh],yy[mask3*maskh],zz[mask3*maskh])                
    plt.draw()   
               

#for kk in range(nframes):
#    plt.pause(1.e-5)
#    update_plot(kk)



ani = animation.FuncAnimation(fig,update_plot,nframes,interval=30)
writer = animation.writers['ffmpeg'](fps=10)
ani.save('demo2.mp4',writer=writer,dpi=100,progress_callback=lambda i, nframes: print(i))

plt.close('all')
        
#ax.plot_trisurf(xx, yy, zz, alpha=0.5)
#ax.plot_surface(xx.reshape(1,len(xx)), yy.reshape(1,len(xx)), zz.reshape(1,len(xx)), alpha=0.5)
