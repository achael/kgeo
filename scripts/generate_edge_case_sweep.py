import matplotlib.pyplot as plt
import numpy as np

from kgeo.kerr_raytracing_ana import raytrace_ana

rout=1.e4
ngeo=100
maxtaufrac=1-1.e-10
npix=1001

plt.close('all')
for i,spin in enumerate([-0.9,0,0.9]):
    for j,inc in enumerate([0,0.5*np.pi,np.pi]):
        for k,b0 in enumerate([True,False]):

            if b0:
                alpha=np.linspace(-6,6,npix)
                beta=0*np.linspace(-6,6,npix)
                titlestr = 'beta=0'
            else:
                alpha=0*np.linspace(-6,6,npix)
                beta=np.linspace(-6,6,npix)
                titlestr = 'alpha=0'
            title=r'a=%.1f, inc=%.1f pi, %s'%(spin,inc/np.pi,titlestr)
            print(title)
            fig=plt.figure(100*i + 10*j + k)
            geos = raytrace_ana(a=spin,observer_coords=[0,rout,inc,0],image_coords=[alpha,beta],ngeo=ngeo,maxtaufrac=maxtaufrac,plotdata=False)
            geos.plotgeos(n_label='equatorial',fig=fig)
            plt.title(title)
            print()

plt.show()
