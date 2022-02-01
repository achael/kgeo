import numpy as np
import scipy.special as sp
import mpmath
import matplotlib.pyplot as plt
from gsl_ellip_binding import ellip_pi_gsl
import h5py

MINSPIN = 1.e-6 # minimum spin for full formulas to work before taking limits. TODO check!
EP = 1.e-12

class Geodesics(object):

    def __init__(self,a, observer_coords, image_coords, mino_times, affine_times, geo_coords):
        # TODO add some consistency checks

        self.a = a
        self.observer_coords = observer_coords
        self.image_coords = image_coords
        self.mino_times = mino_times
        self.affine_times = affine_times
        self.geo_coords = geo_coords

        return

    # observer
    @property
    def t_o(self):
        return self.observer_coords[0]
    @property
    def r_o(self):
        return self.observer_coords[1]
    @property
    def th_o(self):
        return self.observer_coords[2]
    @property
    def ph_o(self):
        return self.observer_coords[3]

    # image
    @property
    def alpha(self):
        return self.image_coords[0]
    @property
    def beta(self):
        return self.image_coords[1]
    @property
    def npix(self):
        return len(self.alpha)
    @property
    def lam(self):
        return -self.alpha*np.sin(self.th_o)
    @property
    def eta(self):
        return (self.alpha**2 - self.a**2)*np.cos(self.th_o)**2 + self.beta**2
    @property
    def n_poloidal(self): # fractional number of poloidal orbits
        n_poloidal = n_poloidal_orbits(self.a, self.th_o, self.alpha, self.beta, self.tausteps)
        return n_poloidal
    @property
    def nmax_eq(self): # number of equatorial crossings
        nmax_eq = n_equatorial_crossings(self.a, self.th_o, self.alpha, self.beta, self.tautot)
        return nmax_eq

    # geodeiscs
    @property
    def tausteps(self):
        return self.mino_times
    @property
    def tautot(self):
        return self.mino_times[-1]
    @property
    def affinesteps(self):
        return self.affine_times
    @property
    def t_s(self):
        return self.geo_coords[0]
    @property
    def r_s(self):
        return self.geo_coords[1]
    @property
    def th_s(self):
        return self.geo_coords[2]
    @property
    def ph_s(self):
        return self.geo_coords[3]
    @property
    def sig_s(self):
        return self.affine_times

    def plotgeos(self,xlim=12,rmax=15,nplot=None,ngeoplot=50,plot_disk=True,
                 plot_inside_cc=True,plot_outside_cc=True):

        a = self.a
        th_o = self.th_o
        nmax_eq = self.nmax_eq
        r_s = self.r_s
        th_s = self.th_s
        ph_s = self.ph_s
        tausteps = self.tausteps

        # horizon
        rplus  = 1 + np.sqrt(1-a**2)

        # convert to cartesian for plotting
        x_s = r_s * np.cos(ph_s) * np.sin(th_s)
        y_s = r_s * np.sin(ph_s) * np.sin(th_s)
        z_s = r_s * np.cos(th_s)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        ax.plot_surface(rplus*np.cos(u) * np.sin(v),  rplus*np.sin(u) * np.sin(v),  rplus*np.cos(v), color='black')

        if plot_disk:
            rr, thth = np.mgrid[0:xlim, 0:2*np.pi:20j]
            xx = rr*np.cos(thth); yy = rr*np.sin(thth)
            zz = np.zeros(xx.shape)
            ax.plot_surface(xx, yy, zz, alpha=0.5)

        ax.set_xlim(-xlim,xlim)
        ax.set_ylim(-xlim,xlim)
        ax.set_zlim(-xlim,xlim)
        ax.auto_scale_xyz([-xlim, xlim], [-xlim, xlim], [-xlim, xlim])
        ax.set_axis_off()


        rmax2 = 3*rmax
        x_o = rmax2 * np.cos(0) * np.sin(th_o)
        y_o = rmax2 * np.sin(0) * np.sin(th_o)
        z_o = rmax2 * np.cos(th_o)
        ax.plot3D([0,x_o],[0,y_o],[0,z_o],'black',ls='dashed')

        maxwraps = int(np.nanmax(nmax_eq))
        colors = ['dimgrey','b','g','orange','r','m','c','y']

        print('maxwraps ', maxwraps)
        if nplot is None:
            nloop = np.min((maxwraps+1,len(colors)))
            nplot = range(nloop)
        else:
            nplot = np.array([nplot]).flatten()
        for j in nplot:
            mask = (nmax_eq==j)
            if not plot_inside_cc: #TODO make nicer
                mask *= (r_s[-1] > 10)
            if not plot_outside_cc:
                mask *= (r_s[-1] < 10)
            if np.sum(mask)==0: continue

            color = colors[j]
            xs = x_s[:,mask];ys = y_s[:,mask];zs = z_s[:,mask];
            rs = r_s[:,mask];tau = tausteps[:,mask]
            #trim = xs.shape[-1]//int(np.floor(ngeoplot*xs.shape[-1]/self.npix))
            trim = max(int(xs.shape[-1]/ngeoplot), 1)
            if xs.shape[-1] < 5 or j>=4:
                geos = range(xs.shape[-1])
            else:
                geos = range(0,xs.shape[-1],trim)

            for i in geos:
                x = xs[:,i]; y=ys[:,i]; z=zs[:,i]
                mask = ((rs[:,i] < rmax) + (tau[:,i] < .5*tau[-1,i]))
                mask *= rs[:,i] < rmax2
                x = x[mask]; y = y[mask]; z = z[mask]
                ax.plot3D(x,y,z,color)
            # do the last geodesic too #TODO WHAT SPACING
            i = xs.shape[-1]-1
            #if i not in geos:
            #    x = xs[:,i]; y=ys[:,i]; z=zs[:,i]
            #    mask = rs[:,i] < rmax
            #    x = x[mask]; y = y[mask]; z = z[mask]
            #    ax.plot3D(x,y,z,color)
        return

    def savegeos(self,path='./'):

        fname = path + 'a%0.2f_th%0.2f_geo.h5'%(self.a,self.th_o*180/np.pi)
        hf = h5py.File(fname,'w')
        hf.create_dataset('spin',data=self.a)
        hf.create_dataset('inc',data=self.th_o)
        hf.create_dataset('alpha',data=self.alpha)
        hf.create_dataset('beta',data=self.beta)
        hf.create_dataset('t',data=self.t_s)
        hf.create_dataset('r',data=self.r_s)
        hf.create_dataset('theta',data=self.th_s)
        hf.create_dataset('phi',data=self.ph_s)
        hf.create_dataset('affine',data=self.sig_s)
        hf.create_dataset('mino',data=self.tausteps)
        #hf.create_dataset('eq_crossings',data=nmax_eq)
        #hf.create_dataset('frac_orbits',data=n_tot)
        hf.close()

def angular_turning(a, th_o, lam, eta):
    """Calculate angular turning theta_pm points for a geodisic with conserved (lam,eta)"""

    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be float in range [0,1)")
    if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
        raise Exception("th_o should be float in range (0,pi/2)")
    if not isinstance(lam, np.ndarray): lam = np.array([lam]).flatten()
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if len(lam) != len(eta):
        raise Exception("lam, eta are different shapes!")

    (u_plus, u_minus, uratio, a2u_minus) = uplus_uminus(a,th_o,lam,eta)

    # angular turning points for normal motion
    th_plus = np.arccos(-np.sqrt(u_plus)) # this is GL 19a th_4
    th_minus = np.arccos(np.sqrt(u_plus)) # this is GL 19a th_1

    # for vortical motion, th_plus = th_2
    # assuming theta_0 is above the equator
    th_plus[eta<0] = np.arccos(np.sqrt(u_minus[eta<0]))

    # angular class
    thclass = np.zeros(lam.shape).astype(int)
    # eta < 0: ordinary motion between the turning points
    thclass[eta>0] = 1
    # eta > 0: vortical motion
    thclass[eta<0] = 2
    # eta == 0: limit of vortical motion in upper half plane unless th_o=90 and beta=0
    thclass[eta==0] = 3

    return(u_plus, u_minus, th_plus, th_minus, thclass)

def uplus_uminus(a,th_o,lam,eta):
    """Calculate u_+, u_- and ratios u_+/u_-, a^2 u_+/u_-
       including in spin->0 limit"""

    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
        raise Exception("th_o should be a float in range (0,pi/2)")
    if not isinstance(lam, np.ndarray): lam = np.array([lam]).flatten()
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if len(lam) != len(eta):
        raise Exception("lam, eta are different lengths!")

    if(a<MINSPIN):  # spin 0 limit
        u_plus = eta / (eta + lam*lam)
        u_minus = u_plus
    else:
        #GL 19b Eqn 11
        Delta_theta = 0.5*(1 - (eta + lam*lam)/(a*a))
        u_plus = Delta_theta + np.sqrt(Delta_theta**2 + eta/(a*a))
        u_minus = Delta_theta - np.sqrt(Delta_theta**2 + eta/(a*a))

    # ensure th_o is inside the turning points [th_1,th_4] exactly
    mask = (np.cos(th_o)**2 - u_plus) > 0
    u_plus[mask] = np.cos(th_o)**2

    # for geodesics with eta==0, exactly u_minus=0.
    # This breaks some equations for th integrals
    # bump up u_minus to a small value # TODO ok?
    u_minus[(u_minus==0)*(eta<0)] = EP
    u_minus[(u_minus==0)*(eta>=0)] = -EP

    # for vortical geodeiscs, ensure th_o is inside [th_minus, th_2] exactly
    maskm = (np.cos(th_o)**2 - u_minus[eta<0]) < 0
    mask = np.zeros(eta.shape).astype(bool)
    mask[eta<0] = maskm
    u_minus[mask] = np.cos(th_o)**2

    # compute ratio
    if(a<MINSPIN):
        uratio = 0. * u_minus
        a2u_minus = -(eta+lam**2)
    else:
        uratio = u_plus/u_minus
        a2u_minus = a**2 * u_minus

    return (u_plus, u_minus, uratio, a2u_minus)

def radial_roots(a, lam, eta):
    """Calculate radial roots r1,r2,r3,r4, for a geodisic with conserved (lam,eta)"""

    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be float in range [0,1)")
    if not isinstance(lam, np.ndarray): lam = np.array([lam]).flatten()
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if len(lam) != len(eta):
        raise Exception("lam, eta are different shapes!")

    # horizon radii
    rplus  = 1 + np.sqrt(1-a**2)
    rminus = 1 - np.sqrt(1-a**2)

    #GL 19a Eqn 95
    A = a**2 - eta - lam**2
    B = 2*(eta + (lam-a)**2) #>0
    C = -(a**2)*eta
    P = -(A**2)/12. - C
    Q = (-A/3.)*((A**2)/36. - C) - (B**2)/8.

    mDelta3 =  (4*(P**3) + 27*(Q**2)).astype(complex)
    omegap3 = -0.5*Q + np.sqrt(mDelta3/108.)
    omegam3 = -0.5*Q - np.sqrt(mDelta3/108.)
    omegap = my_cbrt(omegap3)
    omegam = my_cbrt(omegam3)
    xi0 = omegap + omegam - A/3.
    z = np.sqrt(0.5*xi0)

    r1 = -z - np.sqrt((-0.5*A - z**2 + 0.25*B/z).astype(complex))
    r2 = -z + np.sqrt((-0.5*A - z**2 + 0.25*B/z).astype(complex))
    r3 =  z - np.sqrt((-0.5*A - z**2 - 0.25*B/z).astype(complex))
    r4 =  z + np.sqrt((-0.5*A - z**2 - 0.25*B/z).astype(complex))

    # determine the radial case
    rclass = np.zeros(r1.shape).astype(int)
    # case IV: no real roots
    # inside critical curve, motion between -z and infinity with no radial turning
    rclass[np.imag(r2) != 0 ] = 4
    # case III: two real roots inside horiz.
    # inside critical curve, motion between r2 and infinity with no radial turning
    rclass[(np.imag(r2) == 0) * (np.imag(r4) != 0)] = 3
    # case II: four real roots inside horiz.
    # inside critical curve, motion between r4 and infinity with no radial turning
    rclass[(np.imag(r2) == 0) * (np.imag(r4) == 0) * (np.real(r4) < rplus)] = 2
    # case I: four real roots, two outside horiz.
    # outside critical curve, motion between r4 and infinity with one radial turning
    rclass[(np.imag(r2) == 0) * (np.imag(r4) == 0) * (np.real(r4) > rplus)] = 1

    return (r1,r2,r3,r4,rclass)

def mino_total(a, r_o, eta, r1, r2, r3, r4):
    """Maximal mino time elapsed for geodesic with radial roots r1,r2,r3,r4"""

    # checks
    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    if not (isinstance(r_o,float) and (r_o>=100)):
        raise Exception("r_o should be a float >= 100")
    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if not(len(eta)==len(r1)==len(r2)==len(r3)==len(r4)):
        raise Exception("mino_total input arrays are different lengths!")

    if not isinstance(eta, np.ndarray): eta = np.array([eta]).flatten()
    if not isinstance(r1, np.ndarray): r1 = np.array([r1]).flatten()
    if not isinstance(r2, np.ndarray): r2 = np.array([r2]).flatten()
    if not isinstance(r3, np.ndarray): r3 = np.array([r3]).flatten()
    if not isinstance(r4, np.ndarray): r4 = np.array([r4]).flatten()
    if not(len(eta) == len(r1) == len(r2) == len(r3) == len(r4)):
        raise Exception("inputs to mino_total are different lengths!")

    rplus  = 1 + np.sqrt(1-a**2)
    rminus = 1 - np.sqrt(1-a**2)

    # output array
    Imax_out = np.zeros(r1.shape)

    # TODO what to do if we are on the critical curves of double roots between regions?

    # no real roots -- region IV
    mask_IV = (np.imag(r1) != 0)
    if np.any(mask_IV):

        # GL 19b, appendix A and GL 19a appendix B.4

        # real and imaginary parts of r4,r2, GL19a B10, B11
        a1 = np.imag(r4)[mask_IV] # a1 > 0
        b1 = np.real(r4)[mask_IV] # b1 > 0
        a2 = np.imag(r2)[mask_IV] # a2 > 0
        b2 = np.real(r2)[mask_IV] # b2 < 0

        # parameters for case 4
        CC = np.sqrt((a1-a2)**2 + (b1-b2)**2) # equal to sqrt(r31*r42)>0, GL19a B85
        DD = np.sqrt((a1+a2)**2 + (b1-b2)**2) # equal to sqrt(r32*r41)>0, GL19a B85
        k4 = (4*CC*DD)/((CC+DD)**2) # 0<k4<1, GL19a B87
        g0 = np.sqrt((4*a2**2 - (CC-DD)**2)/ ((CC+DD)**2 - 4*a2**2)) # 0<g0<1, GL19a B88

        x4rp = (rplus + b1)/a2 # GL19a, B83 for origin at horizon
        if r_o==np.infty:
            x4ro = np.infty # GL19a, B83 for observer at r_o
        else:
            x4ro = (r_o + b1)/a2 # GL19a, B83 for observer at r_o

        # antiderivative:
        pref = 2./(CC+DD)
        I4rp = pref*sp.ellipkinc(np.arctan(x4rp) + np.arctan(g0),k4) #GL19a B101
        I4ro = pref*sp.ellipkinc(np.arctan(x4ro) + np.arctan(g0),k4)

        # class IV geodesics terminate at horizon
        Imax = I4ro - I4rp
        Imax_out[mask_IV] = Imax

    # two roots (r3, r4) imaginary -- region III
    mask_III = (np.imag(r3) != 0) * (~mask_IV)
    if np.any(mask_III):

        # GL 19b, Equation A14 and GL 19a appendix B.3

        # real and imaginary parts of r4 GL19a B10
        a1 = np.imag(r4)[mask_III] # a1 > 0
        b1 = np.real(r4)[mask_III] # b1 > 0
        rr1 = np.real(r1)[mask_III] # r1 is real
        rr2 = np.real(r2)[mask_III] # r2 is real
        rr21 = rr2 - rr1
        rrp1 = rplus - rr1
        rrp2 = rplus - rr2
        rrm1 = rminus - rr1
        rrm2 = rminus - rr2
        rro1 = r_o - rr1
        rro2 = r_o - rr2

        # parameters for case 3
        AA = np.sqrt(a1**2 + (b1-rr2)**2) # equal to sqrt(r32*r42)>0, GL19a B85
        BB = np.sqrt(a1**2 + (b1-rr1)**2) # equal to sqrt(r31*r41)>0, GL19a B85
        k3 = ((AA + BB)**2 - rr21**2)/(4*AA*BB) # 0<k3<1, GL19a B59

        x3rp = (AA*rrp1 - BB*rrp2)/(AA*rrp1 + BB*rrp2) # GL19a, B55
        x3rm = (AA*rrm1 - BB*rrm2)/(AA*rrm1 + BB*rrm2) # GL19a, B55
        if r_o==np.infty:
            x3ro = (AA - BB)/(AA + BB)  # GL19a, B55 for observer at r_o
        else:
            x3ro = (AA*rro1 - BB*rro2)/(AA*rro1 + BB*rro2)

        # antiderivatives
        pref = 1. / np.sqrt(AA*BB)
        I3rp = pref * sp.ellipkinc(np.arccos(x3rp), k3) #GL19a B71
        I3ro = pref * sp.ellipkinc(np.arccos(x3ro), k3)

        # class III geodesics terminate at horizon
        Imax = I3ro - I3rp
        Imax_out[mask_III] = Imax

    # all roots real, r3<r4<rh -- region II (case 2)
    mask_II = (np.imag(r3) == 0.) * (r3<rplus) * (~mask_III)
    if np.any(mask_II):

        # GL 19b, Equation A13  and GL 19a appendix B.2

        # roots in this region
        rr1 = np.real(r1)[mask_II]
        rr2 = np.real(r2)[mask_II]
        rr3 = np.real(r3)[mask_II]
        rr4 = np.real(r4)[mask_II]
        rr31 = rr3 - rr1
        rr32 = rr3 - rr2
        rr41 = rr4 - rr1
        rr42 = rr4 - rr2

        # parameters for case 2
        k2 = (rr32*rr41)/(rr31*rr42)# 0<k=k2<1, GL19a B13
        x2rp = np.sqrt((rr31*(rplus-rr4))/(rr41*(rplus-rr3)))# GL19a, B35
        if r_o==np.infty:
            x2ro = np.sqrt(rr31/rr41)# B35, for observer at r_o
        else:
            x2ro = np.sqrt((rr31*(r_o-rr4))/(rr41*(r_o-rr3)))# GL19a, B35

        # antiderivatives
        pref = 2./np.sqrt(rr31*rr42)
        I2rp = pref*sp.ellipkinc(np.arcsin(x2rp), k2) # GL19a B40
        I2ro = pref*sp.ellipkinc(np.arcsin(x2ro), k2)

        # class II geodesics terminate at horizon
        Imax = I2ro - I2rp
        Imax_out[mask_II] = Imax

    # all roots real, r4>r3>rh -- region I (also case 2)
    mask_I = (np.imag(r3) == 0.) * (r3>rplus) * (~mask_II)
    if np.any(mask_I):

        # GL 19b, Equation A10  and GL 19a appendix B.2

         # roots in this region
        rr1 = np.real(r1)[mask_I]
        rr2 = np.real(r2)[mask_I]
        rr3 = np.real(r3)[mask_I]
        rr4 = np.real(r4)[mask_I]
        rr31 = rr3 - rr1
        rr32 = rr3 - rr2
        rr41 = rr4 - rr1
        rr42 = rr4 - rr2

        # parameters for case 2
        k2 = (rr32*rr41)/(rr31*rr42)# 0<k=k2<1, GL19a B13
        if r_o==np.infty:
            x2ro = np.sqrt(rr31/rr41)# B35, for observer at r_o
        else:
            x2ro = np.sqrt((rr31*(r_o-rr4))/(rr41*(r_o-rr3)))# GL19a, B35

        # antiderivatives
        pref = 2./np.sqrt(rr31*rr42)
        I2ro = pref*sp.ellipkinc(np.arcsin(x2ro), k2) # GL19a B40

        # class I geodesics terminate at infinity
        Imax = 2*I2ro
        Imax_out[mask_I] = Imax

    return Imax_out


def n_poloidal_orbits(a, th_o, alpha, beta, tau):
    """the number of poloidal orbits as a function of Mino time tau (GL 19b Eq 35)
       only applies for normal geodesics eta>0"""

    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
        raise Exception("th_o should be a float in range (0,pi/2)")
    if len(alpha) != len(beta):
        raise Exception("alpha, beta are different shapes!")
    if not(tau.shape[1]==len(alpha)):
        raise Exception("tau has incompatible shape in n_poloidal_orbits!")

    lam = -alpha*np.sin(th_o)
    eta = (alpha**2 - a**2)*np.cos(th_o)**2 + beta**2

    (u_plus, u_minus, uratio, a2u_minus) = uplus_uminus(a,th_o,lam,eta)

    K = sp.ellipk(uratio) # gives NaN for eta<0
    n_poloidal = (np.sqrt(-a2u_minus.astype(complex))*tau)/(4*K)
    n_poloidal = np.real(n_poloidal.astype(complex))

    return n_poloidal

def n_equatorial_crossings(a, th_o, alpha, beta, tau):
    """ the fractional number of equatorial crossings
        equation only applies for normal geodesics eta>0"""

    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
        raise Exception("th_o should be a float in range (0,pi/2)")
    if not isinstance(alpha, np.ndarray): alpha = np.array([alpha]).flatten()
    if not isinstance(beta, np.ndarray): beta = np.array([beta]).flatten()
    if len(alpha) != len(beta):
        raise Exception("alpha, beta are different shapes!")
    if not(tau.shape[-1]==len(alpha)):
        raise Exception("tau has incompatible shape in n_equatorial_crossings!")

    lam = -alpha*np.sin(th_o)
    eta = (alpha**2 - a**2)*np.cos(th_o)**2 + beta**2

    (u_plus, u_minus, uratio, a2u_minus) = uplus_uminus(a,th_o,lam,eta)

    s_o = my_sign(beta)  # sign of final angular momentum
    F_o = sp.ellipkinc(np.arcsin(np.cos(th_o)/np.sqrt(u_plus)), uratio) # gives NaN for eta<0
    K = sp.ellipk(uratio) # gives NaN for eta<0
    nmax_eq = ((tau*np.sqrt(-a2u_minus.astype(complex)) + s_o*F_o) / (2*K))  + 1
    nmax_eq[beta>=0] -= 1
    nmax_eq = np.floor(np.real(nmax_eq.astype(complex)))
    nmax_eq[np.isnan(nmax_eq)] = 0

    return nmax_eq

def is_outside_crit(a, th_o, alpha, beta):
    """is the point alpha, beta outside the critical curve?"""

    # checks
    if not (isinstance(a,float) and (0<=a<1)):
        raise Exception("a should be a float in range [0,1)")
    if not (isinstance(th_o,float) and (0<th_o<=np.pi/2.)):
        raise Exception("th_o should be a float in range (0,pi/2)")
    if not isinstance(alpha, np.ndarray): alpha = np.array([alpha]).flatten()
    if not isinstance(beta, np.ndarray): beta = np.array([beta]).flatten()
    if len(alpha) != len(beta):
        raise Exception("alpha, beta are different shapes!")

    # horizon radius
    rh = 1 + np.sqrt(1-a**2)

    # conserved quantities
    lam = -alpha*np.sin(th_o)
    eta = (alpha**2 - a**2)*(np.cos(th_o))**2 + beta**2

    outarr = np.empty(alpha.shape)

    # vortical region is always inside critical curve
    vortmask = eta<0
    outarr[vortmask] = 0

    # points outside the critical curve have real r4>rh
    (r1,r2,r3,r4,rclass) = radial_roots(a,lam,eta)
    mask = (np.imag(r4)==0) * (r4>rh) * ~vortmask
    outarr[mask] = 1
    outarr[~mask] = 0

    return outarr

def my_cbrt(x):

    s = np.sign(x)
    r = np.abs(x)
    th = np.angle(x)
    proot = np.empty(x.shape).astype(complex)

    realmask = (np.imag(x)==0)
    proot[realmask] = (s*np.cbrt(r))[realmask]

    if np.any(~realmask):
        root1 = np.cbrt(r) * np.exp(1j*th/3.)
        root2 = np.cbrt(r) * np.exp(1j*th/3. + 2j*np.pi/3.)
        root3 = np.cbrt(r) * np.exp(1j*th/3. - 2j*np.pi/3.)

        root1mask = (~realmask * (np.real(root1) > np.real(root2)) *
                                 (np.real(root1) > np.real(root3)))
        root2mask = (~realmask * (np.real(root2) > np.real(root1)) *
                                 (np.real(root2) > np.real(root3)))
        root3mask = (~realmask * (np.real(root3) > np.real(root1)) *
                                 (np.real(root3) > np.real(root2)))

        proot[root1mask] = root1[root1mask]
        proot[root2mask] = root2[root2mask]
        proot[root3mask] = root3[root3mask]


    return proot

def my_sign(x):
    # TODO are these signs right?
    out = np.zeros(x.shape)
    #out[x==0] = 0
    out[x>=0] = 1
    #out[x>0] = 1
    out[x<0] = -1
    return out


# def intersect_plane(th_n, ph_n, r_s, th_s, ph_s):
#
#     nint = np.zeros(r_s.shape[-1])
#     for i in range(r_s.shape[-1]): # TODO speed up #TODO right indexing?
#         r = r_s[:,i];th=th_s[:,i];ph=ph_s[:,i]
#
#         # find where geodesic passes through plane with normal at th_n, ph_n through origin.
#         # NOTE: pass in a single geodesic (for now)
#
#         xn = np.array([np.cos(ph_n)*np.sin(th_n),np.sin(ph_n)*np.sin(th_n),np.cos(th_n)])
#         xo = np.array([r*np.cos(th)*np.sin(th),r*np.sin(th)*np.cos(th),r*np.cos(th)]).T
#
#         d = xo[1:] - xo[:-1] # vector between geodeisc points
#         t = np.dot(-xo[:-1],xn)/np.dot(d,xn)
#         intersect = (0 < t) * (t < 1)
#         nintersect = np.sum(intersect)
#         print(xn)
#         nint[i] = nintersect
