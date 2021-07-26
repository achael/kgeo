import numpy as np
import scipy.special as sp
import mpmath
from gsl_ellip_binding import ellip_pi_gsl

MINSPIN = 1.e-6 # minimum spin for full formulas to work before taking limits. TODO check!
EP = 1.e-10

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

    if(a<MINSPIN):  # spin 0 limit
        u_plus = eta / (eta + lam*lam)
        u_minus = u_plus
    else:
        #GL 19b Eqn 11
        Delta_theta = 0.5*(1 - (eta + lam*lam)/(a*a))
        u_plus = Delta_theta + np.sqrt(Delta_theta**2 + eta/(a*a))
        u_minus = Delta_theta - np.sqrt(Delta_theta**2 + eta/(a*a))

    # ensure th_o is inside the turning points [th_1,th_4] exactly
    #xFarg = np.cos(th_o)/np.sqrt(u_plus)
    #mask = (xFarg - 1) > 0 #or use EP?

    mask = (np.cos(th_o)**2 - u_plus) > 0
    u_plus[mask] = np.cos(th_o)**2

    # for geodesics with eta==0, exactly u_minus=0.
    # This breaks some equations for th integrals
    # bump up u_minus to a small value # TODO ok?
    u_minus[(u_minus==0)*(eta<0)] = EP
    u_minus[(u_minus==0)*(eta>=0)] = -EP

    # for vortical geodeiscs, ensure th_o is inside [th_minus, th_2] exactly
    #xFarg2 = (np.cos(th_o)/np.sqrt(u_minus[eta<0]))
    #maskm = (xFarg2 - 1) < 0 #or use EP?

    maskm = (np.cos(th_o)**2 - u_minus[eta<0]) < 0
    mask = np.zeros(eta.shape).astype(bool)
    mask[eta<0] = maskm
    u_minus[mask] = np.cos(th_o)**2


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
            x2ro = np.sqrt(rr31/rr41)# B35, for observer at r_o # TODO -- finite radius observer?
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
            x2ro = np.sqrt(rr31/rr41)# B35, for observer at r_o # TODO -- finite radius observer?
        else:
            x2ro = np.sqrt((rr31*(r_o-rr4))/(rr41*(r_o-rr3)))# GL19a, B35

        # antiderivatives
        pref = 2./np.sqrt(rr31*rr42)
        I2ro = pref*sp.ellipkinc(np.arcsin(x2ro), k2) # GL19a B40

        # class I geodesics terminate at infinity
        Imax = 2*I2ro
        Imax_out[mask_I] = Imax

    return Imax_out

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
