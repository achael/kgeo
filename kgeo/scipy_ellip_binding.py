# Uses newest version of scipy with Carlson elliptic integrals

import mpmath as mp
import numpy as np
import scipy.special as sp

halfpi = 0.5*np.pi

# oldest scalar version, unused; superseded by ellip_pi_arr
#def ellip_pi(n,phi,m,precision=1.e-3):
#    """Incomplete Elliptic Integral of the Third Kind Pi(n;phi|m)
#       Convention following Abramowitz & Stegun & Mathematica
#       DIFFERENT from gsl convention.
#       Implemented using CarlsonRJ and CarlsonRF and periodicities to account for full range
#    """
#
#    # real range is m sin^2(phi) < 1, n sin^2 phi < 1
#    # relation to Carlson symmetric form only works for -pi/2 < phi < pi/2,
#    # need to use periodicity outside this range
#
#    # limit as phi->0
#    if np.abs(phi)<=1.e-10: # TODO what value cutoff?
#        return phi
#
#    # limit as m->-infinity
#    if(m<-1.e14): # TODO what value cutoff?
#        return 0.
#
#    cphi = np.cos(phi)
#    sphi = np.sin(phi)
#
#    x = cphi**2
#    y = 1. - m*sphi**2
#    z = 1.
#    rho = 1. - n*sphi**2
#
#    # check the allowed region
#    if(y<0):
#        raise Exception("m*sin^2(phi) > 1 in ellipticPi!")
#    if(np.abs(phi)>np.pi/2. and m>=1):
#        raise Exception("abs(phi)>pi/2 and m >= 1 in ellipticPi!")
#    if(np.abs(phi)>np.pi/2. and n==1):
#        return np.inf
#    if (rho==0):
#        return np.inf
#    if (np.isinf(n)): # TODO large n limit
#        return 0.
#
#    # account for periodicity
#    if(np.abs(phi)>np.pi):
#        s = np.sign(phi)
#        k = s*np.floor(np.abs(phi)/np.pi)
#        phi2 = phi - k*np.pi
#        comp_part = 2*k*ellip_pi(n,np.pi/2.,m,precision=precision)
#        incomp_part = ellip_pi(n,phi2,m,precision=precision)
#        Pi = comp_part + incomp_part
#
#    elif(np.pi/2.<np.abs(phi)<=np.pi): #TODO can we merge with previous case somehow?
#        s = np.sign(phi)
#        phi2 = -phi + s*np.pi
#        comp_part = s*2*ellip_pi(n,np.pi/2.,m,precision=precision)
#        incomp_part = -ellip_pi(n,phi2,m,precision=precision)
#        Pi = comp_part + incomp_part
#
#    else:
#        # TODO should we just use scipy ellipkinc here?!
#
#        CRF = sp.elliprf(x,y,z)
#
#        if rho>0:
#            CRJ = sp.elliprj(x,y,z,rho)
#
#        else: # transform for rho<0, https://dlmf.nist.gov/19.20#E14 19.20.14
#            q = -rho
#            p = (z*(x+y+q) - x*y) / (z + q)
#
#            CRJ0 = sp.elliprj(x,y,z,p)
#            CRC = sp.elliprc(x*y + p*q,p*q)
#
#
#            CRJ = ((p-z)*CRJ0 - 3*CRF + 3*np.sqrt((x*y*z)/(x*y+p*q))*CRC)/(q+z)
#
#        F = sphi * CRF
#        Pi = F + (n/3.)*(sphi**3)*CRJ
#
#    return Pi

@np.errstate(divide='ignore', invalid='ignore') # x/0, 0/0 in branches that are computed but not used
def ellip_pi_arr(n,phi,m):
    """Incomplete Elliptic Integral of the Third Kind Pi(n;phi|m)
       Convention following Abramowitz & Stegun & Mathematica
       DIFFERENT from gsl convention.
       Implemented using CarlsonRJ and CarlsonRF and periodicities to account for full range
       """

    if not isinstance(n, np.ndarray): n  = np.array([n]).flatten()
    if not isinstance(phi, np.ndarray): phi = np.array([phi]).flatten()
    if not isinstance(m, np.ndarray): m = np.array([m]).flatten()

    #print(n.shape[-1],phi.shape[-1],m.shape[-1])
    if not(n.shape[-1]==phi.shape[-1]==m.shape[-1]):
        raise Exception("inputs to ellip_pi_arr do not have same last dimension!")

    # if dimensions aren't equal, expand parameter arrays # TODO??
    if (len(n.shape)!=len(phi.shape)):
        n = np.outer(np.ones(phi.shape[0]),n)
    if (len(m.shape)!=len(phi.shape)):
        m = np.outer(np.ones(phi.shape[0]),m)

    # real range is m sin^2(phi) < 1, n sin^2 phi < 1
    # relation to Carlson symmetric form only works for -pi/2 < phi < pi/2,
    # need to use periodicity outside this range

    # derived quantities
    cphi = np.cos(phi)
    sphi = np.sin(phi)

    x = cphi**2
    y = 1. - m*sphi**2
    z = np.ones(x.shape)
    rho = 1. - n*sphi**2


    outarr = np.zeros(phi.shape)
    donemask = np.zeros(phi.shape).astype(bool)

    # phi = 0 (the Carlson form below is accurate for small phi)
    #mask1 = np.abs(phi)<=1.e-10
    mask1 = phi==0
    if np.any(mask1):
        outarr[mask1] = phi[mask1]
    donemask += mask1

    # m = -infinity (the Carlson form below is accurate for large |m|)
    #mask2 = (m<-1.e14) * ~donemask
    mask2 = (m==-np.inf) * ~donemask
    if np.any(mask2):
        outarr[mask2] = 0.
    donemask += mask2

    # no real value: m sin^2 phi > 1
    mask3 = (y<0) * ~donemask
    if np.any(mask3):
        outarr[mask3] = np.nan
    donemask += mask3

    # no real value: path crosses m sin^2 = 1
    mask4 = (np.abs(phi)>halfpi) * (m>=1) * ~donemask
    if np.any(mask4):
        outarr[mask4] = np.nan
    donemask += mask4

    # divergent: double pole at pi/2  when n=1
    mask5 = (np.abs(phi)>halfpi) * (n==1) * ~donemask
    if np.any(mask5):
        #outarr[mask5] = np.inf
        outarr[mask5] = np.sign(phi[mask5])*np.inf # Pi is odd in phi
    donemask += mask5

    # divergent: phi on the pole n sin^2 = 1
    mask6 = (rho==0) * ~donemask
    if np.any(mask6):
        #outarr[mask6] = np.inf
        outarr[mask6] = np.sign(phi[mask6])*np.inf # Pi is odd in phi
    donemask += mask6

    # limit as n->+-infinity
    mask7 = np.isinf(n) * ~donemask
    if np.any(mask7):
        outarr[mask7] = 0
    donemask += mask7

    # account for periodicity
    mask8 = (np.abs(phi)>np.pi) * ~donemask
    if np.any(mask8):
        m_m = m[mask8]
        n_m = n[mask8]
        phi_m = phi[mask8]

        s = np.sign(phi_m)
        k = s*np.floor(np.abs(phi_m)/np.pi)
        phi2 = phi_m - k*np.pi
        comp_part = 2*k*ellip_pi_arr(n_m,halfpi*np.ones(phi2.shape),m_m)
        incomp_part = ellip_pi_arr(n_m,phi2,m_m)
        Pi = comp_part + incomp_part
        outarr[mask8] = Pi
    donemask += mask8

    mask9 = (np.abs(phi)>halfpi) * (np.abs(phi)<=np.pi) * ~donemask
    if np.any(mask9):
        m_m = m[mask9]
        n_m = n[mask9]
        phi_m = phi[mask9]

        s = np.sign(phi_m)
        phi2 = -phi_m + s*np.pi
        comp_part = s*2*ellip_pi_arr(n_m,halfpi*np.ones(phi2.shape),m_m)
        incomp_part = -ellip_pi_arr(n_m,phi2,m_m)
        Pi = comp_part + incomp_part
        outarr[mask9] = Pi
    donemask += mask9

    # all others
    nmask = ~donemask
    if np.any(nmask):
        m_m = m[nmask]
        n_m = n[nmask]
        phi_m = phi[nmask]

        sphi_m = sphi[nmask]
        x_m = x[nmask]
        y_m = y[nmask]
        z_m = z[nmask]
        rho_m = rho[nmask]

        q_m = -rho_m
        p_m = (z_m*(x_m+y_m+q_m) - x_m*y_m) / (z_m + q_m)
        aa = (x_m*y_m + p_m*q_m)
        bb = p_m*q_m
        fac1 = p_m - z_m
        fac2_num = (x_m*y_m*z_m)
        fac2_denom = (x_m*y_m+p_m*q_m)
        fac3 = q_m + z_m

        rhomask = rho_m > 0

        CRF = sp.elliprf(x_m,y_m,z_m)
        CRJ = np.zeros(phi_m.shape)
        if np.any(rhomask):
            CRJ[rhomask] = sp.elliprj(x_m[rhomask],y_m[rhomask],z_m[rhomask],rho_m[rhomask])

        rhomask = ~rhomask
        if np.any(rhomask): # transform for rho<0, https://dlmf.nist.gov/19.20#E14 19.20.14
            CRJ0 = sp.elliprj(x_m[rhomask],y_m[rhomask],z_m[rhomask],p_m[rhomask])
            CRC = sp.elliprc(aa[rhomask],bb[rhomask])

            fac2 = 3*np.sqrt(fac2_num[rhomask]/fac2_denom[rhomask])
            CRJ[rhomask] = (fac1[rhomask]*CRJ0 - 3*CRF[rhomask] + fac2*CRC)
            CRJ[rhomask] = CRJ[rhomask] / fac3[rhomask]


        F = sphi_m * CRF
        Pi = F + (n_m/3.)*(sphi_m**3)*CRJ
        outarr[nmask] = Pi

    return outarr

def ellip_pi_scaled(c, nc, phi, m, ctan=None):
    """c * Pi(n;phi|m) in the circular case m < n < 1, for c = +-C with C = sqrt((1-n)(n-m)/n),
       given c and the complementary characteristic nc = m/n instead of n.
       Finite as n -> 1 (C -> 0), where Pi(n;phi|m) itself diverges at phi = pi/2 + j pi.
       Uses DLMF 19.7.8 (Byrd & Friedman 117.02), extended to all real phi by quasi-periodicity:
       Pi(n;phi|m) = F(phi|m) - Pi(m/n;phi|m) + [arctan(C tan(Phi)/Delta(Phi)) + j pi]/C,
       where j = round(phi/pi), Phi = phi - j pi, Delta = sqrt(1 - m sin^2 Phi).
       At c = 0 this is j pi (sign(c) = +1 at c = 0).

       optionally pass ctan = (num, den), where num/den = c tan(phi) for |phi| <= pi/2,
       when c -> 0 and phi -> +-pi/2 together (an observer on the axis), c tan(phi) stays finite.

       c, nc, m have the shape of the last axis of phi.
    """

    if not isinstance(c, np.ndarray): c = np.array([c]).flatten()
    if not isinstance(nc, np.ndarray): nc = np.array([nc]).flatten()
    if not isinstance(phi, np.ndarray): phi = np.array([phi]).flatten()
    if not isinstance(m, np.ndarray): m = np.array([m]).flatten()

    j = np.round(phi/np.pi)
    Phi = phi - j*np.pi # in [-pi/2, pi/2], so cos(Phi) >= 0 and atan2 below is arctan(c tan(Phi)/Delta)
    Delta = np.sqrt(1. - m*np.sin(Phi)**2)
    sgn = np.where(c>=0, 1., -1.)
    if ctan is None:
        arct = np.arctan2(c*np.sin(Phi), np.cos(Phi)*Delta)
    else:
        arct = np.arctan2(ctan[0], ctan[1]*Delta) # arctan(c tan(phi)/Delta), |phi| <= pi/2 so j = 0

    return c*(sp.ellipkinc(phi, m) - ellip_pi_arr(nc, phi, m)) + arct + sgn*j*np.pi

def ellipj_m1(u, m1, niter=16):
    """Jacobi elliptic functions (sn, cn, dn, am) of u for parameter m = 1 - m1, 0 <= m1 <= 1,
       taking the complementary parameter m1 as input (same outputs as scipy.special.ellipj).
       Accurate as m -> 1, where scipy.special.ellipj loses dn at |u| ~ K (e.g. 1e-8 relative
       error for m1 = 1e-9, 2e-4 for m1 = 1e-13).
       Implemented with the AGM / descending Landen transformation (Numerical Recipes sncndn).
    """

    u = np.asarray(u, dtype=float)
    m1 = np.broadcast_to(np.asarray(m1, dtype=float), u.shape)

    # AGM sequence of (1, sqrt(m1))
    a = np.ones(u.shape)
    emc = m1.copy()
    ems = []
    ens = []
    for i in range(niter):
        ems.append(a.copy())
        emc = np.sqrt(emc)
        ens.append(emc.copy())
        c = 0.5*(a + emc)
        emc = emc*a
        a = c

    # back-substitution
    uu = u*c
    sn = np.sin(uu)
    cn = np.cos(uu)
    dn = np.ones(u.shape)
    nz = (sn != 0)
    aa = np.where(nz, cn/np.where(nz, sn, 1), 0.)
    cc = c*aa
    for b, en in zip(reversed(ems), reversed(ens)):
        aa = aa*cc
        cc = cc*dn
        dn = (en + aa)/(b + aa)
        aa = cc/b
    s = 1/np.sqrt(cc*cc + 1)
    sn = np.where(nz, np.where(sn >= 0, s, -s), sn)
    cn = np.where(nz, cc*sn, cn)
    dn = np.where(nz, dn, 1.)  # sn = 0 only at u = 2jK, where dn = 1

    # m = 1 exactly: sn = tanh, cn = dn = sech
    m1zero = (m1 == 0)
    sn = np.where(m1zero, np.tanh(u), sn)
    cn = np.where(m1zero, 1/np.cosh(u), cn)
    dn = np.where(m1zero, 1/np.cosh(u), dn)

    # amplitude: am = atan2(sn, cn) + 2 pi j, with j fixed by |am - pi u/(2K)| < pi/2
    K = sp.ellipkm1(m1)
    ph = np.arctan2(sn, cn)
    am = ph + 2*np.pi*np.round((0.5*np.pi*u/K - ph)/(2*np.pi))

    return (sn, cn, dn, am)



def test_ellip_pi(n = .8, m=-.33):

    args = [[n,.026,m],[n,.1,m],[n,np.pi/2.,m],[n,2.37,m],[n,4.1,m],
            [n,6.,m],[n,7.7,m],[n,14.3,m],[n,215.,m],
            [n,-.026,m],[n,-.1,m],[n,-np.pi/2.,m],[n,-2.37,m],[n,-4.1,m],
            [n,-6.,m],[n,-7.7,m],[n,-14.3,m],[n,-215.,m]]

    for k,arg in enumerate(args):
        mp_pi = mp.ellippi(arg[0],arg[1],arg[2])
        if np.imag(mp_pi)!=0.:
            print(k," complex result")
        else:
            mp_pi = np.real(mp_pi)
            gsl_pi = ellip_pi_arr(arg[0],arg[1],arg[2])
            print("%d | %.6e %.6e %.6e"%(k,mp_pi,gsl_pi,1.-gsl_pi/mp_pi))
    return
