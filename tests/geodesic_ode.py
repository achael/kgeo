"""Independent numerical oracle for Kerr null geodesics (used by tests only).

Integrates the Mino-time equations of motion (GL19a eqs 7-10) with scipy's
DOP853. u = 1/r and theta are integrated in second-order form,

    d2u/dtau2 = Ru'(u)/2,      d2theta/dtau2 = Theta'(theta)/2,

with Ru(u) = u^4 R(1/u), so radial and polar turning points need no sign
bookkeeping and the radial error stays uniform from the screen to the horizon.

Conventions match kgeo's analytic raytracer: Mino time tau >= 0 increases
backward from the observer (tau_kgeo = -tau_forward + const), so
dt/dtau, dphi/dtau and dsigma/dtau are minus the forward-time expressions,
and at the screen dr/dtau = -sqrt(R), dtheta/dtau = -sign(beta) sqrt(Theta).
"""

import numpy as np
from scipy.integrate import solve_ivp


def conserved(a, th_o, alpha, beta):
    """lambda, eta for screen coordinates (alpha, beta) (GL19b eq 55)"""
    lam = -alpha*np.sin(th_o)
    eta = (alpha**2 - a**2)*np.cos(th_o)**2 + beta**2
    return lam, eta


def radial_potential(r, a, lam, eta):
    """R(r), GL19a eq 5"""
    return (r**2 + a**2 - a*lam)**2 - (r**2 - 2*r + a**2)*(eta + (lam - a)**2)


def angular_potential(th, a, lam, eta):
    """Theta(theta), GL19a eq 6; the lambda term is dropped exactly when lambda == 0"""
    out = eta + a**2*np.cos(th)**2
    if lam != 0:
        out = out - lam**2/np.tan(th)**2
    return out


def inverse_radial_potential(u, a, lam, eta):
    """u^4 R(1/u) = 1 + A u^2 + B u^3 + C u^4, with A, B, C as in GL19a eq 95"""
    A = a**2 - eta - lam**2
    B = 2*(eta + (lam - a)**2)
    C = -a**2*eta
    return 1 + A*u**2 + B*u**3 + C*u**4


def _rhs(tau, y, a, lam, eta):
    """d/dtau of the state y = (u=1/r, du/dtau, theta, dtheta/dtau, phi, t, sigma)

    Radial motion uses u = 1/r: (du/dtau)^2 = u^4 R(1/u) is O(1) from the
    screen to the horizon, so integration error doesn't grow as R(r) falls by
    ~10 orders of magnitude from r_o inward (as it would integrating r itself).

    Differentiating (du/dtau)^2 = Ru(u) and (dtheta/dtau)^2 = Theta(theta)
    gives d2u/dtau2 = Ru'(u)/2 and d2theta/dtau2 = Theta'(theta)/2,
    which are regular at turning points and invariant under tau -> -tau.
    """
    u, udot, th, thdot, _, _, _ = y
    r = 1/u
    delta = r**2 - 2*r + a**2
    sth, cth = np.sin(th), np.cos(th)

    # Ru'(u) = 2A u + 3B u^2 + 4C u^3
    A = a**2 - eta - lam**2
    B = 2*(eta + (lam - a)**2)
    C = -a**2*eta
    dRu = 2*A*u + 3*B*u**2 + 4*C*u**3

    # Theta'(theta) = -2 a^2 cos sin + 2 lam^2 cos/sin^3
    dTH = -2*a**2*cth*sth

    # forward-time dphi/dtau (GL19a eq 9) and dt/dtau (eq 10)
    phidot = a*(r**2 + a**2 - a*lam)/delta - a
    if lam != 0:  # skip exactly at lam == 0 so on-axis rays avoid 0*inf
        dTH += 2*lam**2*cth/sth**3
        phidot += lam/sth**2
    tdot = (r**2 + a**2)*(r**2 + a**2 - a*lam)/delta - a*(a*sth**2 - lam)

    # forward-time dsigma/dtau = Sigma (affine parameter)
    sigdot = r**2 + a**2*cth**2

    # kgeo's tau runs backward in time: t, phi, sigma use minus the forward rates
    return [udot, 0.5*dRu, thdot, 0.5*dTH, -phidot, -tdot, -sigdot]


def integrate_geodesic(a, r_o, th_o, alpha, beta, taus, rtol=1e-12, atol=1e-12):
    """Integrate one geodesic back from the observer at (r_o, th_o, phi_o=0)
    and sample it at the Mino times taus (increasing, starting at >= 0).

    Returns a dict of arrays shaped like taus: 't', 'r', 'th', 'ph', 'sig',
    and 'folds', the number of pole crossings folded into th and ph
    (always 0 unless lambda == 0).
    """
    a = float(a); r_o = float(r_o); th_o = float(th_o)
    alpha = float(alpha); beta = float(beta)
    lam, eta = conserved(a, th_o, alpha, beta)

    # initial velocities at the screen; sign(0) = +1 as in kgeo's my_sign
    # dr/dtau = -sqrt(R) <=> du/dtau = -u^2 dr/dtau = +sqrt(u^4 R(1/u))
    u_o = 1/r_o
    sign_beta = 1. if beta >= 0 else -1.
    udot_o = np.sqrt(inverse_radial_potential(u_o, a, lam, eta))
    thdot_o = -sign_beta*np.sqrt(max(angular_potential(th_o, a, lam, eta), 0.))

    taus = np.asarray(taus, dtype=float)
    y0 = [u_o, udot_o, th_o, thdot_o, 0., 0., 0.]
    sol = solve_ivp(_rhs, (0., taus[-1]), y0, method='DOP853', t_eval=taus,
                    rtol=rtol, atol=atol, args=(a, lam, eta))
    if not sol.success:
        raise RuntimeError(f"ODE oracle failed: {sol.message}")

    u, _, th, _, ph, t, sig = sol.y

    # lam == 0: the ray stays in a meridional plane and theta passes smoothly
    # through the poles. A signed theta outside [0, pi] is the same point in the
    # opposite half-plane, so fold it back and add pi to phi per pole crossing.
    folds = np.zeros(th.shape, dtype=int)
    if lam == 0:
        k = np.floor(th/np.pi)
        th_red = th - k*np.pi
        th = np.where(k % 2 == 0, th_red, np.pi - th_red)
        folds = np.abs(k).astype(int)
        ph = ph + np.pi*folds

    return dict(t=t, r=1/u, th=th, ph=ph, sig=sig, folds=folds)
