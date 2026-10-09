"""Shared helpers for the kgeo test suite."""

import contextlib
import io
import pathlib

import numpy as np
import pytest

from kgeo.kerr_raytracing_ana import coords_at_tau
from kgeo.kerr_raytracing_utils import mino_total, radial_roots

from geodesic_ode import conserved

DATA_DIR = pathlib.Path(__file__).parent / "data"


def xfail_until(step, reason, strict=True):
    """Mark a known edge-case failure that plan step `step` should fix.

    strict=True (default): once the fix lands, the unexpected pass fails the suite,
    so the marker must be removed rather than going stale. Use strict=False only
    when some cases pass by roundoff luck before the real fix.
    """
    return pytest.mark.xfail(strict=strict, reason=f"[{step}] {reason}")


def quiet():
    """Silence kgeo's progress prints"""
    return contextlib.redirect_stdout(io.StringIO())


def wrap(x):
    """Wrap angles to (-pi, pi]"""
    return np.angle(np.exp(1j*np.asarray(x)))


def trace(a, th_o, alpha, beta, ngeo=100, r_o=1000., taufrac=0.97):
    """Analytic geodesics from kgeo, sampled at ngeo Mino times in [0, taufrac*tau_tot].

    taufrac < 1 stays clear of the horizon, where t and phi diverge.
    Returns a dict of (ngeo, npix) arrays 'tau', 't', 'r', 'th', 'ph', 'sig',
    and per-pixel 'lam', 'eta', 'rclass', 'tautot'.
    """
    a = float(a); th_o = float(th_o); r_o = float(r_o)
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    lam, eta = conserved(a, th_o, alpha, beta)

    r1, r2, r3, r4, rclass = radial_roots(a, lam, eta)
    tautot = mino_total(a, r_o, eta, r1, r2, r3, r4)
    tau = np.outer(np.linspace(0, taufrac, ngeo), tautot)
    with quiet():
        sig, (t, r, th, ph) = coords_at_tau(a, [0, r_o, th_o, 0], [alpha, beta], tau)

    return dict(tau=tau, t=t, r=r, th=th, ph=ph, sig=sig,
                lam=lam, eta=eta, rclass=rclass, tautot=tautot)
