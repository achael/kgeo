# ADAPTED from GSL Airy function bindings in https://github.com/perey/python-gsl

import numpy as np
from ctypes import CDLL, CFUNCTYPE
from ctypes import c_double, c_int,c_uint, c_void_p,c_char_p
from ctypes import pointer, POINTER, Structure
from enum import IntEnum
from collections.abc import Callable


import mpmath as mp
import scipy.special as sp

# Load the gsl library
# the path to gsl on my system /usr/lib/x86_64-linux-gnu/libgsl.so,
# but CDLL finds it automatically :)
native = CDLL('libgsl.so')

###########################################
## PRECISION STUFF
gsl_mode_t = c_uint
class Mode(IntEnum):
    default = 0
    double = 0
    single = 1
    approx = 2

###########################################
## RESULT DEFNS
class sf_result(Structure):
    _fields_ = [('val', c_double),
                ('err', c_double)]

sf_result_p = POINTER(sf_result)
def make_sf_result_p():
    """Construct and initialise a pointer to a gsl_sf_result struct."""
    return pointer(sf_result(0.0, 0.0))

###########################################
## ERROR HANDLING STUFF
native.gsl_strerror.argtypes = (c_int,)
native.gsl_strerror.restype = c_char_p

ErrorHandler = CFUNCTYPE(None, c_char_p, c_char_p, c_int, c_int)
native.gsl_set_error_handler.argtypes = (ErrorHandler,)
native.gsl_set_error_handler.restype = ErrorHandler

native.gsl_set_error_handler_off.argtypes = ()
native.gsl_set_error_handler_off.restype = ErrorHandler

error_codes = {1: ValueError,
               2: ValueError,
               3: ValueError,
               4: ValueError,
               5: RuntimeError,
               6: ArithmeticError,
               7: RuntimeError,
               8: MemoryError,
               9: RuntimeError,
               10: RuntimeError,
               11: RuntimeError,
               12: ZeroDivisionError,
               13: ValueError,
               14: RuntimeError,
               15: ArithmeticError,
               16: OverflowError,
               17: ArithmeticError,
               18: ArithmeticError,
               19: TypeError,
               20: TypeError,
               21: ArithmeticError,
               22: ArithmeticError,
               23: OSError,
               24: NotImplementedError,
               25: MemoryError,
               26: MemoryError,
               27: RuntimeError,
               28: RuntimeError,
               29: RuntimeError,
               30: RuntimeError,
               31: RuntimeError,
               32: EOFError}

def exception_from_result(error_code):
    """Get an exception instance suitable for a GSL error code."""
    exception_class = error_codes.get(error_code, Exception)
    error_message = native.gsl_strerror(error_code).decode()

    return exception_class(error_message)


def exception_on_error(reason, file, line, errno):
    """Raise a Python exception on GSL errors."""
    exception_class = error_codes.get(errno, Exception)
    exception_text = reason.decode()

    raise exception_class(exception_text)

_handlers = []
def set_error_handler(fn):
    """Set the given function as the GSL error handler.
    The function must accept four arguments:
        reason -- a bytes object
        file -- a bytes object
        line -- an integer
        errno -- an integer
    If no function is given (the sole argument is None), GSL error
    handling is disabled. GSL functions that return an error code will
    still cause python-gsl to raise an exception, however.
    Returns:
        The previously-active error handler, which can be restored with
        another call to set_error_handler().
    """
    if fn is None:
        return native.gsl_set_error_handler_off()
    else:
        # Do the right thing if fn is a ctypes function pointer (as previously
        # returned by this very function), rather than a Python function that
        # needs wrapping with ErrorHandler.
        if isinstance(fn, Callable):
            handler = ErrorHandler(fn)
            _handlers.append(handler)
        else:
            handler = fn

        return native.gsl_set_error_handler(handler)

def sf_error_handler(result, func, arguments):
    """Check the return code."""
    if result:
        raise exception_from_result(result)
    return result

###################################################################################
# Disable crash-on-error and raise Python exceptions instead.
set_error_handler(exception_on_error)

###################################################################################
# Elliptic Integral
###################################################################################

native.gsl_sf_ellint_RJ_e.argtypes = (c_double,c_double,c_double,c_double,gsl_mode_t,sf_result_p)
native.gsl_sf_ellint_RJ_e.restype = c_int
native.gsl_sf_ellint_RJ_e.errcheck = sf_error_handler

native.gsl_sf_ellint_RF_e.argtypes = (c_double,c_double,c_double,gsl_mode_t,sf_result_p)
native.gsl_sf_ellint_RF_e.restype = c_int
native.gsl_sf_ellint_RF_e.errcheck = sf_error_handler

native.gsl_sf_ellint_RC_e.argtypes = (c_double,c_double,gsl_mode_t,sf_result_p)
native.gsl_sf_ellint_RC_e.restype = c_int
native.gsl_sf_ellint_RC_e.errcheck = sf_error_handler

def ellip_pi_gsl(n,phi,m,precision=Mode.default):
    """Incomplete Elliptic Integral of the Third Kind Pi(n;phi|m)
       Convention following Abramowitz & Stegun & Mathematica
       DIFFERENT from gsl convention.
       Implemented using CarlsonRJ and CarlsonRF and periodicities to account for full range
       """

    # real range is m sin^2(phi) < 1, n sin^2 phi < 1
    # relation to Carlson symmetric form only works for -pi/2 < phi < pi/2,
    # need to use periodicity outside this range

    # limit as phi->0
    if np.abs(phi)<=1.e-10: # TODO what value cutoff?
        return phi

    # limit as m->-infinity
    if(m<-1.e14): # TODO what value cutoff?
        return 0.

    cphi = np.cos(phi)
    sphi = np.sin(phi)

    x = cphi**2
    y = 1. - m*sphi**2
    z = 1.
    rho = 1. - n*sphi**2

    # check the allowed region
    if(y<0):
        raise Exception("m*sin^2(phi) > 1 in ellipticPi!")
    if(np.abs(phi)>np.pi/2. and m>=1):
        raise Exception("abs(phi)>pi/2 and m >= 1 in ellipticPi!")
    if(np.abs(phi)>np.pi/2. and n==1):
        return np.infty
    if (rho==0):
        return np.infty
    if (np.isinf(n)): # TODO large n limit
        return 0.

    # account for periodicity
    if(np.abs(phi)>np.pi):
        s = np.sign(phi)
        k = s*np.floor(np.abs(phi)/np.pi)
        phi2 = phi - k*np.pi
        comp_part = 2*k*ellip_pi_gsl(n,np.pi/2.,m,precision=precision)
        incomp_part = ellip_pi_gsl(n,phi2,m,precision=precision)
        Pi = comp_part + incomp_part

    elif(np.pi/2.<np.abs(phi)<=np.pi): #TODO can we merge with previous case somehow?
        s = np.sign(phi)
        phi2 = -phi + s*np.pi
        comp_part = s*2*ellip_pi_gsl(n,np.pi/2.,m,precision=precision)
        incomp_part = -ellip_pi_gsl(n,phi2,m,precision=precision)
        Pi = comp_part + incomp_part

    else:
        # TODO should we just use scipy ellipkinc here?!
        resultRF = make_sf_result_p()
        retRF = native.gsl_sf_ellint_RF_e(x,y,z,precision,resultRF)
        CRF = resultRF.contents.val

        if rho>0:
            resultRJ = make_sf_result_p()
            retRJ = native.gsl_sf_ellint_RJ_e(x,y,z,rho,precision,resultRJ)
            CRJ = resultRJ.contents.val

        else: # transform for rho<0, https://dlmf.nist.gov/19.20#E14 19.20.14
            q = -rho
            p = (z*(x+y+q) - x*y) / (z + q)

            resultRJ = make_sf_result_p()
            retRJ = native.gsl_sf_ellint_RJ_e(x,y,z,p,precision,resultRJ)
            CRJ0 = resultRJ.contents.val

            resultRC = make_sf_result_p()
            retRC = native.gsl_sf_ellint_RC_e(x*y + p*q,p*q,precision,resultRC)
            CRC = resultRC.contents.val

            CRJ = ((p-z)*CRJ0 - 3*CRF + 3*np.sqrt((x*y*z)/(x*y+p*q))*CRC)/(q+z)

        F = sphi * CRF
        Pi = F + (n/3.)*(sphi**3)*CRJ

    return Pi

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
            gsl_pi = ellip_pi_gsl(arg[0],arg[1],arg[2])
            print("%d | %.6e %.6e %.6e"%(k,mp_pi,gsl_pi,1.-gsl_pi/mp_pi))
    return
