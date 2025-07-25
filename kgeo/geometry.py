#establishes geometry of the accretion flow in the drift-frame model (with no parallel boost)
import numpy as np
#from kgeo.solver import *
from kgeo.bfields import Bfield
from kgeo.velocities import Velocity

def lower_spatial_vec(vec, r, th, a): #lowers a contravariant four-vector with zero timelike component
    sig = r**2+a**2*np.cos(th)**2
    delta = r**2-2*r+a**2
    pi = (r**2+a**2)**2-a**2*delta*np.sin(th)**2
    lowering = np.array([sig/delta, sig, pi*np.sin(th)**2/sig])
    return np.transpose(lowering)*vec

def lapse(r, th, a): #lapse function for Kerr metric
    sig = r**2+a**2*np.cos(th)**2
    delta = r**2-2*r+a**2
    pi = (r**2+a**2)**2-a**2*delta*np.sin(th)**2
    return np.sqrt(delta*sig/pi)


#constA is multiplicative factor out front (doesn't affect continuity equation)
#"vel" is either 'driftframe' or 'MHD', and nu is the parallel boost parameter in that case
def densityhere(r, th, a, eta, model='para', vel = 'driftframe', nu_parallel = 0, gammamax=None, pval = 0, sigma = 2): 
    if model == 'para':
        bpara = Bfield("bz_para", C=1)
    elif model == 'mono':
        bpara = Bfield("bz_monopole", C=1)
    else: #power law case
        bpara = Bfield("power", C=1, p=pval)

    velocity=Velocity(vel, bfield=bpara, nu_parallel = nu_parallel, gammamax=gammamax)
    bupper = np.transpose(bpara.bfield_lab(a, r, th=th))
    (u0,u1,u2,u3) = velocity.u_lab(a, r, th=th) 
    
    return np.abs(eta*bupper[:,0]/u1) #solution to continuity equation, need absolute value because eta flips sign at stagnation surface

#returns images contained in order of neq
def sort_image(iobs, qobs, uobs, neqvals, guesses_shape, ashape, neqmax):
    iarr = [] #initialize arrays
    qarr = []
    uarr = []

    for neq in np.arange(0, neqmax, 1):
        ihere = np.copy(iobs)
        qhere = np.copy(qobs)
        uhere = np.copy(uobs)

        ihere[neqvals != neq] = 0 #zero out everything not in this subring
        qhere[neqvals != neq] = 0
        uhere[neqvals != neq] = 0

        iarr.append(np.reshape(np.sum(np.reshape(ihere, guesses_shape), axis=0), ashape)) #sum over all points in the subring
        qarr.append(np.reshape(np.sum(np.reshape(qhere, guesses_shape), axis=0), ashape))
        uarr.append(np.reshape(np.sum(np.reshape(uhere, guesses_shape), axis=0), ashape))

    #add the total image last to the array
    iarr.append(np.reshape(np.sum(np.reshape(iobs, guesses_shape), axis=0), ashape))
    qarr.append(np.reshape(np.sum(np.reshape(qobs, guesses_shape), axis=0), ashape))
    uarr.append(np.reshape(np.sum(np.reshape(uobs, guesses_shape), axis=0), ashape))

    return np.array(iarr), np.array(qarr), np.array(uarr)

