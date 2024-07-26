from kgeo.equatorial_images import *
from kgeo.geometry import *
from kgeo.solver import *
from kgeo.image import *
<<<<<<< HEAD

def Iobs_off(a, r_o, r_s, th_o, alpha, beta, kr_sign, kth_sign,
         emissivity=emis_default, velocity=vel_default, bfield=bfield_default,
         polarization=False, specind=SPECIND, th_s=np.pi/2, density=1):
=======
from kgeo.densityfuncs import *

def Iobs_off(a, r_o, r_s, th_o, alpha, beta, kr_sign, kth_sign,
         emissivity=emis_default, velocity=vel_default, bfield=bfield_default,
         polarization=False,  efluid_nonzero=False, specind=SPECIND, th_s=np.pi/2, density=1, retsin = False):
>>>>>>> cf42a0df8502f93c65168bfa2ae7a0d64c42b250
    """Return (Iobs, g, r_s, Ir, Imax, Nmax) where
       Iobs is Observed intensity for a ring of order mbar, GLM20 Eq 6
       g is the Doppler factor
       r_s is the equatorial emission radius
       Ir is the elapsed Mino time at emission
       Imax is the maximal Mino time on the geodesic
       Nmax is the *maximum* number of equatorial crossings"""

    # checks
    if not (isinstance(a,float) and (0<=np.abs(a)<1)):
        raise Exception("|a| should be a float in range [0,1)")
    if not (isinstance(r_o,float) and (r_o>=100)):
        raise Exception("r_o should be a float > 100")
    if not (isinstance(th_o,float) and (0<th_o<np.pi) and th_o!=0.5*np.pi):
        raise Exception("th_o should be a float in range (0,pi/2) or (pi/2,pi)")
                
    if not isinstance(alpha, np.ndarray): alpha = np.array([alpha]).flatten()
    if not isinstance(beta, np.ndarray): beta = np.array([beta]).flatten()
    if not isinstance(r_s, np.ndarray): r_s = np.array([r_s]).flatten()
    if len(alpha) != len(beta):
        raise Exception("alpha, beta are different lengths!")

    # horizon radius
    rh = 1 + np.sqrt(1-a**2)

    # conserved quantities
    lam = -alpha*np.sin(th_o)
    eta = (alpha**2 - a**2)*np.cos(th_o)**2 + beta**2

    # emission radius & Mino time
   # r_s, Ir, Imax, Nmax = r_equatorial(a, r_o, th_o, mbar, alpha, beta)

    # output arrays
    g = np.zeros(alpha.shape)
    sin_thb = np.zeros(alpha.shape)
    Iobs_here = np.zeros(alpha.shape)
    Qobs = np.zeros(alpha.shape)
    Uobs = np.zeros(alpha.shape)
        
    # apply to all pixels - so the mask here always works
    zeromask = np.abs(alpha)<0
        
    if np.any(~zeromask):

        ###############################
        # get momentum signs
        ###############################        
        # kr_sign = np.ones_like(alpha[~zeromask])#radial_momentum_sign(a, th_o, alpha[~zeromask], beta[~zeromask], Ir[~zeromask], Imax[~zeromask])
        # kth_sign = -1#theta_momentum_sign(th_o, mbar)

        ###############################
        # get velocity and redshift
        ###############################        
        (u0,u1,u2,u3) = velocity.u_lab(a, r_s[~zeromask],th=th_s) 

        gg = calc_redshift(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask], kr_sign, kth_sign, u0, u1, u2, u3, th=th_s)   
        g[~zeromask] = gg

        ###############################
        # get emissivity in local frame
        ###############################
        Iemis = emissivity.jrest(a, r_s[~zeromask])*density

        ###############################
        # get polarization quantities
        # if polarization not used, set sin(theta_b) = 1 everywhere
        ###############################

        if polarization:
<<<<<<< HEAD
            (sinthb, kappa) = calc_polquantities(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask],
                                                 kr_sign, kth_sign, u0, u1, u2, u3, 
                                                 bfield=bfield, th=th_s)
            (cos2chi, sin2chi) = calc_evpa(a, th_o, alpha[~zeromask], beta[~zeromask], kappa)
        else:
            sinthb = 1
        
        sin_thb[~zeromask] = sinthb   
=======
            (sinthb, kappa, pathlength, bsq) = calc_polquantities(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask],
                                                 kr_sign, kth_sign, u0, u1, u2, u3, 
                                                 bfield=bfield,  efluid_nonzero=efluid_nonzero, th=th_s)
            (cos2chi, sin2chi) = calc_evpa(a, th_o, alpha[~zeromask], beta[~zeromask], kappa)

        else:
            sinthb = 1
        
        sin_thb[~zeromask] = sinthb  
        Iemis *= bsq**((1+specind)/2) #for spectral index of 1, we add on factor of B^2 
>>>>>>> cf42a0df8502f93c65168bfa2ae7a0d64c42b250
                             
        ###############################
        # observed emission
        ###############################         
<<<<<<< HEAD
        Iobs_here[~zeromask] = (gg**2) * (gg**specind) * Iemis * (sinthb**(1+specind))       
=======
        Iobs_here[~zeromask] = (gg**3) * (gg**specind) * pathlength * Iemis * (sinthb**(1+specind))       
>>>>>>> cf42a0df8502f93c65168bfa2ae7a0d64c42b250

        if polarization:
            Qobs[~zeromask] = cos2chi*Iobs_here[~zeromask]
            Uobs[~zeromask] = sin2chi*Iobs_here[~zeromask]
<<<<<<< HEAD
        
=======

    
>>>>>>> cf42a0df8502f93c65168bfa2ae7a0d64c42b250
       
    else:
        print("masked all pixels in Iobs! m=%i"%mbar)

    Iobs_2 = np.copy(Iobs_here)
    Qobs_2 = np.copy(Qobs)
    Uobs_2 = np.copy(Uobs)

    Iobs_2[r_s<=rh] = Qobs_2[r_s<=rh] = Uobs_2[r_s<=rh] = 0 #zero out emission coming from within the horizon

<<<<<<< HEAD
    return Iobs_2, Qobs_2, Uobs_2
=======
    if not retsin:
        return Iobs_2, Qobs_2, Uobs_2
    return Iobs_2, Qobs_2, Uobs_2, sinthb    
>>>>>>> cf42a0df8502f93c65168bfa2ae7a0d64c42b250

#returns images contained in order of neq


#get stokes parameters for a grid with off-equatorial emission in BZ model
<<<<<<< HEAD
def getstokes(psitarget, alphavals, betavals, r_o, th_o, a, ngeo, do_phi_and_t = True, model='para', neqmax=1, constA=1, outgeo=None, tol=1e-8, nu_parallel = 0): #neqmax is the maximum number of equatorial crossings
=======
def getstokes(psitarget, alphavals, betavals, r_o, th_o, a, ngeo, do_phi_and_t = True, model='para', neqmax=1, eta=1, outgeo=None, tol=1e-8, 
              nu_parallel = 0, pval=1, gammamax=None, retvals = False, vel='driftframe', sigma=2, sumsubring=True, usemono=False, retsin=False): #neqmax is the maximum number of equatorial crossings
>>>>>>> cf42a0df8502f93c65168bfa2ae7a0d64c42b250
    ashape = alphavals.shape #store shapes for later
    alphavals = alphavals.flatten() #flatten since we need everything to be a vector for our code to work
    betavals = betavals.flatten()

    if outgeo == None:
            outgeo = raytrace_ana(a=a,
                 observer_coords = [0,r_o,th_o,0],
                 image_coords = [alphavals, betavals], #assumes 1D arrays of alpha and beta
                 ngeo=ngeo,
                 do_phi_and_t=do_phi_and_t,
                 savedata=False, plotdata=False)
    
    #solve for crossing points and densities there
<<<<<<< HEAD
    tau, rvals, thvals, signpr, signptheta, neqvals, guesses_shape = findroot(outgeo, psitarget, alphavals, betavals, r_o, th_o, a, ngeo, do_phi_and_t = do_phi_and_t, model=model, neqmax=neqmax, tol=tol)
=======
    tau, rvals, thvals, signpr, signptheta, neqvals, guesses_shape = findroot(outgeo, psitarget, alphavals, betavals, r_o, th_o, a, ngeo, do_phi_and_t = do_phi_and_t, model=model, neqmax=neqmax, tol=tol, pval=pval)
>>>>>>> cf42a0df8502f93c65168bfa2ae7a0d64c42b250

    print('guesses before ', guesses_shape)
    #reshape coordinates
    alphavals = np.tile(alphavals, guesses_shape[0])
    betavals = np.tile(betavals, guesses_shape[0])

<<<<<<< HEAD
    dvals = densityhere(rvals, thvals, a, constA, model=model)
    
    
    #initialize arrays
    bf = Bfield('bz_para') if model == 'para' else Bfield('bz_monopole') #initialize bfield

    outvec = Iobs_off(a, r_o, rvals, th_o, alphavals, betavals, signpr, signptheta,
    emissivity=Emissivity('constant'), velocity=Velocity('driftframe', bfield=bf, nu_parallel = nu_parallel), bfield=bf,
    polarization=True,  specind=SPECIND, th_s=thvals, density=dvals) #generate data
=======
    if model == 'mono' or (model == 'power' and pval == 0):
        dvals = density_mono_all(rvals, thvals, guesses_shape, psitarget, a, nu_parallel, sigma, neqmax=neqmax, gammamax=gammamax)
    elif model == 'para':
        dvals = density_para_all(rvals, thvals, guesses_shape, psitarget, a, nu_parallel, sigma, neqmax=neqmax, gammamax=gammamax)
    elif (model == 'power' and pval > 0):
        dvals = density_power_all(rvals, thvals, guesses_shape, psitarget, a, nu_parallel, sigma, neqmax=neqmax, gammamax=gammamax, pval = pval, usemono=usemono)
    
    #initialize arrays
    if model == 'para':
        bf = Bfield('bz_para')
    elif model == 'mono':
        bf = Bfield('bz_monopole') #initialize bfield
    elif model == 'power':
        bf = Bfield('power', p=pval, usemono=usemono)
    else:
        bf = 0

    outvec = Iobs_off(a, r_o, rvals, th_o, alphavals, betavals, signpr, signptheta,
    emissivity=Emissivity('constant'), velocity=Velocity('driftframe', bfield=bf, nu_parallel = nu_parallel, gammamax=gammamax), bfield=bf,
    polarization=True,  efluid_nonzero=False, specind=SPECIND, th_s=thvals, density=dvals, retsin=retsin) #generate data
>>>>>>> cf42a0df8502f93c65168bfa2ae7a0d64c42b250

    iobs = np.copy(outvec[0])
    qobs = np.copy(outvec[1])
    uobs = np.copy(outvec[2])

    iobs[rvals == 0] = qobs[rvals==0] = uobs[rvals==0] = 0 #zero out geodesics with no fieldline crossings

    iobs = np.real(np.nan_to_num(np.array(iobs))) #get rid of nans
    qobs = np.real(np.nan_to_num(np.array(qobs)))
    uobs = np.real(np.nan_to_num(np.array(uobs)))

<<<<<<< HEAD
    #call sorter here
    ivec, qvec, uvec = sort_image(iobs, qobs, uobs, neqvals, guesses_shape, ashape, neqmax)
    evpa = np.real(np.nan_to_num(0.5*np.arctan(uvec/qvec)))
    
    return ivec, qvec, uvec, evpa
=======
    if not sumsubring:
        return iobs, qobs, uobs, neqvals, guesses_shape  #just return the raw subrings

    #call sorter here
    ivec, qvec, uvec = sort_image(iobs, qobs, uobs, neqvals, guesses_shape, ashape, neqmax)
    evpa = np.nan_to_num(0.5*np.arctan2(uvec, qvec))
    
    if not retvals: #just return stokes
        return ivec, qvec, uvec, evpa
    if not retsin: #return stokes + intersection data
        return ivec, qvec, uvec, evpa, np.nan_to_num(rvals), np.nan_to_num(thvals)
    return ivec, qvec, uvec, evpa, np.nan_to_num(rvals), np.nan_to_num(thvals), np.nan_to_num(outvec[3]) #return stokes+intersection+pitch angle
>>>>>>> cf42a0df8502f93c65168bfa2ae7a0d64c42b250
