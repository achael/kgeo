import numpy as np
from kgeo.bfields import Bfield
from kgeo.velocities import Velocity
from kgeo.emissivities import Emissivity
from kgeo.equatorial_images import calc_redshift, calc_polquantities, calc_evpa
from kgeo.kerr_raytracing_ana import raytrace_ana
from kgeo.densityfuncs import density_mono_all, density_para_all,density_power_all
from kgeo.solver import findroot
from kgeo.geometry import sort_image
bfield_default = Bfield('rad')
vel_default = Velocity('zamo')
emis_default = Emissivity('bpl')
SPECIND = 1 # default (negative) spectral index

def Iobs_off(a, r_o, r_s, th_o, alpha, beta, kr_sign, kth_sign,
         emissivity=emis_default, velocity=vel_default, bfield=bfield_default,
         polarization=False, specind=SPECIND, th_s=np.pi/2, density=1, retsin = False):

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

            (sinthb, kappa, pathlength, bsq) = calc_polquantities(a, r_s[~zeromask], lam[~zeromask], eta[~zeromask],
                                                 kr_sign, kth_sign, 
                                                 velocity=velocity, 
                                                 bfield=bfield, th=th_s)
            (cos2chi, sin2chi) = calc_evpa(a, th_o, alpha[~zeromask], beta[~zeromask], kappa)

        else:
            sinthb = 1
        
        sin_thb[~zeromask] = sinthb  
        Iemis *= bsq**((1+specind)/2) #for spectral index of 1, we add on factor of B^2 

                             
        ###############################
        # observed emission
        ###############################         

        Iobs_here[~zeromask] = (gg**3) * (gg**specind) * pathlength * Iemis * (sinthb**(1+specind))       


        if polarization:
            Qobs[~zeromask] = cos2chi*Iobs_here[~zeromask]
            Uobs[~zeromask] = sin2chi*Iobs_here[~zeromask]

    else:
        print("masked all pixels in Iobs_off!")

    Iobs_2 = np.copy(Iobs_here)
    Qobs_2 = np.copy(Qobs)
    Uobs_2 = np.copy(Uobs)

    Iobs_2[r_s<=rh] = Qobs_2[r_s<=rh] = Uobs_2[r_s<=rh] = 0 #zero out emission coming from within the horizon


    if not retsin:
        return Iobs_2, Qobs_2, Uobs_2
    return Iobs_2, Qobs_2, Uobs_2, sinthb    





#get stokes parameters for a grid with off-equatorial emission in BZ model
#returns images contained in order of neq
def getstokes(psitarget, alphavals, betavals, r_o, th_o, a, ngeo, 
              do_phi_and_t = True, neqmax=1, outgeo=None, tol=1e-8, 
              model='para', pval=1,   
              nu_parallel = 0,  gammamax=None, retvals = False,
              sigma=2, sumsubring=True, usemono=False, retsin=False): #neqmax is the maximum number of equatorial crossings

    """Get stokes parameters for a grid with off-equatorial emission in BZ jet model
       
       Args:
           psitarget (float): Psi of field line to be imaged in jet model
           alphavals (numpy.array): array of image pixel alpha values 
           betavals (numpy.array): array of image pixel beta values 
           r_o (float): camera radius (should be large)
           th_o (float): camera inclination in radian range (0,pi/2) or (pi/2,pi)
           a (float): bh spin in range [0,1)
           ngeo (int): number of points sampled along geodesic
           
           do_phi_and_t (bool): raytrace in phi_and_t or not (must be true??)
           neqmax (int): maximal number of geodesic equatorial (?) crossings
           outgeo (Geodesics): precomputed geodesics
           tol (float): tolerance for Newton solve of crossing points
           
           model (str): define BZ jet model: 'mono', 'para', or 'power'
           pval (float): index of variable width 'power' jet model

           nu_parallel (float): fractional allowable parallel velocity in range [-1,1]       
           gammamax (float): cutoff lorentz factor in jet model
           
           sigma (float): constant sigma for determining density/emissivity
           
           retvals (bool): whether or not to return data beyond stokes params
           sumsubring (bool): whether or not to sum all crossing data in final return
           usemono (bool): if True, fix Omega_field to monopole rate
           retsin (bool): if True, return sin(theta_B)
            
       Returns:

    """
    ashape = alphavals.shape # store shapes for later
    alphavals = alphavals.flatten() # flatten since we need everything to be a vector for our code to work
    betavals = betavals.flatten()

    if outgeo == None:
        outgeo = raytrace_ana(a=a,
                              observer_coords = [0,r_o,th_o,0],
                              image_coords = [alphavals, betavals], #assumes 1D arrays of alpha and beta
                              ngeo=ngeo,
                              do_phi_and_t=do_phi_and_t,
                              savedata=False, plotdata=False)

    #solve for crossing points and densities there
    tau, rvals, thvals, signpr, signptheta, neqvals, guesses_shape = findroot(outgeo, psitarget, alphavals, betavals, r_o, th_o, a, ngeo, 
                                                                              model=model, neqmax=neqmax, tol=tol, pval=pval)


    print('guesses before ', guesses_shape)
    #reshape coordinates
    alphavals = np.tile(alphavals, guesses_shape[0])
    betavals = np.tile(betavals, guesses_shape[0])


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

    #generate intensity data
    outvec = Iobs_off(a, r_o, rvals, th_o, alphavals, betavals, signpr, signptheta,
                      emissivity=Emissivity('constant'), 
                      velocity=Velocity('driftframe', bfield=bf, nu_parallel = nu_parallel, gammamax=gammamax), 
                      bfield=bf,
                      polarization=True, specind=SPECIND, th_s=thvals, density=dvals, retsin=retsin) 


    iobs = np.copy(outvec[0])
    qobs = np.copy(outvec[1])
    uobs = np.copy(outvec[2])

    iobs[rvals == 0] = qobs[rvals==0] = uobs[rvals==0] = 0 #zero out geodesics with no fieldline crossings

    iobs = np.real(np.nan_to_num(np.array(iobs))) #get rid of nans
    qobs = np.real(np.nan_to_num(np.array(qobs)))
    uobs = np.real(np.nan_to_num(np.array(uobs)))

    if not sumsubring:
        #return iobs, qobs, uobs, neqvals, guesses_shape  #just return the raw subrings
        
        return iobs, qobs, uobs, neqvals, np.nan_to_num(rvals), np.nan_to_num(thvals), guesses_shape  #just return the raw subrings #AC edited
        
    #call sorter here
    ivec, qvec, uvec = sort_image(iobs, qobs, uobs, neqvals, guesses_shape, ashape, neqmax)
    evpa = np.nan_to_num(0.5*np.arctan2(uvec, qvec))
    
    if not retvals: #just return stokes
        return ivec, qvec, uvec, evpa
    if not retsin: #return stokes + intersection data
        return ivec, qvec, uvec, evpa, np.nan_to_num(rvals), np.nan_to_num(thvals)
    return ivec, qvec, uvec, evpa, np.nan_to_num(rvals), np.nan_to_num(thvals), np.nan_to_num(outvec[3]) #return stokes+intersection+pitch angle

