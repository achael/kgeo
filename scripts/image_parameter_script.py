import ehtim as eh
import numpy as np

NSEARCH = 10    # number of points in each dimension to search for the ring center
FOVSEARCH = 0.1 # fraction of the image fov to search for the ring center
BLUR = 5       # default blurring kernel

def compute_image_parameters(path_to_fitsfile, blur_kernel_uas=BLUR):
    """load an image from path_to_fitsfile, blur with blur_kernal_uas, 
       then compute the image parameters using the ehtim rex function"""
    
    # load the image from fits file
    im = eh.image.load_image(path_to_fitsfile) 
    
    if blur_kernel_uas:
        imblur = im.blur_circ(blur_kernel_uas*eh.RADPERUAS, blur_kernel_uas*eh.RADPERUAS)
        
    else:
        imblur = im
        
    # find ring profile with rex
    prof = eh.features.rex.FindProfile(imblur,
                       n_search=NSEARCH, fov_search=FOVSEARCH,
                       imsize=im.fovx(), npix=im.xdim,
                       thresh_search=0,
                       rmin=0, 
                       rmin_search=0, rmax_search=40)
    prof.calc_meanprof_and_stats()
    
    # plot ring fit
    prof.plot_img()
    
    # get total intensity parameters
    (d, d_sigma) = prof.RingSize1       # diameter in uas
    (w, w_sigma) = prof.RingWidth       # width in uas
    (eta, eta_sigma) = prof.RingAngle1  # position angle in rad
    (Asym, Asym_sigma) = prof.RingAsym1 # asymmetry
    
    # get polarization beta2 parameter in rad
    argbeta2 = np.angle(imblur.betamodes(ms=[2])[0])
    

    # print output:
    print("==================")
    print("diameter: %.2f +/- %.2f uas"%(d,d_sigma))
    print("width: %.2f +/- %.2f uas"%(w,w_sigma))
    print("pos ang: %.2f +/- %.2f deg"%(eta/eh.DEGREE,eta_sigma/eh.DEGREE))
    print("Asym: %.2f +/- %.2f"%(Asym,Asym_sigma))
    print("argbeta2: %.2f deg"%(argbeta2/eh.DEGREE))   
    
    return (d,w,eta,Asym,argbeta2), (d_sigma, w_sigma, eta_sigma, Asym_sigma) 
    
if __name__=='__main__':
    compute_image_parameters('./m87_model_test.fits')
