"""© 2022 The Trustees of Columbia University in the City of New York.
This work may be reproduced, distributed, and otherwise exploited for
academic non-commercial purposes only, provided that it cites the
original work (IN PUBLICATIONP PROCESS).  To obtain a license to
use this work for commercial purposes, please contact Columbia
Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

import numpy as np
import glob
import os
from inout import read_image, save_image


#This is the top level routine for identifying the occulted pixels
def find_occulted_pixels(config):
    folder_run = config['general']['folder_run']
    method     = config['occultation_mask']['method']

    #if occultation masks have been predefined
    if method != 'predefined':  #if the occultation masks have not been predefined and provided by the user: derive the occultation masks
        #derive the occultation masks
        files = glob.glob(folder_run + '/deconvolved_image/*.npz')
        for file in files:
            img = read_image(file, dtype = np.float16)  
            #here, you can also define your own methods for finding the occulted pixels
            if method == 'threshold':        
                thr        = config['occultation_mask']['threshold']
                use_median = config['occultation_mask']['threshold_usemedian'] 
                occultation_mask = find_occulted_pixels_bythr(img, thr, use_median)
                
            save_image(occultation_mask, folder_run + '/occultation_mask/' + os.path.basename(file), plot_norm = 'lin', dtype = np.int8, keys = 'img')


#In this section, you can put your own algorithms to identify occulted pixels. The function should use the (psf deconvolved) image "img" as input, and return a numpy array with a zero background and the pixels identified as occulted set to one.

def find_occulted_pixels_bythr(img, thr, use_median = True):
    #define if using the median or mean of the illuminated portion of the image as references
    if use_median: avg_function = np.median
    else: avg_function = np.mean

    #define a mask containing the occulted pixels
    occulted_pixels = np.zeros(img.shape)
    
    #derive the mean/median of the image and set the occulted pixels as all pixels smaller than threshold * mean/median. Rederive the mean/median only from the not-occulted pixels, find the occulted pixels, and iterate until the result converges.
    img_mean_last = 0.
    for i in range(100): #use 100 as the max number of iterations. Usually, <10 iterations are sufficient for convergence
        img_mean = avg_function(img[occulted_pixels == 0])
        occulted_pixels = img < (img_mean * thr)
        if img_mean == img_mean_last: 
            break
        else: 
            img_mean_last = img_mean       
    return np.array(occulted_pixels)

