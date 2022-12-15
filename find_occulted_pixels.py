"""© 2022 The Trustees of Columbia University in the City of New York. 
This work may be reproduced, distributed, and otherwise exploited for 
academic non-commercial purposes only, provided that it cites the original work: 

    Stefan J. Hofmeister, Michael Hahn, and Daniel Wolf Savin, "Deriving instrumental point spread functions from partially occulted images," 
    Journal of the Optical Society of America A 39, 2153-2168 (2022), doi: 10.1364/JOSAA.471477.

To obtain a license to use this work for commercial purposes, please contact 
Columbia Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

import numpy as np
import glob
import os
from inout import read_image, save_image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, LogNorm 

#This is the top level routine for identifying the occulted pixels
def find_occulted_pixels(config):
    folder_run = config['general']['folder_run']
    method     = config['occultation_mask']['method']
    if 'add_distance' in config['occultation_mask']: add_distance = config['occultation_mask']['add_distance']
    else: add_distance = 0

    #if occultation masks have been predefined
    if method == 'predefined': files = glob.glob(folder_run + '/occultation_mask/*.npz')
    else:                      files = glob.glob(folder_run + '/deconvolved_image/*.npz')
        
    for file in files:        
        #here, you can also define your own methods for finding the occulted pixels
        if method == 'predefined':        
            #if the occultation mask has been provided by the user
            occultation_mask = read_image(file, dtype = np.int8)
            img = read_image(folder_run + '/deconvolved_image/' + os.path.splitext(os.path.basename(file))[0] + '.npz')
            
        if method == 'threshold':            
            #derive the occultation masks by an adaptive threshold technique
            img = read_image(file, dtype = np.float16)  
            thr        = config['occultation_mask']['threshold']
            use_median = config['occultation_mask']['threshold_usemedian'] 
            occultation_mask = find_occulted_pixels_bythr(img, thr, use_median)
            
        if method == 'eclipse':                  
            #derive the occultation masks by fitting a circle to the eclipse
            img = read_image(file, dtype = np.float16)  
            # thr        = config['occultation_mask']['threshold']
            # use_median = config['occultation_mask']['threshold_usemedian'] 
            # occultation_mask = find_occulted_pixels_byeclipse(img, thr, use_median)
            occultation_mask = fit_eclipse( config, file, thr_eclipse =0.10, morph_radius = 5, distance_from_eclipse_boundary = 20, low_resolution_for_fit = 4096, high_resolution_for_fit = 4096, erosion = 0) 
            # occultation_mask[occultation_mask > 0] = 1
            # occultation_mask[(occultation_mask <= 0) & (occultation_mask != -1)] = 0
	                      
        if add_distance > 0:  
            kernel_size = 2 * np.floor(add_distance).astype(np.int) + 1
            kx, ky = np.meshgrid(np.arange(kernel_size) - kernel_size//2, np.arange(kernel_size) - kernel_size//2)
            kr = kx**2 + ky**2
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kr <= add_distance**2] = 1
            mask_ignore = (occultation_mask == -1)
            occultation_mask = np.logical_not(occultation_mask)
            occultation_mask = ndimage.convolve(occultation_mask.astype(np.int8), kernel.astype(np.int8), mode = 'nearest')
            occultation_mask = np.logical_not(occultation_mask).astype(np.int8)
            occultation_mask[mask_ignore] = -1

        #plot the occultation mask, one time solely the occultation mask and one time overlayed to the image
        filename_figure = os.path.splitext(file)[0] + '.jpg'
        fig, ax = plt.subplots(1, 2, figsize = (12, 5)) 
        ax1 = ax[0] 
        divider = make_axes_locatable(ax1) 
        cax1 = divider.append_axes('right', size='5%', pad=0.05)
        im1 = ax1.imshow(occultation_mask, norm = Normalize(vmin = -1, vmax = 1, clip = True), interpolation = 'nearest', cmap = 'viridis')
        fig.colorbar(im1, cax=cax1, orientation='vertical', ticks=[-1, 0, 1])
        ax2 = ax[1] 
        divider = make_axes_locatable(ax2) 
        cax2 = divider.append_axes('right', size='5%', pad=0.05)
        img_occ_marked = np.copy(img)
        img_occ_marked[occultation_mask == 1] = 10000
        im2 = ax2.imshow(img_occ_marked,  norm=LogNorm(vmin=1, vmax=10000, clip = True), interpolation = 'nearest', cmap = 'viridis')
        fig.colorbar(im2, cax=cax2, orientation='vertical')
        plt.tight_layout()
        plt.savefig(filename_figure, dpi = 600)
        plt.close(fig)
        #finally, save the occultation mask
        save_image(occultation_mask, folder_run + '/occultation_mask/' + os.path.basename(file), plot = False, dtype = np.int8, keys = 'img')

        
        
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

