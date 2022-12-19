"""© 2022 The Trustees of Columbia University in the City of New York. 
This work may be reproduced, distributed, and otherwise exploited for 
academic non-commercial purposes only, provided that it cites the original work: 

    Stefan J. Hofmeister, Michael Hahn, and Daniel Wolf Savin, "Deriving instrumental point spread functions from partially occulted images," 
    Journal of the Optical Society of America A 39, 2153-2168 (2022), doi: 10.1364/JOSAA.471477.

To obtain a license to use this work for commercial purposes, please contact 
Columbia Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

import glob
import numpy as np
import os
from numba import njit
from inout import read_image, save_image, convert_toimage
from deconvolve_image import convolve_image


#Derive the importance mask. It tells for each psf segment which region in the occulted region is most important for fitting the psf segment. Within the illuminated region, we set the distance of the illuminated pixels to the illuminated edge.
def derive_importance_mask(config):
        folder_run = config['general']['folder_run']
        large_psf  = config['general']['large_psf']    
        use_gpu    = config['general']['use_gpu']
        if 'min_dist' in config['importance_mask']: min_dist = config['importance_mask']['min_dist']
        else:                                       min_dist = 0
        if 'max_dist' in config['importance_mask']: max_dist = config['importance_mask']['max_dist']
        else:                                       max_dist = 0
        
        if config['importance_mask']['method'] == 'predefined':         files = config['importance_mask']['importance_file']
        if config['importance_mask']['method'] == 'distances':          files = glob.glob(folder_run + '/distance_from_illuminated_edge/*.npz')
        if config['importance_mask']['method'] == 'psf segmentation':   files = glob.glob(folder_run + '/occultation_mask/*.npz')

        for file in files:
            if config['importance_mask']['method'] == 'predefined':
                importance_mask = convert_toimage(file, folder_run + '/importance_mask/' + os.path.basename(file), dtype = np.int16, plot_norm = 'lin')
    
            if config['importance_mask']['method'] == 'distances':
                #create the importance mask from the distances of the illuminated edges. Basically, this is just a segmentation of the distance_from_illuminated_edge map (which was already derived earlier) into segments given by config_importancemask['distance_edges'].
                distance_edges = config['importance_mask']['distance_edges']
                distances = np.load(file)['dists_from_illuminated_edge'].astype(dtype = np.float16)
                occultation_mask = read_image(folder_run + '/occultation_mask/' + os.path.basename(file), dtype = np.int8)
                importance_mask = np.full(distances.shape, -1)
                for i in range(len(distance_edges) -1):
                    region = (distances >= distance_edges[i]) & (distances < distance_edges[i+1]) & (occultation_mask == 1)
                    importance_mask[region] = i         
            
            if config['importance_mask']['method'] == 'psf segmentation':
                #here, we define the importance mask by the psf segmentation. This should allow to get a good distribution of pixels for both asymmetric PSFs and diffraction patterns, but it is slow.
                #first, load the needed files
                occultation_mask = read_image(file, dtype = np.int8)
                illumination_mask = np.float32(occultation_mask == 0)
                exclude = (occultation_mask == -1)
                dists, angles = read_image(folder_run + '/distance_from_illuminated_edge/' + os.path.basename(file), keys = ['dists_from_illuminated_edge', 'angles_from_illuminated_edge'], dtype = np.float16)
                psf_segments =  np.load(folder_run + '/psf_segmentation/psf_segments.npz', allow_pickle=True)   
                shells_fov = psf_segments['indices_fov']
                shells_edges = psf_segments['shells_edges_inner']
                if np.sum(psf_segments['shells_edges_outer']) != 0: shells_edges = np.concatenate([shells_edges, psf_segments['shells_edges_outer']])
                shells =         {'radius_min': psf_segments['shells'].item()['radius_min_px'],
                                  'radius_mean': psf_segments['shells'].item()['radius_mean_px'],
                                  'radius_max': psf_segments['shells'].item()['radius_max_px'],
                                  'angle_min':  psf_segments['shells'].item()['angle_min'],
                                  'angle_max':  psf_segments['shells'].item()['angle_max'],
                                  'index':      psf_segments['shells'].item()['index']}
                manual_shells =  {'radius_min': psf_segments['manual_shells'].item()['radius_min_px'],
                                  'radius_max': psf_segments['manual_shells'].item()['radius_max_px'],
                                  'angle_min':  psf_segments['manual_shells'].item()['angle_min'],
                                  'angle_max':  psf_segments['manual_shells'].item()['angle_max'],
                                  'index':      psf_segments['manual_shells'].item()['index']}

                
                #then, derive the importance mask for the not-manual shells. We derive them in chunks based on the distance of the PSF segments to the PSF center. Where the importance masks overlap within a chunk, we choose the index randomly.
                #first, we define some arrays
                importance_mask_empty = np.full(occultation_mask.shape, -1, dtype = np.int32)
                importance_mask = np.copy(importance_mask_empty)
                #For each radial distance
                for index_rad in reversed(range(len(shells_edges) -1)):
                    #we look for which segments are within the radial distance
                    indices_of_segments = np.where((shells['radius_mean'] >= shells_edges[index_rad]) &  (shells['radius_mean'] < shells_edges[index_rad + 1]))[0]
                    #for each segment, we create an importance mask where we store temporary results
                    importance_mask_array = np.tile(importance_mask_empty, (len(indices_of_segments), 1, 1) )
                    #For each segment, we now derive the important pixels in the illuminated image. For that, we convolve the illuminated image with the segment.
                    for i, index_segment in enumerate(indices_of_segments):
                        #get a mask of the psf segment associated with index
                        mask_psf_segment = np.float32(shells_fov == shells['index'][index_segment])
                        #convolve the illumination mask with the mask of the individual psf segment
                        importance_mask_for_segment = convolve_image(illumination_mask, mask_psf_segment, use_gpu = use_gpu, pad = True, large_psf = large_psf)
                        #get the important regions for that segments: set the illuminated region to zero as it will anyway not be used, and identify the important region by a threshold of 0.5 (as the fourier based convolution results in very small numbers of not-important regions instead of zeros)
                        importance_mask_for_segment[illumination_mask == 1] = 0
                        importance_mask_for_segment[exclude == 1] = 0
                        importance_mask_for_segment[importance_mask_for_segment > 0.5] = 1
                        importance_mask_array[i, importance_mask_for_segment == 1] = shells['index'][index_segment]    
                    #Having the array of importance masks that are related to the given radial distance, we now look where they overlap, and draw from these one index by random.
                    importance_mask = choose_random_index(importance_mask, importance_mask_array)

                #For the manual shells, which are meant to contain the diffraction pattern, we derive the important pixels by their distance and angle to the illuminated parts. Here, we do not have to care anymore about overlaps.
                for i in range(len(manual_shells['index'])):
                    importance_mask[(dists >= manual_shells['radius_min'][i]) &  (dists <= manual_shells['radius_max'][i]) &  (angles >= manual_shells['angle_min'][i]) &  (angles <= manual_shells['angle_max'][i]) & (occultation_mask == 1)] = manual_shells['index'][i]
            
            if min_dist or max_dist:
                distances = np.load( folder_run + '/distance_from_illuminated_edge/{:}.npz'.format( os.path.splitext(os.path.basename(file))[0] ) )['dists_from_illuminated_edge'].astype(dtype = np.float16)
                if min_dist: importance_mask[distances < min_dist] = -1
                if max_dist: importance_mask[distances > max_dist] = -1

            #Finally, save the result
            save_image(importance_mask, folder_run + '/importance_mask/' + os.path.basename(file), plot_norm = 'lin', dtype = np.int32)
       
@njit                    
def choose_random_index(array, from_array):
    #from_array is a 3d array of shape (n_images, x_image, y_image). array is of shape (x_image, y_image).
    #for each pixel, we look where from_array[:, x, y] is not -1 (== missing value), draw randomly one of the valid values, and set the value accordingly in array.
    xsize, ysize = array.shape
    for x in np.arange(xsize):
        for y in np.arange(ysize):
            choices = np.where(from_array[:, x, y] != -1)[0]
            if len(choices) == 0: continue
            choice = np.random.choice(choices)

            array[x, y] = from_array[choice, x, y]
    return array


  
