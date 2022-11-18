"""© 2022 The Trustees of Columbia University in the City of New York. This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only, provided that it cites the original work: 
Stefan J. Hofmeister, Michael Hahn, and Daniel Wolf Savin, "Deriving instrumental point spread functions from partially occulted images," J. Opt. Soc. Am. A 39, 2153-2168 (2022), doi: 10.1364/JOSAA.471477.
To obtain a license to use this work for commercial purposes, please contact Columbia Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

from __future__ import division

import numpy as np
import os
from numba import cuda, njit, prange
import glob
from inout import read_image


def set_up_system_of_equations(config):
    #Some definitions from the config file
    folder_run                       = config['general']['folder_run']
    resolution                       =  config['general']['resolution']
    large_psf                        = config['general']['large_psf']
    use_gpu                          = config['general']['use_gpu']
    if 'n_px_occulted in config_soe':
        n_occulted_pixels            = config['system_of_equations']['n_px_occulted']
    else:
        n_occulted_pixels = 'max'
    if 'n_px_illuminatededge' in config['system_of_equations']:
        n_illuminatededge_pixels     = config['system_of_equations']['n_px_illuminatededge']
        dist_from_illuminated_edge   = config['system_of_equations']['dist_from_illuminatededge']
    if 'n_px_illuminated' in config['system_of_equations']: 
        n_illuminated_pixels         = config['system_of_equations']['n_px_illuminated']
        dist_from_illuminated_edge   = config['system_of_equations']['dist_from_illuminatededge']
        n_illuminated_bins           = config['system_of_equations']['n_illuminated_bins']
    if 'rebin_factor_occulted_pixels' in config['system_of_equations']:
        rebin_factor_occulted_pixels = config['system_of_equations']['rebin_factor_occulted_pixels']
        rebin_dist_occulted_pixels   = config['system_of_equations']['rebin_dist_occulted'] 
    
    #load the psf segmentation
    psf_segments = np.load(folder_run + '/psf_segmentation/psf_segments.npz', allow_pickle=True)    
    n_segs       = psf_segments['n_shells'] + psf_segments['n_manual_shells']
    indices_fov  = psf_segments['indices_fov']
    indices_fov_flipped = np.flip(indices_fov) 
    
    #find the image pixels and their weighting which contribute to the center pixel when a convolution with the psf is applied by flipping the psf   
    if config['general']['file_psf_existing'] != '':
        psf_existing = read_image(folder_run + '/original_psf.npz', dtype = np.float32)
    else:
        psf_existing = np.zeros((resolution, resolution), dtype = np.float32)
    psf_existing_flipped = np.flip(psf_existing) 
    
    #create file and folder definitions    
    folder_out =                    folder_run + '/system_of_equations/'
    folder_occultation_masks =      folder_run + '/occultation_mask/' 
    folder_importance_masks =       folder_run + '/importance_mask/'                     
    folder_distances =              folder_run + '/distance_from_illuminated_edge/'
    folder_approx_true =            folder_run + '/approximated_true_image/'   
    folder_original =               folder_run + '/original_image/'               
            
    files = glob.glob(folder_original + '/*.npz')
    # process each file
    for file in files:
          try:
              #read in the deconvolved map and the mask
              occ_mask          = read_image(folder_occultation_masks + os.path.basename(file) , dtype = np.bool8)
              illumination_mask = (occ_mask == 0)
              importance_mask   = read_image(folder_importance_masks + os.path.basename(file) , dtype = np.int16)
              distances         = np.load(folder_distances + os.path.basename(file))['dists_from_illuminated_edge'].astype(dtype = np.float16)
              img_approxtrue    = read_image(folder_approx_true+ os.path.basename(file), dtype = np.float32) 
              img_orig          = read_image(folder_original + os.path.basename(file), dtype = np.float32)
          except:
              #it could happen that, e.g., a deconvolution for an image did not converge and thus that the approximated true image does not exist. In that case, continue with the next file.
              continue 
             

          #process the images for 1) all occulted pixels, 2) extra pixels close ot the illuminated edge, 3) extra pixels in the illuminated region
          system_of_equations = list()
          for process in ['occulted', 'illuminated_edge', 'illuminated']: 
                  if process == 'occulted':
                      if n_occulted_pixels != 'max':
                           #draw occulted pixels from the image. We use a statified draw, i.e., we draw  px_per_bin pixels for each index in the importance mask. Thus, the importance mask allows us to constrain the distribution of the pixels over the image, which will be used to set up the system of equations.
                           #first, we derive how many indices we have in our importance mask, how many pixels in the importance mask each index covers, and we sort the values in ascending order, i.e., the indices containing a small number of pixels first.
                           indices_in_importancemask = np.unique(importance_mask[importance_mask != -1])
                           n_importancemask_indices = len(indices_in_importancemask)
                           n_importancemask_px_per_index = np.array([ np.sum(importance_mask == index) for index in indices_in_importancemask ])
                           sort = np.argsort(n_importancemask_px_per_index)
                           indices_in_importancemask, n_importancemask_px_per_index = indices_in_importancemask[sort], n_importancemask_px_per_index[sort]
                                 
                           #we derive the number of pixels to draw from the indices that contain a large number of pixels. This is given by the (total number of pixels to draw - the number of pixels in indices containing a small number of pixels (there, we just draw all the pixels)) divided by the number of bins left to distribute. We iterate over it until the result converges to find the number of pixels per bin for indices related to large segments.
                           n_small_importancemask_segments, n_px_in_small_importancemask_segments, px_per_bin_last = 0, 0, 0
                           while True:
                               px_per_bin = (n_occulted_pixels - n_px_in_small_importancemask_segments) // (n_importancemask_indices - n_small_importancemask_segments)
                               small_segments = np.where(n_importancemask_px_per_index < px_per_bin)[0]
                               n_small_importancemask_segments = len(small_segments)
                               n_px_in_small_importancemask_segments = np.sum(n_importancemask_px_per_index[small_segments])
                               if px_per_bin == px_per_bin_last: break
                               else: px_per_bin_last = px_per_bin
                               
                           #finally, we draw the pixels. For the small segments, we just use all the pixels. For large semgents, we draw px_per_bin pixels.
                           px_x, px_y = [], []
                           for i in range(len(indices_in_importancemask)):
                                    px_in_index = np.where(importance_mask == indices_in_importancemask[i])
                                    if i != len(indices_in_importancemask) -1:
                                        n_random = min(px_per_bin, n_importancemask_px_per_index[i])
                                    random = np.random.choice(n_importancemask_px_per_index[i], int(n_random) , replace = False)
                                    px_x.extend(px_in_index[0][random])
                                    px_y.extend(px_in_index[1][random])    
                           px_2d = (px_x, px_y)     
                                     
                      else:
                          px_2d = np.where(occ_mask == 1)
                          
                  if process == 'illuminated_edge':
                      px_2d = []
                      #if extra pixels along the illuminated edge are requested:
                      if 'n_px_illuminatededge' in config['system_of_equations']:
                        if (n_illuminatededge_pixels > 0):
                          px_2d = np.where( (illumination_mask == 1) & (distances < dist_from_illuminated_edge))   
                          random = np.random.choice(px_2d[0].size, min(n_illuminatededge_pixels, px_2d[0].size), replace = False)
                          px_2d = (px_2d[0][random], px_2d[1][random])
                          
                  if process == 'illuminated':
                      px_2d = []
                      #if extra pixels within the illuminated region are requested:
                      if  'n_px_illuminated' in config['system_of_equations']: 
                        if (n_illuminated_pixels > 0):
                          px_2d = np.where( (illumination_mask == 1) & (distances > dist_from_illuminated_edge))
                          #draw the pixels equally distributed over the histogram
                          #first, find the edges of the histogram bins
                          px_intensities = img_approxtrue[px_2d]
                          int_min, int_max = np.min(px_intensities[px_intensities > 0]), np.max(px_intensities)
                          bin_edges = np.logspace(np.log10(int_min), np.log10(int_max), num = n_illuminated_bins + 1)
                          #now, draw pixels from within these histogram bins
                          px_x, px_y = [], []
                          for bin_i in range(n_illuminated_bins):
                                px_in_bin = np.where( (px_intensities >= bin_edges[bin_i]) & (px_intensities < bin_edges[bin_i+1]) )[0]
                                px_in_bin = np.random.choice(px_in_bin, min(n_illuminated_pixels // n_illuminated_bins, px_in_bin.size), replace = False)
                                px_x.extend(px_2d[0][px_in_bin])
                                px_y.extend(px_2d[1][px_in_bin])
                          px_2d = (px_x, px_y)   
                                                    
                          
                            
                  #if there are pixels to process:
                  if len(px_2d) > 0:
                      #create an empty array for the total shells intensities
                      n_px2d = len(px_2d[0])
                      intensity_shells =      np.ascontiguousarray( np.zeros((n_px2d, n_segs), dtype = np.float64) )
                      intensities_explained = np.ascontiguousarray( np.zeros(n_px2d          , dtype = np.float32) )
                      intensities_observed =  np.ascontiguousarray( np.zeros(n_px2d          , dtype = np.float32) ) 
                      
                      #convert the other arrays to numba-conform formats
                      img_orig =              np.ascontiguousarray( img_orig                 , dtype = np.float32) 
                      px_x =                  np.ascontiguousarray( px_2d[0]                 , dtype = np.int16)
                      px_y =                  np.ascontiguousarray( px_2d[1]                 , dtype = np.int16)
                      true_intensities =      np.ascontiguousarray( img_approxtrue           , dtype = np.float32 )
                      indices_flipped =       np.ascontiguousarray( indices_fov_flipped      , dtype = np.int16) 
                      psf_existing_flipped =  np.ascontiguousarray( psf_existing_flipped     , dtype = np.float32 )
                      resolution = np.int16(resolution)               
                      
                      #derive the requested rebinning for each occulted pixel depending on its distance to the illuminated edge
                      distances_px = distances[px_2d]
                      rebin_factor_occulted_pixels_px = np.ones(n_px2d, dtype = np.int16)
                      if 'rebin_factor_occulted_pixels' in config['system_of_equations']: 
                          for i in range(len(rebin_factor_occulted_pixels)):
                              mask = (distances_px >= rebin_dist_occulted_pixels[i]) & (distances_px < rebin_dist_occulted_pixels[i+1])
                              rebin_factor_occulted_pixels_px[mask] = rebin_factor_occulted_pixels[i]
                         
                      if use_gpu:                          
                          GPU_threadsperblock = 32
                          GPU_blockspergrid =  np.ceil(n_px2d / GPU_threadsperblock).astype(np.int)
                          indices = create_indices_fromfov(indices_flipped)
                          derive_shells_forpx_GPU[GPU_blockspergrid, GPU_threadsperblock](px_x, px_y, true_intensities, indices, intensity_shells, resolution, large_psf)
                          derive_intensities_explained_GPU[GPU_blockspergrid, GPU_threadsperblock](px_x, px_y, true_intensities, psf_existing_flipped, intensities_explained, resolution, large_psf)
                          derive_intensities_observed_GPU[GPU_blockspergrid, GPU_threadsperblock](px_x, px_y, img_orig, intensities_observed, rebin_factor_occulted_pixels_px, resolution)   
                      else:
                          indices = create_indices_fromfov(indices_flipped)
                          derive_shells_forpx_CPU(px_x, px_y, true_intensities, indices, intensity_shells, resolution, large_psf)
                          derive_intensities_explained_CPU(px_x, px_y, true_intensities, psf_existing_flipped, intensities_explained, resolution, large_psf)
                          derive_intensities_observed_CPU(px_x, px_y, img_orig, intensities_observed, rebin_factor_occulted_pixels_px, resolution)

                      

                      #put the results into the correct variables
                      system_of_equations.append({'intensities_shells'              : np.array(intensity_shells, dtype = np.float32),
                                                 'intensities'                      : np.array(intensities_observed, dtype = np.float32),
                                                 'intensities_explained'            : np.array(intensities_explained, dtype = np.float32),
                                                 'intensities_unexplained'          : np.array(intensities_observed - intensities_explained, dtype = np.float32), 
                                                 'px_location'                      : np.array(px_2d, dtype = np.int16).transpose(),
                                                 'distance_to_edge'                 : np.array(distances_px, dtype = np.float16),
                                                 'index_importancemask'             : np.array(importance_mask[px_2d], dtype = np.int16),
                                                 'region'                           : np.repeat(process, len(intensities_observed)) })
          
            #combine the datasets for the occulted, illuminated, and illuminated_edge pixels to a single system of equations. Save the system of equations
          soe = {} 
          for key in system_of_equations[0].keys():
              soe[key] = np.concatenate([item[key] for item in system_of_equations], 0)
          outfile = folder_out + os.path.splitext(os.path.basename(file))[0] + '.npz'
          np.savez_compressed(outfile, system_of_equations = soe)

       
 



    
@cuda.jit
def derive_intensities_observed_GPU(px_x_in, px_y_in, original_intensities, intensities_observed, rebin_factor_occulted_pixels, resolution_img):
    k = cuda.grid(1)  #get thread number. This sets the pixel we process.
    if k < len(px_x_in):  #as we launch many threads, check if the thread number is in the processing range.
        #if yes: set the occulted pixel to process
        px_x = px_x_in[k]
        px_y = px_y_in[k]
        radius = rebin_factor_occulted_pixels[k] //2
        
        int_tmp = 0.
        n_sum = 0.
        for x in range(px_x - radius, px_x + radius + 1):
            for y in range(px_y - radius, px_y + radius + 1):
                if (x>=0) & (y>=0) & (x<resolution_img) & (y<resolution_img):
                    int_tmp += original_intensities[x,y] 
                    n_sum += 1
        int_tmp /= n_sum
        intensities_observed[k] = int_tmp
           
@cuda.jit
def derive_intensities_explained_GPU(px_x_in, px_y_in, intensities, psf_flipped, intensities_explained, resolution_img, large_psf):
    k = cuda.grid(1)  #get thread number. This sets the pixel we process.
    if k < len(px_x_in):  #as we launch many threads, check if the thread number is in the processing range.
        #if yes: set the occulted pixel to process
        px_x = px_x_in[k]
        px_y = px_y_in[k]
        if large_psf: resolution_psf = 2*resolution_img
        else:       resolution_psf = resolution_img
        
        #center the flipped psf on that pixel, and define the valid bounding box of the flipped psf within the image frame
        xmin, xmax = max(px_x - resolution_psf//2 + 1, 0), min(px_x + resolution_psf//2, resolution_img) #np.arange(resolution) - resolution//2 centers the original psf on px_x, i.e., sums from px_x - resolution//2 to px_x + resolution//2 -1. However, the psf is flipped, so that the center pixel in the PSF is not at resolution//2 but (resolution//2 -1). Thus, the summation has to go over px_x - (resolution//2 -1) to px_x + (resolution//2). This is solved by adding the "-1".
        ymin, ymax = max(px_y - resolution_psf//2 + 1, 0), min(px_y + resolution_psf//2, resolution_img) #the min(0) and max(resolution) guarantee correct boundaries for the x and y, i.e., that they stay within the psf.
        #Do the summation of the scattered light from the other pixels to the occulted pixel
        for x in range(xmin, xmax):      
             for y in range(ymin, ymax):
                     intensities_explained[k] += ( intensities[x,y] * psf_flipped[x + resolution_psf//2 -1 - px_x, y + resolution_psf//2 -1 - px_y] ) 


@cuda.jit
def derive_shells_forpx_GPU(px_x_in, px_y_in, intensities, indices, intensity_shells, resolution_img, large_psf):
    k = cuda.grid(1)
    if k < len(px_x_in): 
        # this is the position of the occulted pixel:
        px_x = px_x_in[k]
        px_y = px_y_in[k]
        if large_psf: resolution_psf = 2*resolution_img
        else:       resolution_psf = resolution_img
        
        #indices corresponds to the 2d indices_flipped array of size resolution x resolution. However, it is now transformed to a list. Each element contains (line x, column y_start, column y_end, indices_flipped index). Thus, each element corresponds to a block of pixels which has the same index in indices_flipped. This hugely speeds up computation time (as index does not have to be looked up at each iteration in the following loop).
        #Thus, go through the original indices_flipped by processing each element in indices
        for i_indices in range(indices.shape[0]):
            x = px_x - (resolution_psf//2 -1) + indices[i_indices, 0] 
            y_start = px_y - (resolution_psf//2 -1) + indices[i_indices, 1]
            y_end =  px_y - (resolution_psf//2 -1) + indices[i_indices, 2]           
            index = indices[i_indices, 3]
            
            #if the pixel block that corresponds to indices[i_indices] is outside of the image boundary: continue
            if (x < 0) or (x >= resolution_img) or ((y_start < 0) and (y_end < 0)) or ((y_start >= resolution_img) and (y_end >= resolution_img)):
                continue
            
            #Now, go over the pixel block, and sum the intensities to the correct element in intensity_shells
            intensity_tmp = np.float64(0.)                                  #Use a temporary variable for doing the following summation in the loop. This one can stay in the register of the kernels, and thus the summation is super fast
            y_start, y_end = max(y_start, 0), min(y_end, resolution_img -1)     #if it is inside, derive the maximum boundaries now WITHIN the image boundaries. Note that we use resolution-1 here as y_end was defined correspondingly. Maybe, I should change that definition in the future.
            for y in range(y_start, y_end+1): 
                    intensity_tmp += intensities[x, y]
            intensity_shells[k, index] += intensity_tmp                     #and finally put the temporary variable back into intensity_shells
       
        

@njit(parallel = True)
def derive_intensities_observed_CPU(px_x_in, px_y_in, original_intensities, intensities_observed, rebin_factor_occulted_pixels, resolution_img):
    for k in prange(len(px_x_in)):
        px_x = px_x_in[k]
        px_y = px_y_in[k]
        radius = rebin_factor_occulted_pixels[k] //2
        
        int_tmp = 0.
        n_sum = 0.
        for x in range(px_x - radius, px_x + radius + 1):
            for y in range(px_y - radius, px_y + radius + 1):
                if (x>=0) & (y>=0) & (x<resolution_img) & (y<resolution_img):
                    int_tmp += original_intensities[x,y] 
                    n_sum += 1
        int_tmp /= n_sum
        intensities_observed[k] = int_tmp
           
@njit(parallel = True)
def derive_intensities_explained_CPU(px_x_in, px_y_in, intensities, psf_flipped, intensities_explained, resolution_img, large_psf):
    for k in prange(len(px_x_in)):
        px_x = px_x_in[k]
        px_y = px_y_in[k]
        if large_psf: resolution_psf = 2*resolution_img
        else:       resolution_psf = resolution_img
        
        #center the flipped psf on that pixel, and define the valid bounding box of the flipped psf within the image frame
        xmin, xmax = max(px_x - resolution_psf//2 + 1, 0), min(px_x + resolution_psf//2, resolution_img) #np.arange(resolution) - resolution//2 centers the original psf on px_x, i.e., sums from px_x - resolution//2 to px_x + resolution//2 -1. However, the psf is flipped, so that the center pixel in the PSF is not at resolution//2 but (resolution//2 -1). Thus, the summation has to go over px_x - (resolution//2 -1) to px_x + (resolution//2). This is solved by adding the "-1".
        ymin, ymax = max(px_y - resolution_psf//2 + 1, 0), min(px_y + resolution_psf//2, resolution_img) #the min(0) and max(resolution) guarantee correct boundaries for the x and y, i.e., that they stay within the psf.
        #Do the summation of the scattered light from the other pixels to the occulted pixel
        for x in range(xmin, xmax):      
             for y in range(ymin, ymax):
                     intensities_explained[k] += ( intensities[x,y] * psf_flipped[x + resolution_psf//2 -1 - px_x, y + resolution_psf//2 -1 - px_y] ) 


@njit(parallel = True)
def derive_shells_forpx_CPU(px_x_in, px_y_in, intensities, indices, intensity_shells, resolution_img, large_psf):
    for k in prange(len(px_x_in)):
        px_x = px_x_in[k]
        px_y = px_y_in[k]
        if large_psf: resolution_psf = 2*resolution_img
        else:       resolution_psf = resolution_img
        
        #indices corresponds to the 2d indices_flipped array of size resolution x resolution. However, it is now transformed to a list. Each element contains (line x, column y_start, column y_end, indices_flipped index). Thus, each element corresponds to a block of pixels which has the same index in indices_flipped. This hugely speeds up computation time (as index does not have to be looked up at each iteration in the following loop).
        #Thus, go through the original indices_flipped by processing each element in indices
        for i_indices in range(indices.shape[0]):
            x = px_x - (resolution_psf//2 -1) + indices[i_indices, 0] 
            y_start = px_y - (resolution_psf//2 -1) + indices[i_indices, 1]
            y_end =  px_y - (resolution_psf//2 -1) + indices[i_indices, 2]           
            index = indices[i_indices, 3]
            
            #if the pixel block that corresponds to indices[i_indices] is outside of the image boundary: continue
            if (x < 0) or (x >= resolution_img) or ((y_start < 0) and (y_end < 0)) or ((y_start >= resolution_img) and (y_end >= resolution_img)):
                continue
            
            #Now, go over the pixel block, and sum the intensities to the correct element in intensity_shells
            intensity_tmp = np.float64(0.)                                  #Use a temporary variable for doing the following summation in the loop. This one can stay in the register of the kernels, and thus the summation is super fast
            y_start, y_end = max(y_start, 0), min(y_end, resolution_img -1)     #if it is inside, derive the maximum boundaries now WITHIN the image boundaries. Note that we use resolution-1 here as y_end was defined correspondingly. Maybe, I should change that definition in the future.
            for y in range(y_start, y_end+1): 
                    intensity_tmp += intensities[x, y]
            intensity_shells[k, index] += intensity_tmp                     #and finally put the temporary variable back into intensity_shells

@njit
def create_indices_fromfov(indices_fov):  #indices_flipped array into a list, where each element contains (line x, column y_start, column y_end, indices_flipped index). Basically, it is a simple going over the 2d array indices_flipped, and whenever a new index in indices_flipped appears, generate a new element in indices
    indices = list()
    for x in range(indices_fov.shape[0]):
        y = 0
        indices.append([x, y, -1, indices_fov[x, y]]) #this one is the first element in each line x
        index_last = indices_fov[x, y]
        for y in range(indices_fov.shape[1]):                   #go through the elements in the line x
            if indices_fov[x, y] != index_last:                 #if indices_fov has changed from the previous element
                indices[-1][2] = y-1                            #put the index of the last element of the line into the last element of indices, i.e., close that element
                indices.append([x, y, -1, indices_fov[x, y]])   #create a new element in indices
                index_last = indices_fov[x, y]                  #and update index last
        indices[-1][2] = y                                      #close the last element
    return np.array(indices)
            

