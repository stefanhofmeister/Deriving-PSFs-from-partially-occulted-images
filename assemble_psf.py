"""© 2022 The Trustees of Columbia University in the City of New York. This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only, provided that it cites the
original work: 
Stefan J. Hofmeister, Michael Hahn, and Daniel Wolf Savin, "Deriving instrumental point spread functions from partially occulted images," J. Opt. Soc. Am. A 39, 2153-2168 (2022), doi: 10.1364/JOSAA.471477.
To obtain a license to use this work for commercial purposes, please contact Columbia Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

import numpy as np
from numba import jit, njit, prange, cuda
import scipy as scp
from inout import read_image, save_image


def assemble_psf(config):
 
    #some definitions
    folder_run = config['general']['folder_run']
    resolution = config['general']['resolution'] 
    large_psf = config['general']['large_psf']
    use_gpu = config['general']['use_gpu']
    smooth_iterations = config['postprocessing']['n_smooth_iterations']  
    if 'rebin' in config['postprocessing']: rebin_array = config['postprocessing']['rebin']
    else: rebin_array = [1]
    
    
    #load the original PSF and the new fitted coefficient
    psf_orig = read_image(folder_run + '/original_psf.npz', dtype = np.float32)                 
    coeff = np.load(folder_run + '/fitted_psf_coefficients/fitted_coefficients.npz')['coeff']
    coeff[np.isnan(coeff)] = 0.

    #load the psf segmentation file. This is needed to reconstruct the psf.
    psf_segments    = np.load(folder_run + '/psf_segmentation/psf_segments.npz', allow_pickle=True)
    indices_fov     = psf_segments['indices_fov']
    indices_manual  = psf_segments['manual_shells'].item()['index']

    #if a large psf is requested, adapt the resolution accordingly
    if large_psf: resolution = np.copy(resolution) *2
    
    indices_fov_smoothing = np.copy(indices_fov)
    #as we do not smooth the manual shells, remove them from indices_fov_smoothing
    for i in indices_manual:
        indices_fov_smoothing[indices_fov_smoothing == i] = -1
    
    #create an empty psf for the newly fitted weights and fill it
    psf_fitted = np.zeros((resolution, resolution), dtype = np.float32)
    for i, coeff_tmp in enumerate(coeff):
        psf_fitted[indices_fov_smoothing == i] = coeff_tmp
        
 
    #some manual shells and regions where the psf segmentation was empty shall not be smoothed. However, these regions would also affect the regions that are smoothed. Thus, first identify these gaps and their surrounding (which is needed to fill the gaps later on).
    #First, identify the gaps and their surrounding
    gaps                        = np.logical_not(psf_fitted)    
    surrounding_of_gaps         = scp.signal.convolve2d(gaps, np.ones((3, 3)), mode = 'same')
    #label the regions
    gaps, dummy                 = scp.ndimage.label(gaps, structure = np.ones((3, 3)))
    surrounding_of_gaps, dummy  = scp.ndimage.label(surrounding_of_gaps, structure = np.ones((3, 3)))
    #gaps that extend to the boundary of the image are not gaps but actually a boundary condition. Thus, remove these gaps. 
    mask_boundary               = np.ones(gaps.shape, dtype = np.int8)
    mask_boundary[1:-1, 1:-1]   = 0 
    indices_boundary            = np.unique(gaps[mask_boundary == 1])
    for index in indices_boundary: gaps[gaps == index] = 0
    #relate each gap to its surrounding
    indices_gaps                = np.unique(gaps[gaps != 0])
    indices_surrounding         = list()
    for index in indices_gaps:
        indices_surrounding.append(surrounding_of_gaps[gaps == index][0]) #i.e., the overlap of (gaps == index) with surrounding_of_gaps gives the index of the surrounding. [0] for only taking one value out of it; note that all elements will have the same value, so this should be fine.
    #create a new gaps map, where each gap has an index and its surrounding the negative index of the gap
    new_gaps                    = np.zeros(gaps.shape, dtype = np.int32)
    for index_gap, index_surrounding in zip(indices_gaps, indices_surrounding):    
        new_gaps[surrounding_of_gaps == index_surrounding] = -1 * index_surrounding
        new_gaps[gaps == index_gap] = index_surrounding
    gaps = new_gaps
    indices_gaps = indices_surrounding
    
    #Now, create a psf map where the gaps are filled with the values of its surrounding
    indices_fov_without_gaps = np.copy(indices_fov_smoothing)
    psf_without_gaps = np.copy(psf_fitted)
    for index in indices_gaps:
        mask_gap = (gaps == index)
        mask_surrounding = (gaps == -1 * index) #in the surrounding, the indices are -1 * (the index of associated the gap)
        index_surrounding = indices_fov_smoothing[mask_surrounding]
        index_unique, index_frequency = np.unique(index_surrounding, return_counts = True)
        index_surrounding = index_unique[index_frequency.argmax()]
        psf_without_gaps[mask_gap] = coeff[index_surrounding] 
        indices_fov_without_gaps[mask_gap] = index_surrounding
        indices_fov_without_gaps[mask_surrounding] = index_surrounding
        
    #Now, we start the smoothing. 
    psf_fitted_orig = np.copy(psf_without_gaps)
    psf_fitted = psf_without_gaps
    rebin_array = sorted(rebin_array)
    for rebin in reversed(rebin_array):
        #if rebin == 1, we use a very accurate version of smoothing, where the gaps are continuously derived and refilled during the smoothing process. Thus, we have a separate definition for rebin == 1, i.e., no rebinning.
        if rebin == 1:
              #change the dataype of variables and create temporary variables which are needed for the processing    
              smooth_iterations_i               = np.int32(smooth_iterations)
              resolution_i                      = np.int16(resolution)
              coeff_i                           = coeff.astype(np.float32)
              psf_tmp1_i                        = np.zeros((resolution_i+2, resolution_i+2), dtype = np.float32)
              psf_tmp1_i[1:-1, 1:-1]            = psf_fitted
              psf_tmp2_i                        = np.copy(psf_tmp1_i)
              correction_factors_psf_i          = np.zeros(len(coeff_i), dtype = np.float32)
              n_totalweights_psf_i              = np.zeros(len(coeff_i), dtype = np.int32) 
              indices_fov_smoothing_i           = indices_fov_smoothing
              gaps_i                            = gaps
              if len(indices_gaps) > 0:
                  weights_gaps_i                    = np.zeros(np.max(indices_gaps).astype(np.int16), dtype = np.float32)
                  n_weights_surrounding_of_gaps_i   = np.zeros(np.max(indices_gaps).astype(np.int16), dtype = np.int32)
              else:
                  weights_gaps_i                    = np.array([], dtype = np.int32)
                  n_weights_surrounding_of_gaps_i   = np.array([], dtype = np.int32)
                  
        else:
            #for other rebinnings
            #rebinning is done by collapsing n pixels with n being odd. The image size is not necessarily be dividable by n. As such, create a cutout of the image so that we can collapse it.
            resolution_cutout = resolution  - (resolution % rebin) - ((resolution // rebin  + 1) % 2) * rebin    #this line first cuts off some rows so that resolution becomes divisible by n and (possibly) substracts another n rows and columns so that the collapsed result has an odd image edge size. This guarantees that the center pixel is rebinned in the right fashion.
            halflength = (resolution_cutout - 1) // 2    
            psf_cutout          = psf_fitted[ resolution//2 - halflength :  resolution//2 + halflength + 1, resolution//2 - halflength :  resolution//2 + halflength + 1 ]
            psf_orig_cutout     = psf_fitted_orig[ resolution//2 - halflength :  resolution//2 + halflength + 1, resolution//2 - halflength :  resolution//2 + halflength + 1 ]
            indices_fov_cutout  = indices_fov_without_gaps[ resolution//2 - halflength :  resolution//2 + halflength + 1, resolution//2 - halflength :  resolution//2 + halflength + 1 ] 

            #now, rebin the arrays. For the indices_fov_cutout, use a rebinning by which each pixel becomes the most frequent pixel of its neighbourhood
            shape = (resolution_cutout//rebin, rebin, resolution_cutout//rebin, rebin)
            psf_cutout = psf_cutout.reshape(shape).mean(-1).mean(1) 
            psf_orig_cutout = psf_orig_cutout.reshape(shape).mean(-1).mean(1) 
            indices_fov_cutout = np.percentile(np.percentile(indices_fov_cutout.reshape(shape), 50, axis = -1, interpolation = 'higher'), 50, axis = 1, interpolation = 'higher') 
                        
            #as we have changed indices_fov, we need to rederive the weights associated with each index, i.e., coeff_cutout
            indices_cutout = np.unique(indices_fov_cutout)
            indices_cutout = indices_cutout[indices_cutout != -1]
            coeff_cutout = np.zeros(np.max(indices_cutout) + 1)
            for i in indices_cutout:
                coeff_cutout[i] = np.mean(psf_orig_cutout[indices_fov_cutout == i])
    
            #change the dataype of variables and create temporary variables which are needed for the processing    
            smooth_iterations_i               = np.int32(smooth_iterations)
            resolution_i                      = np.int16(resolution_cutout // rebin)
            coeff_i                           = coeff_cutout.astype(np.float32)
            psf_tmp1_i                        = np.zeros((resolution_i+2, resolution_i+2), dtype = np.float32)
            psf_tmp1_i[1:-1, 1:-1]            = psf_cutout
            psf_tmp2_i                        = np.copy(psf_tmp1_i)
            correction_factors_psf_i          = np.zeros(len(coeff_i), dtype = np.float32)
            n_totalweights_psf_i              = np.zeros(len(coeff_i), dtype = np.int32) 
            indices_fov_smoothing_i           = indices_fov_cutout
            gaps_i                            = np.zeros(psf_cutout.shape, dtype = np.float32)
            weights_gaps_i                    = np.array([], dtype = np.int32)
            n_weights_surrounding_of_gaps_i   = np.array([], dtype = np.int32)
        
      
        #transform gaps and indices_fov into a more suitable format for fast processing of images. The result is a n x 4 array, where the four elements are (line, column start, column end, index of thix pixel block)
        if use_gpu: indices_block_size = 32
        else:       indices_block_size = resolution_i
        indices_psf_i     = create_indices_fromfov(indices_fov_smoothing_i, indices_block_size).astype(np.int32)
        indices_psf_i     = indices_psf_i[indices_psf_i[:, 3] != -1, :]
        indices_gaps_i    = create_indices_fromfov(gaps_i, indices_block_size).astype(np.int32)
        indices_gaps_i    = indices_gaps_i[indices_gaps_i[:, 3] != 0, :]
        
        #smooth the result while keeping the total psf weights in each psf segment constant
        if use_gpu:
            GPU_threadsperblock = (8, 8)
            GPU_blockspergrid = (16, 16) #16x16 blocks utilizes all GPU SMs.
            smooth_and_renormalize_GPU[GPU_blockspergrid, GPU_threadsperblock](psf_tmp1_i, psf_tmp2_i, resolution_i, smooth_iterations_i, indices_psf_i, coeff_i, correction_factors_psf_i, n_totalweights_psf_i, indices_gaps_i, weights_gaps_i, n_weights_surrounding_of_gaps_i)
        else:
            psf_tmp1_i = smooth_and_renormalize_CPU(psf_tmp1_i, resolution_i, smooth_iterations_i, indices_psf_i, coeff_i, indices_gaps_i)
    
        #after the smoothing, reinsert the smoothed result into the original array
        if rebin == 1:
            psf_fitted = psf_tmp1_i[1:-1, 1:-1]
        else:
            #in case that binning was done, we use an interpolation to upscale the rebinned smoothed result to the original resolution
            psf_interpol = scp.interpolate.interp2d(np.arange(0, resolution_cutout, rebin), np.arange(0, resolution_cutout, rebin), psf_tmp1_i[1:-1, 1:-1], kind = 'linear')
            psf_tmp1_i = psf_interpol(np.arange(resolution_cutout) - (rebin -1)//2, np.arange(resolution_cutout) - (rebin -1)//2) 
            psf_fitted[resolution//2 - halflength :  resolution//2 + halflength + 1, resolution//2 - halflength :  resolution//2 + halflength + 1] = psf_tmp1_i
            
    #put the smoothed result in psf_fitted_smoothed
    psf_fitted_smoothed = psf_fitted

    # reinsert the manual shells that have not been smoothed
    for i, coeff_tmp in enumerate(coeff):
        if i in indices_manual:
            psf_fitted_smoothed[indices_fov == i] = coeff_tmp
   
    #combine the newly fitted psf with the existing one. Note that if no psf was provided to work with, psf_orig will be an array of zeros with the center coefficient set to one. This results in that also the center cofficient is derived here in the correct way.
    totalweight_fitted = np.sum(psf_fitted_smoothed)
    psf_new =  psf_orig * (1. - totalweight_fitted) + psf_fitted_smoothed
    psf_new /= np.sum(psf_new)
    
    #finally, double_check if the center coefficient is reasonable. In particular in the first iteration of our algorithm, it can happen that the center coefficient has not the maximum value of the psf, since the approximation of the true image is still bad. In that case, having an entirely wrong center coefficient can result that the deconvolution does not converge in the next iteration and that the algorithm breaks. Thus, in that case, rescale the PSF and set the center coefficient to a reasonable one. Latest after a few iterations, the algorithm will converge and this conditional rescaling will not be applied anymore.
    if psf_new[resolution//2, resolution//2] < np.max(psf_new): 
        psf_new[resolution//2, resolution//2] = np.max(psf_new) 
        psf_new = psf_new / np.sum(psf_new) 
                        
    save_image(psf_new, folder_run + '/fitted_psf/fitted_psf.jpg', plot_norm = 'log', plot_range = [1e-10, 1], dtype = np.float32, keys = 'psf' )






@njit(parallel = True)
def smooth_and_renormalize_CPU(img_1, resolution, iterations, indices, coeff, indices_manual_surrounding):
    img_2 = np.zeros(img_1.shape, dtype = np.float32)
    corr_factor = np.zeros(len(coeff), dtype = np.float32)
    n_totalweights = np.zeros(len(coeff), dtype = np.int32)
    weights_manual_surrounding = np.zeros(int(np.max(indices_manual_surrounding[:, 3])), dtype = np.float32)
    n_weights_manual_surrounding = np.zeros(int(np.max(indices_manual_surrounding[:, 3])), dtype = np.int32)

    for iteration in range(iterations):
        
        # reset the correction factor and the number of weights    
        for i in range(len(corr_factor)):
            corr_factor[i] = 0.
            n_totalweights[i] = 0
                                    
        #set the image boundary conditions: The outer rows and columns are set to the values of the inner rows and columns next to it
        for i in range(resolution + 2): img_1[i, 0] = img_1[i, 1]
        for i in range(resolution + 2): img_1[0, i] = img_1[1, i]
        for i in range(resolution + 2): img_1[i, -1] = img_1[i, -2]
        for i in range(resolution + 2): img_1[-1, i] = img_1[-2, i]
           
        
        #if manual shells are present: replace the weights of the manuals shells with the weight of the surrounding pixels
        if len(indices_manual_surrounding) > 0:
            for i in range(len(weights_manual_surrounding)):
                weights_manual_surrounding[i] = 0.
                n_weights_manual_surrounding[i] = 0 
            
            #get the current total weights in each psf segment of the surrounding pixels
            for i_indices in prange(indices_manual_surrounding.shape[0]):
                index   =  indices_manual_surrounding[i_indices, 3]
                if index < 0: #<0 are indices in the surrounding of gaps
                    index = int(-1 * index)
                    x_line  =  indices_manual_surrounding[i_indices, 0] 
                    y_start =  indices_manual_surrounding[i_indices, 1]
                    y_end   =  indices_manual_surrounding[i_indices, 2]   
                    #Now, go over the pixel block, and sum the psf weights
                    for y_pos in range(y_start, y_end+1):
                        weights_manual_surrounding[index -1] +=  img_1[x_line+1, y_pos+1]
                    n_weights_manual_surrounding[index -1] += y_end - y_start + 1
    
    
            #derive from this the normalization factor
            for i in range(len(weights_manual_surrounding)): 
                if n_weights_manual_surrounding[i] > 0 : weights_manual_surrounding[i] = weights_manual_surrounding[i] / float(n_weights_manual_surrounding[i])
            
            #apply the normalization factor to each pixel
            for i_indices in prange(indices_manual_surrounding.shape[0]):
                index   =  indices_manual_surrounding[i_indices, 3]
                if index > 0:#>0 are indices in the gaps
                    x_line  =  indices_manual_surrounding[i_indices, 0] 
                    y_start =  indices_manual_surrounding[i_indices, 1]
                    y_end   =  indices_manual_surrounding[i_indices, 2]           
                    #Now, go over the pixel block, and sum the psf weights
                    weights_tmp = weights_manual_surrounding[index -1]           
                    for y_pos in range(y_start, y_end+1):
                            img_1[x_line+1, y_pos+1] = weights_tmp 
                
        #apply a laplacian smoothing
        for x in prange(1, resolution + 1):
            for y in range(1, resolution + 1):
                 img_2[x, y] =(img_1[x, y] + 
                            img_1[x-1, y] + 
                            img_1[x+1, y] + 
                            img_1[x, y-1] +
                            img_1[x, y+1] )/5.
                

        # # #get the current total weights in each psf segment. Note that indices is basically a list, where each element contains (xline, ystart, yend, index). Thus, it is just another, shorter view of indices_fov.
        for i_indices in prange(indices.shape[0]):
            x_line  =  indices[i_indices, 0] 
            y_start =  indices[i_indices, 1]
            y_end   =  indices[i_indices, 2]           
            index   =  indices[i_indices, 3]
            #Now, go over the pixel block, and sum the psf weights
            for y_pos in range(y_start, y_end+1):
                    corr_factor[index] += img_2[x_line+1, y_pos+1]
            n_totalweights[index] += y_end - y_start + 1

        #derive from this the normalization factor
        for i in range(len(coeff)):
            if (n_totalweights[i] * corr_factor[i]) > 0 : corr_factor[i] = coeff[i] / (corr_factor[i] / float(n_totalweights[i]))
        
        # #apply the normalization factor to each pixel
        for i_indices in prange(indices.shape[0]):
            x_line  =  indices[i_indices, 0] 
            y_start =  indices[i_indices, 1]
            y_end   =  indices[i_indices, 2]           
            index   =  indices[i_indices, 3]
            for y_pos in range(y_start, y_end+1):
                img_1[x_line+1, y_pos+1] = img_2[x_line+1, y_pos+1] * corr_factor[index]             
    return img_1




@cuda.jit
def smooth_and_renormalize_GPU(img_1, img_2, resolution, iterations, indices, coeff, corr_factor, n_totalweights, indices_manual_surrounding, weights_manual_surrounding, n_weights_manual_surrounding):
#limitations: we need to have more threads than pixels along an edge of the image, and more threads, than coefficients to fit. As we use cooperative groups, the number of threads is limited corresponding to the hardware design of the GPU. But usually, more than enough threads are available.
    this_grid = cuda.cg.this_grid()
    x, y = cuda.grid(2)  #get thread number. This sets the pixel we process.
    xsize, ysize = cuda.gridsize(2)
    xy = ysize * x + y
    xysize = xsize * ysize
    n_sequential =  int(resolution / xsize)     
 
    for _ in range(iterations):
        
        # reset the correction factor and the number of weights          
        if xy < len(coeff):
            corr_factor[xy] = 0.
            n_totalweights[xy] = 0
                                   
        #set the image boundary conditions: The outer rows and columns are set to the values of the inner rows and columns next to it
        if xy < resolution +2:
            img_1[xy, 0] = img_1[xy, 1]
            img_1[0, xy] = img_1[1, xy]   
            img_1[xy, -1] = img_1[xy, -2]
            img_1[-1, xy] = img_1[-2, xy]
        this_grid.sync()
        
        #if manual shells are present: replace the weights of the manuals shells with the weight of the surrounding pixels
        if False: #len(indices_manual_surrounding) > 0:
            if xy < len(weights_manual_surrounding):
                weights_manual_surrounding[xy] = 0.
                n_weights_manual_surrounding[xy] = 0 
            
            #get the current total weights in each psf segment of the surrounding pixels
            n_chunks = (indices_manual_surrounding.shape[0] / xysize) 
            if int(n_chunks) != n_chunks: n_chunks += 1
            for nc in range(n_chunks):
                # i_indices = int(nc + xy * n_chunks)
                i_indices = nc*xysize + xy
                if i_indices < indices_manual_surrounding.shape[0]:        
                    index   =  indices_manual_surrounding[i_indices, 3]
                    if index < 0: #<0 are pixels in the surrounding of gaps
                        index = int(-1 * index)
                        x_line  =  indices_manual_surrounding[i_indices, 0] 
                        y_start =  indices_manual_surrounding[i_indices, 1]
                        y_end   =  indices_manual_surrounding[i_indices, 2]   
                        #Now, go over the pixel block, and sum the psf weights
                        weight_tmp = np.float64(0.)    #Use a temporary variable for doing the following summation in the loop. This one can stay in the register of the kernels, and thus the summation is super fast
                        for y_pos in range(y_start, y_end+1):
                                weight_tmp += img_1[x_line+1, y_pos+1]
                        cuda.atomic.add(weights_manual_surrounding, index -1, weight_tmp)
                        cuda.atomic.add(n_weights_manual_surrounding, index -1, y_end - y_start + 1 )
            this_grid.sync()
    
    
            #derive from this the normalization factor
            if xy < len(weights_manual_surrounding): 
                if n_weights_manual_surrounding[xy] > 0 : weights_manual_surrounding[xy] = weights_manual_surrounding[xy] / float(n_weights_manual_surrounding[xy])
            this_grid.sync()
            
            # #apply the normalization factor to each pixel
            n_chunks = (indices_manual_surrounding.shape[0] / xysize) 
            if int(n_chunks) != n_chunks: n_chunks += 1
            for nc in range(n_chunks):
                i_indices = nc*xysize + xy
                if i_indices < indices_manual_surrounding.shape[0]:
                    index   =  indices_manual_surrounding[i_indices, 3]
                    if index > 0:#>0 are pixels in the gap
                        x_line  =  indices_manual_surrounding[i_indices, 0] 
                        y_start =  indices_manual_surrounding[i_indices, 1]
                        y_end   =  indices_manual_surrounding[i_indices, 2]           
                        #Now, go over the pixel block, and sum the psf weights
                        weights_tmp = weights_manual_surrounding[index -1]           
                        for y_pos in range(y_start, y_end+1):
                                img_1[x_line+1, y_pos+1] = weights_tmp 
            this_grid.sync()
                
        #apply a laplacian smoothing
        for dx in range(n_sequential):
            for dy in range(n_sequential):
                img_2[x*n_sequential+1+dx, y*n_sequential+1+dy] = (img_1[x*n_sequential+1+dx, y*n_sequential+1+dy] + 
                                                          # img_1[x*n_sequential+2+dx,   y*n_sequential+2+dy] + 
                                                          # img_1[x*n_sequential+dx,   y*n_sequential+dy] + 
                                                          # img_1[x*n_sequential+2+dx,   y*n_sequential+dy] + 
                                                          # img_1[x*n_sequential+dx,   y*n_sequential+2+dy] + 
                                                          img_1[x*n_sequential+dx,   y*n_sequential+1+dy] + 
                                                          img_1[x*n_sequential+2+dx, y*n_sequential+1+dy] + 
                                                          img_1[x*n_sequential+1+dx, y*n_sequential+dy] +
                                                          img_1[x*n_sequential+1+dx, y*n_sequential+2+dy] )/5.
        this_grid.sync()
        
        # # #get the current total weights in each psf segment. Note that indices is basically a list, where each element contains (xline, ystart, yend, index). Thus, it is just another, shorter view of indices_fov.
        n_chunks = (indices.shape[0] / xysize)          #it can happen that we have less threads than elements in the list to process. Thus, process them in chunks.
        if int(n_chunks) != n_chunks: n_chunks += 1
        for nc in range(n_chunks):
            # i_indices = int(nc + xy * n_chunks)
            i_indices = nc*xysize + xy
            if i_indices < indices.shape[0]:
                x_line  =  indices[i_indices, 0] 
                y_start =  indices[i_indices, 1]
                y_end   =  indices[i_indices, 2]           
                index   =  indices[i_indices, 3]
                #Now, go over the pixel block, and sum the psf weights
                corr_fac_tmp = np.float64(0.)                                  #Use a temporary variable for doing the following summation in the loop. This one can stay in the register of the kernels, and thus the summation is super fast
                for y_pos in range(y_start, y_end+1):
                        corr_fac_tmp += img_2[x_line+1, y_pos+1]
                cuda.atomic.add(corr_factor, index, corr_fac_tmp)
                cuda.atomic.add(n_totalweights, index, y_end - y_start + 1 )
        this_grid.sync()


        #derive from this the normalization factor
        if xy < len(coeff):
            if (n_totalweights[xy] * corr_factor[xy]) > 0 : corr_factor[xy] = coeff[xy] / (corr_factor[xy] / float(n_totalweights[xy]))
        this_grid.sync()
        
        # #apply the normalization factor to each pixel
        n_chunks = (indices.shape[0] / xysize) 
        if int(n_chunks) != n_chunks: n_chunks += 1
        for nc in range(n_chunks):
            i_indices = nc*xysize + xy
            if i_indices < indices.shape[0]:
                x_line  =  indices[i_indices, 0] 
                y_start =  indices[i_indices, 1]
                y_end   =  indices[i_indices, 2]           
                index   =  indices[i_indices, 3]
                corr_fac_tmp = corr_factor[index]           
                for y_pos in range(y_start, y_end+1):
                        img_1[x_line+1, y_pos+1] = img_2[x_line+1, y_pos+1] * corr_fac_tmp
        this_grid.sync()
                
    

#This version is memory aligned                   
@jit
def create_indices_fromfov(indices_fov, maxsize = 0):  #indices_flipped array into a list, where each element contains (line x, column y_start, column y_end, indices_flipped index). Basically, it is a simple going over the 2d array indices_flipped, and whenever a new index in indices_flipped appears, generate a new element in indices
    indices = list()
    for x in range(indices_fov.shape[0]):
        y = 0
        indices.append([x, y, -1, indices_fov[x, y]]) #this one is the first element in each line x
        index_last = indices_fov[x, y]
        n = 0
        for y in range(indices_fov.shape[1]):                                                       #go through the elements in the line x
            n += 1                                                                                  #count the number of elements
            if (indices_fov[x, y] != index_last) or ((maxsize != 0) and (n > maxsize)):             #if indices_fov has changed from the previous element
                indices[-1][2] = y-1                                                                #put the index of the last element of the line into the last element of indices, i.e., close that element
                indices.append([x, y, -1, indices_fov[x, y]])                                       #create a new element in indices
                index_last = indices_fov[x, y]                                                      #and update index last
                n = 1
        indices[-1][2] = y                                                                          #close the last element
    return np.array(indices)


