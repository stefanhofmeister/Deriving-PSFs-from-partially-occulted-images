"""© 2022 The Trustees of Columbia University in the City of New York. 
This work may be reproduced, distributed, and otherwise exploited for 
academic non-commercial purposes only, provided that it cites the original work: 

    Stefan J. Hofmeister, Michael Hahn, and Daniel Wolf Savin, "Deriving instrumental point spread functions from partially occulted images," 
    Journal of the Optical Society of America A 39, 2153-2168 (2022), doi: 10.1364/JOSAA.471477.

To obtain a license to use this work for commercial purposes, please contact 
Columbia Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""


import numpy as np
import os
from matplotlib import cm
from inout import read_image, plot_image


def parametrize_psf(config):
    folder_run = config['general']['folder_run']
    resolution = config['general']['resolution']
    large_psf  = config['general']['large_psf']

    psf_discretization_file = config['psf_discretization']['psfdisc_file']

    #usually, the PSF kernel is the same size as the image. However, this only allows light to be scattered by half of the image size. If set to true, double the size of the PSF so that scattered light over the full image range is allowed.
    if large_psf:
        resolution *= 2
        
    #This is the segmentation array which is to be filled in this routine
    indices_large_psf = np.full( (resolution, resolution) ,-1)
    
    #first, create a distance map for the PSF from the PSF center       
    xy = np.arange(-resolution//2, resolution//2)
    x_large_psf, y_large_psf = np.meshgrid(xy, xy)
    dist_large_psf = np.sqrt(x_large_psf**2 + y_large_psf**2)
    angles_large_psf = np.arctan2(x_large_psf, y_large_psf) / np.pi * 180.
    angles_large_psf += 180
       
    if ('radius_segments_edges' in  config['psf_discretization']) and ('shape' in  config['psf_discretization']):
        #create definitions of the PSF core
        inner = {}
        inner['shape'] = config['psf_discretization']['shape']
        inner['shells_edges'] = config['psf_discretization']['radius_segments_edges'] 
        inner['radius_min'] = 0
        inner['radius_max'] = np.sqrt(2) * resolution + 1
        if 'radius_inner_outer_boundary' in config['psf_discretization']:
            inner['radius_max'] =  config['psf_discretization']['radius_inner_outer_boundary']
        index_rad_min = [i for i,v in enumerate(inner['shells_edges']) if v >inner['radius_min']][0] -1
        index_rad_max = [i for i,v in enumerate(inner['shells_edges']) if v <inner['radius_max']][-1] +1
        inner['shells_edges'] = inner['shells_edges'][index_rad_min : index_rad_max +1] 
        inner['n_shells'] = len(inner['shells_edges']) -1
        if 'shell_segments' in inner['shape']: inner['shell_segments_angle'] = config['psf_discretization']['width_segments_angle']
        else:                               inner['shell_segments_angle'] = 360
        inner['n_shellsegments'] = 360 // inner['shell_segments_angle']
        inner['n_segments'] = inner['n_shells'] * inner['n_shellsegments']

        #and of the outer PSF region. If the outer one is not define, the core region has to be defined to span the entire PSF.
        outer = {}
        if 'radius_inner_outer_boundary' in config['psf_discretization']:
            outer['shape'] = config['psf_discretization']['outer_shape']
            outer['shells_edges'] = config['psf_discretization']['outer_radius_segments_edges'] 
            outer['radius_min'] = config['psf_discretization']['radius_inner_outer_boundary']
            outer['radius_max'] = np.sqrt(2) * resolution + 1
            index_rad_min = [i for i,v in enumerate(outer['shells_edges']) if v >outer['radius_min']][0] -1
            index_rad_max = [i for i,v in enumerate(outer['shells_edges']) if v <outer['radius_max']][-1] +1
            outer['shells_edges'] = outer['shells_edges'][index_rad_min : index_rad_max +1] 
            outer['n_shells'] = len(outer['shells_edges']) -1
            if 'outer_shell_segments' in outer['shape']: outer['shell_segments_angle'] = config['psf_discretization']['outer_width_segments_angle']
            else:                                        outer['shell_segments_angle'] = 360
            outer['n_shellsegments'] = 360 // outer['shell_segments_angle']
            outer['n_segments'] = outer['n_shells'] * outer['n_shellsegments']
        else:
            outer['n_segments'] = 0
            outer['shape'] = ''
            outer['n_shells'] = 0
            outer['n_shellsegments'] = 0
            outer['shell_segments_angle'] = 0
            outer['shells_edges'] = 0

        #define an output array
        n_segments_total = int(inner['n_segments']) + int(outer['n_segments'])
        shells = {'index': np.full(n_segments_total, -1, dtype = np.int32), 'npix': np.full(n_segments_total, 0, dtype = np.int32), 'radius_mean_px': np.full(n_segments_total, np.nan, dtype = np.float16), 'radius_min_px': np.full(n_segments_total, np.nan, dtype = np.float16), 'radius_max_px': np.full(n_segments_total, np.nan, dtype = np.float16), 'angle_mean': np.full(n_segments_total, np.nan, dtype = np.float16), 'angle_min': np.full(n_segments_total, np.nan, dtype = np.float16), 'angle_max': np.full(n_segments_total, np.nan, dtype = np.float16)}

        #Now, derive the PSF segmentation for the core and outer region
        index_segment = 0
        for region in [inner, outer]:
            if region['n_segments'] == 0: continue

            #Now, create the segments in the psf kernel
            if region['shape'] == 'shells':
                #if one uses simple shells, no segmentation of the individual shells is needed
                for ind_shell in range(region['n_shells']): 
                    mask_shell = (dist_large_psf >= region['shells_edges'][ind_shell]) & (dist_large_psf < region['shells_edges'][ind_shell+1]) & (dist_large_psf >= region['radius_min']) & (dist_large_psf <= region['radius_max'])
                    indices_large_psf[mask_shell] = index_segment
                    shells['index'][index_segment] = index_segment
                    index_segment += 1

            if region['shape'] == 'shell_segments':    
                #splits each shell into segments of shell_segments_angle degrees
                #first create a map of the angles, i.e., of each pixel to the image center
                #then, create the accroding maps for each angle: first, set the shell segment around angle==0 manually, and use for all others the remaining loop iterations
                for ind_angle in range(region['n_shellsegments']):
                    for ind_shell in range(region['n_shells']):
                        mask_shell = (dist_large_psf >= region['shells_edges'][ind_shell]) & (dist_large_psf < region['shells_edges'][ind_shell+1])  & (dist_large_psf >= region['radius_min']) & (dist_large_psf <= region['radius_max'])
                        mask_angle = (angles_large_psf >= ind_angle*region['shell_segments_angle']) & (angles_large_psf <= (ind_angle + 1)*region['shell_segments_angle'])
                        mask_segment = (mask_shell == 1) & (mask_angle == 1)
                        indices_large_psf[mask_segment] = index_segment
                        shells['index'][index_segment] = index_segment
                        index_segment += 1
    else:
        n_segments_total = 0
        psf_shape_inner = ''
        inner = {}
        inner['shape'] = ''
        inner['n_shells'] = 0
        inner['n_shellsegments'] = 0
        inner['shell_segments_angle'] = np.nan
        inner['shells_edges'] = []
        outer = {}
        outer['shape'] = ''
        outer['n_shells'] = 0
        outer['n_shellsegments'] = 0
        outer['shell_segments_angle'] = np.nan
        outer['shells_edges'] = []      
        shells = {}
        shells['index'] = []
        shells['npix'] = []        
        shells['radius_mean_px'] = []
        shells['radius_min_px'] = []
        shells['radius_max_px'] = []
        shells['angle_mean'] = [] 
        shells['angle_min'] = []
        shells['angle_max'] = []

    
    #if a manual discretization file was provided, use the segments defined there to overwrite the shells and shell segments defined before. This allows to stack a manual segmentation, e.g., for the diffraction pattern, with simple shells.
    #in the manual segmentation file, all pixel values <= are ignored. Each value > 0 is an index for the segmentation map, i.e., all pixels with the same value belong to the same segment.
    if psf_discretization_file:
        segments = read_image(folder_run + os.path.splitext(os.path.basename(psf_discretization_file))[0] + '.npz' )
        mask_disc = (segments != -1)
        indices = np.unique(segments[mask_disc])
        n_manual_shells = len(indices)
        manual_shells = {'index': np.full(n_manual_shells, -1, dtype = np.int32), 'npix': np.full(n_manual_shells, 0, dtype = np.int32), 'radius_mean_px': np.full(n_manual_shells, np.nan, dtype = np.float16), 'radius_min_px': np.full(n_manual_shells, np.nan, dtype = np.float16), 'radius_max_px': np.full(n_manual_shells, np.nan, dtype = np.float16), 'angle_mean': np.full(n_manual_shells, np.nan, dtype = np.float16), 'angle_min': np.full(n_manual_shells, np.nan, dtype = np.float16), 'angle_max': np.full(n_manual_shells, np.nan, dtype = np.float16)}
        indices_large_psf_max = np.max(indices_large_psf)
        for index_new, index in enumerate(indices):
            mask = (segments == index)
            indices_large_psf[mask] = indices_large_psf_max + 1 + index_new
            manual_shells['index'][index_new] = indices_large_psf_max + 1 + index_new
            manual_shells['npix'][index_new] = np.sum(mask)            
            if np.sum(mask):
                manual_shells['radius_mean_px'][index_new] = np.mean(dist_large_psf[mask]) 
                manual_shells['radius_min_px'][index_new] = np.min(dist_large_psf[mask])
                manual_shells['radius_max_px'][index_new] = np.max(dist_large_psf[mask])
                manual_shells['angle_mean'][index_new] =  np.mean(angles_large_psf[mask]) 
                manual_shells['angle_min'][index_new] = np.min(angles_large_psf[mask])
                manual_shells['angle_max'][index_new] = np.max(angles_large_psf[mask])
    else:
        n_manual_shells = 0
        manual_shells = {'index': [], 'npix': [], 'radius_mean_px': [], 'radius_min_px': [], 'radius_max_px': [], 'angle_mean': [], 'angle_min': [], 'angle_max': []}

    #For each PSF segment, derive its statistics
    for index_segment in shells['index']:
        mask_segment = (indices_large_psf == index_segment)
        if np.sum(mask_segment):
            shells['npix'][index_segment] = np.sum(mask_segment) 
            shells['radius_mean_px'][index_segment] = np.mean(dist_large_psf[mask_segment]) 
            shells['radius_min_px'][index_segment] = np.min(dist_large_psf[mask_segment])
            shells['radius_max_px'][index_segment] = np.max(dist_large_psf[mask_segment]) 
            shells['angle_mean'][index_segment] =  np.mean(angles_large_psf[mask_segment]) 
            shells['angle_min'][index_segment] = np.min(angles_large_psf[mask_segment])
            shells['angle_max'][index_segment] = np.max(angles_large_psf[mask_segment])

    #to speed up computiation, convert it to uint16. Allows up to 65565 shells 
    indices_large_psf = indices_large_psf.astype(np.int16) 

    #finally, save everything
    if large_psf:
        resolution = int( resolution // 2 )
        
    indices_large_psf, resolution, n_shells, n_manual_shells,  = np.array(indices_large_psf, dtype = np.int32), np.int16(resolution), np.int16(n_segments_total), np.int32(n_manual_shells)
    n_shells_inner, n_shellsegments_inner, shell_segments_angle_inner, shells_edges_inner = np.int16(inner['n_shells']), np.int16(inner['n_shellsegments']), np.float16(inner['shell_segments_angle']), np.array(inner['shells_edges']).astype(np.float16)
    n_shells_outer, n_shellsegments_outer, shell_segments_angle_outer, shells_edges_outer = np.int16(outer['n_shells']), np.int16(outer['n_shellsegments']), np.float16(outer['shell_segments_angle']), np.array(outer['shells_edges']).astype(np.float16)


    plot_image(indices_large_psf, folder_run + '/psf_segmentation/psf_segments.jpg', plot_norm = 'lin', cmap = cm.flag)
    np.savez_compressed(folder_run+'/psf_segmentation/psf_segments.npz', 
             indices_fov=indices_large_psf, resolution = resolution, large_psf = large_psf, n_shells = n_shells, n_manual_shells = n_manual_shells,
             psf_shape_inner = inner['shape'], n_shells_inner = n_shells_inner, n_shellsegments_inner = n_shellsegments_inner, shell_segments_angle_inner = shell_segments_angle_inner, shells_edges_inner = shells_edges_inner,
             psf_shape_outer = outer['shape'], n_shells_outer = n_shells_outer, n_shellsegments_outer = n_shellsegments_outer, shell_segments_angle_outer = shell_segments_angle_outer, shells_edges_outer = shells_edges_outer,
             shells = shells, manual_shells = manual_shells)

    
