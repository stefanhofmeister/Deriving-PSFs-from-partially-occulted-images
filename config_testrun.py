"""© 2022 The Trustees of Columbia University in the City of New York. 
This work may be reproduced, distributed, and otherwise exploited for 
academic non-commercial purposes only, provided that it cites the original work: 

    Stefan J. Hofmeister, Michael Hahn, and Daniel Wolf Savin, "Deriving instrumental point spread functions from partially occulted images," 
    Journal of the Optical Society of America A 39, 2153-2168 (2022), doi: 10.1364/JOSAA.471477.

To obtain a license to use this work for commercial purposes, please contact 
Columbia Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

#This configuration file will be imported in python and sets up the parameters for deriving the PSF. For each run, adapt this configuration file to your needs. 
#As this file will be imported, you can use all python commands to define arrays. 

import numpy as np

config =  {}
config['general'] = {
   'folder_run':                     '/data/sjh/datasets/2021_PSF_general/zernike_multilinear_newalgorithm/',
   'files_occulted':                 ''  
   'file_psf_existing':              '',
   'resolution':                     1024,           
   'n_iterations':                   3,
   'large_psf':                      False, 
   'use_gpu':                        True }
    
config['testrun'] = {
   'PSF_testrun':                    True,
   'file_psf':                       '', 
   'PSF_shape':                      ['gaussian_core', 'circle'],
   'occmask_shape':                  ['large_hole'],
   'occmask_size_largehole':         0.05*1024,    
   'occmask_size_pinholes':          1,
   'occmask_n_pinholes':             49,
   'occmask_signaltonoise':          0 }

config['psf_discretization'] = {
   'psfdisc_file':                   '',              
   'shape':                          'shells',        #shells or shell_segments
   'radius_segments_edges':          [0, 1, 1.4, 1.8, 2.2, 2.6, 3., 3.4, 3.8, 4.2, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, *np.logspace(np.log10(11.), np.log10(np.sqrt(2) * config['general']['resolution']//2 + 1), num = 25)],       
   'width_segments_angle':           30}
    
config['occultation_mask'] = {
   'method':                         'threshold',     #'predefined', 'threshold'
   'occ_file':                       '',             
   'threshold':                      0.5,
   'threshold_usemedian':            False }    
                            
config['importance_mask'] = {
   'method':                        'psf segmentation',     #'psf segmentation', 'distances', 'predefined'
   'importance_file':               '',             
   'distance_edges':                 config['psf_discretization']['radius_segments_edges'] }
                                            
config['deconvolution'] = {
   'n_iterations':                   100,
   'pad':                            True }
                                                  
config['system_of_equations'] = {
   'n_px_occulted':                  10000, #   100000,            
   'n_px_illuminatededge':           0,           
   'dist_from_illuminatededge':      10,            
   'n_px_illuminated':               0,              
   'n_illuminated_bins':             10,            
   'rebin_occulted':                 [1],           
   'rebin_dist_occulted':            [0, np.sqrt(2) * config['general']['resolution']] }
    
config['fitting'] = {
   'fit_function':                   'multilinear_cpu',
   'n_fit_repetitions':              1,
   'n_samples_occulted':             1000,
   'n_samples_illuminatededge':      0,
   'n_samples_illuminated':          0,
   'only_use_best_fraction':         1.,
   'split_training_evaluation':      1.,
   'regularization_alpha':           .001,
   'fit_repetition_chunksize':       10,
   'tolerance':                      1e-5, 
   'max_iter':                       100000,
   'piecewise_min_samples': 10,
   'piecewise_min_dist': 10,
   'piecewise_min_coeff': 10} 
   # '1-2_fit_function':               'multilinear_cpu',
   # '1-2_n_fit_repetitions':          1,
   # '1-2_n_samples_occulted':         1500,
   # }
    
config['postprocessing'] = {
   'n_smooth_iterations':            0,
   'rebin':                          [1] }  # [101, 75, 25, 11, 5, 1]}

config['finalizing'] = {
   'create_PSF_deconvolved_images':  True}

 
from derive_psf_from_partially_occulted_image import derive_psf_from_partially_occulted_image
derive_psf_from_partially_occulted_image(config)
