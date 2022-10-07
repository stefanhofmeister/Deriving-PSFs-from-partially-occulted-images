"""© 2022 The Trustees of Columbia University in the City of New York.
This work may be reproduced, distributed, and otherwise exploited for
academic non-commercial purposes only, provided that it cites the
original work (IN PUBLICATIONP PROCESS).  To obtain a license to
use this work for commercial purposes, please contact Columbia
Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

from initialize import read_config, initialize, load_psf
from parametrize_psf import parametrize_psf
from create_testrun import create_psf, create_artificial_image
from deconvolve_image import deconvolve_images 
from find_occulted_pixels import find_occulted_pixels
from derive_distance_from_illuminated_edge import derive_distance_from_illuminated_edge
from derive_importance_mask import derive_importance_mask
from estimate_true_image import estimate_true_image             
from set_up_system_of_equations import set_up_system_of_equations
from solve_system_of_equations import solve_system_of_equations
from assemble_psf import assemble_psf
from finalize import finalize 
    
import warnings
warnings.filterwarnings("ignore")

def derive_psf_from_partially_occulted_image(config_in):           
        #read the configuration file and transform all paths in the config file to absolute paths. config_in can either be a structure containing the configuration or a path to a configuration file.
        config = read_config(config_in) 
   
        # #intialize the program. Set up the run directory and copy the required files and images
        print('Initialize')
        initialize(config)
                
        # #Defines a testrun with predefined artificial PSFs and occulted images
        if 'testrun' in config:
            if config['testrun']['PSF_testrun'] == True: 
                print('Set up the testrun')
                create_psf(config) 
                create_artificial_image(config)

        # #parametrize the PSF into segments
        print('Segment the PSF')
        parametrize_psf(config)

        #if a PSF is already existing as first guess, load it. If not, create a dummy PSF where all the energy is in the central pixel
        load_psf(config)
                                              
        #Now, iterate over our methodology:
        for iteration in range(config['general']['n_iterations']):
            print('Iteration {}'.format(iteration))
                        
            print('   Deconvolve the images with the PSF')
            deconvolve_images(config)
            
            print('   Find the occulted pixels')
            find_occulted_pixels(config)
            
            print('   Derive the distance from the illuminated edge')
            derive_distance_from_illuminated_edge(config)
            
            print('   Derive the importance mask')
            derive_importance_mask(config   )
            
            print('   Estimate the true image')
            estimate_true_image(config)
            
            print('   Set up the system of equations')
            set_up_system_of_equations(config)

            print('   Solve the system of equations')
            solve_system_of_equations(config, iteration)

            print('   Postprocess the PSF')
            assemble_psf(config)
            
        print('Finalizing')
        finalize(config)  
        
        print('The result can be foudn in {}'.format(config['general']['folder_run']))
            