"""© 2022 The Trustees of Columbia University in the City of New York. 
This work may be reproduced, distributed, and otherwise exploited for 
academic non-commercial purposes only, provided that it cites the original work: 

    Stefan J. Hofmeister, Michael Hahn, and Daniel Wolf Savin, "Deriving instrumental point spread functions from partially occulted images," 
    Journal of the Optical Society of America A 39, 2153-2168 (2022), doi: 10.1364/JOSAA.471477.

To obtain a license to use this work for commercial purposes, please contact 
Columbia Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

import numpy as np
import scipy as scp 
from functools import partial
import glob
import sys
import os
from numba import jit 
from scipy.interpolate import RectBivariateSpline, interp1d
try:
    import gpufit as gf
except:
    def gf(*dummy, **dummy2):
        print('Gpufit is not installed. Exiting.')
        sys.exit()
        

from inout import read_image
import matplotlib.pyplot as plt

def solve_system_of_equations(config, iteration):

    #some definitions
    folder_run                  = config['general']['folder_run']
    if (iteration < 2) and ('1-2_fit_function' in config['fitting']):                fit_function = config['fitting']['1-2_fit_function']
    else:                                                                            fit_function = config['fitting']['fit_function']
    if (iteration < 2) and ('1-2_n_fit_repetitions' in config['fitting']):           total_iterations = config['fitting']['1-2_n_fit_repetitions']
    else:                                                                            total_iterations = config['fitting']['n_fit_repetitions']
    if (iteration < 2) and ('1-2_n_samples_occulted' in config['fitting']):          samples = config['fitting']['1-2_n_samples_occulted']
    else:                                                                            samples = config['fitting']['n_samples_occulted'] 
    if (iteration < 2) and ('1-2_n_samples_illuminatededge' in config['fitting']):   n_samples_illuminatededge = config['fitting']['1-2_n_samples_illuminatededge']
    elif 'n_samples_illuminatededge' in config['fitting']:                           n_samples_illuminatededge = config['fitting']['n_samples_illuminatededge']
    else:                                                                            n_samples_illuminatededge = 0
    if (iteration < 2) and ('1-2_n_samples_illuminated' in config['fitting']):       n_samples_illuminated = config['fitting']['1-2_n_samples_illuminated']
    elif 'n_samples_illuminated' in config['fitting']:                               n_samples_illuminated = config['fitting']['n_samples_illuminated']
    else:                                                                            n_samples_illuminated = 0
    if (iteration < 2) and ('1-2_tolerance' in config['fitting']):                   tolerance = config['fitting']['1-2_tolerance']
    elif 'tolerance' in config['fitting']:                                           tolerance = config['fitting']['tolerance']
    else:                                                                            tolerance = 1e-5 
    if 'max_iter' in config['fitting']:                                              max_iter = config['fitting']['max_iter']
    else:                                                                            max_iter = 10000 
    if 'split_training_evaluation' in config['fitting']:                             split_training_evaluation = config['fitting']['split_training_evaluation']
    else:                                                                            split_training_evaluation = 1
    if 'best_fraction' in config['fitting']:                                         best_fraction = config['fitting']['only_use_best_fraction']
    else:                                                                            best_fraction = 1.
    if'regularization_alpha' in config['fitting']:                                        alpha = config['fitting']['regularization_alpha']
    if 'fit_repetition_chunksize' in config['fitting']:                              chunksize = config['fitting']['fit_repetition_chunksize']
    else:                                                                            chunksize = total_iterations
    if (iteration < 2) and ('1-2_piecewise_min_samples' in config['fitting']):       piecewise_min_samples = config['fitting']['1-2_piecewise_min_samples']
    elif 'piecewise_min_samples' in config['fitting']:                               piecewise_min_samples = config['fitting']['piecewise_min_samples']
    else:                                                                            piecewise_min_samples = 1
    if (iteration < 2) and ('1-2_piecewise_min_coefficients' in config['fitting']):  piecewise_min_coefficients = config['fitting']['1-2_piecewise_min_coefficients']
    elif 'piecewise_min_coefficients' in config['fitting']:                          piecewise_min_coefficients = config['fitting']['piecewise_min_coefficients']
    else:                                                                            piecewise_min_coefficients = 1
    if (iteration < 2) and ('1-2_piecewise_min_dist' in config['fitting']):          piecewise_min_dist = config['fitting']['1-2_piecewise_min_dist']
    elif 'piecewise_min_dist' in config['fitting']:                                  piecewise_min_dist = config['fitting']['piecewise_min_dist']
    else:                                                                            piecewise_min_dist = 3
    if 'x_scaling_factor' in config['fitting']:                                      x_scaling_factor = config['fitting']['x_scaling_factor']
    else:                                                                            x_scaling_factor = 1e5
    

    #load the psf segmentation
    psf_segments        = np.load(folder_run + '/psf_segmentation/psf_segments.npz', allow_pickle=True)    
    n_coeffs            = psf_segments['n_shells'] + psf_segments['n_manual_shells']
    shells_npix_orig    =  np.array([*psf_segments['shells'].item()['npix'], *psf_segments['manual_shells'].item()['npix']])
    shells_radii_px     =  np.array([*psf_segments['shells'].item()['radius_mean_px'], *psf_segments['manual_shells'].item()['radius_mean_px']])
    shells_radii_px_min =  np.array([*psf_segments['shells'].item()['radius_min_px'], *psf_segments['manual_shells'].item()['radius_min_px']])
    shells_radii_px_max =  np.array([*psf_segments['shells'].item()['radius_max_px'], *psf_segments['manual_shells'].item()['radius_max_px']])

    #these are only needed for fits wich use a support grid. As such, in these parameters, the values are divided between the support grid (radii, n_shells, n_shellsegments) and n_manualshells
    shells_radii_px_for_supportgrid = psf_segments['shells'].item()['radius_mean_px'] #used only for support grid for psf_shape == 'shell_segments' or 'shells', thus without manual shells 
    n_shells_for_supportgrid        = psf_segments['n_shells_inner']
    n_shellsegments_for_supportgrid = psf_segments['n_shellsegments_inner']
    n_manual_shells                 = psf_segments['n_manual_shells']
    
    
    #check if the system of equations is well determined
    if (samples < n_coeffs) and (samples != -1):
        print('System of equations is underdetermined: Samples < Coefficients. Exiting')
        sys.exit()
     
    #load the equations and derive the weights for each line, i.e., occulted pixel. Also add a file number for each file/system of equations.
    files_system_of_equations = glob.glob(folder_run + '/system_of_equations/*.npz')
    nfiles = len(files_system_of_equations) 
    datacube = list()  
    for i_file, file in enumerate(files_system_of_equations): 
        equations = np.load(file, allow_pickle = True)['system_of_equations'].item()
        equations = add_weights(equations)
        equations['file_index'] = np.repeat(i_file, len(equations['intensities']))
        datacube.append(equations)
        
    #combine all the system of equations into a single one
    datacube_tmp = {}
    for key in datacube[0].keys():
        datacube_tmp[key] = np.concatenate([item[key] for item in datacube], 0)
    datacube = datacube_tmp   

    coeff, rmse = list(), list()
    #in case that we use many fit repetitions, we process them in chunks. Thus, derive the number of chunks needed from the chunk size, and start processing the chunks
    n_chunks = int(np.ceil(total_iterations / chunksize))
    for index_chunk in range(n_chunks):
        #derive the number of iterations within the chunk
        iterations = min(chunksize, total_iterations - index_chunk * chunksize)
        
        #for each iteration, we split the files containing the individual system of equations into a training dataset and an evaluation dataset, and extract the parameters needed for the fitting
        datacubes_training, datacubes_evaluation = list(), list()
        for i in range(iterations):
            #First, split the dataset into a training and evaluation dataset
            if (split_training_evaluation == 1) or (nfiles == 1): #if only one input file exists or no splitting was requested, use the same file(s) as training and evaluation dataset
                training_files, evaluation_files = np.arange(nfiles), np.arange(nfiles)
            else:
                training_files = np.zeros(nfiles)
                n_training_files = np.round(split_training_evaluation * nfiles) 
                n_training_files = max(1, n_training_files)  #so that at least one file is in training_files and evaluation_files
                n_training_files = min(n_training_files, nfiles - 1)
                training_files[np.random.choice(nfiles, n_training_files, replace = False)] = 1
                evaluation_files = np.where(training_files == 0)[0]
                training_files = np.where(training_files == 1)[0]
            
            #draw samples which are used for the fitting from the training dataset
            n_occulted, n_illuminatededge, n_illuminated = split_training_evaluation * np.array([samples, n_samples_illuminatededge, n_samples_illuminated])
            datacube_training = draw_samples(datacube, file_indices = training_files, n_occulted = n_occulted, n_illuminatededge = n_illuminatededge, n_illuminated = n_illuminated)
            #if no splitting was requested, the evaluation dataset is the training dataset. 
            if (split_training_evaluation == 1):
                datacube_evaluation = datacube_training
            #else: draw samples which are used for the evaluation dataset
            else:           
                n_occulted, n_illuminatededge, n_illuminated = (1. - split_training_evaluation) * np.array([samples, n_samples_illuminatededge, n_samples_illuminated])
                datacube_evaluation = draw_samples(datacube, file_indices = evaluation_files, n_occulted = n_occulted, n_illuminatededge = n_illuminatededge, n_illuminated = n_illuminated)
            datacubes_training.append(datacube_training)
            datacubes_evaluation.append(datacube_evaluation)
        
        #extract the parameters which are needed for the fitting
        X_training_dataset = np.array([dc['intensities_shells']               for dc in datacubes_training], dtype = np.float32)
        Y_training_dataset = np.array([dc['intensities']                      for dc in datacubes_training], dtype = np.float32)
        Y_diff_training_dataset = np.array([dc['intensities_explained']       for dc in datacubes_training], dtype = np.float32)
        weights_training_dataset = np.array([dc['weights']                    for dc in datacubes_training], dtype = np.float32)
        distance_to_edge_training_dataset = np.array([dc['distance_to_edge']  for dc in datacubes_training], dtype = np.float32)

        X_evaluation_dataset = np.array([dc['intensities_shells']         for dc in datacubes_evaluation], dtype = np.float32)
        Y_evaluation_dataset = np.array([dc['intensities']                for dc in datacubes_evaluation], dtype = np.float32)
        Y_diff_evaluation_dataset = np.array([dc['intensities_explained'] for dc in datacubes_evaluation], dtype = np.float32)
        weights_evaluation_dataset = np.array([dc['weights']              for dc in datacubes_evaluation], dtype = np.float32)
    
        shells_npix_orig = np.array(shells_npix_orig, dtype = np.float32)
        #normalize the X data. This is required to get reasonable termination condition for the fit, i.e., when it assumes to be converged. This will also affect the values of the fitted coefficients; consequently, the normalization will be reversed after the fitting (i.e., at the end of this program)
        if 'cpu' in fit_function:
                file_coeffs = folder_run + '/fitted_psf_coefficients/fitted_coefficients.npz'
                if os.path.isfile(file_coeffs): xscale = 1./ np.load(file_coeffs)['coeff']
                else:                           xscale = (shells_radii_px.astype(np.float64))**2
                xscale *= x_scaling_factor 
                xscale = xscale.astype(np.float32)
                xscale[(np.isfinite(xscale) == 0) | (xscale == 0.)] = 1
                X_training_dataset /= xscale
                X_evaluation_dataset /= xscale
                shells_npix = shells_npix_orig / xscale
        else:
                shells_npix = shells_npix_orig

        
        #finally perform the fitting
        if fit_function == 'multilinear_cpu':
                p0 = np.ones((iterations, n_coeffs), dtype = np.float64)
                shells_npix[np.isfinite(shells_npix) == 0] = 0
                for i in range(iterations):
                    coeff_fit = 0
                    multilinear_fit_partial = partial(multilinear_fit, y_diff = Y_diff_training_dataset[i, :], shells_npix = shells_npix)
                    coeff_fit, cov = scp.optimize.curve_fit(multilinear_fit_partial, X_training_dataset[i, :, :], Y_training_dataset[i, :], p0= p0[i, :],  bounds = (np.zeros(n_coeffs), np.full(n_coeffs, np.inf)),  method = 'trf', max_nfev = max_iter, sigma = 1./weights_training_dataset[i, :], ftol = None, gtol = None, xtol = tolerance, diff_step = 1e-3)
                    coeff_fit[np.sum(X_training_dataset[i, :, :], axis = 0) == 0] = np.nan
                    Y_pred = multilinear_fit(X_evaluation_dataset[i, :, :], *coeff_fit, y_diff = Y_diff_evaluation_dataset[i, :], shells_npix = shells_npix)
                    coeff.append(coeff_fit)
                    rmse.append(np.sqrt(np.nanmean((Y_pred - Y_evaluation_dataset[i, :])**2 * weights_evaluation_dataset[i, :]**2)))
                    
        
        if fit_function == 'multilinear_piecewise_cpu':
            #here, we split the system of equations up into multiple smaller ones
            p0 = np.ones((iterations, n_coeffs), dtype = np.float64)
            shells_npix[np.isfinite(shells_npix) == 0] = 0
            for i in range(iterations):
                #first, some definitions
                coeff_fit = 0
                X_training_dataset_i = X_training_dataset[i, :, :].astype(np.float32)
                Y_training_dataset_i = Y_training_dataset[i, :].astype(np.float32)
                Y_diff_training_dataset_i = Y_diff_training_dataset[i, :].astype(np.float32)
                weights_training_dataset_i = weights_training_dataset[i, :].astype(np.float32)
                p0_i = p0[i, :]
                distance_to_edge_training_dataset_i = distance_to_edge_training_dataset[i, :]

                #get the distances of the edges of the PSF segments. These will be used to define the boundaries between the system of equationes
                dist_coefficients = np.unique(np.concatenate([shells_radii_px_min, shells_radii_px_max]))
                dist_coefficients = dist_coefficients[np.isfinite((dist_coefficients))]
                
                #here, we put the definitions of the sub-system of equations          
                fit_definition = list()
            
                #Now, lets go through the dist_coefficients in reversed order
                edge = [len(dist_coefficients)-2, len(dist_coefficients)-1]
                while True:
                    #look how many samples of occulted pixels have distances larger than dist_coefficient[edge[0]] and smaller than dist_coefficient[edge[1]]
                    y_selected = (distance_to_edge_training_dataset_i >= dist_coefficients[edge[0]]) & (distance_to_edge_training_dataset_i <= dist_coefficients[edge[1]])
                    n_y_selected = np.sum(y_selected)
                    #look on which coefficients they depend
                    coeffs_selected = np.sum(X_training_dataset_i[y_selected, :], axis = 0) != 0                    
                    #look which coefficients are within the required distance to fit. The other ones have already been fitted in previous iterations, as we go from large distances from the illuminated edge to smaller ones.
                    coeffs_to_fit = (shells_radii_px >= dist_coefficients[edge[0]]) & (shells_radii_px <= dist_coefficients[edge[1]])
                    n_coeffs_to_fit = np.sum(coeffs_to_fit)
                    
                    #if the lower edge is at zero, this will be the last sub-system of equations to add
                    if edge[0] == 0:
                        if n_coeffs_to_fit >0:
                            fit_definition.append( {'dist_range': [dist_coefficients[edge[0]], dist_coefficients[edge[1]]], 'y_selected': y_selected, 'coeffs_selected': coeffs_selected, 'coeffs_to_fit': coeffs_to_fit, 'coeffs_constant': (coeffs_selected & (coeffs_to_fit == 0)) } )
                        break
                                  
                    #if several conditions, i.e., the minimum number of coefficients to fit, the minimum distance of the coefficients to fit in the sub-system of equations, and the minimum number of occulted pixel per coeffient to fit are fullfilled, add it to the list of system of equations
                    if (n_y_selected > 0 ) and (n_y_selected >= n_coeffs_to_fit * piecewise_min_samples) and (n_coeffs_to_fit > piecewise_min_coefficients) and ((dist_coefficients[edge[1]] - dist_coefficients[edge[0]]) >= piecewise_min_dist):
                        fit_definition.append( {'dist_range': [dist_coefficients[edge[0]], dist_coefficients[edge[1]]], 'y_selected': y_selected, 'coeffs_selected': coeffs_selected, 'coeffs_to_fit': coeffs_to_fit, 'coeffs_constant': (coeffs_selected & (coeffs_to_fit == 0)) } )
                        edge[1] = edge[0]
                        edge[0] -= 1
                    #else: reduce the distance of the lower boundary of the system of equations to get more samples and coefficients into the sub-system of equations
                    else:
                        edge[0] -= 1  
                    
                #having created the sub-system of equations, we fit one by one
                for fitdef in fit_definition:
                    if np.any(fitdef['coeffs_constant']):
                        x_constant = X_training_dataset_i[fitdef['y_selected'], :]
                        x_constant = x_constant[:, fitdef['coeffs_constant']]
                        y_constant = np.matmul(x_constant, p0_i[fitdef['coeffs_constant']])
                        totalweight_longscatter_constant = np.sum(p0_i[fitdef['coeffs_constant']] * shells_npix.flatten()[fitdef['coeffs_constant']]) 
                    else:
                        y_constant = 0
                        totalweight_longscatter_constant = 0
                    
                    multilinear_piecewise_fit_partial = partial(multilinear_piecewise_fit, 
                                                      y_diff = Y_diff_training_dataset_i[fitdef['y_selected']], 
                                                      shells_npix = shells_npix.flatten()[fitdef['coeffs_to_fit']],
                                                      y_constant = y_constant, 
                                                      totalweight_longscatter_constant = totalweight_longscatter_constant                               
                                                      )
 
                    
                    p0_tmp = p0_i[fitdef['coeffs_to_fit']]
                    X_tmp =  X_training_dataset_i[fitdef['y_selected'], :]
                    X_tmp =  X_tmp[:, fitdef['coeffs_to_fit']]
                    Y_tmp =  Y_training_dataset_i[fitdef['y_selected']]
                    bounds_tmp =  (np.zeros(len(p0_i[fitdef['coeffs_to_fit']]), dtype = np.float32), np.full(len(p0_i[fitdef['coeffs_to_fit']]), np.inf, dtype = np.float32))
                    weights_tmp = weights_training_dataset_i[fitdef['y_selected']]
    
                    coeff_fit, cov = scp.optimize.curve_fit(multilinear_piecewise_fit_partial, X_tmp, Y_tmp, p0= p0_tmp,  bounds =bounds_tmp,  method = 'trf', max_nfev = 10000, sigma = 1./weights_tmp, ftol = None, gtol = None, xtol = tolerance, diff_step = 1e-3)
                    p0_i[fitdef['coeffs_to_fit']] = coeff_fit

                #finally, add the solution to the output array and derive the errors
                p0_i[np.sum(X_training_dataset[i, :, :], axis = 0) == 0] = np.nan
                Y_pred = multilinear_fit(X_evaluation_dataset[i, :, :], *p0_i, y_diff = Y_diff_evaluation_dataset[i, :], shells_npix = shells_npix)
                coeff.append(p0_i)
                rmse.append(np.sqrt(np.nanmean((Y_pred - Y_evaluation_dataset[i, :])**2 * weights_evaluation_dataset[i, :]**2)))
                           
                    
        if fit_function == 'powerlaw_gaussian_cpu':
                p0 =np.array([ 1,  1., 1., 1., 2.0]).astype(np.float128)
                p0 = np.tile(p0, (iterations, 1))
                shells_npix[np.isfinite(shells_npix) == 0] = 0
                shells_radii_px[np.isfinite(shells_radii_px) == 0] = -1
                for i in range(iterations):
                    coeff_fit = 0
                    powerlaw_gaussian_partial = partial(powerlaw_gaussian_cpu, y_diff = Y_diff_training_dataset[i, :], shells_npix = shells_npix,  r = shells_radii_px.flatten())
                    coeff_fit, cov = scp.optimize.curve_fit(powerlaw_gaussian_partial, X_training_dataset[i, :, :], Y_training_dataset[i, :], p0= p0[i, :],  method = 'trf', max_nfev = max_iter, sigma = 1./weights_training_dataset[i, :], ftol = tolerance, gtol = None, xtol = None)
                    coeff_fit = coeff_fit[2] * 1./ (coeff_fit[3]+((shells_radii_px.flatten() +1.e-5))**coeff_fit[4]) + coeff_fit[0]*np.exp(-1*( (shells_radii_px.flatten()+1.e-5 )**2 / 2. / (coeff_fit[1])**2. ))
                    coeff_fit[0] = 0
                    coeff_fit[np.sum(X_training_dataset[i, :, :], axis = 0) == 0] = np.nan
                    Y_pred = multilinear_fit(X_evaluation_dataset[i, :, :], *coeff_fit, y_diff = Y_diff_evaluation_dataset[i, :], shells_npix = shells_npix) #0 in evaluation dataset is correct. At the moment, all iterations use the same evaluation
                    coeff.append(coeff_fit)
                    rmse.append(np.sqrt(np.nanmean((Y_pred - Y_evaluation_dataset[i, :])**2 * weights_evaluation_dataset[i, :]**2)))
        
             
        if fit_function == 'multilinear_regularization_cpu':
                p0 =np.ones((iterations, n_coeffs), dtype = np.float64)
                shells_npix[np.isfinite(shells_npix) == 0] = 0
                for i in range(iterations):
                    coeff_fit = 0
                    Y_training_expanded = np.concatenate(([0.], Y_training_dataset[i, :]))
                    weights_expanded = np.concatenate(( [1.], weights_training_dataset[i, :]))
                    alpha_reg = alpha * len(Y_training_dataset[i, :])
                    
                    multilinear_fit_reg_partial = partial(multilinear_fit_reg, y_diff = Y_diff_training_dataset[i, :], shells_npix = shells_npix, alpha = alpha_reg)
                    coeff_fit, cov = scp.optimize.curve_fit(multilinear_fit_reg_partial, X_training_dataset[i, :, :], Y_training_expanded, p0= p0[i, :], bounds = (np.zeros(n_coeffs), np.full(n_coeffs, np.inf)),  method = 'trf', max_nfev = max_iter, sigma = 1./weights_expanded, ftol = None, gtol = None, xtol = tolerance, diff_step = 1e-3)#, loss = 'cauchy')
                    coeff_fit[np.sum(X_training_dataset[i, :, :], axis = 0) == 0] = np.nan
                    Y_pred = multilinear_fit(X_evaluation_dataset[i, :, :], *coeff_fit, y_diff = Y_diff_evaluation_dataset[i, :], shells_npix = shells_npix) #0 in evaluation dataset is correct. At the moment, all iterations use the same evaluation
                    coeff.append(coeff_fit)
                    rmse.append(np.sqrt(np.mean((Y_pred - Y_evaluation_dataset[i, :])**2 * weights_evaluation_dataset[i, :]**2)))
            
    
        if fit_function == 'multilinear_supportgrid_cpu':         
                shells_radii = shells_radii_px_for_supportgrid.reshape((n_shellsegments_for_supportgrid, n_shells_for_supportgrid))
                shells_angles = np.linspace(0., 2*np.pi, n_shellsegments_for_supportgrid, endpoint = False)
                shells_radii_2d, shells_angles_2d  = np.meshgrid( np.nanmean(shells_radii, axis = 0), shells_angles)
                shells_radii_2d, shells_angles_2d  =  shells_radii_2d.flatten(), shells_angles_2d.flatten()
    
                p_loc_rad = np.nanmean(shells_radii, axis = 0)
                p_loc_rad = p_loc_rad[np.isfinite(p_loc_rad)]
                p_loc_tan = shells_angles
                p_loc_rad_arr = list([p_loc_rad])
                p_loc_tan_arr = list([p_loc_tan])
                while len(p_loc_rad_arr[-1]) > 20: p_loc_rad_arr.append(p_loc_rad_arr[-1][::2])
                while len(p_loc_tan_arr[-1]) > 1: p_loc_tan_arr.append(p_loc_tan_arr[-1][::2])
                for k in range(len(p_loc_rad_arr) - len(p_loc_tan_arr)): p_loc_tan_arr.append(p_loc_tan_arr[-1])
                for k in range(len(p_loc_tan_arr) - len(p_loc_rad_arr)): p_loc_rad_arr.append(p_loc_rad_arr[-1])
            
                shells_npix[np.isfinite(shells_npix) == 0] = 0
                for i in range(iterations):
                    for k in reversed(range(len(p_loc_rad_arr))):
                        if k == (len(p_loc_rad_arr) -1):  #this is the first iteration                    
                            p0_i = np.ones(len(p_loc_tan_arr[-1])* len(p_loc_rad_arr[-1]) + n_manual_shells)     
                        else:   # for the other iterations, we have to gradually increase the resolution from k+1 (the previous iteration) to k
                            if n_manual_shells:    
                                p0_i = np.concatenate([ interpolate_to_resolution(coeff_fit[0 : len(p_loc_tan_arr[k+1])* len(p_loc_rad_arr[k+1])], (p_loc_tan_arr[k+1], p_loc_rad_arr[k+1]), (p_loc_tan_arr[k], p_loc_rad_arr[k])),   coeff_fit[-n_manual_shells:] ])
                            else:
                                p0_i = interpolate_to_resolution(coeff_fit, (p_loc_tan_arr[k+1], p_loc_rad_arr[k+1]), (p_loc_tan_arr[k], p_loc_rad_arr[k]))
                            
                        multilinear_supportgrid_partial = partial(multilinear_supportgrid, y_diff = Y_diff_training_dataset[i, :], p_loc_tan = p_loc_tan_arr[k], p_loc_rad = p_loc_rad_arr[k], shells_npix = shells_npix, shell_angles = shells_angles, shells_radii =  np.nanmean(shells_radii, axis = 0), n_manual_shells = n_manual_shells)
                        coeff_fit, cov = scp.optimize.curve_fit(multilinear_supportgrid_partial, X_training_dataset[i, :, :], Y_training_dataset[i, :], p0 = p0_i,  bounds =  (np.zeros(len(p0_i)), np.full(len(p0_i), np.inf)),  method = 'trf', max_nfev = max_iter, sigma = 1./weights_training_dataset[i, :], ftol = None, gtol = None, xtol = tolerance, diff_step = 1e-3)
                        coeff_fit_tmp = coeff_fit[ 0 : len(p_loc_tan_arr[k]) * len(p_loc_rad_arr[k]) ].reshape((len(p_loc_tan_arr[k]), len(p_loc_rad_arr[k]))) 
                        coeff_fit_tmp[:, 0] = 0 
                        coeff_fit[ 0 : len(p_loc_tan_arr[k]) * len(p_loc_rad_arr[k])] = coeff_fit_tmp.flatten()

                           
                    if n_manual_shells:
                        coeff_fit = np.concatenate([ interpolate_to_resolution(coeff_fit[ 0: len(p_loc_tan_arr[0])* len(p_loc_rad_arr[0])], (p_loc_tan_arr[0], p_loc_rad_arr[0]), (shells_angles, np.nanmean(shells_radii, axis = 0))),    coeff_fit[-n_manual_shells:] ])
                    else:
                        coeff_fit = interpolate_to_resolution(coeff_fit, (p_loc_tan_arr[0], p_loc_rad_arr[0]), (shells_angles, np.nanmean(shells_radii, axis = 0)))

                    coeff_fit[np.sum(X_training_dataset[i, :, :], axis = 0) == 0] = np.nan
                    Y_pred = multilinear_fit(X_evaluation_dataset[i, :, :], *coeff_fit, y_diff = Y_diff_evaluation_dataset[i, :], shells_npix = shells_npix) #0 in evaluation dataset is correct. At the moment, all iterations use the same evaluation
                    coeff.append(coeff_fit)
                    rmse.append(np.sqrt(np.nanmean((Y_pred - Y_evaluation_dataset[i, :])**2 * weights_evaluation_dataset[i, :]**2)))
                    
 
    
        if fit_function == 'multilinear_gpu':        
              model_id = gf.ModelID.MULTILINEAR_1024PARAMETERS
                        
              estimator_id = gf.EstimatorID.LSE #MLE, LSE
              constraints = np.zeros((iterations, 2*1024))
              constraints[:, 1::2] = 1.
              constraints = constraints.astype(np.float32)
              constraint_types = np.full(1024, gf.ConstraintType.LOWER_UPPER, dtype=np.int32)
              max_number_iterations = max_iter
           
    
              p0 = np.zeros(1024)
              p0 = p0.astype(np.float32)
              p0 = np.tile(p0, (iterations, 1))
              initial_parameters = p0
              
              shells_npix[np.isfinite(shells_npix) == 0] = 0
              nshells_ydiff_x = np.array([])
              nshells_ydiff_x = np.append(nshells_ydiff_x, n_coeffs)
              nshells_ydiff_x = np.append(nshells_ydiff_x, shells_npix)
              nshells_ydiff_x = np.append(nshells_ydiff_x, np.concatenate( (Y_diff_training_dataset, X_training_dataset.reshape((iterations, n_coeffs * len(Y_diff_training_dataset[0, :]))) ), axis= 1 ).flatten())
              nshells_ydiff_x = nshells_ydiff_x.astype(np.float32)
              user_info = nshells_ydiff_x
              
              Y_training_dataset = Y_training_dataset.astype(np.float32)
              data = Y_training_dataset
              
              weights_training = weights_training_dataset.astype(np.float32)
              weights = weights_training
                            
              parameters_to_fit = np.zeros(1024)
              parameters_to_fit[0:n_coeffs] = np.sum(X_training_dataset[:, :, :], axis = (0, 1)) > 0
              parameters_to_fit = parameters_to_fit.astype(np.int32)
              
              #def fit(data,          weights,       model_id:ModelID, initial_parameters, tolerance:float=None, max_number_iterations:int=None, parameters_to_fit=None, estimator_id:EstimatorID=None, user_info=None):
              parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data, weights, model_id, initial_parameters=initial_parameters, tolerance=tolerance, max_number_iterations= max_number_iterations, parameters_to_fit= parameters_to_fit, constraints=constraints, constraint_types=constraint_types, estimator_id = estimator_id, user_info = user_info) #, 
              parameters = parameters[:, 0:n_coeffs]
              
              converged = np.where(states == 0)[0]
              for i in converged: #multilinear_fit is here correct. For deriving the RMSE, the regularization is not needed.
                  parameters[i, np.sum(X_training_dataset[i, :, :], axis = 0) == 0]  = np.nan
                  Y_pred = multilinear_fit(X_evaluation_dataset[i, :, :], *parameters[i , :], y_diff = Y_diff_evaluation_dataset[i, :], shells_npix = shells_npix) #0 in evaluation dataset is correct. At the moment, all iterations use the same evaluation
                  coeff.append(parameters[i , :])
                  rmse.append(np.sqrt(np.nanmean((Y_pred - Y_evaluation_dataset[i, :])**2 * weights_evaluation_dataset[i, :]**2)))
                      
        
        
        if fit_function == 'multilinear_supportgrid_gpu':
              #for the gpu, there was no interpolation function ready at the time I have written this code. Thus, we programed the interpolation manually. For each index to fit, we identify the four nearest neighbors within our support grid, and interpolate the index by a bilinear interpolation. Thus, we get the indices of the four neighbours and the weights of the four neighbours for the interpolation. We can this pass on to our gpu fitting algorithm which then derives the interpolated values on the fly.
              shells_radii = shells_radii_px_for_supportgrid.reshape((n_shellsegments_for_supportgrid, n_shells_for_supportgrid))
              shells_angles = np.linspace(0., 2*np.pi, n_shellsegments_for_supportgrid, endpoint = False)
              shells_radii_2d, shells_angles_2d  = np.meshgrid( np.nanmean(shells_radii, axis = 0), shells_angles)
              shells_radii_2d, shells_angles_2d  =  shells_radii_2d.flatten(), shells_angles_2d.flatten()
    
              p_loc_rad = np.nanmean(shells_radii, axis = 0)
              p_loc_rad = p_loc_rad[np.isfinite(p_loc_rad)]
              p_loc_tan = shells_angles
              p_loc_rad_arr = list([p_loc_rad])
              p_loc_tan_arr = list([p_loc_tan])
              while len(p_loc_rad_arr[-1]) > 20: p_loc_rad_arr.append(p_loc_rad_arr[-1][::2])
              while len(p_loc_tan_arr[-1]) > 1: p_loc_tan_arr.append(p_loc_tan_arr[-1][::2])
              for k in range(len(p_loc_rad_arr) - len(p_loc_tan_arr)): p_loc_tan_arr.append(p_loc_tan_arr[-1])
              for k in range(len(p_loc_tan_arr) - len(p_loc_rad_arr)): p_loc_rad_arr.append(p_loc_rad_arr[-1])
              
       
              for k in reversed(range(len(p_loc_rad_arr))):    
                    n_segments = len(p_loc_rad_arr[k]) * len(p_loc_tan_arr[k])
                    model_id = gf.ModelID.MULTILINEAR_SUPPORTGRID_1024PARAMETERS
                    
                    estimator_id = gf.EstimatorID.LSE #MLE, LSE
                    constraints = np.zeros((iterations, 2*1024))
                    constraints[:, 1::2] = 1.
                    constraints = constraints.astype(np.float32)
                    constraint_types = np.full(1024, gf.ConstraintType.LOWER_UPPER, dtype=np.int32)
                    max_number_iterations = max_iter 
                    
                    #create the p0. In the first iteration, it has to be set up. In the following iterations, it has to be interpolated from the results fitted from the previous lower resolution.
                    if k == (len(p_loc_rad_arr) -1):  #this is the first iteration
                        p0 = np.full(1024, 1.e-10 )
                        p0 = np.tile(p0, (iterations, 1))
                    else:   # for the other iterations, we have to gradually increase the resolution from k+1 (the previous iteration) to k
                            p0 = np.zeros((iterations, 1024))
                            for i_p0 in range(iterations): #we have to interpolate each set of fitted parameters from the low resolution to the next higher resolution. #iteration gives the number of sets.
                                p0[i_p0, 0:len(p_loc_tan_arr[k]) * len(p_loc_rad_arr[k])] = interpolate_to_resolution(coeff_fit[i_p0, :], (p_loc_tan_arr[k+1], p_loc_rad_arr[k+1]), (p_loc_tan_arr[k], p_loc_rad_arr[k]))
                                if n_manual_shells:
                                    p0[len(p_loc_tan_arr[k]) * len(p_loc_rad_arr[k]) : len(p_loc_tan_arr[k]) * len(p_loc_rad_arr[k]) + n_manual_shells] = coeff_fit[ -n_manual_shells : ]
                    initial_parameters = p0.astype(np.float32)
        
        
                    #first, we extend p_loc_tan by one element in front and one in the back -> this corresponds to wrapping of the angles at 2pi
                    p_loc_tan_tmp = np.array([ (-1)*(2*np.pi - p_loc_tan_arr[k][-1]) ,*p_loc_tan_arr[k], 2*np.pi + p_loc_tan_arr[k][0] ]) #add the first element to the back (+ 2 pi to account for the wrapping)            
                    ind_angles_upper = np.searchsorted(p_loc_tan_tmp, shells_angles_2d) #this one now refers to p_loc_tan_tmp, not p_loc_tan
                    ind_angles_lower = ind_angles_upper -1                              #this one now refers to p_loc_tan_tmp, not p_loc_tan
                    ind_rad_right = np.searchsorted(p_loc_rad_arr[k], shells_radii_2d)      
                    ind_rad_left = ind_rad_right -1
                    
                    #In the radial direction, there is no wrapping. Thus, set the indices at the radial boundary so that they both ind_rad_right and ind_rad_left are within our domain.
                    ind_rad_left[np.isfinite(shells_radii_2d) == 0] = -1 #When shells_radii_2d is nan, set it to -1. The following to lines set the boundaries then to the start of our domain.
                    ind_rad_right[np.isfinite(shells_radii_2d) == 0] = -1 #When shells_radii_2d is nan, set it to -1. The following to lines set the boundaries then to the start of our domain.
                    ind_rad_right[ind_rad_left < 0] = 1
                    ind_rad_left[ind_rad_left < 0] = 0
                    ind_rad_left[ind_rad_right >= len(p_loc_rad_arr[k])] = len(p_loc_rad_arr[k]) -2
                    ind_rad_right[ind_rad_right >= len(p_loc_rad_arr[k])] = len(p_loc_rad_arr[k]) -1
     
                    x2x = p_loc_rad_arr[k][ind_rad_right] - shells_radii_2d
                    xx1=  shells_radii_2d - p_loc_rad_arr[k][ind_rad_left]
                    y2y = p_loc_tan_tmp[ind_angles_upper] - shells_angles_2d
                    yy1 = shells_angles_2d - p_loc_tan_tmp[ind_angles_lower]         
                    x2x1 = p_loc_rad_arr[k][ind_rad_right] - p_loc_rad_arr[k][ind_rad_left]
                    y2y1 = p_loc_tan_tmp[ind_angles_upper] - p_loc_tan_tmp[ind_angles_lower]
                    
                    x2x[np.isfinite(x2x) == 0] = 0
                    xx1[np.isfinite(xx1) == 0] = 0
                    y2y[np.isfinite(y2y) == 0] = 0
                    yy1[np.isfinite(yy1) == 0] = 0
                    x2x1[np.isfinite(x2x1) == 0] = 0
                    y2y1[np.isfinite(y2y1) == 0] = 0
    
                    w_upperleft = xx1 * y2y / (x2x1 * y2y1)
                    w_lowerleft = x2x * y2y / (x2x1 * y2y1)
                    w_upperright = xx1 * yy1 / (x2x1 * y2y1)
                    w_lowerright = x2x * yy1 / (x2x1 * y2y1) 
                    
                    #wrap the indices at the angles, and if radius = 0 is present, refer all indices with radius = 0 to one index
                    ind_angles_upper -= 1 #this transforms the indices from p_loc_tan_tmp to p_loc_tan
                    ind_angles_lower -= 1
                    ind_angles_upper[ind_angles_upper > len(p_loc_tan_arr[k]) -1] = 0 
                    ind_angles_lower[ind_angles_lower > len(p_loc_tan_arr[k]) -1] = 0 
                    ind_angles_upper[ind_angles_upper < 0] = len(p_loc_tan_arr[k]) -1
                    ind_angles_lower[ind_angles_lower < 0] = len(p_loc_tan_arr[k]) -1
    
                    
                    indices = np.arange(n_segments).reshape((len(p_loc_tan_arr[k]), len(p_loc_rad_arr[k])))
                    indices[:, p_loc_rad_arr[k] == 0] = indices[0, p_loc_rad_arr[k] == 0] #At zero, polar coordinates are not well defined. Use only one coefficient, i.e., at p_lod_tan = 0, for the pole.
                    ind_upperleft  = indices[ind_angles_upper, ind_rad_left]
                    ind_lowerleft  = indices[ind_angles_lower, ind_rad_left]
                    ind_upperright = indices[ind_angles_upper, ind_rad_right]
                    ind_lowerright = indices[ind_angles_lower, ind_rad_right]
                    
                    if n_manual_shells:
                        ind_upperleft = np.concatenate([ind_upperleft, np.arange(n_segments, n_segments + n_manual_shells)])
                        ind_lowerleft = np.concatenate([ind_lowerleft, np.arange(n_segments, n_segments + n_manual_shells)])
                        ind_upperright = np.concatenate([ind_upperright, np.arange(n_segments, n_segments + n_manual_shells)])
                        ind_lowerright = np.concatenate([ind_lowerright, np.arange(n_segments, n_segments + n_manual_shells)])
                        w_upperleft = np.concatenate([w_upperleft, np.repeat(1.0, n_manual_shells)])
                        w_lowerleft = np.concatenate([w_lowerleft, np.repeat(0.0, n_manual_shells)])
                        w_upperright = np.concatenate([w_upperright, np.repeat(0.0, n_manual_shells)])
                        w_lowerright = np.concatenate([w_lowerright, np.repeat(0.0, n_manual_shells)])
                  
                    interp_coefficients = np.array([ind_upperleft, ind_lowerleft, ind_upperright, ind_lowerright, w_upperleft, w_lowerleft, w_upperright, w_lowerright]).transpose().flatten()  
    
                    shells_npix[np.isfinite(shells_npix) == 0] = 0
                    nshells_ydiff_x = np.array([])
                    nshells_ydiff_x = np.append(nshells_ydiff_x, n_coeffs)
                    nshells_ydiff_x = np.append(nshells_ydiff_x, shells_npix)
                    nshells_ydiff_x = np.append(nshells_ydiff_x, interp_coefficients.flatten())
                    nshells_ydiff_x = np.append(nshells_ydiff_x, np.concatenate( (Y_diff_training_dataset, X_training_dataset.reshape((iterations, n_coeffs * len(Y_diff_training_dataset[0, :]))) ), axis= 1 ).flatten())
                    nshells_ydiff_x = nshells_ydiff_x.astype(np.float32)
                    user_info = nshells_ydiff_x
    
    
                    parameters_to_fit = np.zeros(1024)
                    parameters_to_fit[np.unique(np.array( [*np.delete(ind_upperleft, (w_upperleft == 0) | (np.sum(X_training_dataset, axis = (0, 1)) == 0) ), 
                                                            *np.delete(ind_lowerleft, (w_lowerleft == 0) | (np.sum(X_training_dataset, axis = (0, 1)) == 0) ),
                                                            *np.delete(ind_upperright, (w_upperright == 0) | (np.sum(X_training_dataset, axis = (0, 1)) == 0) ),
                                                            *np.delete(ind_lowerright, (w_lowerright == 0) | (np.sum(X_training_dataset, axis = (0, 1)) == 0) ) ] ))] = 1
                    parameters_to_fit = parameters_to_fit.astype(np.int32)                
                                   
                    
                    Y_training_dataset = Y_training_dataset.astype(np.float32)
                    data = Y_training_dataset
                  
                    weights_training = weights_training_dataset.astype(np.float32)
                    weights = weights_training
     
                    #def fit(data,          weights,       model_id:ModelID, initial_parameters, tolerance:float=None, max_number_iterations:int=None, parameters_to_fit=None, estimator_id:EstimatorID=None, user_info=None):
                    parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data, weights, model_id, initial_parameters=initial_parameters, tolerance=tolerance, max_number_iterations= max_number_iterations, parameters_to_fit= parameters_to_fit, constraints=constraints, constraint_types=constraint_types, estimator_id = estimator_id, user_info = user_info) #, 
                    converged = states == 0
                    converged_parameters = parameters[converged, 0:n_segments + n_manual_shells]
                    for i_p in range(np.sum(converged)):
                        coeff_tmp = converged_parameters[i, : n_segments]
                        coeff_tmp = coeff_tmp.reshape(len(p_loc_tan_arr[k]), len(p_loc_rad_arr[k]))
                        coeff_tmp[:, p_loc_rad_arr[k] == 0] = coeff_tmp[0, p_loc_rad_arr[k] == 0]
                        converged_parameters[i, : n_segments] = coeff_tmp.flatten()
                    coeff_fit = converged_parameters
                    iterations = np.sum(converged)
                    X_training_dataset, Y_training_dataset, Y_diff_training_dataset, weights_training_dataset = X_training_dataset[converged, :, :], Y_training_dataset[converged, :], Y_diff_training_dataset[converged, :], weights_training_dataset[converged, :]

    
              coeff_final = np.zeros((iterations, n_shellsegments_for_supportgrid * n_shells_for_supportgrid + n_manual_shells))       
              for i_p0 in range(iterations): #we have to interpolate each set of fitted parameters from the low resolution to the next higher resolution. #iteration gives the number of sets.
                  if n_manual_shells:
                      coeff_final[i_p0, :] = np.concatenate( [interpolate_to_resolution(coeff_fit[i_p0, 0 : p_loc_tan_arr[0]*p_loc_rad_arr[0]], (p_loc_tan_arr[0], p_loc_rad_arr[0]), (shells_angles, np.nanmean(shells_radii, axis = 0))),      coeff_fit[-n_manual_shells:] ])
                  else:
                      coeff_final[i_p0, :] = interpolate_to_resolution(coeff_fit[i_p0, 0:n_coeffs], (p_loc_tan_arr[0], p_loc_rad_arr[0]), (shells_angles, np.nanmean(shells_radii, axis = 0)))
    
              for i in range(iterations): #multilinear_fit is here correct. For deriving the RMSE, the regularization is not needed.
                  coeff_final[i, np.sum(X_training_dataset[i, :, :], axis = 0) == 0]  = np.nan
                  Y_pred = multilinear_fit(X_evaluation_dataset[i, :, :], *coeff_final[i , :], y_diff = Y_diff_evaluation_dataset[i, :], shells_npix = shells_npix) #0 in evaluation dataset is correct. At the moment, all iterations use the same evaluation
                  coeff.append(coeff_final[i , :])
                  rmse.append(np.sqrt(np.nanmean((Y_pred - Y_evaluation_dataset[i, :])**2 * weights_evaluation_dataset[i, :]**2)))
    
        
        if fit_function == 'multilinear_regularization_gpu':    
              model_id = gf.ModelID.MULTILINEAR_REG_1024PARAMETERS
    
              estimator_id = gf.EstimatorID.LSE #MLE, LSE
              constraints = np.zeros((iterations, 2*1024))
              constraints[:, 1::2] = 1.
              constraints = constraints.astype(np.float32)
              constraint_types = np.full(1024, gf.ConstraintType.LOWER_UPPER, dtype=np.int32)
              max_number_iterations = max_iter
           
    
              p0 =np.zeros(1024)
              p0 = p0.astype(np.float32)
              p0 = np.tile(p0, (iterations, 1))
              initial_parameters = p0
              
              shells_npix[np.isfinite(shells_npix) == 0] = 0
              alpha_reg = alpha * len(Y_training_dataset[0, :])
              alpha_nshells_ydiff_x = np.array([])
              alpha_nshells_ydiff_x = np.append(alpha_nshells_ydiff_x, n_coeffs)
              alpha_nshells_ydiff_x = np.append(alpha_nshells_ydiff_x, alpha_reg)
              alpha_nshells_ydiff_x = np.append(alpha_nshells_ydiff_x, shells_npix)
              alpha_nshells_ydiff_x = np.append(alpha_nshells_ydiff_x, np.concatenate( (Y_diff_training_dataset, X_training_dataset.reshape((iterations, n_coeffs * len(Y_diff_training_dataset[0, :]))) ), axis= 1 ).flatten())
              alpha_nshells_ydiff_x = alpha_nshells_ydiff_x.astype(np.float32)
              user_info = alpha_nshells_ydiff_x
              
              Y_training_dataset = np.concatenate((np.zeros((Y_training_dataset.shape[0], 1)), Y_training_dataset), axis = 1).astype(np.float32)
              data = Y_training_dataset
              
              weights_training = np.concatenate((np.ones((weights_training_dataset.shape[0], 1)), weights_training_dataset), axis = 1).astype(np.float32)
              weights = weights_training
              
              parameters_to_fit = np.zeros(1024)
              parameters_to_fit[0:n_coeffs] = np.sum(X_training_dataset, axis = (0, 1)) > 0
              parameters_to_fit = parameters_to_fit.astype(np.int32)
                                                                      #def fit(data,          weights,       model_id:ModelID, initial_parameters, tolerance:float=None, max_number_iterations:int=None, parameters_to_fit=None, estimator_id:EstimatorID=None, user_info=None):
              parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data, weights, model_id, initial_parameters=initial_parameters, tolerance=tolerance, max_number_iterations= max_number_iterations, parameters_to_fit= parameters_to_fit, constraints=constraints, constraint_types=constraint_types, estimator_id = estimator_id, user_info = user_info) #, 
              parameters = parameters[:, 0:n_coeffs]
              
              converged = np.where(states == 0)[0]
              for i in converged: #multilinear_fit is here correct. For deriving the RMSE, the regularization is not needed.
                  parameters[i, np.sum(X_training_dataset[i, :, :], axis = 0) == 0]  = np.nan
                  Y_pred = multilinear_fit(X_evaluation_dataset[i, :, :], *parameters[i , :], y_diff = Y_diff_evaluation_dataset[i, :], shells_npix = shells_npix) #0 in evaluation dataset is correct. At the moment, all iterations use the same evaluation
                  coeff.append(parameters[i , :])
                  rmse.append(np.sqrt(np.nanmean((Y_pred - Y_evaluation_dataset[i, :])**2 * weights_evaluation_dataset[i, :]**2)))
      
    #undo the normalization
    if 'cpu' in fit_function:
        coeff = np.array(coeff) / xscale
    
    #Having done the fitting, we create an array of structures with all the fitted cofficients and associated RMSEs.
    coeff_rmse_for_each_iteration =  [ dict({'coeff': a, 'rmse': b, 'percentage_scattered': c}) for a, b, c in zip(coeff, rmse, np.nansum(coeff * shells_npix_orig, axis = 1)) ]       
 
    #We derive the mean coefficients from all the fits as final fit result. If requested, we use only the fits associated with the best fraction of the RMSEs to calculate the mean coefficients.
    if len(coeff) > 1:
        coeff = np.array(coeff)
        rmse = np.array(rmse)
        rmse_ind_sorted = rmse.argsort()
        rmse_ind_sorted = rmse_ind_sorted[0: int(np.ceil(len(rmse_ind_sorted) * best_fraction ))]
        coeff = coeff[rmse_ind_sorted, :]
        rmse = rmse[rmse_ind_sorted]
    coeff = np.nanmean(coeff, axis = 0) 
    
    np.savez_compressed(folder_run + '/fitted_psf_coefficients/fitted_coefficients.npz', coeff = coeff, coeff_rmse_for_each_iteration = coeff_rmse_for_each_iteration )
    

def split_array_by_r_alpha(dist_ext):
    n_elements, n_columns = dist_ext.shape
    out = list()
    last_element = dist_ext[0, 1:]
    subset = list()
    for i in range(n_elements):
        dist_index = dist_ext[i, 0]
        dist_r = dist_ext[i, 1]
        dist_alpha = dist_ext[i, 2]
        if (dist_r != last_element[0]) or (dist_alpha != last_element[1]): 
            out.append(subset) #if the actual element differs from the last element in r or alpha, start a new list  
            subset = list()
            last_element = [dist_r, dist_alpha]
        subset.append(dist_index) #add the current element to the current list
    if len(subset) > 0: out.append(subset)
    return out
      

#derive the distance of an occulted pixel to the illuminated pixels by looking in which first shell around the occulted pixel we have recoreded intensity
@jit
def derive_dist(X):
    #X is of size (n_occulted_pixlels, n_shellsegments, n_shells)
    n_data, n_shellsegments, n_shells = X.shape
    
    #prepare the dist_r and dist_alpha array for output
    dist = np.zeros((n_data, 2), dtype = np.int32)
    
    #for each occulted pixel
    for i in range(n_data):
        #first look at which minimum distance, i.e., shell, we have intensity
        for k in range(n_shells):
            intensity_shell = 0  #if a shell was split up into multiple shell segments, add the intensities of the shell segments into a total shell intensity
            for l in range(n_shellsegments):
                intensity_shell += X[i, l, k]
            if intensity_shell != 0: 
                dist_r = k
                break
            
        #having the shell distance, look in which shell segments within the shell we have intensity. To do so:
        valid_shellsegments = np.arange(n_shellsegments) #create an index array for all shellsegments within the minimum distance shell
        for l in range(n_shellsegments):   #for the shell segments which do not have intensity, set the index array to -1
            if X[i, l, dist_r] == 0: valid_shellsegments[l] = -1
        valid_shellsegments = valid_shellsegments[valid_shellsegments != -1]  #and remove these indices. What is left are the indices where we actually have intensity
        dist_alpha = np.random.choice(valid_shellsegments)#of these remaining indices for the valid shell segments, select one randomly. 
        dist[i, 0] = dist_r
        dist[i, 1] = dist_alpha

    return dist
                    


#multilinear fit, where all coefficients have to be positive
def multilinear_fit(x, *p, y_diff, shells_npix):
# def multilinear_fit(x, *p, y_diff, dist_shell_px,  dist_arr_interp, hist_dist_arr_forinterp):
     p = np.array(p) 
     p[np.isnan(p)] = 0.
     y = np.matmul(x, p)
     totalweight_longscatter = np.sum(p*shells_npix)
     y += (1 - totalweight_longscatter)* y_diff
     return y
 
    
#multilinear fit, where some parameters given by y_constant and totalweight_longscatter_constant have already been fitted and thus could pre-calculated. 
def multilinear_piecewise_fit(x, *p, y_constant, totalweight_longscatter_constant, y_diff, shells_npix):
     p = np.array(p)  
     y = np.matmul(x, p) + y_constant
     totalweight_longscatter = np.sum(p*shells_npix) + totalweight_longscatter_constant
     y += (1 - totalweight_longscatter)* y_diff
     return y
 
#multilinear fit, where all coefficients have to be positive. + regularization
def multilinear_fit_reg(x, *p, y_diff, shells_npix, alpha):
     p = np.array(p)  
     y = np.matmul(x, p)

     totalweight_longscatter = np.sum(p*shells_npix)
     y += (1 - totalweight_longscatter)* y_diff
     y = np.concatenate(([alpha*totalweight_longscatter], y))
     return y
    
def powerlaw_gaussian_cpu(x, *p, y_diff, shells_npix, r):
    notvalid = np.isfinite(r) == 0
    r[notvalid] = 0.
    weights = p[2] *1./ (p[3]+ ((r +1.e-5 ))**p[4])  + p[0]*np.exp(-1*( (r+1.e-5)**2 / 2 / (p[1])**2 ))
    weights[0] = 0.
    weights[notvalid] = 0.
    y = np.matmul(x, weights)
    totalweight_longscatter = np.sum(weights*shells_npix)
    y += (1 - totalweight_longscatter)* y_diff   
    return y

def multilinear_supportgrid(x, *p, y_diff, p_loc_tan, p_loc_rad, shells_npix, shell_angles, shells_radii, n_manual_shells):
     coeff_fit = np.array(p)  
     if n_manual_shells:
         coeff_fit = np.concatenate([ interpolate_to_resolution(coeff_fit[0 : len(p_loc_tan) * len(p_loc_rad)], (p_loc_tan, p_loc_rad), (shell_angles, shells_radii)),    coeff_fit[-n_manual_shells:] ]) 
     else:
         coeff_fit = interpolate_to_resolution(coeff_fit, (p_loc_tan, p_loc_rad), (shell_angles, shells_radii))

     y = np.matmul(x, coeff_fit)
     totalweight_longscatter = np.sum(coeff_fit*shells_npix)
     y += (1 - totalweight_longscatter)* y_diff
     return y   



def polar_bilinear(theta, rad, x, smooth=0.):
    #this is a very rough estimate. Basically, one would have to do this by transforming the axes to a polar coordinate system, and do all the coordinate calculations in polar coordinates.
    #extand the axes for periodic boundary conditions for theta
     if len(theta) >= 2:
               x =  np.concatenate((np.expand_dims(x[-2, :], axis = 0), np.expand_dims(x[-1, :], axis = 0), x, np.expand_dims(x[0, :], axis = 0), np.expand_dims(x[1, :], axis = 0)), axis = 0)
               theta = np.append(np.insert(theta, 0, theta[-2:] - 2*np.pi), theta[0:2] + 2*np.pi)
     else:
               x = np.repeat(x, 5, axis = 0)
               theta = theta[0] #as the array has only one element, transform the array into the number for convenience
               theta = np.array([theta - 2*np.pi, theta - np.pi, theta, theta + np.pi, theta + 2*np.pi])
     # Note that for all r = 0, theta -> p[:, 0] = const
     ind_r0 = np.where(rad == 0.)
     if len(ind_r0[0]) > 0: x[:, ind_r0] = np.nanmax(x[:, ind_r0])
     
     spline = RectBivariateSpline(theta, rad, x, kx=1, ky=1, s=smooth)
     return spline    

def interpolate_to_resolution(data, xy_old, xy_new):
        data = np.copy(data)
        x_old, y_old = xy_old
        x_new, y_new = xy_new
        x_new, y_new = interpolate_nans(x_new), interpolate_nans(y_new)
        data_2d = np.reshape(data, (len(x_old), len(y_old)))
        data_2d = np.log10(data_2d)
        data_2d = interpolate_nans(data_2d)
        spline = polar_bilinear(x_old, y_old, data_2d)
        data_2d = spline(x_new, y_new) 
        data_2d = 10**np.array(data_2d)
        data_2d[np.isfinite(data_2d) ==0] = 0.e-10
        return data_2d.flatten() 
        
    
def interpolate_nans(ar):
    ar = np.copy(ar)
    ok = np.isfinite(ar)
    xok = ok.ravel().nonzero()[0]
    fok = ar[ok]
    
    notok = np.isfinite(ar) == 0
    x_notok  = notok.ravel().nonzero()[0]
    if (np.sum(notok) > 0):
        interp = interp1d(xok, fok, fill_value = 'extrapolate')
        ar[notok] = interp(x_notok)
    return ar

def add_weights(equation):
        #derive the weights for the fitting: 1/ intensities. As many intensities will be close to zero, derive the weights by using the average intensities of similar occulted pixels, i.e., which have the same importance mask index.
        importance_indices_occulted = equation['index_importancemask'][equation['region'] == 'occulted']
        intensities_occulted        = equation['intensities'][equation['region'] == 'occulted']
        weights_occulted            = np.zeros(len(importance_indices_occulted))

        #find similar pixels for the occulted pixels and derive their average weight
        for index in np.unique(importance_indices_occulted):
            mask = (importance_indices_occulted == index)
            if np.sum(mask) > 0: weights_occulted[mask] = 1./np.nanmean(np.abs(intensities_occulted[mask].astype(np.float32)))

        #derive the weights for illuminated_edge and illuminated pixels
        weights_illuminated_edge = 1./ equation['intensities'][equation['region'] == 'illuminated_edge']
        weights_illuminated = 1./ equation['intensities'][equation['region'] == 'illuminated']
        
        
        #for pixels which have not been assigned a weight and the illuminated_edge and illuminated pixels, set weights into reasonable bounds based on the occulted pixels
        min_weight, max_weight = np.nanmin(weights_occulted[weights_occulted > 0]), np.nanmax(weights_occulted[weights_occulted > 0])
        weights_occulted[weights_occulted == 0] = min_weight
        weights_illuminated_edge[weights_illuminated_edge > max_weight] = max_weight
        weights_illuminated[weights_illuminated > max_weight] = max_weight

        #add the weights to the equations
        equation['weights'] = np.zeros(len(equation['region']), dtype = np.float32)
        equation['weights'][equation['region'] == 'occulted'] = weights_occulted
        equation['weights'][equation['region'] == 'illuminated_edge'] = weights_illuminated_edge
        equation['weights'][equation['region'] == 'illuminated'] = weights_illuminated
        # equation['weights'] = np.ones(len(equation['region']), dtype = np.float32)

        return equation
    
      
def draw_samples(datacube, file_indices, n_occulted, n_illuminatededge, n_illuminated):
    #define masks for the files to include, pixels in the occulted region, illuminated region, and illuminted_edge region
    mask_files = np.isin(datacube['file_index'], file_indices)
    mask_illuminated_edge = (datacube['region'] == 'illuminated_edge')
    mask_illuminated = (datacube['region'] == 'illuminated')
    mask_occulted = (datacube['region'] == 'occulted')
    
    #get the pixels which are in the requested files and in one of the (occulted, illuminated, illuminated_edgge) region
    mask_illuminatededge = np.where(mask_files & mask_illuminated_edge)[0] 
    mask_illuminated = np.where(mask_files & mask_illuminated)[0]
    mask_occulted = np.where(mask_files & mask_occulted)[0]
    
    #let's first draw pixels from the occulted region. These shall be distributed according to the importance mask index. So, let's first have a few definitions
    indices_importancemask = datacube['index_importancemask'][mask_occulted]
    indices_in_importancemask = np.unique(indices_importancemask)
    indices_in_importancemask = indices_in_importancemask[indices_in_importancemask != -1]
    n_importancemask_indices = len(indices_in_importancemask)
    n_importancemask_px_per_index = np.array([ np.sum(indices_importancemask == index) for index in indices_in_importancemask ])
    n_small_importancemask_segments, n_px_in_small_importancemask_segments, px_per_bin_last = 0, 0, 0
    #derive the number of pixels to draw for each index in the importance mask: px_per_bin. We use an iterative approach, considering that some importancemask segments might have less pixels than px_per_bin. This increases the number of pixels to draw from the other bins.
    while True:
        px_per_bin = (n_occulted - n_px_in_small_importancemask_segments) // (n_importancemask_indices - n_small_importancemask_segments)
        small_segments = np.where(n_importancemask_px_per_index < px_per_bin)[0]
        n_small_importancemask_segments = len(small_segments)
        n_px_in_small_importancemask_segments = np.sum(n_importancemask_px_per_index[small_segments])
        if px_per_bin == px_per_bin_last: break
        else: px_per_bin_last = px_per_bin
        
    #we draw the pixels. For the small segments, we just use all the pixels. For large semgents, we draw px_per_bin pixels. The last bin (in the else statement) is adjusted so that the total number of pixels drawn is equal to the lines of system of equations requested.
    random = list()
    for i in range(len(indices_in_importancemask)):
              px_in_index = np.where(indices_importancemask == indices_in_importancemask[i])[0] 
              if i != len(indices_in_importancemask) -1:
                  n_random = min(px_per_bin, n_importancemask_px_per_index[i])
              random.extend(np.random.choice(mask_occulted[px_in_index], int(n_random)))
            
    #we add random pixels from the illuminated edge and illumianted region
    random.extend(np.random.choice(mask_illuminatededge, int(n_illuminatededge))) 
    random.extend(np.random.choice(mask_illuminated, int(n_illuminated)))
    
    #finally, we create a subdatacube and return it
    subdatacube = {}
    for key in datacube.keys():
        subdatacube[key] = datacube[key][random]
    return subdatacube
