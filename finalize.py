"""© 2022 The Trustees of Columbia University in the City of New York. This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only, provided that it cites the original work: 
Stefan J. Hofmeister, Michael Hahn, and Daniel Wolf Savin, "Deriving instrumental point spread functions from partially occulted images," J. Opt. Soc. Am. A 39, 2153-2168 (2022), doi: 10.1364/JOSAA.471477.
To obtain a license to use this work for commercial purposes, please contact Columbia Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm 
from inout import read_image, save_image
from deconvolve_image import deconvolve_richardson_lucy

def finalize(config):
    #create some overview plots regarding the fitted PSF
    folder_run = config['general']['folder_run']
    resolution = config['general']['resolution']
    large_psf  = config['general']['large_psf']
    if large_psf: resolution *= 2
    
    #read the fitted PSF
    psf = read_image(folder_run + '/fitted_psf/fitted_psf.npz', dtype = np.float32)
    save_image(psf, folder_run + '/final_result/fitted_psf.npz', plot_norm = 'log', plot_range = [1e-10, 1], dtype = np.float32)
    
    #First, create an image of the fitted PSF, and a plot showing the weights of the PSF along a horizontal line through the center of the PSF
    fig, ax = plt.subplots(1, 2, figsize = (12, 5)) 
    ax1 = ax[0]
    divider = make_axes_locatable(ax1) 
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    im = ax1.imshow(psf, norm=LogNorm(vmin=np.min(psf[psf>0]), vmax=np.max(psf)), interpolation = 'nearest')
    cbar1 = fig.colorbar(im, cax=cax1, orientation='vertical')
    cbar1.set_label('PSF weight')
    
    ax2 = ax[1]
    ax2.plot(psf[resolution//2, resolution//2:])
    ax2.set_yscale('log')
    ax2.set_xscale('symlog')
    ax2.set_xlim([-0.1, resolution//2])
    ax2.set_xlabel('Distance from PSF center [px]')
    ax2.set_ylabel('PSF weight')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.text(1, .1, 'Scattered photons: {:2.1f}%'.format((1. - psf[resolution//2, resolution//2])*100.))
    
    plt.tight_layout()
    plt.savefig(folder_run + '/final_result/fitted_psf.jpg', dpi = 600)
    plt.close(fig)
    
    #if requested, create PSF deconvolved images from the original images, and show images of them
    if 'create_PSF_deconvolved_images' in config['finalizing']:
        if config['finalizing']['create_PSF_deconvolved_images'] == True:
            large_psf  = config['general']['large_psf']    
            use_gpu    = config['general']['use_gpu']
            n_iterations = config['deconvolution']['n_iterations']
            pad        = config['deconvolution']['pad']
            original_images = glob.glob(folder_run + '/original_image/*.npz')
            for img_file in original_images:
                img = read_image(img_file, dtype = np.float64)
                img_dec = deconvolve_richardson_lucy(img, psf.astype(np.float64),  iterations = n_iterations, use_gpu = use_gpu, pad = pad, large_psf = large_psf)
                save_image(img_dec, folder_run + '/final_result/' + os.path.splitext(os.path.basename(img_file))[0] + '_PSF_deconvolved.jpg', dtype = np.float16)
            
                img_min, img_max = 1., np.max(img_dec) * 2
                fig, ax = plt.subplots(1, 3, figsize = (18, 5)) 
                ax1 = ax[0]
                divider = make_axes_locatable(ax1) 
                cax1 = divider.append_axes('right', size='5%', pad=0.05)
                im1 = ax1.imshow(img, norm=LogNorm(vmin=img_min, vmax=img_max), interpolation = 'nearest') 
                cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical') 
                cbar1.set_label('Intensity')
                
                ax2 = ax[1]
                divider = make_axes_locatable(ax2) 
                cax2 = divider.append_axes('right', size='5%', pad=0.05)
                im2 = ax2.imshow(img_dec, norm=LogNorm(vmin=img_min, vmax=img_max), interpolation = 'nearest') 
                cbar2 = fig.colorbar(im2, cax=cax2, orientation='vertical') 
                cbar2.set_label('Intensity')
                
                ax3 = ax[2]
                ax3.plot(img[resolution//2, :], label = 'Original image')
                ax3.plot(img_dec[resolution//2, :], label = 'PSF deconvolved image')
                ax3.set_yscale('log')
                ax3.set_ylim(img_min, img_max)
                ax3.set_xlabel('X Position [px]')
                ax3.set_ylabel('Intensity')
                ax3.spines['right'].set_visible(False)
                ax3.spines['top'].set_visible(False)
                ax3.legend()
    
                plt.tight_layout()
                plt.savefig(folder_run + '/final_result/' + os.path.splitext(os.path.basename(img_file))[0] + '_PSF_deconvolved.jpg', dpi = 600)
                plt.close(fig)    

    #If this was a testrun, i.e., the true PSF is known, also create a plot showing the percentage error of the fitted to the true PSF coefficients
    if 'testrun' in config:
        if config['testrun']['PSF_testrun'] == True: 
            true_psf = read_image(folder_run + '/true_image_and_psf/true_psf.npz', dtype = np.float32)
            indices_fov =  np.load(folder_run + '/psf_segmentation/psf_segments.npz')['indices_fov']
    
            #derive the fitted and true PSF coefficients from the fitted and true PSF using the PSF segmentation mask: each index in the PSF segmentation mask corresponds to one coefficient 
            n_coeff = np.max(indices_fov)
            coeff_newpsf = np.zeros(n_coeff + 1)
            coeff_oldpsf = np.zeros(n_coeff + 1)
            for index in np.unique(indices_fov):
                if index == -1: continue
                mask = (indices_fov == index)
                coeff_newpsf[index] = np.mean(psf[mask])
                coeff_oldpsf[index] = np.mean(true_psf[mask])
            #derive the percentage error in the PSF coefficients
            correctness_of_coeff = np.divide(coeff_newpsf  - coeff_oldpsf, coeff_oldpsf)
            #plot the errors
            fig, ax = plt.subplots(figsize = (6, 5)) 
            ax.plot(correctness_of_coeff * 100, marker = 'x', linestyle = 'None')
            ax.axhline(y=0, color='grey', linestyle='-')
            ax.set_yscale('symlog', linthresh = 10)
            ax.set_ylim(-100, 100)
            ax.set_yticks([-90, -80, -70, -60, -50, -40, -30, -20, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70, 80, 90], minor = True)
            ax.set_xlabel('Coefficient')
            ax.set_ylabel('Error [%]')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.text(1, 60, 'Mean absolute percentage error: {:2.1f}%'.format(np.nanmean(np.abs(correctness_of_coeff * 100))))
            plt.tight_layout()
            plt.savefig(folder_run + '/final_result/psf_error_analysis.jpg', dpi = 600)
            plt.close(fig)
            
          
