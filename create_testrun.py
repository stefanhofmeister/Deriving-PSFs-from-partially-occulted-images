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
from scipy.ndimage import gaussian_filter
from inout import save_image, read_image, convert_toimage
from deconvolve_image import convolve_image

def create_psf(config):
    folder_run = config['general']['folder_run']
    resolution = config['general']['resolution']
    large_psf = config['general']['large_psf']
    
    if 'file_psf' in config['testrun']:
        if config['testrun']['file_psf'] != '':
            file_psf = config['testrun']['file_psf']
            psf = convert_toimage(file_psf, dtype = np.float32, save = False) 
    
    if 'PSF_shape' in config['testrun']:
        shape = config['testrun']['PSF_shape']
        
        #PSF shape should contain a gaussian core, and either circle or ellipse. Adding cross on top is optional. The parameters for these PSF have been hardcoded, as they result in reasonable PSFs. If you want to change these, keep track that the PSF stays reasonable, particularly that the center coefficient is the largest coefficient
        if large_psf:
            res = resolution *2
        else:
            res = resolution
        if not isinstance(shape, list): shape = list([shape])
        psf = np.zeros((res, res), dtype = np.float32)
        psf_x, psf_y = np.meshgrid(*np.tile(np.arange(res) - res//2, [2, 1]))
        psf_dist = np.sqrt(psf_x**2 + psf_y**2)
            
        if 'circle' in shape:
            psf += 1./(1 + (psf_dist / 256 * 2e3)**2)
    
        if 'ellipse' in shape:
            psf += 1./(1 + (np.sqrt(psf_x**2 + 2*psf_y**2 ) / 256 * 2e3)**2)
                
            
        if 'cross' in shape:
            mask = np.zeros((res, res))
            widths = 4
            angles = np.pi / 8., np.pi / 8. + np.pi/2.
            for angle in angles:
                for x_offset in np.arange(-widths, widths+1, .25):
                    for y_offset in np.arange(-widths, widths+1, .25):
                        dist = np.linspace(-np.sqrt(2)*res//2, np.sqrt(2)*res//2, 41, endpoint=True)
                        x, y = x_offset + (res//2) - np.sqrt(2)*dist*np.cos(angle+np.pi/2.), y_offset + (res//2) - np.sqrt(2)*dist*np.sin(angle+np.pi/2.)
                        x, y = np.round(x).astype('int'), np.round(y).astype('int')
                        valid = (x >=0) & (x < (res-1)) & (y >=0) & (y <(res-1))
                        x, y = x[valid], y[valid]
                        mask[x, y] = 1
            mask[res//2, res//2] = 0
            mask = scp.ndimage.label(mask)[0]
            for step in range(0, int(np.max(psf_dist)) -11, 10):
                indices_tmp = mask[(mask != 0) & (psf_dist >= step) & (psf_dist < step + 10) ]
                indices_tmp = np.unique(indices_tmp)
                for index_tmp in indices_tmp:
                    mask[mask == index_tmp] = indices_tmp[0]
            for index_new, index in enumerate(np.unique(mask)):
                mask[mask == index] = index_new
            indices = np.unique(mask)
            distances = np.zeros(len(indices))
            for ind in indices[1:]:  
                distances[ind] = np.mean(psf_dist[mask == ind])
            for _ in range(2):
                ind_tmp = np.where(distances != np.min(distances))
                distances = distances[ind_tmp]
                indices = indices[ind_tmp]
            for ind in indices:  
                distance = np.mean(psf_dist[mask == ind])
                psf[mask == ind] += (np.sin(np.pi * distance / 10.) / (np.pi * distance/10.) )**2 /100. *100
    
        if 'gaussian_core' in shape:
            psf_tmp = np.zeros((res, res))
            psf_tmp[res//2, res//2] = .3
            if 'circle' in shape: psf_tmp = gaussian_filter(psf_tmp, [10, 10], truncate = 5 )
            if 'ellipse' in shape:  psf_tmp = gaussian_filter(psf_tmp, [10, np.sqrt(2)*10],  truncate = 5 )
            psf += psf_tmp
        else:
            psf[res//2, res//2] += 10*np.nanmax(psf)
        
        psf = psf / np.sum(psf)

    save_image(psf, folder_run + '/true_image_and_psf/true_psf.jpg', dtype = np.float32, keys = 'psf')


def create_artificial_image(config):
    folder_run = config['general']['folder_run']
    large_psf = config['general']['large_psf']
    use_gpu = config['general']['use_gpu']
    signaltonoise = config['testrun']['occmask_signaltonoise'] 
    
    psf = read_image(folder_run + '/true_image_and_psf/true_psf.npz', dtype = np.float32)
    #create the artificial images
    occultation_mask = create_occultation_mask(config)
    image_true = create_true_image(occultation_mask)    
    image_observed = convolve_image(image_true.astype(np.float32), psf.astype(np.float32), use_gpu = use_gpu, pad = True, large_psf = large_psf).astype(np.float32) #need to redefine the datatype, as it gets lost in the convolution   
    image_observed = add_noise(image_observed, occultation_mask, signaltonoise) 
    save_image(image_true, folder_run + '/true_image_and_psf/true_image.jpg', plot_norm = 'lin', dtype = np.float32)
    save_image(image_observed, folder_run + '/original_image/image.jpg', plot_norm = 'log', dtype = np.float32)


def create_occultation_mask(config):
    resolution = config['general']['resolution']
    occmask_shape = config['testrun']['occmask_shape']
    occultation_mask = np.ones((resolution, resolution))
    dist_x, dist_y = np.meshgrid(*np.tile(np.arange(resolution) - resolution//2, [2, 1]))
    dist = np.sqrt(dist_x**2 + dist_y**2)

    if 'pinhole_array' in occmask_shape:
        n_pinholes = config['testrun']['occmask_n_pinholes']
        size_pinholes = config['testrun']['occmask_size_pinholes']    

        #create a list containing the positions of the pinholes
        small_pinholes = list()
        if n_pinholes == 1: 
            small_pinholes = np.array([resolution//2, resolution//2])
        else: #else on a grid
            small_pinholes = np.meshgrid(*np.tile(np.linspace(int(resolution * 0.2), int(resolution * 0.8), int(np.sqrt(n_pinholes))), [2,1]).astype(np.int))
        small_pinholes = [small_pinholes[0].flatten(), small_pinholes[1].flatten()]
        occultation_mask[tuple(small_pinholes)] = 0

        if size_pinholes >1:
            for ph_y, ph_x in zip(small_pinholes[0], small_pinholes[1]):
                ph_mask = (dist_x - dist_x[0, ph_x])**2 + (dist_y - dist_y[ph_y, 0])**2 <= size_pinholes**2
                occultation_mask[ph_mask] = 0
                
    if 'large_hole' in occmask_shape:
        size_largehole = config['testrun']['occmask_size_largehole']
        occultation_mask[dist <= size_largehole] = 0
        
    return occultation_mask
    
    
def create_true_image(occultation_mask):
    #define the true image
    image_true = np.zeros(occultation_mask.shape, dtype = np.float32)
    image_true[occultation_mask == 0] = 1e4
    return image_true

def add_noise(img, occultation_mask, signaltonoise = 0):
    if signaltonoise != 0:
        signal = np.mean(img[occultation_mask == 1])
        noise = signal / signaltonoise 
        img[:, :] =  np.round(np.random.poisson(img[:, :] / noise) * noise)
    return img


