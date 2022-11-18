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
import sys
import cupy
from inout import read_image, save_image


def deconvolve_images(config):
     folder_run = config['general']['folder_run']
     large_psf  = config['general']['large_psf']    
     use_gpu    = config['general']['use_gpu']
     iterations = config['deconvolution']['n_iterations']
     pad        = config['deconvolution']['pad']
     
     #for both the image and the psf, float64 is required to get a good convergence!
     psf = read_image(folder_run + '/fitted_psf/fitted_psf.npz', dtype = np.float64)
     files_original = glob.glob(folder_run + '/original_image/*.npz')
     for file in files_original:
         n_iterations = iterations
         img = read_image(file, dtype = np.float64) 
         if psf[psf.shape[0]//2, psf.shape[1]//2] == 1:  
             #if the PSF center is 1, it follows that all other segments are zero. Then, the deconvolved image is the input image.
             img_deconvolved = img
         else:
             # Deconvolve the image with the PSF, and check if the deconvolution did converge. If the number of iterations is set to a too large value and it does not converge, it can produce nan values or very large and small values. If it did not converge, redo the deconvolution with less iterations.
             while True:
                   img_deconvolved = deconvolve_richardson_lucy(img, psf, iterations = n_iterations, use_gpu = use_gpu, pad = pad, large_psf = large_psf)
                   if np.all(np.isfinite(img_deconvolved)): break
                   else: n_iterations //= 2
                   if n_iterations < 1: sys.exit('PSF deconvolution did not converge.' )

         img_deconvolved = img_deconvolved.astype(np.float16) #by setting the datatype to np.float16, all values > 6.55040e+04 are set to inf. As the intensities in the original images were normalized to 0 - 1e4, this allows for PSFs that scatters maximal about 85% of the photons. But it is also the best location to check if the PSF has converged.
         if np.all(np.isfinite(img_deconvolved)):
             save_image(img_deconvolved, folder_run + '/deconvolved_image/' +  os.path.basename(file), dtype = np.float16, keys = 'img')
         else:
             print('{} created NaNs during PSF deconvolution. Skipping the file.'.format(os.path.basename(file)))

def deconvolve_richardson_lucy(img, psf, iterations=25, use_gpu = True, pad = True, large_psf = False):
    """
    Deconvolve an image with the point spread function

    Perform image deconvolution on an image with the instrument
    point spread function using the Richardson-Lucy deconvolution
    algorithm

    Parameters
    ----------
    img : 'numpy 2d array'
        An image.
    psf : `~numpy.ndarray`, optional
        The point spread function. 
    iterations: `int`
        Number of iterations in the Richardson-Lucy algorithm
    use_gpu: True/False
        If True, the deconvolution will be performed on the GPU.
    pad: True/False
        If true, increase the size of both the psf and the image by a factor of two, and pad the psf and image accordingly with zeros. As this is a fourier-based method, this breaks the symmetric boundary conditions involved in the fourier transform.
    large_psf: True/False
        Usually, the PSF has the same dimension as the image, restricting scattered light to half of the image size. If set to true, the PSF given to the deconvolution has to be double the image size (that allows scattering over the full image range). The image will be padded with zeros to match the size of the full psf, and deconvolution is done over the full psf.

    Returns
    -------
    `~sunpy.map.Map`
        Deconvolved image

    Comments:
        Based on the aiapy.deconvolve method, as described in Cheung, M., 2015, *GPU Technology Conference Silicon Valley*, `GPU-Accelerated Image Processing for NASA's Solar Dynamics Observatory <https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s5209-gpu-accelerated+imaging+processing+for+nasa%27s+solar+dynamics+observatory>`_
    """
    img, psf = np.copy(img), np.copy(psf)
    im_size = img.shape[0]
    psf_size = psf.shape[0]
    padsize_pad, padsize_large_psf = int(0.25*im_size), int(0.5*im_size)
    
    if large_psf:
        img = np.pad(img, padsize_large_psf)
        img[img == 0] = np.finfo(img.dtype).tiny
        im_size = im_size +2*padsize_large_psf
                 
    #padding is only required if the PSF is not large_psf. Else, the padding of the image has already be done above in the large_psf block.
    if pad and not large_psf:  
        psf, img = np.pad(psf, padsize_pad), np.pad(img, padsize_pad)
        im_size = im_size +2*padsize_pad
        psf_size = psf_size +2*padsize_pad

    if use_gpu:
        img = cupy.array(img)
        psf = cupy.array(psf)
        
    # Center PSF at pixel (0,0)
    psf = np.roll(np.roll(psf, psf.shape[0]//2, axis=0),
                  psf.shape[1]//2,
                  axis=1)
    
    # Convolution requires FFT of the PSF
    psf = np.fft.rfft2(psf)
    psf_conj = psf.conj()

    img_decon = np.copy(img)
    for _ in range(iterations):
        ratio = img/np.fft.irfft2(np.fft.rfft2(img_decon)*psf)
        img_decon = img_decon*np.fft.irfft2(np.fft.rfft2(ratio)*psf_conj)


    if use_gpu:
        img_decon = cupy.asnumpy(img_decon)
    
    if large_psf:
        img_decon = img_decon[padsize_large_psf : im_size - padsize_large_psf, padsize_large_psf : im_size - padsize_large_psf]
    
    if pad and not large_psf:
        img_decon = img_decon[padsize_pad : im_size - padsize_pad, padsize_pad : im_size - padsize_pad]
                    
    img_decon = img_decon.astype(img.dtype)
    
    return img_decon


def convolve_image(img, psf, use_gpu = False, pad = True, large_psf = False):
    img = np.copy(img)
    psf = np.copy(psf)
    im_size = img.shape[0]
    psf_size = psf.shape[0]
    padsize_pad, padsize_large_psf = int(0.25*im_size), int(0.5*im_size)
    
    if large_psf:
        img = np.pad(img, padsize_large_psf)
        img[img == 0] = np.finfo(img.dtype).tiny
        im_size = im_size +2*padsize_large_psf
    
    if pad and not large_psf:  
        psf, img = np.pad(psf, padsize_pad), np.pad(img, padsize_pad)
        im_size = im_size +2*padsize_pad
        psf_size = psf_size +2*padsize_pad
    
    if use_gpu:
        img = cupy.array(img)
        psf = cupy.array(psf)

    # Center PSF at pixel (0,0)
    psf = np.roll(np.roll(psf, psf.shape[0]//2, axis=0),
                  psf.shape[1]//2,
                  axis=1)
    # Convolution requires FFT of the PSF
    psf = np.fft.rfft2(psf)
    img_con = np.fft.rfft2(img)
    img_con = img_con * psf
    img_con = np.fft.irfft2(img_con)

    if use_gpu:
        img_con = cupy.asnumpy(img_con)
        
    if large_psf:
        img_con = img_con[padsize_large_psf : im_size - padsize_large_psf, padsize_large_psf : im_size - padsize_large_psf]
        
    if pad and not large_psf:
        img_con = img_con[padsize_pad : im_size - padsize_pad, padsize_pad : im_size - padsize_pad]

    img_con = img_con.astype(img.dtype)
    return img_con

  
           
