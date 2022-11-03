"""© 2022 The Trustees of Columbia University in the City of New York.
This work may be reproduced, distributed, and otherwise exploited for
academic non-commercial purposes only, provided that it cites the
original work (IN PUBLICATIONP PROCESS).  To obtain a license to
use this work for commercial purposes, please contact Columbia
Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

import numpy as np
import os
from astropy.io import fits
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize


def convert_toimage(file, new_filename = '', dtype = np.float32, normalize = False, plot_norm = 'log', keys = 'img', save = True):
    #converts any fits file, image format, or numpy save file into a numpy format
    if not new_filename: new_filename = file
    file_extension = os.path.splitext(file)[1]
    if (file_extension.lower() == '.fit') or (file_extension.lower() == '.fits') or (file_extension.lower() == '.fts'):
        with fits.open(file) as hdu:
          for i_ext in range(len(hdu)):
              img = hdu[i_ext].data
              if isinstance(img, np.ndarray): break
    elif (file_extension.lower() == '.npz'):
        img = np.load(file)['img']
    elif (file_extension.lower() == '.npy'):
        img = np.load(file) 
    else:
        img = Image.open(file)
    img = np.array(img)
    if normalize: img = img / np.max(img) * 10000.
    if save:
        save_image(img, new_filename, plot_norm =plot_norm, dtype = dtype, keys = keys)
    return img

def save_image(img, filename, plot_norm = 'log', plot_range = None, dtype = np.float32, keys = 'img'):
    #saves a 2d numpy array, i.e., an image. Multiple images can be saved at once if 'keys' is given.
    if isinstance(keys, str): 
        keys = [keys]
        img = [img]
    plot_image(img[0], filename, plot_norm = plot_norm, plot_range = plot_range)
    save_dict = {}
    for i in range(len(img)): 
        save_dict[keys[i]] = img[i].astype(dtype)
    filename = os.path.splitext(filename)[0]
    np.savez_compressed(filename  + '.npz', **save_dict)   
    
def read_image(filename, dtype = np.float16, keys = ''):
    #reads a numpy file to extract images. Multiple specific images from one file can be loaded if 'keys' is given. If 'keys' is not given, all saved images in the file are returned as a tuple.
    file = np.load(filename)
    if keys == '': keys = file.files
    if   isinstance(keys, str):   data = file[keys].astype(dtype) 
    elif len(keys) == 1:          data = file[keys[0]].astype(dtype) 
    elif isinstance(keys, list):  data = (file[key].astype(dtype) for key in keys)
    return data
    
def plot_image(img, filename, plot_norm = 'log', cmap = 'viridis', plot_range = None):
    #plots an image with a colorbar.
    filename = os.path.splitext(filename)[0]
    fig, ax = plt.subplots(figsize = (6, 5)) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if plot_range:
        vmin = plot_range[0]
        vmax = plot_range[1]
    else:
        if plot_norm == 'log': vmin = np.nanmin(img[img>0])
        if plot_norm == 'lin': vmin = np.nanmin(img)        vmax = np.nanmax(img)
    if plot_norm == 'log': im = ax.imshow(img, norm=LogNorm(vmin=vmin, vmax=vmax, clip = True), interpolation = 'nearest', cmap = cmap)
    if plot_norm == 'lin': im = ax.imshow(img, norm = Normalize(vmin = vmin, vmax = vmax, clip = True), interpolation = 'nearest', cmap = cmap) 
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.tight_layout()
    plt.savefig(filename + '.jpg', dpi = 600)
    plt.close(fig)
