"""© 2022 The Trustees of Columbia University in the City of New York. 
This work may be reproduced, distributed, and otherwise exploited for 
academic non-commercial purposes only, provided that it cites the original work: 

    Stefan J. Hofmeister, Michael Hahn, and Daniel Wolf Savin, "Deriving instrumental point spread functions from partially occulted images," 
    Journal of the Optical Society of America A 39, 2153-2168 (2022), doi: 10.1364/JOSAA.471477.

To obtain a license to use this work for commercial purposes, please contact 
Columbia Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

import glob
import os
import numpy as np 
from inout import read_image, save_image  

def estimate_true_image(config):
    #the approximated true image is the deconvolved image with the intensities in the occultation set to zero. Note that our algorithm basically works with all images where the true intensities can be approximated in advance, not only partially occulted images. Thus, if you have another approximation, you can add your own function here.
    folder_run = config['general']['folder_run']
    files = glob.glob(folder_run + '/deconvolved_image/*.npz')
    for file in files:
        img_dec = read_image(file, dtype = np.float16)
        occ_mask = read_image(folder_run + '/occultation_mask/' + os.path.basename(file), dtype = np.int8)
        img_approx_true  = img_dec
        img_approx_true[occ_mask == 1] = 0
        save_image(img_approx_true, folder_run + '/approximated_true_image/' + os.path.basename(file), dtype = np.float16, plot_norm = 'log')
