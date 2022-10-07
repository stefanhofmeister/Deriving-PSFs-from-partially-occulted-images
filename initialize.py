"""© 2022 The Trustees of Columbia University in the City of New York.
This work may be reproduced, distributed, and otherwise exploited for
academic non-commercial purposes only, provided that it cites the
original work (IN PUBLICATIONP PROCESS).  To obtain a license to
use this work for commercial purposes, please contact Columbia
Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

import os
import shutil
import numpy as np
from importlib.machinery import SourceFileLoader
from inout import convert_toimage, save_image 
  
 
user_provided_files = [{'fname': 'files_occulted',    'config_key': 'general',            'dir': 'original_image',   'shall_be_array': True,  'normalize': True, 'dtype': np.float32 }, 
                       {'fname': 'occ_file',          'config_key': 'occultation_mask',   'dir': 'occultation_mask', 'shall_be_array': True,  'normalize': False, 'dtype': np.bool8   },
                       {'fname': 'importance_file',   'config_key': 'importance_mask',    'dir': 'importance_mask',  'shall_be_array': True,  'normalize': False, 'dtype': np.int32   },
                       {'fname': 'psfdisc_file',      'config_key': 'psf_discretization', 'dir': '',                 'shall_be_array': False, 'normalize': False, 'dtype': np.int32   },
                       {'fname': 'file_psf_existing', 'config_key': 'general',            'dir': '',                 'shall_be_array': False, 'normalize': False, 'dtype': np.float32 }, 
                       {'fname': 'config_file',       'config_key': 'general',            'dir': '',                 'shall_be_array': False, 'normalize': None,  'dtype': None       },
                       {'fname': 'file_psf',          'config_key': 'testrun',            'dir': '',                 'shall_be_array': False, 'normalize': None,  'dtype': np.float32 }] 

def read_config(configuration):
    #load the configuration file
    if isinstance(configuration, dict) == False:
        config = SourceFileLoader(os.path.basename(configuration), configuration).load_module()
        config = config.config
        config['general']['config_file'] = configuration
    else:
        config = configuration
        config['general']['config_file'] = ''
        
    #if a variable requests a list of files but only a single file was provided, convert it to a list. Also transform the filenames to absolute paths.
    for i in range(len(user_provided_files)): 
           #get the information from the user_provided_files arrray
           fname, config_key, shall_be_array = (user_provided_files[i][key] for key in ['fname', 'config_key', 'shall_be_array'])
           #foreach file
           if fname in config[config_key]:
               #if it is not an array or empty and it shall be an array, transform it to an array. Else, simply transform them to their absolute paths.
               if isinstance(config[config_key][fname], list) == False:
                   if (config[config_key][fname] != ''):
                       if  shall_be_array == True: config[config_key][fname] = [os.path.abspath(config[config_key][fname])]
                       else:                       config[config_key][fname] =  os.path.abspath(config[config_key][fname])
               else:
                   config[config_key][fname] = [os.path.abspath(file) for file in config[config_key][fname]]      
    return config

def initialize(config):
        folder_run      = config['general']['folder_run']

        #build the working directory structure
        os.makedirs(folder_run, exist_ok = True)    
        os.makedirs(folder_run + '/original_image', exist_ok=True)
        os.makedirs(folder_run + '/psf_segmentation', exist_ok=True)
        os.makedirs(folder_run + '/occultation_mask', exist_ok=True)
        os.makedirs(folder_run + '/importance_mask', exist_ok=True)
        os.makedirs(folder_run + '/distance_from_illuminated_edge', exist_ok=True)
        os.makedirs(folder_run + '/deconvolved_image', exist_ok=True)
        os.makedirs(folder_run + '/approximated_true_image', exist_ok=True)
        os.makedirs(folder_run + '/system_of_equations', exist_ok=True)
        os.makedirs(folder_run + '/fitted_psf_coefficients', exist_ok=True)
        os.makedirs(folder_run + '/fitted_psf', exist_ok=True)
        os.makedirs(folder_run + '/final_result', exist_ok=True)

        
        #if it is a testrun, add a folder for the true image and psf. If it is not one, make sure that this folder does not exist.
        if 'testrun' in config:
            if config['testrun']['PSF_testrun'] == True: 
                os.makedirs(folder_run + '/true_image_and_psf', exist_ok=True)
            else:
                if os.path.isdir(folder_run + '/true_image_and_psf'): shutil.rmtree(folder_run + '/true_image_and_psf')
        else:
            if os.path.isdir(folder_run + '/true_image_and_psf'): shutil.rmtree(folder_run + '/true_image_and_psf')

        #always make a clean start, i.e., delete all existing files in the working directory. But keep the folder structure if it already exists, as well as the userdefined files. 
        #walk through all directories in our project folder
        for path, dirs, files in os.walk(folder_run):
            for file in files: 
                delete_file = True
                #if the file is listed in the userdefined_files, do not delete it
                for i in range(len(user_provided_files)):
                    fname, config_key = (user_provided_files[i][key] for key in ['fname', 'config_key'])
                    if fname in config[config_key]:
                        if file in config[config_key][fname]: delete_file = False
                if delete_file == True: os.remove(path + '/' + file)
                
        # copy (or create) the configuration file
        if 'config_file' in config['general']:
            config_file = config['general']['config_file']
            if (folder_run not in config_file) and (config_file != ''): shutil.copy(config_file, folder_run)
            else: write_configuration(config)
         
        #copy the other input files into the run directory
        for i in range(len(user_provided_files)):
            fname, config_key, directory,  shall_be_array, normalize, dtype =  (user_provided_files[i][key] for key in['fname', 'config_key', 'dir', 'shall_be_array', 'normalize', 'dtype']) 
            if fname == 'config_file': continue
            if fname in config[config_key]:
                files_to_copy = config[config_key][fname]
                if shall_be_array == False and files_to_copy != '': files_to_copy = [files_to_copy]
                for file_to_copy in files_to_copy:
                    if not os.path.abspath(folder_run + '/' + directory) in file_to_copy: convert_toimage(file_to_copy, folder_run + '/' + directory + '/' + os.path.basename(file_to_copy), normalize = normalize, dtype = dtype)  

def load_psf(config):
    folder_run = config['general']['folder_run']
    file = config['general']['file_psf_existing']
    resolution = config['general']['resolution']
    large_psf = config['general']['large_psf']
     
    if large_psf:
        res = 2* resolution
    else:
        res = resolution
     
    if os.path.isfile(file):
          psf_original = convert_toimage(file, new_filename = config.folder_run + '/original_psf.jpg', dtype = np.float32, keys = 'psf')
    else:
          psf_original = np.zeros((res, res), dtype = np.float32) 
          psf_original[res//2, res//2] = 1.
          save_image(psf_original, folder_run + '/original_psf.jpg', dtype = np.float32, keys = 'psf') 
    save_image(psf_original, folder_run + '/fitted_psf/fitted_psf.jpg', dtype = np.float32, keys = 'psf')
    return psf_original       

def write_configuration(config):
    with open(config['general']['folder_run'] + 'configuration.config', 'w') as file:
        file.write('#This file was automatically generated from a user-defined configuration dictionary.\n\n')
        file.write('config = {}\n')
        for config_key in config.keys():
            file.write('config[' + config_key + '] = {\n')
            for key in config[config_key]:
                value = config[config_key][key]
                if isinstance(value, str): value = '\'' + value + '\'' 
                else: value = str(value)
                if key == list(config[config_key])[-1]: file.write('   ' + key + ': ' + value + ' }\n\n')
                else:                                     file.write('   ' + key + ': ' + value + ',\n'   ) 
 