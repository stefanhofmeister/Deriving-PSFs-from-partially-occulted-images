"""© 2022 The Trustees of Columbia University in the City of New York. This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only, provided that it cites the original work: 
Stefan J. Hofmeister, Michael Hahn, and Daniel Wolf Savin, "Deriving instrumental point spread functions from partially occulted images," J. Opt. Soc. Am. A 39, 2153-2168 (2022), doi: 10.1364/JOSAA.471477.
To obtain a license to use this work for commercial purposes, please contact Columbia Technology Ventures at techventures@columbia.edu <mailto:techventures@columbia.edu>. 
"""

import glob
import os
import numpy as np
from numba import njit
import skimage.morphology as morph
from inout import read_image, save_image


def derive_distance_from_illuminated_edge(config):
    folder_run = config['general']['folder_run']
    files = glob.glob(folder_run + '/occultation_mask/*.npz')
     for file in files:
        occultation_mask = read_image(file, dtype = np.int8)
        exclude           = (occultation_mask == -1)
        illumination_mask = (occultation_mask == 0)
        occultation_mask  = (occultation_mask == 1)
        illuminated_edge  = np.where( (illumination_mask & morph.binary_dilation(occultation_mask)) & (exclude == 0) ) # The ^ is the xor operator
        mask_of_wanted_distances = np.ones(illumination_mask.shape, dtype = np.bool8)
        dists, angles = derive_dist_from_edge(mask_of_wanted_distances, np.array(illuminated_edge))  #this function needs numpy arrays as input (limitation of numba), but np.where gives a list of tuples. Thus, convert it to a numpy array.
        filename =  folder_run + '/distance_from_illuminated_edge/' + os.path.basename(file)
        save_image([dists, angles], filename, plot_norm = 'log', keys = ['dists_from_illuminated_edge', 'angles_from_illuminated_edge'], dtype = np.float16)
    
#derive the distance from the illuminated edge. To do so, we use a growing edges algorithm. We start at the illuminated edge, and let it grow. In each growth step, we derive the distance of the pixels in the new growth band to the illuminated edge. 
#This can be most easily achieved by not deriving the distance directly, but the distance in x and y direction to the illuminated edge separately. This allows to take the distance of the preceeding growth band as a-priori knwon true distances, and thereby to derive the new distances iteratively from the known distances of the last band.
@njit
def derive_dist_from_edge(mask, edges):
    #pad the image by one row and column. This will avoid boundary issues later on.
    dim_x, dim_y = mask.shape
    mask_tmp = np.zeros((dim_x + 2, dim_y + 2))
    mask_tmp[1:-1, 1:-1] = mask
    mask = mask_tmp
    mask = mask.astype(np.uint8)
    
    #correct the edge position accordingly to the padding
    edges = edges + 1
    edges = edges.transpose().astype(np.uint16)
    edges = np.ascontiguousarray(edges)
    
    #define empty dist arrays which will be filled later. Initialize theses with greater than maximum possible distances
    len_edges = mask.shape[0] + mask.shape[1]
    dist_x, dist_y, dist =  (np.full(mask.shape, len_edges, dtype = np.int16), 
                                    np.full(mask.shape,  len_edges, dtype = np.int16), 
                                    np.full(mask.shape,  len_edges**2, dtype = np.float32))
    
    #Set the distances at the illuminated edges in the dist arrays to zero. This is our first band.
    for x, y in edges:
        dist_x[x, y] = 0
        dist_y[x, y] = 0
        dist[x, y] = 0

    #define the direction in which the edge shall be extended in each iteration
    directions = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)]).astype(np.int8)
 
    #now, in each iteration go through the edge line. If a neighboring pixel to the edge line is set in the mask, derive its distance to the original edge.
    #the trick is not to derive the distance to the entire edge line, as this would be computationally very expensive. Instead, use the x an y distances of the edge line which already correspond to minimum distances, and increase these x and y distances by dx and dy to get the minimum distance of the new pixel.
    #having derived the distance of a new pixel, add it to the new edge line, i.e., shift the edge line into that direction. Iterate until the edge line has reached the edges of the mask, i.e., the length of the edge line becomes zero. Then, the entire image has been processed.
    while True:
        new_edges = list()
        #look in each direction
        for dx, dy in directions:
            #for each pixel in the current edge band
            for x, y in edges:
                #if the neighbouring pixel is within the illuminated region, derive the distance to the illuminated edge by using the known x and y distance of the current band
                if mask[x + dx, y + dy] == 1:
                    dist_x_tmp = dist_x[x, y] + dx
                    dist_y_tmp = dist_y[x, y] + dy
                    dist_tmp = np.float32(dist_x_tmp)**2 + np.float32(dist_y_tmp)**2
                    #if the derived distance in the neighbouring pixel is smaller than the one which is currently in, update it and add the neighbouring pixel to the list which will be the next edge band
                    if dist_tmp < dist[x + dx, y + dy]:
                        dist_x[x + dx, y + dy] = dist_x_tmp
                        dist_y[x + dx, y + dy] = dist_y_tmp
                        dist[x + dx, y + dy] = dist_tmp
                        new_edges.append([x + dx, y + dy])
                        
        #iterate over the preceding procedure until the length of the new edge band becomes zero, i.e., all pixels have been processed
        if len(new_edges) == 0:
            break
        else:
            edges = np.array(new_edges).astype(np.uint16)
            edges = np.ascontiguousarray(edges)
            
    #finally, derive the real distances by taking the square root of the distances, set all distances which were not needed (mask == 0) to -1, and remove the padding.
    dist = np.sqrt(dist)
    for x in range(dist.shape[0]):
        for y in range(dist.shape[1]):
            if mask[x, y] == 0:
                dist[x, y] = -1
    angle = np.arctan2(dist_x, dist_y) / np.pi * 180. + 180.
    dist = dist[1:-1, 1:-1]
    angle = angle[1:-1, 1:-1]
    
    return (dist, angle)


