#ifndef GPUFIT_MULTILINEAR_1024PARAMETERS
#define GPUFIT_MULTILINEAR_1024PARAMETERS

/* Description of the 
* ===================================================
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: offset
*             p[1]: slope
*
* n_fits: The number of fits; I think which are running concurrently on the GPU, not the total number of fits (!). This is why chunksize is needed.
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index.
*
* chunk_index: The chunk index. Used for indexing of user_info.
*
* user_info: An input vector containing user information.
*
* user_info_size: The size of user_info in bytes.
*
* Calling the calculate_linear1d function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_multilinear_1024parameters(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    REAL * user_info_float = (REAL*) user_info;
    REAL * current_derivatives = derivative + point_index;

    int n_parameters = user_info_float[0];
    REAL * shells_npix = &user_info_float[1]; //this array has n_parameter values
    int const chunk_begin = 1 + n_parameters + chunk_index * n_fits * (n_points + n_parameters*n_points) ;
    int const fit_begin = fit_index * (n_points + n_parameters*n_points);  
    REAL y_diff = user_info_float[chunk_begin + fit_begin + point_index];
    REAL * x = &user_info_float[chunk_begin + fit_begin + n_points + n_parameters*point_index];
    
    double y = 0, totalweight_longscatter = 0;
    REAL x_i = 0, para_i = 0, shells_npix_i = 0;
    for (int i_para = 0; i_para < n_parameters; i_para++)
    {
        x_i = x[i_para];
        para_i = parameters[i_para];
        shells_npix_i = shells_npix[i_para];
        
        y += para_i * x_i;
        totalweight_longscatter += para_i * shells_npix_i;
        current_derivatives[i_para * n_points] = x_i - y_diff * shells_npix_i;
    }
    y += (1. - totalweight_longscatter) * y_diff;
    value[point_index] = y;
}

#endif
