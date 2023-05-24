#include "sgm_gpu/disparity_to_depth.hpp"
#include <stdio.h>

namespace sgm_gpu
{
__global__ void disparityToDepth(const uint8_t* __restrict__ d_input, float* __restrict__ d_out, const float Tx,uint32_t rows, uint32_t cols,const float delta_cx = 0.0f)
{
    const int x = blockIdx.x*blockDim.x+threadIdx.x;
    const int y = blockIdx.y*blockDim.y+threadIdx.y;
    
    if (x >= cols || y >= rows)
    return;
    
    if (d_input[y*cols + x] == 0)
      d_out[y*cols + x] = bigZ;
    else
      d_out[y*cols + x] = -Tx / (d_input[y*cols + x] - (delta_cx));
}
}