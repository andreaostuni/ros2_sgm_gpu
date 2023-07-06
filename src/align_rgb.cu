
#include "sgm_gpu/align_rgb.hpp"

namespace sgm_gpu
{
__inline__ __device__ static float atomicMin(float* address, float val)
{
  int* address_as_i = (int*)address;
  int old = *address_as_i, assumed;
  do
  {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ void kernel_transfer_pixels(int2* mapped_pixels, const camera_intrinsics* depth_intrin,
    const camera_intrinsics* other_intrin, const Transform* depth_to_other, float depth_val, int depth_x, int depth_y, int block_index)
{
    // block_index is used to map the pixels to the correct location in the mapped_pixels array
    float shift = block_index ? 0.5 : -0.5; 
    auto depth_size = depth_intrin->width * depth_intrin->height;
    auto mapped_index = block_index * depth_size + (depth_y * depth_intrin->width + depth_x);

    // auto depth_size = depth_intrin->width * depth_intrin->height;
    // auto mapped_index = block_index * depth_size + (depth_y * depth_intrin->width + depth_x);

    if (mapped_index >= depth_size * 2)
        return;

    // Skip over depth pixels with the value of zero, we have no depth data so we will not write anything into our aligned images
    if (depth_val == 0)
    {
        mapped_pixels[mapped_index] = { -1, -1 };
        return;
    }

    //// Map the top-left corner of the depth pixel onto the other image
    float depth_pixel[2] = { depth_x + shift, depth_y + shift }, depth_point[3], other_point[3], other_pixel[2];
    deproject_pixel_to_point(depth_point, depth_intrin, depth_pixel, depth_val);
    transform_point_to_point(other_point, depth_to_other, depth_point);
    project_point_to_pixel(other_pixel, other_intrin, other_point);
    mapped_pixels[mapped_index].x = static_cast<int>(other_pixel[0] + 0.5f); // pixel coordinates are rounded to the next integer
    mapped_pixels[mapped_index].y = static_cast<int>(other_pixel[1] + 0.5f); // pixel coordinates are rounded to the next integer
}

// call this kernel to obtain the mapped pixels for each depth pixel

__global__  void kernel_map_depth_to_other(int2* mapped_pixels, const float* depth_in, const camera_intrinsics* depth_intrin, const camera_intrinsics* other_intrin,
    const Transform* depth_to_other)
{
    int depth_x = blockIdx.x * blockDim.x + threadIdx.x; // depth_x is the x coordinate of the depth pixel
    int depth_y = blockIdx.y * blockDim.y + threadIdx.y; // depth_y is the y coordinate of the depth pixel

    int depth_pixel_index = depth_y * depth_intrin->width + depth_x;
    if (depth_pixel_index >= depth_intrin->height * depth_intrin->width)
        return;
    float depth_val = depth_in[depth_pixel_index];
    kernel_transfer_pixels(mapped_pixels, depth_intrin, other_intrin, depth_to_other, depth_val, depth_x, depth_y, blockIdx.z);
}

// image alignment to depth
__global__  void kernel_other_to_depth(uint8_t* aligned, const uint8_t* other, const int2* mapped_pixels, const camera_intrinsics* depth_intrin, const camera_intrinsics* other_intrin)
{
    int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

    auto depth_size = depth_intrin->width * depth_intrin->height;
    int depth_pixel_index = depth_y * depth_intrin->width + depth_x;

    if (depth_pixel_index >= depth_intrin->width * depth_intrin->height)
        return;

    int2 p0 = mapped_pixels[depth_pixel_index];              // p0 is the top-left corner of the rectangle on the other image    
    int2 p1 = mapped_pixels[depth_size + depth_pixel_index]; // p1 is the bottom-right corner of the rectangle on the other image

    if (p0.x < 0 || p0.y < 0 || p1.x >= other_intrin->width || p1.y >= other_intrin->height) // if the rectangle is outside the other image, we skip it
        return;
    
    // Transfer between the depth pixels and the pixels inside the rectangle on the other image
    auto in_other = (const rgb_pixel*)(other); // pointer to the other image 
    auto out_other = (rgb_pixel*)(aligned);    // pointer to the aligned image
    for (int y = p0.y; y <= p1.y; ++y)                                  // iterate over the pixels inside the rectangle on the other image
    {
        for (int x = p0.x; x <= p1.x; ++x)  
        {
            auto other_pixel_index = y * other_intrin->width + x;       // index of the pixel inside the rectangle on the other image
            out_other[depth_pixel_index] = in_other[other_pixel_index]; // transfer the pixel from the other image to the aligned image
        }
    }
}

// depth alignment to other
__global__  void kernel_depth_to_other(float* aligned_out, const float* depth_in, const int2* mapped_pixels, const camera_intrinsics* depth_intrin, const camera_intrinsics* other_intrin)
{
    int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

    auto depth_size = depth_intrin->width * depth_intrin->height;
    int depth_pixel_index = depth_y * depth_intrin->width + depth_x;

    if (depth_pixel_index >= depth_intrin->width * depth_intrin->height)
        return;

    int2 p0 = mapped_pixels[depth_pixel_index];
    int2 p1 = mapped_pixels[depth_size + depth_pixel_index];

    if (p0.x < 0 || p0.y < 0 || p1.x >= other_intrin->width || p1.y >= other_intrin->height)
        return;

    // Transfer between the depth pixels and the pixels inside the rectangle on the other image

    float new_val = depth_in[depth_pixel_index];
    // float* arr = (float*)aligned_out;
    for (int y = p0.y; y <= p1.y; ++y)
    {
        for (int x = p0.x; x <= p1.x; ++x)
        {
            auto other_pixel_index = y * other_intrin->width + x; // index of the pixel inside the rectangle on the other image
            atomicMin(&aligned_out[other_pixel_index], new_val);      // transfer the depth value to the aligned image
        }
    }
}


// replace all the pixels that are not mapped to zero
__global__  void kernel_replace_to_zero(float* aligned_out, const camera_intrinsics* other_intrin)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    auto other_pixel_index = y * other_intrin->width + x;
    if (aligned_out[other_pixel_index] == FLT_MAX)
        aligned_out[other_pixel_index] = 0.0f;
}

__device__ void transform_point_to_point(float to_point[3], const Transform* extrin, const float from_point[3])
{
  to_point[0] = extrin->rotation[0] * from_point[0] + extrin->rotation[3] * from_point[1] +
                extrin->rotation[6] * from_point[2] + extrin->translation[0];
  to_point[1] = extrin->rotation[1] * from_point[0] + extrin->rotation[4] * from_point[1] +
                extrin->rotation[7] * from_point[2] + extrin->translation[1];
  to_point[2] = extrin->rotation[2] * from_point[0] + extrin->rotation[5] * from_point[1] +
                extrin->rotation[8] * from_point[2] + extrin->translation[2];
}

__device__ void project_point_to_pixel(float pixel[2], const camera_intrinsics* intrin, const float point[3])
{
  float x = point[0] / point[2], y = point[1] / point[2];
  pixel[0] = x * intrin->fx + intrin->ppx;
  pixel[1] = y * intrin->fy + intrin->ppy;
}

__device__ void deproject_pixel_to_point(float point[3], const camera_intrinsics* intrin, const float pixel[2],
                                         float depth)
{
  float x = (pixel[0] - intrin->ppx) / intrin->fx;
  float y = (pixel[1] - intrin->ppy) / intrin->fy;

  point[0] = depth * x;
  point[1] = depth * y;
  point[2] = depth;
}

} // namespace