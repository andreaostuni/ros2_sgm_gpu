#ifndef SGM_GPU__ALIGN_RGB_HPP_
#define SGM_GPU__ALIGN_RGB_HPP_

#include <stdint.h>
#include <memory>
#include "sgm_gpu/align_helper.hpp"

namespace sgm_gpu
{
__device__ void kernel_transfer_pixels(int2* mapped_pixels, const camera_intrinsics* depth_intrin,
                                       const camera_intrinsics* other_intrin, const Transform* depth_to_other,
                                       float depth_val, int depth_x, int depth_y, int block_index);

__global__ void kernel_map_depth_to_other(int2* mapped_pixels, const float* depth_in,
                                          const camera_intrinsics* depth_intrin, const camera_intrinsics* other_intrin,
                                          const Transform* depth_to_other);

__global__ void kernel_other_to_depth(uint8_t* aligned, const uint8_t* other, const int2* mapped_pixels,
                                      const camera_intrinsics* depth_intrin, const camera_intrinsics* other_intrin);

__global__ void kernel_depth_to_other(float* aligned_out, const float* depth_in, const int2* mapped_pixels,
                                      const camera_intrinsics* depth_intrin, const camera_intrinsics* other_intrin);

__global__ void kernel_replace_to_zero(float* aligned_out, const camera_intrinsics* other_intrin);

/* Transform 3D coordinates relative to one sensor to 3D coordinates relative to another viewpoint */
__device__ void transform_point_to_point(float to_point[3], const Transform* extrin, const float from_point[3]);

__device__ void project_point_to_pixel(float pixel[2], const camera_intrinsics* intrin, const float point[3]);

/* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the
 * corresponding point in 3D space relative to the same camera */
__device__ void deproject_pixel_to_point(float point[3], const camera_intrinsics* intrin, const float pixel[2],
                                         float depth);
}  // namespace sgm_gpu

#endif  // SGM_GPU__ALIGN_HPP_