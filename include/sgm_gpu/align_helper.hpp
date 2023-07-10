#ifndef SGM_GPU__ALIGN_HELPER_HPP_
#define SGM_GPU__ALIGN_HELPER_HPP_
#include <memory>
#include <stdint.h>
#include <sensor_msgs/msg/camera_info.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace sgm_gpu
{

typedef struct camera_intrinsics
{
  int width;
  int height;
  float fx;
  float fy;
  float ppx;
  float ppy;
} camera_intrinsics;

typedef struct Transform
{
  float rotation[9];
  float translation[3];
} Transform;

struct rgb_pixel
{
  uint8_t data[3];
};

void fromCameraInfo(const sensor_msgs::msg::CameraInfo& camera_info, camera_intrinsics& intrinsics);

void fromTransform(const tf2::Transform& transform, Transform& transform_out);

}  // namespace sgm_gpu
#endif  // SGM_GPU__ALIGN_HELPER_HPP_