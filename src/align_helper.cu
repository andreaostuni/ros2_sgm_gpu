#include "sgm_gpu/align_helper.hpp"


namespace sgm_gpu
{

void fromCameraInfo(const sensor_msgs::msg::CameraInfo& camera_info, camera_intrinsics& intrinsics)
{
  intrinsics.width = camera_info.width;
  intrinsics.height = camera_info.height;
  intrinsics.fx = camera_info.k[0];
  intrinsics.fy = camera_info.k[4];
  intrinsics.ppx = camera_info.k[2];
  intrinsics.ppy = camera_info.k[5];
}

void fromTransform(const tf2::Transform& transform, Transform& transform_out)
{
  transform_out.rotation[0] = transform.getBasis().getRow(0).getX();
  transform_out.rotation[1] = transform.getBasis().getRow(0).getY();
  transform_out.rotation[2] = transform.getBasis().getRow(0).getZ();
  transform_out.rotation[3] = transform.getBasis().getRow(1).getX();
  transform_out.rotation[4] = transform.getBasis().getRow(1).getY();
  transform_out.rotation[5] = transform.getBasis().getRow(1).getZ();
  transform_out.rotation[6] = transform.getBasis().getRow(2).getX();
  transform_out.rotation[7] = transform.getBasis().getRow(2).getY();
  transform_out.rotation[8] = transform.getBasis().getRow(2).getZ();
  transform_out.translation[0] = transform.getOrigin().getX();
  transform_out.translation[1] = transform.getOrigin().getY();
  transform_out.translation[2] = transform.getOrigin().getZ();
}



} // namespace sgm_gpu