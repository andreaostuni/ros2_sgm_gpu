// Copyright 2020 Hironori Fujimoto
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef SGM_GPU__SGM_GPU_NODE_HPP_
#define SGM_GPU__SGM_GPU_NODE_HPP_

#include "sgm_gpu/sgm_gpu.hpp"

#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "tf2_ros/transform_listener.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>

namespace sgm_gpu
{

class SgmGpuNode : public rclcpp::Node
{
public:
  explicit SgmGpuNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  std::shared_ptr<SgmGpu> sgm_gpu_;

  rclcpp::Publisher<stereo_msgs::msg::DisparityImage>::SharedPtr disparity_pub_;
  image_transport::Publisher depth_pub_;
  image_transport::Publisher depth_aligned_to_color_pub_, color_aligned_to_depth_pub_;

  image_transport::SubscriberFilter left_img_sub_;
  image_transport::SubscriberFilter right_img_sub_;
  image_transport::SubscriberFilter color_img_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> left_caminfo_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> right_caminfo_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> color_caminfo_sub_;
  // tf related
  tf2_ros::Buffer::SharedPtr tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  // std::string depth_frame_id_, color_frame_id_;
  tf2::Transform depth_to_color_tf_, color_to_depth_tf_;
  bool transform_depth_to_left_camera_ = false;
  // stereo related
  using StereoSynchronizer =
      message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo,
                                        sensor_msgs::msg::CameraInfo>;
  std::shared_ptr<StereoSynchronizer> stereo_synch_;

  void stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_image,
                       const sensor_msgs::msg::Image::ConstSharedPtr& right_image,
                       const sensor_msgs::msg::CameraInfo::ConstSharedPtr& left_info,
                       const sensor_msgs::msg::CameraInfo::ConstSharedPtr& right_info);
  // stereo + color related
  using ColorStereoSynchronizer =
      message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image,
                                        sensor_msgs::msg::CameraInfo, sensor_msgs::msg::CameraInfo,
                                        sensor_msgs::msg::CameraInfo>;
  std::shared_ptr<ColorStereoSynchronizer> color_stereo_synch_;
  void color_stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_image,
                             const sensor_msgs::msg::Image::ConstSharedPtr& right_image,
                             const sensor_msgs::msg::Image::ConstSharedPtr& color_image,
                             const sensor_msgs::msg::CameraInfo::ConstSharedPtr& left_info,
                             const sensor_msgs::msg::CameraInfo::ConstSharedPtr& right_info,
                             const sensor_msgs::msg::CameraInfo::ConstSharedPtr& color_info);
};

}  // namespace sgm_gpu

#endif  // SGM_GPU__SGM_GPU_NODE_HPP_
