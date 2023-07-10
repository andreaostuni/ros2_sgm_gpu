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

#include "sgm_gpu_node.hpp"

#include <image_geometry/stereo_camera_model.h>
#include <functional>

namespace sgm_gpu
{

SgmGpuNode::SgmGpuNode(const rclcpp::NodeOptions& options) : Node("sgm_gpu_node", options)
{
  rclcpp::Logger parent_logger = this->get_logger();
  sgm_gpu_.reset(new SgmGpu(parent_logger));

  disparity_pub_ = this->create_publisher<stereo_msgs::msg::DisparityImage>("disparity", 1);
  depth_pub_ = image_transport::create_publisher(this, "depth");

  bool publish_aligned = declare_parameter("publish_aligned_depth", false);
  std::string img_transport_type = declare_parameter("image_transport", "raw");

  left_img_sub_.subscribe(this, "left_image", img_transport_type);
  right_img_sub_.subscribe(this, "right_image", img_transport_type);
  left_caminfo_sub_.subscribe(this, "left_camera_info");
  right_caminfo_sub_.subscribe(this, "right_camera_info");

  using namespace std::placeholders;

  if (!publish_aligned)
  {
    stereo_synch_.reset(
        new StereoSynchronizer(left_img_sub_, right_img_sub_, left_caminfo_sub_, right_caminfo_sub_, 5));
    stereo_synch_->registerCallback(std::bind(&SgmGpuNode::stereo_callback, this, _1, _2, _3, _4));
  }
  if (publish_aligned)
  {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    // get transform from depth to color camera frame
    std::string depth_frame_id = declare_parameter("depth_frame_id", "");
    std::string color_frame_id = declare_parameter("color_frame_id", "");

    while (!transform_depth_to_left_camera_)
    {
      RCLCPP_INFO(this->get_logger(), "Waiting for transform depth to color camera frame");
      try
      {
        geometry_msgs::msg::TransformStamped depth_to_color_msg =
            tf_buffer_->lookupTransform(depth_frame_id.c_str(), color_frame_id.c_str(), tf2::TimePointZero);
        tf2::fromMsg(depth_to_color_msg.transform, depth_to_color_tf_);
        color_to_depth_tf_ = depth_to_color_tf_.inverse();
      }
      catch (tf2::TransformException& ex)
      {
        RCLCPP_WARN(this->get_logger(), "%s", ex.what());
        rclcpp::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      if (sgm_gpu_->setTransform(depth_to_color_tf_))
      {
        RCLCPP_INFO(this->get_logger(), "Set transform depth to color camera frame");
        transform_depth_to_left_camera_ = true;
      }
      else
      {
        RCLCPP_ERROR(this->get_logger(), "Failed to set transform depth to color camera frame");
      }
    }
    std::string camera_color_namespace = declare_parameter("camera_color_namespace", "");
    std::string color_image_topic = camera_color_namespace + "/color";
    // subscribe to color image
    color_img_sub_.subscribe(this, color_image_topic, img_transport_type);
    color_caminfo_sub_.subscribe(this, camera_color_namespace + "/camera_info");
    // pub in the same namespace of the camera
    depth_aligned_to_color_pub_ =
        image_transport::create_publisher(this, camera_color_namespace + "/depth_aligned_to_color");
    // pub in the same namespace of the camera
    color_aligned_to_depth_pub_ = image_transport::create_publisher(this, "color_aligned_to_depth");

    color_stereo_synch_.reset(new ColorStereoSynchronizer(
        left_img_sub_, right_img_sub_, color_img_sub_, left_caminfo_sub_, right_caminfo_sub_, color_caminfo_sub_, 5));
    color_stereo_synch_->registerCallback(std::bind(&SgmGpuNode::color_stereo_callback, this, _1, _2, _3, _4, _5, _6));
  }
}

void SgmGpuNode::stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_image,
                                 const sensor_msgs::msg::Image::ConstSharedPtr& right_image,
                                 const sensor_msgs::msg::CameraInfo::ConstSharedPtr& left_info,
                                 const sensor_msgs::msg::CameraInfo::ConstSharedPtr& right_info)
{
  if (depth_pub_.getNumSubscribers() == 0 && disparity_pub_->get_subscription_count() == 0)
    return;

  stereo_msgs::msg::DisparityImage disparity_msg;
  sensor_msgs::msg::Image depth_msg;

  sgm_gpu_->computeDisparity(*left_image, *right_image, *left_info, *right_info, disparity_msg, depth_msg);
  disparity_pub_->publish(disparity_msg);
  depth_pub_.publish(depth_msg);
  return;
}

void SgmGpuNode::color_stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_image,
                                       const sensor_msgs::msg::Image::ConstSharedPtr& right_image,
                                       const sensor_msgs::msg::Image::ConstSharedPtr& color_image,
                                       const sensor_msgs::msg::CameraInfo::ConstSharedPtr& left_info,
                                       const sensor_msgs::msg::CameraInfo::ConstSharedPtr& right_info,
                                       const sensor_msgs::msg::CameraInfo::ConstSharedPtr& color_info)
{
  if (depth_pub_.getNumSubscribers() == 0 && disparity_pub_->get_subscription_count() == 0 &&
      depth_aligned_to_color_pub_.getNumSubscribers() == 0 && color_aligned_to_depth_pub_.getNumSubscribers() == 0)
    return;

  stereo_msgs::msg::DisparityImage disparity_msg;
  sensor_msgs::msg::Image depth_msg;

  // TODO: check if publish align color image or not
  if (depth_aligned_to_color_pub_.getNumSubscribers() == 0 && color_aligned_to_depth_pub_.getNumSubscribers() == 0)
  {
    // RCLCPP_WARN(this->get_logger(), "Not Publishing aligned color image");
    sgm_gpu_->computeDisparity(*left_image, *right_image, *left_info, *right_info, disparity_msg, depth_msg);
    disparity_pub_->publish(disparity_msg);
    depth_pub_.publish(depth_msg);
    return;
  }
  // TODO: check transform depth to left camera frame
  if (transform_depth_to_left_camera_)
  {
    sensor_msgs::msg::Image color_aligned_to_depth_msg;
    sensor_msgs::msg::Image depth_aligned_to_color_msg;
    // RCLCPP_INFO(this->get_logger(), "Publish aligned color image");

    sgm_gpu_->computeDisparity(*left_image, *right_image, *color_image, *left_info, *right_info, *color_info,
                               disparity_msg, depth_msg, color_aligned_to_depth_msg, depth_aligned_to_color_msg);
    disparity_pub_->publish(disparity_msg);
    depth_pub_.publish(depth_msg);
    color_aligned_to_depth_pub_.publish(color_aligned_to_depth_msg);
    depth_aligned_to_color_pub_.publish(depth_aligned_to_color_msg);
    return;
  }
}

}  // namespace sgm_gpu

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(sgm_gpu::SgmGpuNode)
