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

#ifndef SGM_GPU__SGM_GPU_HPP_
#define SGM_GPU__SGM_GPU_HPP_

#include "sgm_gpu/configuration.hpp"
#include "sgm_gpu/align_helper.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <image_geometry/stereo_camera_model.h>

#include <opencv2/opencv.hpp>

namespace sgm_gpu
{

class SgmGpu
{
private:
  rclcpp::Logger private_logger_;
  image_geometry::StereoCameraModel stereo_model_;
  // AlignCudaHelper align_cuda_helper_;
  /**
   * @brief Parameter used in SGM algorithm
   *
   * See SGM paper.
   */
  const uint8_t P1_ = 6;
  /**
   * @brief Parameter used in SGM algorithm
   *
   * See SGM paper.
   */
  const uint8_t P2_ = 96;

  // Memory for disparity computation
  // d_: for GPU device
  uint8_t* d_im0_;
  uint8_t* d_im1_;
  uint32_t* d_transform0_;
  uint32_t* d_transform1_;
  uint8_t* d_cost_;
  uint8_t* d_disparity_;
  uint8_t* d_disparity_filtered_uchar_;
  uint8_t* d_disparity_right_;
  uint8_t* d_L0_;
  uint8_t* d_L1_;
  uint8_t* d_L2_;
  uint8_t* d_L3_;
  uint8_t* d_L4_;
  uint8_t* d_L5_;
  uint8_t* d_L6_;
  uint8_t* d_L7_;
  uint16_t* d_s_;

  float* d_depth_;
  // for aligning depth image to color image
  uint8_t* d_color_in_;
  // int2* d_pixel_map_;
  float* d_aligned_depth_to_color_;
  uint8_t* d_aligned_color_to_depth_;

  // intrinsics_ and extrinsics_ are used for aligning depth image to color image
  camera_intrinsics h_color_intrinsics_;
  camera_intrinsics h_depth_intrinsics_;
  camera_intrinsics* d_color_intrinsics_;
  camera_intrinsics* d_depth_intrinsics_;

  bool intrinsics_set_;
  Transform h_depth_color_extrinsics_;
  Transform* d_depth_color_extrinsics_;
  // transform h_color_depth_extrinsics_;
  bool transform_set_;

  bool memory_allocated_;
  bool memory_for_align_allocated_;

  uint32_t cols_, rows_;
  uint32_t color_cols_, color_rows_;

  // for depth computation
  float Tx_;
  float delta_cx_;

  void allocateMemory(uint32_t cols, uint32_t rows);
  void allocateMemoryForAlign(uint32_t cols, uint32_t rows, uint32_t color_cols, uint32_t color_rows);
  void freeMemory();
  void freeMemoryForAlign();

  /**
   * @brief Resize images to be width and height divisible by 4 for limit of CUDA code
   */
  void resizeToDivisibleBy4(cv::Mat& left_image, cv::Mat& right_image);

  void resizeToDivisibleBy4(cv::Mat& image);

  void convertToMsg(const cv::Mat_<unsigned char>& disparity, const sensor_msgs::msg::CameraInfo& left_camera_info,
                    const sensor_msgs::msg::CameraInfo& right_camera_info,
                    stereo_msgs::msg::DisparityImage& disparity_msg);

  void convertToDepthMsg(const cv::Mat_<float>& depth, const sensor_msgs::msg::CameraInfo& camera_info,
                         sensor_msgs::msg::Image& depth_msg);

  void convertToColorMsg(const cv::Mat_<cv::Vec3b>& color, const sensor_msgs::msg::CameraInfo& camera_info,
                         sensor_msgs::msg::Image& color_msg);

public:
  /**
   * @brief Constructor which use namespace <parent>/libsgm_gpu for logging
   */
  SgmGpu(rclcpp::Logger& parent_logger);
  ~SgmGpu();

  bool computeDisparity(const sensor_msgs::msg::Image& left_image, const sensor_msgs::msg::Image& right_image,
                        const sensor_msgs::msg::CameraInfo& left_camera_info,
                        const sensor_msgs::msg::CameraInfo& right_camera_info,
                        stereo_msgs::msg::DisparityImage& disparity_msg, sensor_msgs::msg::Image& depth_msg);
  // set transform from color to depth
  bool setTransform(const tf2::Transform& depth_to_color);

  // set intrinsics of camera from camera_info and width and height of image
  bool setIntrinsics(const sensor_msgs::msg::CameraInfo& camera_info, camera_intrinsics& intrinsics, uint32_t cols,
                     uint32_t rows);

  // TODO: implement overload function for cv::Mat
  // bool computeDisparity(const cv::Mat& left_image, const cv::Mat& right_image,
  //                       const sensor_msgs::msg::CameraInfo& left_camera_info,
  //                       const sensor_msgs::msg::CameraInfo& right_camera_info,
  //                       stereo_msgs::msg::DisparityImage& disparity_msg,
  //                       sensor_msgs::msg::Image& depth_msg);

  // TODO: implement overload funcion with aligned image
  bool computeDisparity(const sensor_msgs::msg::Image& left_image, const sensor_msgs::msg::Image& right_image,
                        const sensor_msgs::msg::Image& color_image,
                        const sensor_msgs::msg::CameraInfo& left_camera_info,
                        const sensor_msgs::msg::CameraInfo& right_camera_info,
                        const sensor_msgs::msg::CameraInfo& color_camera_info,
                        stereo_msgs::msg::DisparityImage& disparity_msg, sensor_msgs::msg::Image& depth_msg,
                        sensor_msgs::msg::Image& aligned_color_to_depth_image,
                        sensor_msgs::msg::Image& aligned_depth_to_color_image);
};

}  // namespace sgm_gpu

#endif  // SGM_GPU__SGM_GPU_HPP_
