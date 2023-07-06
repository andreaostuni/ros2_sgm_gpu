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


#include "sgm_gpu/sgm_gpu.hpp"

#include "sgm_gpu/costs.hpp"
#include "sgm_gpu/hamming_cost.hpp"
#include "sgm_gpu/median_filter.hpp"
#include "sgm_gpu/cost_aggregation.hpp"
#include "sgm_gpu/left_right_consistency.hpp"
#include "sgm_gpu/disparity_to_depth.hpp"
#include "sgm_gpu/align_rgb.hpp"

#include <image_geometry/stereo_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>

namespace sgm_gpu
{
// Variables which have CUDA-related type are put here
//   in order to include sgm_gpu.h from non-CUDA package
cudaStream_t stream1_;
cudaStream_t stream2_;
cudaStream_t stream3_;

SgmGpu::SgmGpu(rclcpp::Logger& parent_logger) : 
  memory_allocated_(false), cols_(0), rows_(0),Tx_(0.0), stereo_model_(),
  private_logger_(parent_logger.get_child("libsgm_gpu"))
{
  cudaStreamCreate(&stream1_);
  cudaStreamCreate(&stream2_);
  cudaStreamCreate(&stream3_);
}

SgmGpu::~SgmGpu()
{
  freeMemoryForAlign();

  cudaStreamDestroy(stream1_);
  cudaStreamDestroy(stream2_);
  cudaStreamDestroy(stream3_);
}

void SgmGpu::allocateMemory(uint32_t cols, uint32_t rows)
{
  freeMemory();

  cols_ = cols;
  rows_ = rows;

  int total_pixel = cols_ * rows_;
  cudaMalloc((void **)&d_im0_, sizeof(uint8_t) * total_pixel);
  cudaMalloc((void **)&d_im1_, sizeof(uint8_t) * total_pixel);

  cudaMalloc((void **)&d_transform0_, sizeof(cost_t) * total_pixel);
  cudaMalloc((void **)&d_transform1_, sizeof(cost_t) * total_pixel);

  int cost_volume_size = total_pixel * MAX_DISPARITY;
  cudaMalloc((void **)&d_cost_, sizeof(uint8_t) * cost_volume_size);

  cudaMalloc((void **)&d_L0_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L1_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L2_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L3_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L4_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L5_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L6_, sizeof(uint8_t) * cost_volume_size);
  cudaMalloc((void **)&d_L7_, sizeof(uint8_t) * cost_volume_size);

  cudaMalloc((void **)&d_s_, sizeof(uint16_t) * cost_volume_size);

  cudaMalloc((void **)&d_disparity_, sizeof(uint8_t) * total_pixel);
  cudaMalloc((void **)&d_disparity_filtered_uchar_, sizeof(uint8_t) * total_pixel);
  cudaMalloc((void **)&d_disparity_right_, sizeof(uint8_t) * total_pixel);
  cudaMalloc((void **)&d_depth_, sizeof(float) * total_pixel);

  memory_allocated_ = true;
}

//* NEW CODE
void SgmGpu::allocateMemoryForAlign(uint32_t cols, uint32_t rows, uint32_t color_cols, uint32_t color_rows){
  freeMemoryForAlign();
  color_cols_ = color_cols;
  color_rows_ = color_rows;
  int total_color_pixel = color_cols_ * color_rows_;
  int total_pixel = cols * rows;
  cudaMalloc((void **)&d_color_in_, sizeof(uint8_t) * total_color_pixel * 3);
  cudaMalloc((void **)&d_aligned_depth_to_color_, sizeof(float) * total_color_pixel);
  cudaMemset(d_aligned_depth_to_color_, 0xff, sizeof(float) * total_color_pixel);
  cudaMalloc((void **)&d_aligned_color_to_depth_, sizeof(uint8_t) * total_pixel * 3);
  cudaMemset(d_aligned_color_to_depth_, 0, sizeof(uint8_t) * total_pixel * 3);
  
  // intrinsics and extrinsics
  cudaMalloc((void **)&d_depth_intrinsics_, sizeof(camera_intrinsics));
  cudaMalloc((void **)&d_color_intrinsics_, sizeof(camera_intrinsics));
  cudaMalloc((void **)&d_depth_color_extrinsics_, sizeof(Transform));


  allocateMemory(cols, rows);
  // TODO set memory for transform extrinsics
  memory_for_align_allocated_ = true;
}
//* END NEW CODE

void SgmGpu::freeMemory() {
  if (!memory_allocated_)
    return;

  cudaFree(d_im0_);
  cudaFree(d_im1_);
  cudaFree(d_transform0_);
  cudaFree(d_transform1_);
  cudaFree(d_L0_);
  cudaFree(d_L1_);
  cudaFree(d_L2_);
  cudaFree(d_L3_);
  cudaFree(d_L4_);
  cudaFree(d_L5_);
  cudaFree(d_L6_);
  cudaFree(d_L7_);
  cudaFree(d_disparity_);
  cudaFree(d_disparity_filtered_uchar_);
  cudaFree(d_disparity_right_);
  cudaFree(d_cost_);
  cudaFree(d_s_);

  cudaFree(d_depth_);
  memory_allocated_ = false;
}
//* NEW CODE
void SgmGpu::freeMemoryForAlign(){
  freeMemory();
  if (!memory_for_align_allocated_)
    return;
  cudaFree(d_color_in_);
  cudaFree(d_aligned_depth_to_color_);
  cudaFree(d_aligned_color_to_depth_);
  // cudaFree(d_pixel_map_);
  cudaFree(d_depth_intrinsics_);
  cudaFree(d_color_intrinsics_);
  cudaFree(d_depth_color_extrinsics_);
  memory_for_align_allocated_ = false;
}
//* END NEW CODE

bool SgmGpu::computeDisparity(
  const sensor_msgs::msg::Image& left_image, 
  const sensor_msgs::msg::Image& right_image,
  const sensor_msgs::msg::CameraInfo& left_camera_info,
  const sensor_msgs::msg::CameraInfo& right_camera_info,
  stereo_msgs::msg::DisparityImage& disparity_msg,
  sensor_msgs::msg::Image& depth_msg
)
{
  if (left_image.width != right_image.width || left_image.height != right_image.height)
  {
    RCLCPP_ERROR_STREAM(private_logger_,
      "Image dimension of left and right are not same: \n" << 
      "Left: " << left_image.width << "x" << left_image.height << "\n" <<
      "Right: " << right_image.width << "x" << right_image.height
    );
    return false;
  }
  
  if (left_image.encoding != right_image.encoding)
  {
    RCLCPP_ERROR_STREAM(private_logger_,
      "Image encoding of left and right are not same: \n" << 
      "Left: " << left_image.encoding << "\n" <<
      "Right: " << right_image.encoding
    );
    return false;
  }

  // Convert to 8 bit grayscale image
  cv_bridge::CvImagePtr left_mono8 = cv_bridge::toCvCopy(
    left_image, 
    sensor_msgs::image_encodings::MONO8
  );
  cv_bridge::CvImagePtr right_mono8 = cv_bridge::toCvCopy(
    right_image, 
    sensor_msgs::image_encodings::MONO8
  );
  
  // Resize images to their width and height divisible by 4 for limit of CUDA code
  resizeToDivisibleBy4(left_mono8->image, right_mono8->image);

  if (!stereo_model_.initialized())
  {
    stereo_model_.fromCameraInfo(left_camera_info, right_camera_info);
    Tx_ = stereo_model_.right().Tx();
    delta_cx_ = stereo_model_.left().cx() - stereo_model_.right().cx();
    if(delta_cx_ > FLT_EPSILON)
    {
      RCLCPP_INFO_STREAM(private_logger_,
        "cx of left and right camera are not same: \n" << 
        "Left: " << stereo_model_.left().cx() << "\n" <<
        "Right: " << stereo_model_.right().cx()
      );
      delta_cx_ = 0.0f;
    }
  }

  // Reallocate memory if needed
  bool size_changed = (cols_ != left_mono8->image.cols || rows_ != left_mono8->image.rows);
  if (!memory_allocated_ || size_changed)
    allocateMemory(left_mono8->image.cols, left_mono8->image.rows);
  
  // Copy image to GPU device
  size_t mono8_image_size = left_mono8->image.total() * sizeof(uint8_t);
  cudaMemcpyAsync(d_im0_, left_mono8->image.ptr<uint8_t>(), 
    mono8_image_size, cudaMemcpyHostToDevice, stream1_);
  cudaMemcpyAsync(d_im1_, right_mono8->image.ptr<uint8_t>(), 
    mono8_image_size, cudaMemcpyHostToDevice, stream1_);

  dim3 block_size;
  block_size.x = 32;
  block_size.y = 32;

  dim3 grid_size;
  grid_size.x = (cols_ + block_size.x-1) / block_size.x;
  grid_size.y = (rows_ + block_size.y-1) / block_size.y;

  CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, stream1_>>>(d_im0_, d_im1_, d_transform0_, d_transform1_, rows_, cols_);

  cudaStreamSynchronize(stream1_);
  HammingDistanceCostKernel<<<rows_, MAX_DISPARITY, 0, stream1_>>>(d_transform0_, d_transform1_, d_cost_, rows_, cols_);

  const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;
  const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;

  // Cost Aggregation
  CostAggregationKernelLeftToRight<<<(rows_+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream2_>>>(d_cost_, d_L0_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelRightToLeft<<<(rows_+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream3_>>>(d_cost_, d_L1_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelUpToDown<<<(cols_+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1_>>>(d_cost_, d_L2_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDownToUp<<<(cols_+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1_>>>(d_cost_, d_L3_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalDownUpLeftRight<<<(cols_+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1_>>>(d_cost_, d_L4_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalUpDownLeftRight<<<(cols_+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1_>>>(d_cost_, d_L5_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalDownUpRightLeft<<<(cols_+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1_>>>(d_cost_, d_L6_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalUpDownRightLeft<<<(cols_+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1_>>>(d_cost_, d_L7_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);

  // Left-right consistency check
  int total_pixel = rows_ * cols_;
  ChooseRightDisparity<<<grid_size, block_size, 0, stream1_>>>(d_disparity_right_, d_s_, rows_, cols_);
  LeftRightConsistencyCheck<<<grid_size, block_size, 0, stream1_>>>(d_disparity_, d_disparity_right_, rows_, cols_);
  
  MedianFilter3x3<<<(total_pixel+MAX_DISPARITY-1)/MAX_DISPARITY, MAX_DISPARITY, 0, stream1_>>>(d_disparity_, d_disparity_filtered_uchar_, rows_, cols_);
  
  // Disparity to Depth

  disparityToDepth<<<grid_size, block_size, 0, stream1_>>>(d_disparity_filtered_uchar_, d_depth_, Tx_, rows_, cols_, delta_cx_);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    RCLCPP_ERROR(private_logger_, "%s %d\n", cudaGetErrorString(err), err);
    return false;
  }

  cudaDeviceSynchronize();
  //copy disparity
  cv::Mat disparity(rows_, cols_, CV_8UC1);
  cudaMemcpy(disparity.data, d_disparity_filtered_uchar_, sizeof(uint8_t)*total_pixel, cudaMemcpyDeviceToHost);
  // copy depth
  cv::Mat depth(rows_, cols_, CV_32FC1);
  cudaMemcpy(depth.data, d_depth_, sizeof(float)*total_pixel, cudaMemcpyDeviceToHost);
  // Restore image size if resized to be divisible by 4
  if (cols_ != left_image.width || rows_ != left_image.height)
  {
    cv::Size input_size(left_image.width, left_image.height);
    cv::resize(disparity, disparity, input_size, 0, 0, cv::INTER_AREA);
    cv::resize(depth, depth, input_size, 0, 0, cv::INTER_AREA);
  }
  
  convertToMsg(disparity, left_camera_info, right_camera_info, disparity_msg);
  convertToDepthMsg(depth, left_camera_info, depth_msg);
  return true;
}

bool SgmGpu::computeDisparity(
  const sensor_msgs::msg::Image& left_image, 
  const sensor_msgs::msg::Image& right_image,
  const sensor_msgs::msg::Image& color_image,
  const sensor_msgs::msg::CameraInfo& left_camera_info,
  const sensor_msgs::msg::CameraInfo& right_camera_info,
  const sensor_msgs::msg::CameraInfo& color_camera_info,
  stereo_msgs::msg::DisparityImage& disparity_msg,
  sensor_msgs::msg::Image& depth_msg,
  sensor_msgs::msg::Image& aligned_color_to_depth_image,
  sensor_msgs::msg::Image& aligned_depth_to_color_image
)
{
  if (left_image.width != right_image.width || left_image.height != right_image.height)
  {
    RCLCPP_ERROR_STREAM(private_logger_,
      "Image dimension of left and right are not same: \n" << 
      "Left: " << left_image.width << "x" << left_image.height << "\n" <<
      "Right: " << right_image.width << "x" << right_image.height
    );
    return false;
  }
  
  if (left_image.encoding != right_image.encoding)
  {
    RCLCPP_ERROR_STREAM(private_logger_,
      "Image encoding of left and right are not same: \n" << 
      "Left: " << left_image.encoding << "\n" <<
      "Right: " << right_image.encoding
    );
    return false;
  }

  //* NEW CODE
  if (!transform_set_)
  {
    RCLCPP_ERROR(private_logger_, "Transform not set");
    return false;
  }
  //* END NEW CODE

  // Convert to 8 bit grayscale image
  cv_bridge::CvImagePtr left_mono8 = cv_bridge::toCvCopy(
    left_image, 
    sensor_msgs::image_encodings::MONO8
  );
  cv_bridge::CvImagePtr right_mono8 = cv_bridge::toCvCopy(
    right_image, 
    sensor_msgs::image_encodings::MONO8
  );
  //* NEW CODE
    cv_bridge::CvImagePtr color_rgb8 = cv_bridge::toCvCopy(
    color_image, 
    sensor_msgs::image_encodings::RGB8
  );
  //* END NEW CODE
  // Resize images to their width and height divisible by 4 for limit of CUDA code
  resizeToDivisibleBy4(left_mono8->image, right_mono8->image);
  //* NEW CODE
  resizeToDivisibleBy4(color_rgb8->image); 
  //* END NEW CODE

  if (!stereo_model_.initialized())
  {
    stereo_model_.fromCameraInfo(left_camera_info, right_camera_info);
    Tx_ = stereo_model_.right().Tx();
    delta_cx_ = stereo_model_.left().cx() - stereo_model_.right().cx();
    if(delta_cx_ > FLT_EPSILON)
    {
      RCLCPP_INFO_STREAM(private_logger_,
        "cx of left and right camera are not same: \n" << 
        "Left: " << stereo_model_.left().cx() << "\n" <<
        "Right: " << stereo_model_.right().cx()
      );
      delta_cx_ = 0.0f;
    }
  }
  //* NEW CODE

  //set camera intrinsics
  //TODO: check if camera intrinsics are same
  // if(!intrinsics_set_)
  // {
  //   intrinsics_set_ = setIntrinsics(color_camera_info,h_color_intrinsics_, color_rgb8->image.cols, color_rgb8->image.rows) &&
  //            setIntrinsics(left_camera_info,h_depth_intrinsics_, left_mono8->image.cols, left_mono8->image.rows);
  // }


  //* END NEW CODE



  // Reallocate memory if needed
  bool size_changed = (cols_ != left_mono8->image.cols || rows_ != left_mono8->image.rows);
  if(!memory_allocated_ || size_changed){
    allocateMemory(left_mono8->image.cols, left_mono8->image.rows);
    // setIntrinsics(left_camera_info,h_depth_intrinsics_, left_mono8->image.cols, left_mono8->image.rows);
  }
  //* NEW CODE
  bool size_changed_color = (color_cols_ != color_rgb8->image.cols || color_rows_ != color_rgb8->image.rows);
  if(!memory_for_align_allocated_ || size_changed_color){
    allocateMemoryForAlign(left_mono8->image.cols, left_mono8->image.rows,color_rgb8->image.cols, color_rgb8->image.rows);
    intrinsics_set_ = setIntrinsics(color_camera_info,h_color_intrinsics_, color_rgb8->image.cols, color_rgb8->image.rows) &&
             setIntrinsics(left_camera_info,h_depth_intrinsics_, left_mono8->image.cols, left_mono8->image.rows);
    if(intrinsics_set_)
    {
    cudaMemcpy(d_color_intrinsics_, &h_color_intrinsics_, sizeof(camera_intrinsics), cudaMemcpyHostToDevice);
    cudaMemcpy(d_depth_intrinsics_, &h_depth_intrinsics_, sizeof(camera_intrinsics), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_depth_color_extrinsics_, &h_depth_color_extrinsics_, sizeof(Transform), cudaMemcpyHostToDevice);

  }


  //* END NEW CODE

  // Copy image to GPU device
  size_t mono8_image_size = left_mono8->image.total() * sizeof(uint8_t);
  cudaMemcpyAsync(d_im0_, left_mono8->image.ptr<uint8_t>(), 
    mono8_image_size, cudaMemcpyHostToDevice, stream1_);
  cudaMemcpyAsync(d_im1_, right_mono8->image.ptr<uint8_t>(), 
    mono8_image_size, cudaMemcpyHostToDevice, stream1_);
  //* NEW CODE
  size_t color_image_size = color_rgb8->image.total() * color_rgb8->image.channels() * sizeof(uint8_t);
  cudaMemcpyAsync(d_color_in_, color_rgb8->image.ptr<uint8_t>(), 
    color_image_size, cudaMemcpyHostToDevice, stream1_);
  //* END NEW CODE
  dim3 block_size;
  block_size.x = 32;
  block_size.y = 32;


  dim3 grid_size;
  grid_size.x = (cols_ + block_size.x-1) / block_size.x;
  grid_size.y = (rows_ + block_size.y-1) / block_size.y;

  //* NEW CODE
  dim3 grid_size_color;
  grid_size_color.x = (color_cols_ + block_size.x-1) / block_size.x;
  grid_size_color.y = (color_rows_ + block_size.y-1) / block_size.y;
  //* END NEW CODE

  CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0, stream1_>>>(d_im0_, d_im1_, d_transform0_, d_transform1_, rows_, cols_);

  cudaStreamSynchronize(stream1_);
  HammingDistanceCostKernel<<<rows_, MAX_DISPARITY, 0, stream1_>>>(d_transform0_, d_transform1_, d_cost_, rows_, cols_);

  const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;
  const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;

  // Cost Aggregation
  CostAggregationKernelLeftToRight<<<(rows_+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream2_>>>(d_cost_, d_L0_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelRightToLeft<<<(rows_+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream3_>>>(d_cost_, d_L1_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelUpToDown<<<(cols_+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1_>>>(d_cost_, d_L2_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDownToUp<<<(cols_+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1_>>>(d_cost_, d_L3_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalDownUpLeftRight<<<(cols_+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1_>>>(d_cost_, d_L4_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalUpDownLeftRight<<<(cols_+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1_>>>(d_cost_, d_L5_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalDownUpRightLeft<<<(cols_+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1_>>>(d_cost_, d_L6_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);
  CostAggregationKernelDiagonalUpDownRightLeft<<<(cols_+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1_>>>(d_cost_, d_L7_, d_s_, P1_, P2_, rows_, cols_, d_transform0_, d_transform1_, d_disparity_, d_L0_, d_L1_, d_L2_, d_L3_, d_L4_, d_L5_, d_L6_);

  // Left-right consistency check
  int total_pixel = rows_ * cols_;
  ChooseRightDisparity<<<grid_size, block_size, 0, stream1_>>>(d_disparity_right_, d_s_, rows_, cols_);
  LeftRightConsistencyCheck<<<grid_size, block_size, 0, stream1_>>>(d_disparity_, d_disparity_right_, rows_, cols_);
  
  MedianFilter3x3<<<(total_pixel+MAX_DISPARITY-1)/MAX_DISPARITY, MAX_DISPARITY, 0, stream1_>>>(d_disparity_, d_disparity_filtered_uchar_, rows_, cols_);
  
  // Disparity to Depth

  disparityToDepth<<<grid_size, block_size, 0, stream1_>>>(d_disparity_filtered_uchar_, d_depth_, Tx_, rows_, cols_, delta_cx_);
  
  //* NEW CODE
  int total_color_pixel = color_cols_ * color_rows_;
  int2 *d_pixel_map_;
  cudaMalloc((void **)&d_pixel_map_, sizeof(int2) * total_pixel * 2);
  // use this in intrinsics parameters
  // Align depth to color

  dim3 mapping_blocks(grid_size.x, grid_size.y, 2);

  kernel_map_depth_to_other<<<mapping_blocks, block_size, 0, stream1_>>>(d_pixel_map_, d_depth_, d_depth_intrinsics_, d_color_intrinsics_, d_depth_color_extrinsics_);
  kernel_other_to_depth<<<grid_size, block_size, 0, stream1_>>>(d_aligned_color_to_depth_, d_color_in_, d_pixel_map_, d_depth_intrinsics_, d_color_intrinsics_);
  kernel_depth_to_other<<<grid_size, block_size, 0, stream1_>>>(d_aligned_depth_to_color_, d_depth_, d_pixel_map_, d_depth_intrinsics_, d_color_intrinsics_);
  kernel_replace_to_zero<<<grid_size_color, block_size, 0, stream1_>>>(d_aligned_depth_to_color_, d_color_intrinsics_);
  
  cudaFree(d_pixel_map_);
  //* END NEW CODE

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    RCLCPP_ERROR(private_logger_, "%s %d\n", cudaGetErrorString(err), err);
    return false;
  }

  cudaDeviceSynchronize();
  //copy disparity
  cv::Mat disparity(rows_, cols_, CV_8UC1);
  cudaMemcpy(disparity.data, d_disparity_filtered_uchar_, sizeof(uint8_t)*total_pixel, cudaMemcpyDeviceToHost);
  // copy depth
  cv::Mat depth(rows_, cols_, CV_32FC1);
  cudaMemcpy(depth.data, d_depth_, sizeof(float)*total_pixel, cudaMemcpyDeviceToHost);


  //* NEW CODE
  // copy depth aligned
  cv::Mat depth_aligned_to_color(color_rows_, color_cols_, CV_32FC1);
  cudaMemcpy(depth_aligned_to_color.data, d_aligned_depth_to_color_, sizeof(float)*total_color_pixel, cudaMemcpyDeviceToHost);
  // copy color aligned
  cv::Mat color_aligned_to_depth(rows_, cols_, CV_8UC3);
  cudaMemcpy(color_aligned_to_depth.data, d_aligned_color_to_depth_, sizeof(uint8_t)*total_pixel*3, cudaMemcpyDeviceToHost);

  //* END NEW CODE
  
  
  
  // Restore image size if resized to be divisible by 4
  if (cols_ != left_image.width || rows_ != left_image.height)
  {
    cv::Size input_size(left_image.width, left_image.height);
    cv::resize(disparity, disparity, input_size, 0, 0, cv::INTER_AREA);
    cv::resize(depth, depth, input_size, 0, 0, cv::INTER_AREA);
    //* NEW CODE
    cv::resize(color_aligned_to_depth, color_aligned_to_depth, input_size, 0, 0, cv::INTER_AREA);
    //* END NEW CODE
  }
  // Restore image size if resized to be divisible by 4

  //* NEW CODE
  if (color_cols_ != color_image.width || color_rows_ != color_image.height)
  {
    RCLCPP_WARN(private_logger_, "Resizing aligned images");

    cv::Size input_size(color_image.width, color_image.height);
    cv::resize(depth_aligned_to_color, depth_aligned_to_color, input_size, 0, 0, cv::INTER_AREA);
  }
  //* END NEW CODE 
  convertToMsg(disparity, left_camera_info, right_camera_info, disparity_msg);
  convertToDepthMsg(depth, left_camera_info, depth_msg);
  //* NEW CODE
  convertToColorMsg(color_aligned_to_depth, left_camera_info, aligned_color_to_depth_image);
  convertToDepthMsg(depth_aligned_to_color, color_camera_info, aligned_depth_to_color_image);

  return true;
}

void SgmGpu::resizeToDivisibleBy4(cv::Mat& left_image, cv::Mat& right_image)
{
  bool need_resize = false;
  cv::Size original_size, resized_size; 

  original_size = cv::Size(left_image.cols, left_image.rows);
  resized_size = original_size;
  if (original_size.width % 4 != 0)
  {
    need_resize = true;
    resized_size.width = (original_size.width / 4 + 1) * 4;
  }
  if (original_size.height % 4 != 0)
  {
    need_resize = true;
    resized_size.height = (original_size.height / 4 + 1) * 4;
  }

  if (need_resize)
  {
    cv::resize(left_image, left_image, resized_size, 0, 0, cv::INTER_LINEAR);
    cv::resize(right_image, right_image, resized_size, 0, 0, cv::INTER_LINEAR);
  }
}

void SgmGpu::resizeToDivisibleBy4(cv::Mat& color_image)
{
  bool need_resize = false;
  cv::Size original_size, resized_size; 

  original_size = cv::Size(color_image.cols, color_image.rows);
  resized_size = original_size;
  if (original_size.width % 4 != 0)
  {
    need_resize = true;
    resized_size.width = (original_size.width / 4 + 1) * 4;
  }
  if (original_size.height % 4 != 0)
  {
    need_resize = true;
    resized_size.height = (original_size.height / 4 + 1) * 4;
  }

  if (need_resize)
  {
    cv::resize(color_image, color_image, resized_size, 0, 0, cv::INTER_LINEAR);
  }
}


bool SgmGpu::setTransform(const tf2::Transform& depth_to_color){
  // Set transforms
  fromTransform(depth_to_color, h_depth_color_extrinsics_);
  transform_set_ = true;
  return true;
}

bool SgmGpu::setIntrinsics(const sensor_msgs::msg::CameraInfo& camera_info, camera_intrinsics& intrinsics, uint32_t cols, uint32_t rows){
  // Set intrinsics
  fromCameraInfo(camera_info, intrinsics);
  if(cols == camera_info.width && rows == camera_info.height){
    return true;
  }
  intrinsics.width = cols;
  intrinsics.height = rows;
  intrinsics.fx = intrinsics.fx * (static_cast<float>(cols) / static_cast<float>(camera_info.width));
  intrinsics.fy = intrinsics.fy * (static_cast<float>(rows) / static_cast<float>(camera_info.height));
  intrinsics.ppx = intrinsics.ppx * (static_cast<float>(cols) / static_cast<float>(camera_info.width));
  intrinsics.ppy = intrinsics.ppy * (static_cast<float>(rows) / static_cast<float>(camera_info.height));
  return true;
}

void SgmGpu::convertToMsg(
  const cv::Mat_<unsigned char>& disparity, 
  const sensor_msgs::msg::CameraInfo& left_camera_info,
  const sensor_msgs::msg::CameraInfo& right_camera_info,
  stereo_msgs::msg::DisparityImage& disparity_msg
)
{
  cv::Mat disparity_32f;
  disparity.convertTo(disparity_32f, CV_32F);
  cv_bridge::CvImage disparity_converter(
    left_camera_info.header, 
    sensor_msgs::image_encodings::TYPE_32FC1, 
    disparity_32f
  );
  disparity_converter.toImageMsg(disparity_msg.image);

  disparity_msg.header = left_camera_info.header;

  disparity_msg.f = stereo_model_.left().fx();
  disparity_msg.t = stereo_model_.baseline();

  disparity_msg.min_disparity = 0.0;
  disparity_msg.max_disparity = MAX_DISPARITY;
  disparity_msg.delta_d = 1.0;
}

void SgmGpu::convertToDepthMsg(const cv::Mat_<float>& depth, 
  const sensor_msgs::msg::CameraInfo& camera_info,
  sensor_msgs::msg::Image& depth_msg){

  cv_bridge::CvImage depth_converter(
    camera_info.header, 
    sensor_msgs::image_encodings::TYPE_32FC1, 
    depth
  );
  depth_converter.toImageMsg(depth_msg);
  }

void SgmGpu::convertToColorMsg(const cv::Mat_<cv::Vec3b>& color, 
  const sensor_msgs::msg::CameraInfo& camera_info,
  sensor_msgs::msg::Image& color_msg){
  cv_bridge::CvImage color_converter(
    camera_info.header, 
    sensor_msgs::image_encodings::RGB8, 
    color
  );
  color_converter.toImageMsg(color_msg);
  }

} // namespace sgm_gpu
