# CMake>=3.8 supports CUDA C++ as intrinsically supported language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(sgm_gpu CUDA CXX)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# FLAGS for CUDA
set(CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -O3 -lineinfo
  -gencode=arch=compute_30,code=sm_30
  -gencode=arch=compute_35,code=sm_35
  -gencode=arch=compute_50,code=sm_50
  -gencode=arch=compute_52,code=sm_52
  -gencode=arch=compute_61,code=sm_61
)

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_geometry REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)

find_package(OpenCV REQUIRED)

set(dependencies 
  cv_bridge
  image_geometry
  image_transport
  message_filters
  rclcpp
  rclcpp_components
  sensor_msgs
  stereo_msgs
  OpenCV
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(include)

# libsgm_gpu
add_library(${PROJECT_NAME} SHARED
  src/sgm_gpu.cu
  src/costs.cu
  src/hamming_cost.cu
  src/left_right_consistency.cu
  src/median_filter.cu
)
ament_target_dependencies(${PROJECT_NAME} ${dependencies})

# A component: sgm_gpu::SgmGpuNode
add_library(${PROJECT_NAME}_component SHARED src/${PROJECT_NAME}_node.cpp)
ament_target_dependencies(${PROJECT_NAME}_component ${dependencies})
target_link_libraries(${PROJECT_NAME}_component ${PROJECT_NAME})
rclcpp_components_register_node(${PROJECT_NAME}_component PLUGIN "sgm_gpu::SgmGpuNode" EXECUTABLE ${PROJECT_NAME}_node)
# Standalone node version of sgm_gpu::SgmGpuNode


install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_component 
  LIBRARY DESTINATION lib)
install(TARGETS ${PROJECT_NAME}_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY test_data launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/test_stereo_publisher DESTINATION lib/${PROJECT_NAME})

ament_package()
