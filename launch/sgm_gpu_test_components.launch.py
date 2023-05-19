# Copyright 2020 Hironori Fujimoto
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    this_pkg_name = "sgm_gpu"
    package_dir = get_package_share_directory(this_pkg_name)
    configured_params = os.path.join(
        package_dir, 'config', 'sensors_sim.yaml')

    container_sgm = ComposableNodeContainer(
        name='sgm_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package=this_pkg_name,
                plugin='sgm_gpu::SgmGpuNode',
                name='sgm_gpu_node',
                remappings=[
                    ('left_image', '/test_stereo_publisher/left/image_raw'),
                    ('right_image', '/test_stereo_publisher/right/image_raw'),
                    ('left_camera_info', '/test_stereo_publisher/left/camera_info'),
                    ('right_camera_info', '/test_stereo_publisher/right/camera_info')
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='stereo_image_proc',
                plugin='stereo_image_proc::PointCloudNode',
                name='pointcloud_node',
                remappings=[
                    ('left/image_rect_color', '/test_stereo_publisher/left/image_raw'),
                    ('left/camera_info', '/test_stereo_publisher/left/camera_info'),
                    ('right/camera_info', '/test_stereo_publisher/right/camera_info'),
                    ('disparity', '/sgm_gpu/disparity'),
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            # ComposableNode(
            #     package='image_view',
            #     plugin='image_view::DisparityViewNode',
            #     name='disparity_view',
            #     remappings=[('image', '/sgm_gpu/disparity')],
            #     parameters=[{'window_name': 'Disparity'}],
            #     extra_arguments=[{'use_intra_process_comms': True}],
            # ),
            ComposableNode(
                package='image_view',
                plugin='image_view::StereoViewNode',
                name='stereo_view',
                remappings=[('stereo/left/image', '/test_stereo_publisher/left/image_raw'),
                            ('stereo/right/image',
                             '/test_stereo_publisher/right/image_raw'),
                            ('stereo/disparity', '/sgm_gpu/disparity'), ],
                parameters=[{'window_name': 'Stereo'}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            # ComposableNode(
            #     package='image_view',
            #     plugin='image_view::ImageViewNode',
            #     name='right_view',
            #     remappings=[
            #         ('image', '/test_stereo_publisher/right/image_raw')],
            #     parameters=[{'window_name': 'Right image'}],
            #     extra_arguments=[{'use_intra_process_comms': True}],
            # ),
            # ComposableNode(
            #     package='image_view',
            #     plugin='image_view::ImageViewNode',
            #     name='left_view',
            #     remappings=[
            #         ('image', '/test_stereo_publisher/left/image_raw')],
            #     parameters=[{'window_name': 'Left image'}],
            #     extra_arguments=[{'use_intra_process_comms': True}],
            # ),
        ],
        output='screen',
    )

    return LaunchDescription([

        Node(
            package='sgm_gpu',
            executable='test_stereo_publisher',
            name='test_stereo_publisher'
        ),
        container_sgm,
    ])
