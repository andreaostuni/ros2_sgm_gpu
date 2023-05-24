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
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    this_pkg_name = "sgm_gpu"

    container_sgm = ComposableNodeContainer(
        name='sgm_container',
        namespace='camera_stereo',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package=this_pkg_name,
                plugin='sgm_gpu::SgmGpuNode',
                namespace='camera_stereo',
                name='sgm_gpu_node',
                remappings=[
                    ('left_image', 'left/image_rect'),
                    ('right_image', 'right/image_rect'),
                    ('left_camera_info', 'left/camera_info'),
                    ('right_camera_info', 'right/camera_info')
                ],
                parameters=[{'use_sim_time': True}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='image_proc',
                plugin='image_proc::RectifyNode',
                namespace='camera_stereo/left',
                name='left_rectify_node',
                remappings=[
                    ('image', 'image_raw'),
                ],
                parameters=[{'use_sim_time': True}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='image_proc',
                plugin='image_proc::RectifyNode',
                namespace='camera_stereo/right',
                name='right_rectify_node',
                remappings=[
                     ('image', 'image_raw'),
                ],
                parameters=[{'use_sim_time': True}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='stereo_image_proc',
                plugin='stereo_image_proc::PointCloudNode',
                namespace='camera_stereo',
                name='pointcloud_node',
                remappings=[
                    ('left/image_rect_color', 'left/image_raw'),
                ],
                parameters=[{'use_sim_time': True}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),

            ComposableNode(
                package='image_view',
                plugin='image_view::StereoViewNode',
                name='stereo_view',
                namespace='camera_stereo',
                remappings=[('stereo/left/camera_stereo/image', 'left/image_raw'),
                            ('stereo/right/camera_stereo/image',
                             'right/image_raw'),
                            ('stereo/disparity', 'disparity'), ],
                parameters=[{'use_sim_time': True}, {'window_name': 'Stereo'}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='image_view',
                plugin='image_view::ImageViewNode',
                name='depth_view',
                namespace='camera_stereo',
                remappings=[('image', 'depth')],
                parameters=[{'use_sim_time': True}, {'window_name': 'Depth'}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],

        output='screen',
    )

    return LaunchDescription([
        container_sgm,
    ])
