"""MoveIt and robot TF only; this launch never starts the hardware driver."""
import importlib.util
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    profile = Path(LaunchConfiguration("profile").perform(context))
    helper = Path(__file__).resolve().parents[1] / "joint1_moveit.py"
    spec = importlib.util.spec_from_file_location("joint1_moveit_launch_helper", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parameters = module.moveit_parameters(profile)
    topic = "/" + module.NAMESPACE + "/joint_states"
    return [
        Node(package="tf2_ros", executable="static_transform_publisher",
             arguments=["0", "0", "0", "0", "0", "0", "world", "base_link"]),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": parameters["robot_description"]}],
             remappings=[("/joint_states", topic)]),
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             parameters=[parameters], remappings=[("/joint_states", topic)]),
    ]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("profile"), OpaqueFunction(function=setup)])
