import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pinocchio as pin
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint1_moveit as trial


class Joint1MoveItTests(unittest.TestCase):
    def profile(self):
        return dict(lower_rad=-0.98, upper_rad=0.56, goal_margin_rad=0.05)

    def trajectory(self):
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = ["joint1"]
        trajectory.joint_trajectory.points = [
            JointTrajectoryPoint(positions=[0.4], velocities=[0.0]),
            JointTrajectoryPoint(positions=[0.3], velocities=[0.0]),
        ]
        trajectory.joint_trajectory.points[-1].time_from_start.sec = 6
        return trajectory

    def test_frozen_model_preserves_every_link_pose(self):
        source = (trial.ROOT / "rebotarm_ros2/src/rebotarm_bringup/description/RS/urdf/ReBot_Arm_RS.urdf").read_text()
        pose = dict(joint1=0.5, joint2=-0.004, joint3=0.003,
                    joint4=0.001, joint5=0.017, joint6=0.076, gripper=0.0007)
        original = pin.buildModelFromXML(source)
        frozen = pin.buildModelFromXML(trial.freeze_description(source, pose, -0.98, 0.56))
        self.assertEqual(frozen.nq, 1)
        for base in (-0.9, 0.0, 0.5):
            q = pin.neutral(original)
            for name, value in pose.items():
                if name != "gripper":
                    q[original.joints[original.getJointId(name)].idx_q] = base if name == "joint1" else value
            for name, limit in (("gripper_joint1", 0.05), ("gripper_joint2", 0.0715)):
                q[original.joints[original.getJointId(name)].idx_q] = pose["gripper"] / 2.7 * limit
            a, b = original.createData(), frozen.createData()
            pin.framesForwardKinematics(original, a, q)
            pin.framesForwardKinematics(frozen, b, np.array([base]))
            for frame in original.frames:
                if frame.type == pin.FrameType.BODY:
                    np.testing.assert_allclose(a.oMf[original.getFrameId(frame.name)].homogeneous,
                                               b.oMf[frozen.getFrameId(frame.name)].homogeneous, atol=1e-12)

    def test_request_rejects_edge_goal_large_step_and_nonfinite_start(self):
        for current, target in ((0.5, 0.55), (0.5, -0.5), (float("nan"), 0.3)):
            with self.subTest(current=current, target=target), self.assertRaises(ValueError):
                trial.plan_request(current, target, self.profile())
        request = trial.plan_request(0.4, 0.3, self.profile()).motion_plan_request
        self.assertEqual(request.group_name, "base_rotation")
        self.assertEqual(list(request.start_state.joint_state.name), ["joint1"])

    def test_trajectory_rejects_nonfinite_outside_bounds_other_axis_or_fast_motion(self):
        changes = (
            lambda t: setattr(t.joint_trajectory.points[0], "positions", [float("nan")]),
            lambda t: setattr(t.joint_trajectory.points[-1], "positions", [-0.99]),
            lambda t: setattr(t.joint_trajectory, "joint_names", ["joint2"]),
            lambda t: setattr(t.joint_trajectory.points[-1].time_from_start, "sec", 1),
            lambda t: setattr(t.joint_trajectory.points[0], "velocities", [0.021]),
            lambda t: setattr(t.joint_trajectory.points[0], "velocities", [0.0, 0.0]),
            lambda t: setattr(t.joint_trajectory.points[-1].time_from_start, "sec", 0),
        )
        for change in changes:
            t = self.trajectory()
            change(t)
            with self.assertRaises(ValueError):
                trial.validate_trajectory(t, 0.4, 0.3, self.profile())

    def test_valid_trajectory_and_start_end_mismatches(self):
        t = self.trajectory()
        self.assertEqual(trial.validate_trajectory(t, 0.4, 0.3, self.profile()), 6)
        for current, target in ((0.42, 0.3), (0.4, 0.32)):
            with self.assertRaises(ValueError):
                trial.validate_trajectory(t, current, target, self.profile())

    def test_offline_fake_controller_refuses_real_hardware_dds_domain(self):
        with patch.dict(os.environ, ROS_DOMAIN_ID="42"), self.assertRaises(RuntimeError):
            trial.offline_probe(Path("missing"), Path("unused"))


if __name__ == "__main__":
    unittest.main()
