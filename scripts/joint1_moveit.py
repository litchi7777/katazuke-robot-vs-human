#!/usr/bin/env python3
"""Prepare a measured, single-axis MoveIt model, or probe it on DDS domain 43.

No CAN or motor APIs. Other joints are rigidly frozen at the measured pose.
The offline probe includes a synthetic controller and refuses domain 42.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
GROUP = "base_rotation"
NAMESPACE = "rebotarm_joint1_trial"
MAX_SPEED = 0.02
GOAL_MARGIN = 0.05


def freeze_description(source, pose, lower, upper):
    import numpy as np
    import pinocchio as pin
    tree = ET.fromstring(source)
    if not -2.8 <= lower < 0 < upper <= 2.8:
        raise ValueError("measured joint1 limits must straddle zero inside URDF limits")
    required = {f"joint{i}" for i in range(1, 7)} | {"gripper"}
    if set(pose) != required or not all(math.isfinite(x) for x in pose.values()):
        raise ValueError("all seven finite measured positions required")
    if any(abs(pose[f"joint{i}"]) > 0.1 for i in range(2, 7)):
        raise ValueError("other axes must be in the supported folded pose")
    fingers = min(1.0, max(0.0, pose["gripper"] / 2.7))
    frozen = {f"joint{i}": pose[f"joint{i}"] for i in range(2, 7)}
    frozen.update(gripper_joint1=fingers * 0.05, gripper_joint2=fingers * 0.0715)
    seen = set()
    for joint in tree.findall("joint"):
        name = joint.attrib["name"]
        if name == "joint1":
            limit = joint.find("limit")
            limit.set("lower", str(lower))
            limit.set("upper", str(upper))
            limit.set("velocity", "0.1")  # request scaling 0.2 -> 0.02 rad/s
        elif joint.attrib["type"] != "fixed":
            if name not in frozen:
                raise ValueError(f"unrecognized movable joint: {name}")
            seen.add(name)
            origin = joint.find("origin")
            if origin is None:
                origin = ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
            axis /= np.linalg.norm(axis)
            rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
            rotation = pin.rpy.rpyToMatrix(rpy)
            if joint.attrib["type"] in ("revolute", "continuous"):
                origin.set("rpy", " ".join(map(str, pin.rpy.matrixToRpy(
                    rotation @ pin.exp3(axis * frozen[name])))))
            elif joint.attrib["type"] == "prismatic":
                xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
                origin.set("xyz", " ".join(map(str, xyz + rotation @ axis * frozen[name])))
            else:
                raise ValueError("unsupported joint type")
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "mimic", "dynamics", "safety_controller", "calibration"):
                for child in joint.findall(tag):
                    joint.remove(child)
    if seen != set(frozen):
        raise ValueError("expected all five arm joints and two fingers")
    return ET.tostring(tree, encoding="unicode")


def prepare(range_path, output_dir):
    data = json.loads(range_path.read_text())
    lower, upper = data["operator_selected_interval_rad"]
    pose = data["endpoint_B"]["pose_median_rad"]
    source = ROOT / "rebotarm_ros2/src/rebotarm_bringup/description/RS/urdf/ReBot_Arm_RS.urdf"
    description = freeze_description(source.read_text(), pose, lower, upper)
    srdf_source = ROOT / "rebotarm_ros2/src/rebotarm_moveit_config/config/rebotarm_rs.srdf"
    srdf = ET.Element("robot", name=ET.fromstring(description).get("name"))
    group = ET.SubElement(srdf, "group", name=GROUP)
    ET.SubElement(group, "joint", name="joint1")
    ET.SubElement(srdf, "virtual_joint", name="FixedBase", type="fixed",
                  parent_frame="world", child_link="base_link")
    links = {e.get("name") for e in ET.fromstring(description).findall("link")}
    for item in ET.parse(srdf_source).getroot().findall("disable_collisions"):
        if item.get("link1") in links and item.get("link2") in links:
            srdf.append(copy.deepcopy(item))
    output_dir.mkdir(exist_ok=False)
    (output_dir / "robot.urdf").write_text(description)
    (output_dir / "robot.srdf").write_text(ET.tostring(srdf, encoding="unicode"))
    profile = dict(group=GROUP, driver_namespace=NAMESPACE,
                   lower_rad=lower, upper_rad=upper, goal_margin_rad=GOAL_MARGIN,
                   max_commanded_speed_rad_s=MAX_SPEED, frozen_motor_pose_rad=pose,
                   initial_joint1_rad=min(upper, max(lower, pose["joint1"])),
                   planned_targets_rad=[0.45, 0.30, 0.15, 0.0],
                   range_source=str(range_path.resolve()),
                   note="Only joint1 moves in this model; other axes are rigid at the measured folded pose")
    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    import yaml
    hardware = yaml.safe_load((ROOT / "scripts/config/joint1_range_trial.yaml").read_text())
    hardware["models"]["rs"]["overrides"]["joint1_trial_position_limits"] = [lower, upper]
    (output_dir / "hardware.yaml").write_text(yaml.safe_dump(hardware, sort_keys=False))
    return profile


def load_profile(path):
    p = json.loads(path.read_text())
    lo, hi = p["lower_rad"], p["upper_rad"]
    if (p["group"] != GROUP or p["driver_namespace"] != NAMESPACE
            or not -2.8 <= lo < 0 < hi <= 2.8
            or p["max_commanded_speed_rad_s"] != MAX_SPEED
            or p["goal_margin_rad"] < GOAL_MARGIN):
        raise ValueError("invalid isolated MoveIt profile")
    return p


def plan_request(current, target, profile):
    from moveit_msgs.msg import Constraints, JointConstraint
    from moveit_msgs.srv import GetMotionPlan
    lo, hi = profile["lower_rad"], profile["upper_rad"]
    margin = profile["goal_margin_rad"]
    if not all(map(math.isfinite, (current, target))):
        raise ValueError("finite start and target required")
    if not lo - 0.002 <= current <= hi + 0.002:
        raise ValueError("measured start outside the selected interval")
    if not lo + margin <= target <= hi - margin:
        raise ValueError("goal must stay 0.05 rad inside both measured endpoints")
    if abs(target - current) > 0.20:
        raise ValueError("MoveIt trial leg must be at most 0.20 rad")
    request = GetMotionPlan.Request()
    req = request.motion_plan_request
    req.group_name = GROUP
    req.pipeline_id = "ompl"
    req.planner_id = "RRTConnectkConfigDefault"
    req.allowed_planning_time = 5.0
    req.num_planning_attempts = 3
    req.max_velocity_scaling_factor = 0.2
    req.max_acceleration_scaling_factor = 0.2
    req.start_state.joint_state.name = ["joint1"]
    req.start_state.joint_state.position = [min(hi, max(lo, current))]
    req.goal_constraints = [Constraints(joint_constraints=[JointConstraint(
        joint_name="joint1", position=target, tolerance_above=0.0001,
        tolerance_below=0.0001, weight=1.0)])]
    return request


def validate_trajectory(trajectory, current, target, profile):
    joint = trajectory.joint_trajectory
    if list(joint.joint_names) != ["joint1"] or len(joint.points) < 2:
        raise ValueError("planned trajectory must contain only joint1 and at least two points")
    if trajectory.multi_dof_joint_trajectory.points:
        raise ValueError("multi-DOF motion not allowed")
    previous = None
    for point in joint.points:
        if len(point.positions) != 1:
            raise ValueError("wrong position length")
        if len(point.velocities) not in (0, 1):
            raise ValueError("wrong velocity length")
        q = point.positions[0]
        seconds = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
        if not math.isfinite(q) or not profile["lower_rad"] <= q <= profile["upper_rad"]:
            raise ValueError("planned position outside measured bounds")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid planned timestamp")
        if any(not math.isfinite(v) or abs(v) > MAX_SPEED + 1e-6 for v in point.velocities):
            raise ValueError("planned velocity exceeds 0.02 rad/s")
        if previous is not None:
            dt = seconds - previous[0]
            if dt <= 0 or abs(q - previous[1]) / dt > MAX_SPEED + 1e-5:
                raise ValueError("planned times/slope violate speed bound")
        previous = seconds, q
    if abs(joint.points[0].positions[0] - current) > 0.005:
        raise ValueError("planned start does not match measured joint1")
    if abs(joint.points[-1].positions[0] - target) > 0.001:
        raise ValueError("planned endpoint does not match goal")
    return previous[0]


def moveit_parameters(profile_path):
    p = load_profile(profile_path)
    return {
        "robot_description": (profile_path.parent / "robot.urdf").read_text(),
        "robot_description_semantic": (profile_path.parent / "robot.srdf").read_text(),
        "robot_description_kinematics": {},
        "robot_description_planning": {"joint_limits": {"joint1": {
            "has_velocity_limits": True, "max_velocity": 0.1,
            "has_acceleration_limits": True, "max_acceleration": 0.05}}},
        "planning_pipelines": ["ompl"], "default_planning_pipeline": "ompl",
        "ompl": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": " ".join([
                "default_planner_request_adapters/AddTimeOptimalParameterization",
                "default_planner_request_adapters/ResolveConstraintFrames",
                "default_planner_request_adapters/FixWorkspaceBounds",
                "default_planner_request_adapters/FixStartStateBounds",
                "default_planner_request_adapters/FixStartStateCollision"]),
            "start_state_max_bounds_error": 0.002,
            "planner_configs": {"RRTConnectkConfigDefault": {"type": "geometric::RRTConnect", "range": 0.05}},
            GROUP: {"planner_configs": ["RRTConnectkConfigDefault"],
                    "longest_valid_segment_fraction": 0.005}},
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {
            "controller_names": [p["driver_namespace"]],
            p["driver_namespace"]: {"type": "FollowJointTrajectory", "action_ns": "follow_joint_trajectory",
                                    "default": True, "joints": ["joint1"]}},
        "trajectory_execution": {"allowed_execution_duration_scaling": 1.5,
                                 "allowed_goal_duration_margin": 3.0,
                                 "allowed_start_tolerance": 0.02,
                                 "execution_duration_monitoring": True},
        "publish_robot_description": True, "publish_robot_description_semantic": True,
        "publish_planning_scene": True, "publish_geometry_updates": True,
        "publish_state_updates": True, "publish_transforms_updates": True,
    }


class MoveItStopper:
    """Stop the execution manager and cancel this isolated controller directly.

    Humble ExecuteTrajectory cancellation alone may leave execution running.
    Keep retrying until both MoveIt and the dedicated driver report termination.
    """
    def __init__(self, node, emit):
        from action_msgs.srv import CancelGoal
        from std_msgs.msg import String
        self.publisher = node.create_publisher(String, "/trajectory_execution_event", 10)
        self.cancel_client = node.create_client(CancelGoal, f"/{NAMESPACE}/follow_joint_trajectory/_action/cancel_goal")
        self.emit = emit
        self.next_send = 0.0
        self.pending = None

    def ready(self):
        return self.publisher.get_subscription_count() > 0 and self.cancel_client.service_is_ready()

    def request(self):
        from action_msgs.srv import CancelGoal
        from std_msgs.msg import String
        if time.monotonic() < self.next_send:
            return
        self.next_send = time.monotonic() + 0.2
        self.publisher.publish(String(data="stop"))
        self.emit("moveit_stop_event")
        if self.cancel_client.service_is_ready() and (self.pending is None or self.pending.done()):
            # Zero goal ID/stamp cancels every goal on this trial-only server.
            self.pending = self.cancel_client.call_async(CancelGoal.Request())
            self.pending.add_done_callback(self._reply)
            self.emit("driver_cancel_all_sent")

    def _reply(self, future):
        try:
            reply = future.result()
            self.emit("driver_cancel_all_reply", return_code=reply.return_code,
                      goals_canceling=len(reply.goals_canceling))
        except Exception as error:
            self.emit("driver_cancel_all_error", error=str(error))


def offline_probe(profile_path, output_dir):
    if os.environ.get("ROS_DOMAIN_ID") != "43":
        raise RuntimeError("offline synthetic controller requires isolated ROS_DOMAIN_ID=43")
    import rclpy
    from rclpy.action import ActionClient, ActionServer, CancelResponse
    from rclpy.node import Node
    from rclpy.task import Future
    from control_msgs.action import FollowJointTrajectory
    from moveit_msgs.action import ExecuteTrajectory
    from moveit_msgs.msg import RobotState
    from moveit_msgs.srv import GetMotionPlan, GetStateValidity
    from rosidl_runtime_py.convert import message_to_ordereddict
    from sensor_msgs.msg import JointState
    from joint1_range_trial import TrackedMoveItGoal
    profile = load_profile(profile_path)
    output_dir.mkdir(exist_ok=False)
    rclpy.init()
    node = Node("joint1_moveit_offline_probe")
    current = [profile["initial_joint1_rad"]]
    published = node.create_publisher(JointState, f"/{NAMESPACE}/joint_states", 10)
    def publish():
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name, msg.position = ["joint1"], list(current)
        published.publish(msg)
    timer = node.create_timer(0.02, publish)
    cancellation = {"testing": False, "started": False, "controller_canceled": False}
    async def execute(handle):
        # Tests controller routing only, with synthetic instantaneous motion.
        if list(handle.request.trajectory.joint_names) != ["joint1"]:
            handle.abort()
            return FollowJointTrajectory.Result(error_code=-2)
        if cancellation["testing"]:
            cancellation["started"] = True
            future = Future()
            def poll_cancel():
                if handle.is_cancel_requested and not future.done():
                    future.set_result(True)
            cancel_timer = node.create_timer(0.02, poll_cancel)
            await future
            node.destroy_timer(cancel_timer)
            handle.canceled()
            cancellation["controller_canceled"] = True
            return FollowJointTrajectory.Result(error_code=0)
        current[0] = handle.request.trajectory.points[-1].positions[0]
        publish()
        handle.succeed()
        return FollowJointTrajectory.Result(error_code=0)
    server = ActionServer(node, FollowJointTrajectory, f"/{NAMESPACE}/follow_joint_trajectory", execute,
                          cancel_callback=lambda _: CancelResponse.ACCEPT)
    planner = node.create_client(GetMotionPlan, "/plan_kinematic_path")
    validity = node.create_client(GetStateValidity, "/check_state_validity")
    executor = ActionClient(node, ExecuteTrajectory, "/execute_trajectory")
    def await_future(future, timeout=15):
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        if not future.done():
            raise TimeoutError("offline probe timeout")
        return future.result()
    result = {"mode": "offline_synthetic_controller", "ROS_DOMAIN_ID": "43", "CAN_access": False,
              "checked_states": [], "plans": []}
    try:
        if not planner.wait_for_service(timeout_sec=15) or not validity.wait_for_service(timeout_sec=15):
            raise RuntimeError("MoveIt services unavailable")
        if not executor.wait_for_server(timeout_sec=15):
            raise RuntimeError("MoveIt ExecuteTrajectory unavailable")
        until = time.monotonic() + 2
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.02)
        for i in range(21):
            q = profile["lower_rad"] + i / 20 * (profile["upper_rad"] - profile["lower_rad"])
            request = GetStateValidity.Request(group_name=GROUP)
            request.robot_state.joint_state.name = ["joint1"]
            request.robot_state.joint_state.position = [q]
            response = await_future(validity.call_async(request))
            result["checked_states"].append({"joint1_rad": q, "valid": response.valid,
                                             "contacts": [f"{c.contact_body_1}/{c.contact_body_2}" for c in response.contacts]})
            if not response.valid:
                raise RuntimeError(f"collision/state validity failed at {q}: {response.contacts}")
        for index, target in enumerate(profile["planned_targets_rad"]):
            response = await_future(planner.call_async(plan_request(current[0], target, profile)))
            response = response.motion_plan_response
            if response.error_code.val != 1:
                raise RuntimeError(f"planning failed: {response.error_code.val}")
            duration = validate_trajectory(response.trajectory, current[0], target, profile)
            (output_dir / f"trajectory_{index}.json").write_text(json.dumps(
                message_to_ordereddict(response.trajectory), indent=2) + "\n")
            handle = await_future(executor.send_goal_async(ExecuteTrajectory.Goal(trajectory=response.trajectory)))
            if not handle.accepted:
                raise RuntimeError("MoveIt rejected offline execution")
            reply = await_future(handle.get_result_async())
            if reply.status != 4 or reply.result.error_code.val != 1:
                raise RuntimeError(f"offline execution failed: {reply}")
            result["plans"].append({"target_rad": target, "duration_s": duration,
                                    "execution": "synthetic_controller_succeeded"})
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0.02)
        stop_events = []
        stopper = MoveItStopper(node, lambda event, **fields: stop_events.append(dict(event=event, **fields)))
        until = time.monotonic() + 3
        while not stopper.ready() and time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.02)
        if not stopper.ready():
            raise RuntimeError("offline stop routes unavailable")
        response = await_future(planner.call_async(plan_request(current[0], 0.10, profile))).motion_plan_response
        if response.error_code.val != 1:
            raise RuntimeError("cancel-test planning failed")
        cancellation["testing"] = True
        track = TrackedMoveItGoal(lambda event, **fields: stop_events.append(dict(event=event, **fields)), stopper.request)
        track.attach(executor.send_goal_async(ExecuteTrajectory.Goal(trajectory=response.trajectory)))
        until = time.monotonic() + 5
        while not cancellation["started"] and time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.02)
        if not cancellation["started"]:
            raise RuntimeError("offline cancel-test controller never started")
        track.cancel()
        until = time.monotonic() + 5
        while (not track.terminal or not cancellation["controller_canceled"]) and time.monotonic() < until:
            stopper.request()
            rclpy.spin_once(node, timeout_sec=0.02)
        result["cancellation_test"] = dict(cancellation, moveit_terminal=track.terminal,
                                            moveit_success=track.success, events=stop_events)
        if not track.terminal or track.success or not cancellation["controller_canceled"]:
            raise RuntimeError("stop did not terminate both MoveIt and the mock driver")
        result["status"] = "ok"
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        server.destroy()
        node.destroy_node()
        rclpy.try_shutdown()
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--range-summary", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    probe = sub.add_parser("offline-probe")
    probe.add_argument("--profile", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.range_summary, args.output_dir), indent=2))
    else:
        offline_probe(args.profile, args.output_dir)


if __name__ == "__main__":
    main()
