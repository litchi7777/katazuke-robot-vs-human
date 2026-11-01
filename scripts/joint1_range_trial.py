#!/usr/bin/env python3
"""joint1を小刻みに動かす常駐ROSクライアント。既定は接続しないdry-run。

ドライバの起動・enable・グリッパ指令は行わない。実機ドライバの起動は
全7台を有効化するため、具体的な試験内容について了承を得てから行う。
"""

import argparse
import json
import math
import select
import signal
import sys
import time
from pathlib import Path

from check_can_stability_readonly import check_link
from probe_can_readonly import link_state


JOINTS = tuple(f"joint{i}" for i in range(1, 7))
MAX_STEP = 0.05
MAX_SPEED = 0.02  # rad/s, commanded quintic trajectory peak
STALE_S = 0.5
OTHER_DRIFT = 0.04
GRIPPER_DRIFT = 0.08


def verified_preflight(path, max_age_s=300):
    summary = json.loads(path.read_text())
    age = time.time() - path.stat().st_mtime
    if (summary.get("status") != "ok" or summary.get("error_frames", -1) != 0
            or not 0 <= age <= max_age_s):
        raise ValueError(f"a successful, error-free CAN preflight from the last {max_age_s} s is required")
    positions = {item["motor_id"]: item["value"]
                 for item in summary.get("initial_readings", [])
                 if item.get("parameter") == "0x7019" and item.get("status") == "ok"}
    if set(positions) != set(range(1, 8)) or not all(map(math.isfinite, positions.values())):
        raise ValueError("preflight must contain all seven finite motor positions")
    return summary


def others(values, index):
    return [v for i, v in enumerate(values) if i != index]


def bounded_plan(current, anchor, offset, lower, upper, index=0):
    """100ms以下の間隔で五次補間。変更する目標は index の関節だけ。"""
    values = [*current, *anchor, offset, lower, upper]
    if len(current) not in (1, 6) or len(anchor) != len(current) or not all(map(math.isfinite, values)):
        raise ValueError("one or six finite current/anchor angles and finite bounds required")
    if not 0 <= index < len(current):
        raise ValueError("moving joint index outside the reported joints")
    if not lower <= offset <= upper or lower >= upper:
        raise ValueError("offset outside this session's approved window")
    target = anchor[index] + offset
    if not -2.8 <= target <= 2.8:
        raise ValueError("target outside URDF limits")
    if abs(target - current[index]) > MAX_STEP + 1e-9:
        raise ValueError("one move may change the joint by at most 0.05 rad")
    if any(abs(a - b) > OTHER_DRIFT for a, b in zip(others(current, index), others(anchor, index))):
        raise ValueError("another joint has drifted; do not send a new target")
    duration = max(3.0, 1.875 * abs(target - current[index]) / MAX_SPEED)
    count = math.ceil(duration / 0.1)
    points = []
    for step in range(count + 1):
        phase = step / count
        ratio = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
        positions = list(anchor)
        positions[index] = current[index] + (target - current[index]) * ratio
        points.append((duration * phase, positions))
    return points


class TrackedGoal:
    """返事が遅いgoalも保持し、後から受理されたら必ずcancelを送る。"""

    def __init__(self, emit):
        self.emit = emit
        self.send_future = None
        self.handle = None
        self.result_future = None
        self.cancel_future = None
        self.cancel_requested = False
        self.terminal = False
        self.success = False

    def attach(self, future):
        self.send_future = future
        future.add_done_callback(self._accepted)

    def _accepted(self, future):
        try:
            self.handle = future.result()
            if self.handle is None or not self.handle.accepted:
                self.terminal = True
                self.emit("goal_rejected")
                return
            self.result_future = self.handle.get_result_async()
            self.result_future.add_done_callback(self._result)
            self.emit("goal_accepted")
            if self.cancel_requested:
                self.cancel()
        except Exception as error:
            # Unknown acceptance must block new goals and client exit.
            self.emit("goal_accept_error", error=str(error))

    def _result(self, future):
        try:
            reply = future.result()
            self.success = (reply.status == 4 and reply.result.error_code == 0
                            and not self.cancel_requested)
            self.terminal = True
            self.emit("goal_result", status=reply.status,
                      error_code=reply.result.error_code,
                      message=reply.result.error_string, success=self.success)
        except Exception as error:
            self.emit("goal_result_error", error=str(error))

    def cancel(self):
        self.cancel_requested = True
        if self.handle is not None and not self.terminal and self.cancel_future is None:
            try:
                self.cancel_future = self.handle.cancel_goal_async()
                self.cancel_future.add_done_callback(self._canceled)
                self.emit("cancel_sent")
            except Exception as error:
                self.emit("cancel_send_error", error=str(error))

    def _canceled(self, future):
        try:
            reply = future.result()
            self.emit("cancel_reply", return_code=reply.return_code,
                      goals_canceling=len(reply.goals_canceling))
        except Exception as error:
            self.emit("cancel_reply_error", error=str(error))
        # A cancel reply alone never means motion has stopped. Wait for result.


class TrackedMoveItGoal(TrackedGoal):
    def __init__(self, emit, stop_execution=None):
        super().__init__(emit)
        self.stop_execution = stop_execution

    def cancel(self):
        if self.stop_execution is not None:
            self.stop_execution()
        super().cancel()

    def _result(self, future):
        try:
            reply = future.result()
            self.success = (reply.status == 4 and reply.result.error_code.val == 1
                            and not self.cancel_requested)
            self.terminal = True
            self.emit("moveit_result", status=reply.status,
                      error_code=reply.result.error_code.val, success=self.success)
        except Exception as error:
            self.emit("moveit_result_error", error=str(error))


def fresh_snapshot(node):
    now = time.monotonic()
    single = getattr(node, "joint1_only", False)
    arm_hold = getattr(node, "arm_hold", False)
    with_gripper = not (single or arm_hold)
    if node.positions is None or (with_gripper and node.gripper is None):
        raise RuntimeError("waiting for configured joints and gripper (if present)")
    if now - node.joint_received > STALE_S or (with_gripper and now - node.gripper_received > STALE_S):
        raise RuntimeError("joint/gripper telemetry is stale")
    if node.status is None or not node.status.enabled or not node.status.control_loop_active:
        raise RuntimeError("driver is not enabled with an active control loop")
    if node.status.error_codes or node.status.state_machine in ("ERROR", "SAFE_HOMING", "GRAVITY_COMP"):
        raise RuntimeError("driver reports an error or incompatible state")
    if single and list(node.status.joint_names) != ["joint1"]:
        raise RuntimeError("single-axis trial requires the isolated joint1 driver configuration")
    if arm_hold and list(node.status.joint_names) != list(JOINTS):
        raise RuntimeError("arm hold trial requires the six-joint arm hold driver configuration")
    if node.invalid_message:
        raise RuntimeError(node.invalid_message)
    return list(node.positions), node.gripper if with_gripper else 0.0


def check_drift(current, gripper, anchor, gripper_anchor, lower, upper, index=0):
    if not all(map(math.isfinite, [*current, gripper])):
        raise RuntimeError("non-finite telemetry")
    if not -2.8 <= current[index] <= 2.8:
        raise RuntimeError("measured moving joint outside URDF limits")
    if not anchor[index] + lower - 0.03 <= current[index] <= anchor[index] + upper + 0.03:
        raise RuntimeError("measured moving joint left the session window + 0.03 rad tolerance")
    if any(abs(a - b) > OTHER_DRIFT for a, b in zip(others(current, index), others(anchor, index))):
        raise RuntimeError("another joint drifted by more than 0.04 rad")
    if abs(gripper - gripper_anchor) > GRIPPER_DRIFT:
        raise RuntimeError("gripper drifted by more than 0.08 rad")


def ros_node(namespace, emit, joint1_only=False, arm_hold=False):
    # Imported only for --execute; dry-run and safety tests never initialize ROS.
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from control_msgs.action import FollowJointTrajectory
    from rebotarm_msgs.msg import ArmStatus, JointMotorState
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger

    class TrialNode(Node):
        def __init__(self):
            super().__init__("joint1_range_trial")
            self.joint1_only = joint1_only
            self.arm_hold = arm_hold
            self.joint_names = ["joint1"] if joint1_only else list(JOINTS)
            self.positions = self.gripper = self.status = None
            self.joint_received = self.gripper_received = 0.0
            self.invalid_message = ""
            self.last_stamp = -1
            self.last_gripper_stamp = -1
            self.last_logged = 0.0
            self.create_subscription(JointState, f"/{namespace}/joint_states", self.on_joints,
                                     qos_profile_sensor_data)
            if not (joint1_only or arm_hold):
                self.create_subscription(JointMotorState, f"/{namespace}/gripper/state", self.on_gripper,
                                         qos_profile_sensor_data)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(ArmStatus, f"/{namespace}/arm_status", self.on_status, qos)
            self.action = ActionClient(self, FollowJointTrajectory,
                                       f"/{namespace}/follow_joint_trajectory")
            self.home = self.create_client(Trigger, f"/{namespace}/safe_home")
            self.disable = self.create_client(Trigger, f"/{namespace}/disable")
            self.hold = self.create_client(Trigger, f"/{namespace}/hold")

        def on_joints(self, msg):
            try:
                if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
                    raise ValueError("malformed joint state")
                mapping = dict(zip(msg.name, msg.position))
                positions = [mapping[name] for name in self.joint_names]
                if not all(map(math.isfinite, positions)):
                    raise ValueError("non-finite joint state")
                stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                if stamp <= self.last_stamp:
                    return  # Replayed data cannot refresh the watchdog.
                self.last_stamp = stamp
                self.positions = positions
                self.joint_received = time.monotonic()
                if self.joint_received - self.last_logged >= 0.1:
                    emit("joints", positions=positions, source_stamp_ns=stamp)
                    self.last_logged = self.joint_received
            except (KeyError, ValueError) as error:
                self.invalid_message = str(error)

        def on_gripper(self, msg):
            if not math.isfinite(msg.position):
                self.invalid_message = "non-finite gripper state"
                return
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if stamp <= self.last_gripper_stamp:
                return
            self.last_gripper_stamp = stamp
            self.gripper = msg.position
            self.gripper_received = time.monotonic()

        def on_status(self, msg):
            self.status = msg
            emit("arm_status", enabled=msg.enabled, state=msg.state_machine,
                 errors=list(msg.error_codes))

    return TrialNode()


def execute(args, emit):
    import rclpy
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from rclpy.signals import SignalHandlerOptions
    from std_srvs.srv import Trigger
    from trajectory_msgs.msg import JointTrajectoryPoint

    preflight = verified_preflight(args.preflight)
    profile = None
    if args.moveit_profile:
        from joint1_moveit import load_profile
        profile = load_profile(args.moveit_profile)
        if not args.joint1_only or args.namespace != profile["driver_namespace"]:
            raise ValueError("MoveIt trial requires the isolated joint1 driver namespace")
        measured = {f"joint{x['motor_id']}" if x['motor_id'] < 7 else "gripper": x['value']
                    for x in preflight['initial_readings'] if x['parameter'] == '0x7019'}
        if any(abs(measured[name] - value) > 0.01
               for name, value in profile["frozen_motor_pose_rad"].items() if name != "joint1"):
            raise ValueError("other axes changed from the frozen MoveIt pose; regenerate model")
    # Netlink state queries only: this client never opens a CAN transmit socket.
    before = link_state(args.channel)
    check_link(before, preflight["link_after"])
    emit("can_before", link=before)
    next_can_check = time.monotonic() + 0.5
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = ros_node(args.namespace, emit, args.joint1_only, args.arm_hold)
    k = 0 if args.joint1_only else JOINTS.index(args.move_joint)
    if profile:
        from joint1_moveit import MoveItStopper
        from rclpy.action import ActionClient
        from moveit_msgs.action import ExecuteTrajectory
        from moveit_msgs.srv import GetMotionPlan
        planner = node.create_client(GetMotionPlan, "/plan_kinematic_path")
        moveit_action = ActionClient(node, ExecuteTrajectory, "/execute_trajectory")
        stopper = MoveItStopper(node, emit)
    interrupted = []
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, lambda signum, _frame: interrupted.append(signum))
    tracker = None
    anchor = grip_anchor = None
    fault = None
    hold_needed = False
    hold_future = None
    hold_deadline = 0.0
    stdin_open = True
    last_display = 0.0

    def spin():
        nonlocal next_can_check, hold_needed, hold_future, hold_deadline
        rclpy.spin_once(node, timeout_sec=0.02)
        if (profile and fault is not None and (
            (tracker is not None and not tracker.terminal)
            or (node.status is not None and node.status.state_machine == "TRAJ_RUNNING")
        )):
            stopper.request()
        # Once a goal has ended, cancel has no effect on its retained target.
        # Request measured-position hold explicitly, without another motion goal.
        if (hold_needed and (tracker is None or tracker.terminal)
                and node.status is not None and node.status.state_machine == "IDLE"):
            hold_needed = False
            if node.hold.service_is_ready():
                hold_future = node.hold.call_async(Trigger.Request())
                hold_deadline = time.monotonic() + 2
                emit("hold_sent")
            else:
                emit("hold_unavailable")
                print("保持サービスに接続できません。必要ならすぐ電源を切ってください。", flush=True)
        if hold_future is not None:
            if hold_future.done():
                try:
                    response = hold_future.result()
                    emit("hold_reply", success=response.success, message=response.message)
                    if not response.success:
                        print("保持指令が失敗しました。すぐ電源を切れる状態を維持してください。", flush=True)
                finally:
                    hold_future = None
            elif time.monotonic() > hold_deadline:
                emit("hold_timeout")
                hold_future = None
                print("保持指令の応答がありません。すぐ電源を切ってください。", flush=True)
        if interrupted:
            interrupted.clear()
            raise RuntimeError("signal received; cancel requested, client stays alive")
        if fault is None and time.monotonic() >= next_can_check:
            check_link(link_state(args.channel), before)
            next_can_check = time.monotonic() + 0.5

    def latch(reason):
        nonlocal fault, hold_needed
        if fault is None:
            hold_needed = True
        fault = str(reason)
        emit("trial_fault", reason=fault)
        if tracker is not None and not tracker.terminal:
            tracker.cancel()
        print("停止要求: " + fault + "。新しい移動は禁止。応答がなければ電源を切ってください。", flush=True)

    def await_goal(deadline):
        while not tracker.terminal:
            spin()
            current, grip = fresh_snapshot(node)
            check_measured(current, grip)
            if time.monotonic() > deadline:
                raise RuntimeError("goal deadline exceeded; cancel and retain client")
            # While moving, any typed line requests stop; never queues another move.
            if stdin_open and select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                raise RuntimeError("operator stop during motion")
        if not tracker.success:
            raise RuntimeError("goal did not finish successfully")

    def monitored_tick():
        spin()
        if stdin_open and select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            raise RuntimeError("operator stop during hold/settle")
        current, grip = fresh_snapshot(node)
        check_measured(current, grip)
        return current, grip

    def check_measured(current, grip):
        check_drift(current, grip, anchor, grip_anchor, args.lower_offset, args.upper_offset, k)
        if profile and not profile["lower_rad"] - 0.002 <= current[0] <= profile["upper_rad"] + 0.002:
            raise RuntimeError("measured joint1 outside the MoveIt trial bounds")

    def service(client, name, timeout_s):
        if not client.service_is_ready():
            raise RuntimeError(name + " service unavailable")
        future = client.call_async(Trigger.Request())
        emit("service_sent", name=name)
        deadline = time.monotonic() + timeout_s
        while not future.done():
            spin()
            if time.monotonic() > deadline:
                raise RuntimeError(name + " service timed out; service requests cannot be canceled")
        response = future.result()
        emit("service_reply", name=name, success=response.success, message=response.message)
        if not response.success:
            raise RuntimeError(name + ": " + response.message)

    print("ドライバは起動しません。commands: arm / status / move OFFSET_RAD / moveit ABS_RAD / stop / mark LABEL / finish / disable / quit", flush=True)
    print("move中はEnterでcancel。Ctrl+Cでもクライアントは終了せずcancelを追跡します。", flush=True)
    try:
        while rclpy.ok():
            try:
                spin()
                if anchor is not None and fault is None:
                    current, grip = fresh_snapshot(node)
                    check_measured(current, grip)
                now = time.monotonic()
                if node.positions is not None and now - last_display >= 1:
                    q = node.positions
                    name = node.joint_names[k]
                    print(f"{name}={q[k]:+.4f} rad ({math.degrees(q[k]):+.2f} deg) "
                          f"others={others(q, k)} gripper={node.gripper}", flush=True)
                    last_display = now
                if not stdin_open or not select.select([sys.stdin], [], [], 0)[0]:
                    continue
                line = sys.stdin.readline()
                if not line:
                    stdin_open = False
                    raise RuntimeError("stdin closed; client retained to track pending goals")
                words = line.split()
                if not words:
                    continue
                command = words[0]
                if command == "stop":
                    latch("operator stop")
                elif command == "status":
                    emit("snapshot", positions=node.positions, gripper=node.gripper, fault=fault,
                         pending_goal=tracker is not None and not tracker.terminal)
                elif command == "arm":
                    if anchor is not None or fault is not None:
                        raise ValueError("baseline is fixed for this session; restart only after disable")
                    if not node.hold.service_is_ready():
                        raise ValueError("updated hold service is required before arming this client")
                    anchor, grip_anchor = fresh_snapshot(node)
                    if node.status.state_machine != "IDLE":
                        anchor = None
                        raise ValueError("baseline capture requires driver IDLE")
                    if profile:
                        args.lower_offset = profile["lower_rad"] - anchor[0]
                        args.upper_offset = profile["upper_rad"] - anchor[0]
                    if profile is None and any(abs(q) > 0.1 for q in anchor):
                        anchor = None
                        raise ValueError("first trial requires a folded pose: all six angles within 0.1 rad of zero")
                    if not anchor[k] + args.lower_offset <= 0 <= anchor[k] + args.upper_offset:
                        anchor = None
                        raise ValueError("session window must include home angle zero")
                    emit("hold_start", anchor=anchor, gripper=grip_anchor)
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        monitored_tick()
                    emit("hold_checked", seconds=5, anchor=anchor, gripper=grip_anchor)
                    print("保持確認完了。" + ("moveit ABS_RADでMoveIt軌道を計画・実行します。" if profile else "moveは1回0.05rad以内。"), flush=True)
                elif command == "move":
                    if profile:
                        raise ValueError("MoveIt session uses 'moveit ABSOLUTE_RAD', not direct move")
                    if fault is not None or anchor is None or (tracker is not None and not tracker.terminal):
                        raise ValueError("no verified baseline, fault, or unresolved goal; no move sent")
                    if len(words) != 2 or not node.action.server_is_ready():
                        raise ValueError("use move OFFSET_RAD with a ready action server")
                    current, grip = fresh_snapshot(node)
                    if node.status.state_machine != "IDLE":
                        raise ValueError("new motion requires driver IDLE")
                    check_drift(current, grip, anchor, grip_anchor, args.lower_offset, args.upper_offset, k)
                    points = bounded_plan(current, anchor, float(words[1]), args.lower_offset,
                                          args.upper_offset, k)
                    goal = FollowJointTrajectory.Goal()
                    goal.trajectory.joint_names = list(node.joint_names)
                    for seconds, positions in points:
                        stamp = round(seconds * 10**9)
                        goal.trajectory.points.append(JointTrajectoryPoint(
                            positions=positions, time_from_start=Duration(
                                sec=stamp // 10**9, nanosec=stamp % 10**9)))
                    tracker = TrackedGoal(emit)
                    emit("move_sent", target=points[-1][1], duration_s=points[-1][0])
                    tracker.attach(node.action.send_goal_async(goal))
                    await_goal(time.monotonic() + points[-1][0] + 2)
                    # Server success means targets sent; verify actual arrival separately.
                    deadline = time.monotonic() + 2
                    settled_since = None
                    while True:
                        current, grip = monitored_tick()
                        if abs(current[k] - points[-1][1][k]) <= 0.02:
                            settled_since = settled_since or time.monotonic()
                            if time.monotonic() - settled_since >= 0.5:
                                break
                        else:
                            settled_since = None
                        if time.monotonic() > deadline:
                            raise RuntimeError("actual joint1 failed to settle within 0.02 rad")
                    emit("move_settled", positions=current, gripper=grip)
                elif command == "moveit":
                    if not profile or fault is not None or anchor is None or (tracker is not None and not tracker.terminal):
                        raise ValueError("MoveIt profile, verified baseline and healthy idle session required")
                    if len(words) != 2 or node.status.state_machine != "IDLE":
                        raise ValueError("use moveit ABSOLUTE_RAD while driver is IDLE")
                    if not planner.service_is_ready() or not moveit_action.server_is_ready() or not stopper.ready():
                        raise ValueError("MoveIt planning/execution not ready")
                    from joint1_moveit import plan_request, validate_trajectory
                    from rosidl_runtime_py.convert import message_to_ordereddict
                    current, grip = fresh_snapshot(node)
                    check_measured(current, grip)
                    target = float(words[1])
                    request = plan_request(current[0], target, profile)
                    future = planner.call_async(request)
                    emit("moveit_plan_requested", current_rad=current[0], target_rad=target)
                    deadline = time.monotonic() + 10
                    while not future.done():
                        monitored_tick()
                        if time.monotonic() > deadline:
                            raise RuntimeError("MoveIt planning timed out; no execution sent")
                    response = future.result().motion_plan_response
                    if response.error_code.val != 1:
                        raise RuntimeError(f"MoveIt planning failed: {response.error_code.val}")
                    current, grip = fresh_snapshot(node)
                    duration = validate_trajectory(response.trajectory, current[0], target, profile)
                    emit("moveit_plan_validated", target_rad=target, duration_s=duration,
                         trajectory=message_to_ordereddict(response.trajectory))
                    tracker = TrackedMoveItGoal(emit, stopper.request)
                    tracker.attach(moveit_action.send_goal_async(ExecuteTrajectory.Goal(trajectory=response.trajectory)))
                    emit("moveit_execution_sent", target_rad=target, duration_s=duration)
                    await_goal(time.monotonic() + duration + 5)
                    deadline, settled_since = time.monotonic() + 2, None
                    while True:
                        current, grip = monitored_tick()
                        if abs(current[0] - target) <= 0.02:
                            settled_since = settled_since or time.monotonic()
                            if time.monotonic() - settled_since >= 0.5:
                                break
                        else:
                            settled_since = None
                        if time.monotonic() > deadline:
                            raise RuntimeError("actual joint1 failed MoveIt arrival check")
                    emit("moveit_settled", target_rad=target, positions=current)
                elif command == "mark":
                    if fault is not None or anchor is None or (tracker is not None and not tracker.terminal):
                        raise ValueError("mark requires an armed, healthy, stationary session")
                    current, grip = fresh_snapshot(node)
                    if node.status.state_machine != "IDLE":
                        raise ValueError("mark requires driver IDLE")
                    emit("operator_mark", label=" ".join(words[1:]), positions=current, gripper=grip,
                         scope="operator-selected clearance for this pose; not a verified global safe limit")
                elif command == "finish":
                    if fault is not None or (tracker is not None and not tracker.terminal):
                        raise ValueError("fault/unresolved goal: do not start homing; inspect and use emergency power stop if needed")
                    current, grip = fresh_snapshot(node)
                    if anchor is None or any(abs(q) > 0.1 for q in others(current, k)):
                        raise ValueError("finish homing requires the verified folded pose")
                    if profile and abs(current[0]) > 0.04:
                        raise ValueError("MoveIt must return joint1 near zero before finish/safe_home")
                    if args.arm_hold and abs(current[k]) > 0.04:
                        # A failed home ends in disable; keep any drop under gravity small.
                        raise ValueError("move the joint back within 0.04 rad of zero before finish/safe_home")
                    # Explicit normal sequence required by the project; do not
                    # hide these movements in signal handlers or client exit.
                    try:
                        service(node.home, "safe_home", 15)
                    finally:
                        service(node.disable, "disable", 5)
                    anchor = None
                    emit("normal_finish")
                elif command == "disable":
                    # Explicit emergency request only. Never home automatically
                    # after a fault or unresolved motion. Caller must support
                    # the arm; CAN loss may require cutting physical power.
                    if tracker is not None and not tracker.terminal:
                        tracker.cancel()
                    service(node.disable, "disable", 5)
                    anchor = None
                    fault = "explicit disable; no further moves this session"
                    emit("explicit_disable")
                elif command == "quit":
                    if tracker is not None and not tracker.terminal:
                        raise ValueError("pending goal: keep the client alive to cancel late acceptance")
                    if node.status is None or node.status.enabled:
                        raise ValueError("disable confirmation required before client exit")
                    emit("client_exit")
                    return 0
                else:
                    print("commands: arm / status / move OFFSET_RAD / moveit ABS_RAD / stop / mark LABEL / finish / disable / quit", flush=True)
            except ValueError as error:
                print("指令拒否: " + str(error), flush=True)
                emit("command_rejected", reason=str(error))
            except Exception as error:
                latch(error)
    finally:
        if tracker is not None and not tracker.terminal:
            tracker.cancel()
            emit("unresolved_goal_at_context_shutdown")
        # ROS/context loss can prevent cancellation; never claim the arm stopped.
        node.destroy_node()
        rclpy.try_shutdown()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="connect ROS after explicit trial approval")
    parser.add_argument("--namespace", help="default rebotarm_joint1_trial in isolated mode, otherwise rebotarm")
    parser.add_argument("--joint1-only", action="store_true", help="requires scripts/config/joint1_range_trial.yaml")
    parser.add_argument("--arm-hold", action="store_true",
                        help="requires scripts/config/arm_hold_trial.yaml: six joints hold, one moves")
    parser.add_argument("--move-joint", default="joint1", choices=JOINTS,
                        help="joint moved in an --arm-hold session")
    parser.add_argument("--channel", default="can0", help="read-only link status checks")
    parser.add_argument("--lower-offset", type=float, default=-0.05)
    parser.add_argument("--upper-offset", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight", type=Path, help="successful CAN stability summary, at most 5 minutes old")
    parser.add_argument("--moveit-profile", type=Path, help="use MoveIt planning/ExecuteTrajectory for an isolated joint1 trial")
    args = parser.parse_args(argv)
    if args.arm_hold and (args.joint1_only or args.moveit_profile):
        parser.error("--arm-hold cannot be combined with --joint1-only or --moveit-profile")
    if not args.arm_hold and args.move_joint != "joint1":
        parser.error("--move-joint other than joint1 requires --arm-hold")
    default_namespace = ("rebotarm_joint1_trial" if args.joint1_only
                         else "rebotarm_arm_hold_trial" if args.arm_hold else "rebotarm")
    args.namespace = (args.namespace or default_namespace).strip("/")
    if not args.namespace:
        parser.error("namespace cannot be empty")
    if not (math.isfinite(args.lower_offset) and math.isfinite(args.upper_offset)
            and -0.15 <= args.lower_offset < 0 < args.upper_offset <= 0.15):
        parser.error("first trial offsets must straddle zero and be within +/-0.15 rad")
    plan = {"status": "prepared_only", "joint": args.move_joint, "lower_offset_rad": args.lower_offset,
            "upper_offset_rad": args.upper_offset, "maximum_step_rad": MAX_STEP,
            "maximum_commanded_speed_rad_s": MAX_SPEED, "initial_hold_s": 5,
            "driver_launch_or_enable": False, "gripper_command": False,
            "joint1_only": args.joint1_only,
            "arm_hold": args.arm_hold,
            "namespace": args.namespace,
            "note": ("isolated driver enables only motor 1; other axes rest unpowered"
                     if args.joint1_only else
                     "arm hold driver enables motors 1-6 (no gripper); only the moving joint changes"
                     if args.arm_hold else "normal driver separately enables all joints")}
    if args.moveit_profile:
        from joint1_moveit import load_profile
        plan["moveit"] = load_profile(args.moveit_profile)
        plan["maximum_step_rad"] = 0.20
        plan["absolute_interval_rad"] = [plan["moveit"]["lower_rad"], plan["moveit"]["upper_rad"]]
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.output_dir is None or args.preflight is None:
        parser.error("--execute requires a new --output-dir and --preflight summary")
    args.namespace = args.namespace.strip("/")
    if not args.namespace:
        parser.error("namespace cannot be empty")
    args.output_dir.mkdir(exist_ok=False)
    with (args.output_dir / "events.jsonl").open("x", buffering=1) as output:
        def emit(event, **fields):
            output.write(json.dumps(dict(event=event, wall_ns=time.time_ns(),
                                         monotonic_s=time.monotonic(), **fields), ensure_ascii=False) + "\n")
        emit("session_plan", **plan)
        return execute(args, emit)


if __name__ == "__main__":
    raise SystemExit(main())
