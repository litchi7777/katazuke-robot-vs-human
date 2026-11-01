#!/usr/bin/env python3
"""Move all six arm joints slowly to a named or given pose (arm hold trial driver).

Resident ROS client like joint1_range_trial.py: it never launches the driver or
enables motors. The gripper is commanded only with --gripper and `grip`.
Default is a dry-run plan.

Commands on stdin:
  arm                 capture the folded pose, check a 5 s hold
  goto NAME|q1..q6    quintic joint trajectory, peak speed <= MAX_SPEED
  step q1..q6         like goto but never adds the --correct move (contact probing)
  home                goto all zeros (folded)
  grip open [RAD]     open to RAD (default GRIP_OPEN), within the driver's limits
  grip close [SQZ [FROM]]  close in small steps (from FROM rad, reached in one move, if given);
                      stop SQZ rad (default GRIP_SQUEEZE) past first contact
  grip resqueeze T S  if the fingers ran on to near target T, command S rad past where they are now
  grip state          print the measured gripper angle (an object held keeps it above the target)
  snap LABEL          record joint angles and grab one frame per --snap-cameras device
  finish              safe_home -> disable; only within 0.04 rad of zero
  status / disable / quit
Any line typed during a motion stops it (cancel, then hold measured pose).

Every planned path is checked with forward kinematics before it is sent:
the gripper tip must stay above --min-tcp-z and inside the driver's limits.
"""
import argparse
import json
import math
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from check_can_stability_readonly import check_link
from joint1_range_trial import JOINTS, TrackedGoal, fresh_snapshot, ros_node, verified_preflight
from probe_can_readonly import link_state

MAX_SPEED = 0.1        # rad/s, commanded quintic peak for the fastest joint
MAX_TRACKING = 0.12    # rad, measured vs commanded during motion
SPEED_CAP = 0.4       # rad/s; 10/28 tracking was 0.033 rad at 0.2 against the 0.12 watchdog (0.3 until 11/01)
SAG_TOLERANCE = 0.05   # rad past a limit that a resumed or moving arm may start from (pulled back in)
MIN_DURATION = 0.4     # s, shortest move (3.0 -> 0.6 -> 0.4 on 2026-11-01: probe steps were half a grasp's time)
PROBE_MIN_DURATION = 0.6  # s, shortest probing `step`: at 0.4 s a stretched arm settled up to ~1 cm short at the
                          # tip and the probe took it for contact above the desk (2026-11-01 real, a pen 41 cm out)
JOINT_SPEED_SHARE = (1.7, 1.0, 1.0, 1.5, 2.0, 2.0)  # of --max-speed per joint; j2/j3 hold the arm up
SETTLE_HOLD = 0.15     # s the joints must stay within SETTLE before a move counts as done
SETTLE = 0.03          # rad, per joint after the goal ends
FOLDED = 0.1           # rad, every joint at `arm`
CORRECT_MIN = 0.004    # rad residual that triggers one correction move (--correct)
CORRECT_MAX = 0.03     # rad, largest per-joint overshoot of the correction
GRIP_OPEN = 1.2        # rad, default opening for a shogi piece
GRIP_OPEN_LIMIT = 4.5  # rad, never opened further whatever is asked (2026-11-01: 5.8 rad with a -0.45 rad zero
                       # drift drove the fingers past their mechanical stop and they cracked)
GRIP_ZERO_TOLERANCE = 0.15  # rad below closed or above GRIP_OPEN_LIMIT the gripper may read at start
GRIP_STEP = 0.05       # rad per closing step
GRIP_STEP_WAIT = 0.15  # s after each step before reading (0.3 until 2026-11-01; the fingers move ~4 rad/s free)
GRIP_OPEN_STEP = 0.3   # rad per opening step (~1.5 rad/s); one jump to 5.8 rad spun the gripper alarmingly fast
GRIP_LEAD_WAIT = 1.5   # s at most for the fingers to reach `grip close SQZ FROM`'s FROM
GRIP_CONTACT = 0.08    # rad the gripper lags its step target -> contact
GRIP_SQUEEZE = 0.08    # rad commanded past contact; squeeze torque ~= kp * this
GRIP_SQUEEZE_RANGE = (0.02, 0.15)  # rad accepted by `grip close SQZ` (kp 8 -> at most ~1.2 Nm)
GRIP_RESQUEEZE = 2     # extra squeezes when the object settles into the fingers after the first


def driver_limits(config_path):
    data = yaml.safe_load(Path(config_path).read_text())
    overrides = data["models"]["rs"]["overrides"]
    if not overrides.get("arm_hold_trial"):
        raise ValueError("driver config is not an arm hold trial configuration")
    limits = overrides["arm_hold_trial_position_limits"]
    return [tuple(map(float, limits[name])) for name in JOINTS]


def plan(current, target, limits, max_speed=None, min_duration=MIN_DURATION):
    values = [*current, *target]
    if len(current) != 6 or len(target) != 6 or not all(map(math.isfinite, values)):
        raise ValueError("six finite current and target angles required")
    for name, q, (lo, hi) in zip(JOINTS, target, limits):
        if not lo <= q <= hi:
            raise ValueError(f"{name} target {q:.3f} outside driver limits [{lo}, {hi}]")
    # A joint that sagged just past its limit (gravity on a stretched arm, 2026-10-28: j2 2.622 > 2.60)
    # starts from the limit, otherwise the driver rejects the whole trajectory.
    current = [min(hi, max(lo, c)) if lo - SAG_TOLERANCE <= c <= hi + SAG_TOLERANCE else c
               for c, (lo, hi) in zip(current, limits)]
    # Each joint at its own share of max_speed (2026-11-01: one cap for all made the base turn to a box
    # the slowest part of a real tidy, while j1 lifts nothing and the wrist joints carry only the hand).
    speed = max_speed or MAX_SPEED
    duration = max([min_duration] + [1.875 * abs(t - c) / (speed * share)
                                     for t, c, share in zip(target, current, JOINT_SPEED_SHARE)])
    count = math.ceil(duration / 0.1)
    points = []
    for step in range(count + 1):
        phase = step / count
        ratio = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
        points.append((duration * phase, [c + (t - c) * ratio for c, t in zip(current, target)]))
    return points


def commanded_at(points, elapsed):
    if elapsed <= 0:
        return points[0][1]
    for (t0, q0), (t1, q1) in zip(points, points[1:]):
        if elapsed <= t1:
            r = (elapsed - t0) / (t1 - t0)
            return [a + (b - a) * r for a, b in zip(q0, q1)]
    return points[-1][1]


def check_path(points, min_tcp_z):
    """Lowest gripper-tip height along the path (FK), rejected below min_tcp_z."""
    import topdown_ik
    lowest = min(topdown_ik.fk(q).translation[2] for _, q in points)
    if lowest < min_tcp_z:
        raise ValueError(f"gripper tip would reach z={lowest:.3f} m < {min_tcp_z} m on this path")
    return lowest


def corrected_command(target, measured, limits, threshold=CORRECT_MIN, cap=CORRECT_MAX):
    """Re-aim past the target by the settled residual (friction/gravity shortfall).

    Returns None when every residual is already below `threshold`.
    """
    residual = [t - m for t, m in zip(target, measured)]
    if max(abs(r) for r in residual) < threshold:
        return None
    command = [t + max(-cap, min(cap, r)) for t, r in zip(target, residual)]
    return [max(lo, min(hi, q)) for q, (lo, hi) in zip(command, limits)]


def soft_close(read_position, command, start, squeeze=GRIP_SQUEEZE):
    """Step the gripper closed; on first contact hold GRIP_SQUEEZE past it.

    read_position() returns the measured gripper angle after a short wait;
    command(target) sends one gripper target. Returns (result, final_target).
    """
    target = start
    while target > 0.0:
        target = max(0.0, target - GRIP_STEP)
        command(target)
        position = read_position()
        if position - target > GRIP_CONTACT:
            final = max(0.0, position - squeeze)
            command(final)
            # A box first touched at its corners turns square between the fingers and the squeeze is
            # used up (sim 2026-10-28: a 180 g block grasped 4.6 deg off dropped on the lift). If the
            # fingers ran on toward the target, squeeze again from where they are now; the command is
            # still never more than `squeeze` past the measured position.
            for _ in range(GRIP_RESQUEEZE):
                now = read_position()
                if now - final > squeeze / 2 or final == 0.0:
                    break
                final = max(0.0, now - squeeze)
                command(final)
            return "contact", final
    return "closed_empty", 0.0


def resqueeze_target(position, target, squeeze):
    """New gripper target when the fingers ran on toward `target` (the object turned or settled), else None.

    Lifting off the desk, a box held a few degrees off turns square between the fingers and the squeeze
    is gone (sim 2026-10-28: a 180 g block). The new target is `squeeze` past where the fingers are now,
    so the torque never exceeds the first close's.
    """
    if position - target > squeeze / 2 or target <= 0.0:
        return None
    return max(0.0, position - squeeze)


def soft_open(command, start, target, step=GRIP_OPEN_STEP):
    """Move the gripper to target in steps of at most `step` rad; command(t) waits for each."""
    position = start
    while True:
        position += max(-step, min(step, target - position))
        reply = command(position)
        if abs(target - position) < 1e-9:
            return reply


def parse_target(words, poses):
    if len(words) == 2:
        if words[1] not in poses:
            raise ValueError(f"unknown pose {words[1]!r}; known: {sorted(poses)}")
        return [float(v) for v in poses[words[1]]]
    if len(words) == 7:
        return [float(v) for v in words[1:]]
    raise ValueError("use goto NAME or goto q1 q2 q3 q4 q5 q6")


def execute(args, limits, poses, emit):
    import rclpy
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from rclpy.signals import SignalHandlerOptions
    from std_srvs.srv import Trigger
    from trajectory_msgs.msg import JointTrajectoryPoint

    # A resumed session takes over a driver that already owns can0, so no new
    # read-only preflight is possible; accept this session's (link still checked).
    preflight = verified_preflight(args.preflight, 3600 if args.resume else 300)
    before = link_state(args.channel)
    check_link(before, preflight["link_after"])
    emit("can_before", link=before)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = ros_node(args.namespace, emit, arm_hold=True)
    gripper = dict(position=None, received=0.0)
    if args.gripper:
        from rclpy.qos import qos_profile_sensor_data
        from rebotarm_msgs.msg import JointMotorState
        from rebotarm_msgs.srv import GripperCommand

        def on_gripper(msg):
            if math.isfinite(msg.position):
                gripper.update(position=float(msg.position), received=time.monotonic())

        node.create_subscription(JointMotorState, f"/{args.namespace}/gripper/state", on_gripper,
                                 qos_profile_sensor_data)
        grip_open = node.create_client(GripperCommand, f"/{args.namespace}/gripper/open")
        grip_close = node.create_client(GripperCommand, f"/{args.namespace}/gripper/close")
    interrupted = []
    previous = {s: signal.signal(s, lambda n, _f: interrupted.append(n)) for s in (signal.SIGINT, signal.SIGTERM)}
    # recoverable: the only fault so far is an operator stop, so `home`/`finish`
    # may still bring the raised arm back instead of dropping it with disable.
    state = dict(tracker=None, fault=None, recoverable=False, armed=False, hold_needed=False,
                 next_can=time.monotonic() + 0.5)
    stdin_open = True
    last_display = 0.0

    def spin():
        rclpy.spin_once(node, timeout_sec=0.02)
        tracker = state["tracker"]
        if (state["hold_needed"] and (tracker is None or tracker.terminal) and node.status is not None
                and node.status.state_machine == "IDLE"):
            state["hold_needed"] = False
            if node.hold.service_is_ready():
                node.hold.call_async(Trigger.Request())
                emit("hold_sent")
            else:
                print("保持サービスに接続できません。必要ならすぐ電源を切ってください。", flush=True)
        if interrupted:
            interrupted.clear()
            raise RuntimeError("signal received; cancel requested, client stays alive")
        if (state["fault"] is None or state["recoverable"]) and time.monotonic() >= state["next_can"]:
            check_link(link_state(args.channel), before)
            state["next_can"] = time.monotonic() + 0.5

    def latch(reason):
        if state["fault"] is None:
            state["hold_needed"] = True
        state["recoverable"] = (str(reason).startswith("operator stop")
                                and (state["fault"] is None or state["recoverable"]))
        state["fault"] = str(reason)
        emit("trial_fault", reason=state["fault"])
        if state["tracker"] is not None and not state["tracker"].terminal:
            state["tracker"].cancel()
        print("停止要求: " + state["fault"] + "。新しい移動は禁止。応答がなければ電源を切ってください。", flush=True)

    def operator_line():
        return stdin_open and select.select([sys.stdin], [], [], 0)[0]

    def service(client, name, timeout_s):
        if not client.service_is_ready():
            raise RuntimeError(name + " service unavailable")
        future = client.call_async(Trigger.Request())
        emit("service_sent", name=name)
        deadline = time.monotonic() + timeout_s
        while not future.done():
            spin()
            if time.monotonic() > deadline:
                raise RuntimeError(name + " service timed out")
        reply = future.result()
        emit("service_reply", name=name, success=reply.success, message=reply.message)
        if not reply.success:
            raise RuntimeError(name + ": " + reply.message)

    def gripper_position():
        if gripper["position"] is None or time.monotonic() - gripper["received"] > 0.5:
            raise RuntimeError("gripper telemetry missing or stale")
        return gripper["position"]

    def gripper_command(client, name, target, timeout_s):
        # The driver bounds the target to its gripper position_limits.
        if not client.service_is_ready():
            raise RuntimeError(name + " service unavailable")
        request = GripperCommand.Request(position=float(target), timeout=float(timeout_s))
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_s + 2
        while not future.done():
            spin()
            if time.monotonic() > deadline:
                raise RuntimeError(name + " service timed out")
        reply = future.result()
        emit("gripper_reply", name=name, target=target, success=reply.success,
             reached=reply.reached_position, message=reply.message)
        return reply

    def grip(words):
        if not args.gripper:
            raise ValueError("start the client with --gripper to command the gripper")
        current, _ = fresh_snapshot(node)
        if node.status.state_machine != "IDLE":
            raise ValueError("gripper commands require driver IDLE")
        if len(words) >= 2 and words[1] == "open":
            target = float(words[2]) if len(words) == 3 else GRIP_OPEN
            if not 0.0 < target <= args.grip_open_max:
                raise ValueError(f"open target must be in (0, {args.grip_open_max}] rad")
            reply = soft_open(lambda t: gripper_command(grip_open, "gripper/open", t, 0.5),
                              gripper_position(), target)
            print(f"グリッパ: {reply.reached_position:+.3f} rad（目標 {target:.2f}）", flush=True)
        elif len(words) == 2 and words[1] == "state":
            spin()
            print(f"グリッパ: state、位置 {gripper_position():+.3f} rad", flush=True)
        elif len(words) in (2, 3, 4) and words[1] == "close":
            squeeze = float(words[2]) if len(words) >= 3 else GRIP_SQUEEZE
            if not GRIP_SQUEEZE_RANGE[0] <= squeeze <= GRIP_SQUEEZE_RANGE[1]:
                raise ValueError(f"squeeze must be in {GRIP_SQUEEZE_RANGE} rad")

            def read():
                end = time.monotonic() + GRIP_STEP_WAIT
                while time.monotonic() < end:
                    spin()
                return gripper_position()

            close = lambda t: gripper_command(grip_close, "gripper/close", t, 0.2)
            start = gripper_position()
            if len(words) == 4:
                # Jump to FROM (just outside the object's width) before stepping. Something wider than
                # expected stops the fingers early; soft_close's first step then sees it as contact.
                lead = float(words[3])
                if not 0.0 <= lead < start:
                    raise ValueError("close FROM must be below the current opening")
                close(lead)
                end = time.monotonic() + GRIP_LEAD_WAIT
                while time.monotonic() < end and gripper_position() - lead > GRIP_CONTACT / 2:
                    spin()
                start = max(lead, gripper_position())
            result, final = soft_close(read, close, start, squeeze)
            emit("grip_close", result=result, final_target=final, position=gripper_position())
            print(f"グリッパ: {result}、位置 {gripper_position():+.3f} rad、目標 {final:+.3f} rad", flush=True)
        elif len(words) == 4 and words[1] == "resqueeze":
            target, squeeze = float(words[2]), float(words[3])
            if not GRIP_SQUEEZE_RANGE[0] <= squeeze <= GRIP_SQUEEZE_RANGE[1]:
                raise ValueError(f"squeeze must be in {GRIP_SQUEEZE_RANGE} rad")
            spin()
            position = gripper_position()
            new = resqueeze_target(position, target, squeeze)
            if new is not None:
                gripper_command(grip_close, "gripper/close", new, 0.2)
                target = new
            emit("grip_resqueeze", squeezed=new is not None, final_target=target, position=gripper_position())
            print(f"グリッパ: {'resqueezed' if new is not None else 'held'}、位置 {gripper_position():+.3f} rad、"
                  f"目標 {target:+.3f} rad", flush=True)
        else:
            raise ValueError("use grip open [RAD], grip close [SQUEEZE], grip resqueeze TARGET SQUEEZE or grip state")

    def snap(label):
        if not label.replace("_", "").replace("-", "").isalnum():
            raise ValueError("snap label must be alphanumeric, '_' or '-'")
        current, grip_q = fresh_snapshot(node)
        if node.status.state_machine != "IDLE":
            raise ValueError("snap requires the arm to be idle")
        folder = args.output_dir / "snaps"
        folder.mkdir(exist_ok=True)
        files = {}
        for device in args.snap_cameras:
            name = Path(device).name
            path = folder / f"{label}_{name}.jpg"
            # ffmpeg reopens the camera; give auto exposure ~3 s before the frame.
            # Keep spinning meanwhile so joint telemetry and the watchdogs stay live.
            proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "v4l2",
                                     "-input_format", "mjpeg", "-video_size", "1280x960", "-framerate", "30",
                                     "-i", device, "-ss", "3", "-frames:v", "1", str(path)])
            deadline = time.monotonic() + 20
            while proc.poll() is None:
                spin()
                if time.monotonic() > deadline:
                    proc.kill()
                    raise RuntimeError(f"camera capture from {device} timed out")
            if proc.returncode != 0:
                raise RuntimeError(f"camera capture from {device} failed")
            files[name] = str(path)
        after, _ = fresh_snapshot(node)
        if max(abs(a - b) for a, b in zip(current, after)) > 0.005:
            raise RuntimeError("arm moved during snap; discard it")
        emit("snap", label=label, positions=current, positions_after=after,
             gripper=gripper["position"], files=files)
        print(f"撮影: {label} " + " ".join(files), flush=True)

    def goto(target, command=None, correct=True):
        # `command` differs from `target` only for the residual correction move.
        command = target if command is None else command
        current, _ = fresh_snapshot(node)
        if node.status.state_machine != "IDLE":
            raise ValueError("new motion requires driver IDLE")
        points = plan(current, command, limits, args.max_speed, MIN_DURATION if correct else PROBE_MIN_DURATION)
        lowest = check_path(points, args.min_tcp_z)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINTS)
        for seconds, positions in points:
            stamp = round(seconds * 10**9)
            goal.trajectory.points.append(JointTrajectoryPoint(
                positions=positions, time_from_start=Duration(sec=stamp // 10**9, nanosec=stamp % 10**9)))
        tracker = TrackedGoal(emit)
        state["tracker"] = tracker
        emit("goto_sent", start=current, target=target, duration_s=points[-1][0], lowest_tcp_z=lowest)
        print(f"移動開始: {points[-1][0]:.1f}秒、手先の最低高さ {lowest:.3f} m。何か入力すると停止。", flush=True)
        sent = time.monotonic()
        tracker.attach(node.action.send_goal_async(goal))
        deadline = sent + points[-1][0] + 5
        while not tracker.terminal:
            spin()
            measured, _ = fresh_snapshot(node)
            expected = commanded_at(points, time.monotonic() - sent)
            worst = max(abs(m - e) for m, e in zip(measured, expected))
            if worst > args.max_tracking:
                raise RuntimeError(f"tracking error {worst:.3f} rad exceeds {args.max_tracking}")
            if time.monotonic() > deadline:
                raise RuntimeError("goal deadline exceeded")
            if operator_line():
                sys.stdin.readline()
                raise RuntimeError("operator stop during motion")
        if not tracker.success:
            raise RuntimeError("goal did not finish successfully")
        settle_deadline, since = time.monotonic() + 3, None
        while True:
            spin()
            measured, _ = fresh_snapshot(node)
            if max(abs(m - t) for m, t in zip(measured, target)) <= SETTLE:
                since = since or time.monotonic()
                if time.monotonic() - since >= SETTLE_HOLD:
                    break
            else:
                since = None
            if time.monotonic() > settle_deadline:
                if correct:
                    raise RuntimeError("joints did not settle within 0.03 rad of the target")
                # A probing `step` that ends pressed on something: hold where it stopped (no push).
                service(node.hold, "hold", 3)
                end = time.monotonic() + 0.3
                while time.monotonic() < end:
                    spin()
                measured, _ = fresh_snapshot(node)
                emit("step_blocked", target=target, positions=measured)
                break
        emit("goto_settled", target=target, command=command, positions=measured)
        if args.correct and correct and command is target:
            corrected = corrected_command(target, measured, limits)
            if corrected is not None:
                emit("goto_correction", target=target, measured=measured, command=corrected)
                return goto(target, corrected)
        print("到達: " + " ".join(f"{q:+.3f}" for q in measured), flush=True)

    print("commands: arm / goto NAME|q1..q6 / step q1..q6 / home / grip open [RAD]|close / finish / status / disable / quit"
          f"  poses={sorted(poses)}", flush=True)
    try:
        while rclpy.ok():
            try:
                spin()
                if node.positions is not None and time.monotonic() - last_display >= 2:
                    print("q=" + " ".join(f"{q:+.3f}" for q in node.positions), flush=True)
                    last_display = time.monotonic()
                if not operator_line():
                    continue
                line = sys.stdin.readline()
                if not line:
                    stdin_open = False
                    raise RuntimeError("stdin closed; client retained to track pending goals")
                words = line.split()
                if not words:
                    continue
                command = words[0]
                busy = state["tracker"] is not None and not state["tracker"].terminal
                if command == "status":
                    emit("snapshot", positions=node.positions, fault=state["fault"], pending_goal=busy)
                elif command == "arm":
                    if state["armed"] or state["fault"]:
                        raise ValueError("already armed or faulted; restart only after disable")
                    if not node.hold.service_is_ready():
                        raise ValueError("hold service is required before arming")
                    if args.gripper:
                        spin()
                        g = gripper_position()
                        if not -GRIP_ZERO_TOLERANCE <= g <= GRIP_OPEN_LIMIT + GRIP_ZERO_TOLERANCE:
                            # A gripper reading below closed or past the lock is a zero drift (2026-11-01: -0.45 rad
                            # after a power cycle): every opening would land that much further. Stop for a person.
                            raise ValueError(f"gripper reads {g:+.3f} rad, outside [{-GRIP_ZERO_TOLERANCE}, "
                                             f"{GRIP_OPEN_LIMIT + GRIP_ZERO_TOLERANCE}]: zero drift or forced open; "
                                             "close it by hand / re-zero before moving")
                    current, _ = fresh_snapshot(node)
                    if node.status.state_machine != "IDLE":
                        raise ValueError("arm requires driver IDLE")
                    if args.resume:
                        # Taking over a raised arm from an earlier client: only `home` until folded.
                        for name, q, (lo, hi) in zip(JOINTS, current, limits):
                            if not lo - SAG_TOLERANCE <= q <= hi + SAG_TOLERANCE:
                                raise ValueError(f"{name} {q:.3f} outside driver limits; cannot resume")
                        state["resume_only"] = True
                    elif any(abs(q) > FOLDED for q in current):
                        raise ValueError("arm requires every joint within 0.1 rad of zero (or --resume)")
                    emit("hold_start", anchor=current)
                    end = time.monotonic() + 5
                    while time.monotonic() < end:
                        spin()
                        measured, _ = fresh_snapshot(node)
                        if max(abs(m - a) for m, a in zip(measured, current)) > 0.03:
                            raise RuntimeError("joint drifted by more than 0.03 rad during the hold check")
                    state["armed"] = True
                    emit("hold_checked")
                    print("保持確認完了。goto で移動できます。", flush=True)
                elif command in ("goto", "home", "step"):
                    if state.get("resume_only") and command not in ("home", "step"):  # step: lift clear first
                        raise ValueError("resumed session: run home before any other motion")
                    allowed = state["fault"] is None or (command == "home" and state["recoverable"])
                    if not state["armed"] or not allowed or busy:
                        raise ValueError("arm first; no fault (home only after an operator stop) "
                                         "and no pending goal allowed")
                    goto([0.0] * 6 if command == "home" else parse_target(words, poses),
                         correct=command != "step")
                    if command == "home":
                        state["resume_only"] = False
                elif command == "snap":
                    if not state["armed"] or state["fault"] or busy or len(words) != 2:
                        raise ValueError("use snap LABEL while armed, idle and without fault")
                    snap(words[1])
                elif command == "grip":
                    if not state["armed"] or state["fault"] or busy:
                        raise ValueError("arm first; no fault and no pending goal allowed")
                    grip(words)
                elif command == "finish":
                    if (state["fault"] and not state["recoverable"]) or busy:
                        raise ValueError("fault/unresolved goal: do not start homing")
                    current, _ = fresh_snapshot(node)
                    if any(abs(q) > 0.04 for q in current):
                        raise ValueError("use home first; finish needs every joint within 0.04 rad of zero")
                    try:
                        service(node.home, "safe_home", 20)
                    finally:
                        service(node.disable, "disable", 5)
                    state["armed"] = False
                    emit("normal_finish")
                    print("終了処理完了: safe_home -> disable", flush=True)
                elif command == "disable":
                    if busy:
                        state["tracker"].cancel()
                    service(node.disable, "disable", 5)
                    state["armed"] = False
                    state["fault"] = "explicit disable; no further moves this session"
                    emit("explicit_disable")
                elif command == "quit":
                    if busy:
                        raise ValueError("pending goal: keep the client alive")
                    if node.status is None or node.status.enabled:
                        raise ValueError("disable confirmation required before client exit")
                    emit("client_exit")
                    return 0
                else:
                    print("commands: arm / goto NAME|q1..q6 / step q1..q6 / home / grip open [RAD]|close / finish / status / disable / quit", flush=True)
            except ValueError as error:
                print("指令拒否: " + str(error), flush=True)
                emit("command_rejected", reason=str(error))
            except Exception as error:
                latch(error)
    finally:
        tracker = state["tracker"]
        if tracker is not None and not tracker.terminal:
            tracker.cancel()
            emit("unresolved_goal_at_context_shutdown")
        node.destroy_node()
        rclpy.try_shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true", help="connect ROS after explicit trial approval")
    p.add_argument("--driver-config", type=Path, required=True, help="arm hold trial YAML used by the driver")
    p.add_argument("--poses", type=Path, help="JSON {name: [q1..q6]}")
    p.add_argument("--namespace", default="rebotarm_arm_hold_trial")
    p.add_argument("--channel", default="can0")
    p.add_argument("--min-tcp-z", type=float, default=0.08, help="lowest allowed gripper tip height (m)")
    p.add_argument("--gripper", action="store_true", help="enable grip commands (driver must register the gripper)")
    p.add_argument("--grip-open-max", type=float, default=1.6, help="largest grip open target (rad)")
    p.add_argument("--max-speed", type=float, default=MAX_SPEED,
                   help="commanded peak joint speed (rad/s), at most SPEED_CAP")
    p.add_argument("--max-tracking", type=float, default=MAX_TRACKING,
                   help="rad measured-vs-commanded watchdog; recovery of a sagging stretched arm may need up to 0.2")
    p.add_argument("--correct", action="store_true",
                   help="after each goto, one small move re-aimed past the target by the residual")
    p.add_argument("--resume", action="store_true",
                   help="take over an enabled arm away from the folded pose; only `home` until folded")
    p.add_argument("--snap-cameras", default="/dev/video2",
                   help="comma separated V4L2 devices captured by `snap`")
    p.add_argument("--preflight", type=Path)
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args(argv)
    args.snap_cameras = [d for d in args.snap_cameras.split(",") if d]
    if not 0 < args.max_tracking <= 0.2:
        p.error("--max-tracking must be in (0, 0.2] rad")
    if not 0 < args.grip_open_max <= GRIP_OPEN_LIMIT:
        p.error(f"--grip-open-max must be in (0, {GRIP_OPEN_LIMIT}] rad (hard lock)")
    if not 0 < args.max_speed <= SPEED_CAP:
        p.error(f"--max-speed must be in (0, {SPEED_CAP}] rad/s")
    limits = driver_limits(args.driver_config)
    poses = json.loads(args.poses.read_text()) if args.poses else {}
    report = {"status": "prepared_only", "max_speed_rad_s": MAX_SPEED, "min_tcp_z_m": args.min_tcp_z,
              "limits": dict(zip(JOINTS, limits)), "poses": {}}
    for name, q in poses.items():
        points = plan([0.0] * 6, q, limits)
        entry = {"q": q, "duration_from_folded_s": round(points[-1][0], 1)}
        try:
            entry["lowest_tcp_z_m"] = round(check_path(points, args.min_tcp_z), 3)
        except ValueError as error:
            # Listed for reference only; goto re-checks and refuses it this session.
            entry["refused"] = str(error)
        report["poses"][name] = entry
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0
    if args.output_dir is None or args.preflight is None:
        p.error("--execute requires a new --output-dir and --preflight summary")
    args.output_dir.mkdir(exist_ok=False)
    with (args.output_dir / "events.jsonl").open("x", buffering=1) as output:
        def emit(event, **fields):
            output.write(json.dumps(dict(event=event, wall_ns=time.time_ns(),
                                         monotonic_s=time.monotonic(), **fields), ensure_ascii=False) + "\n")
        emit("session_plan", **report)
        return execute(args, limits, poses, emit)


if __name__ == "__main__":
    raise SystemExit(main())
