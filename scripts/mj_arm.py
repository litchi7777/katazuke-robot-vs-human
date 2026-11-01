#!/usr/bin/env python3
"""Stand-in for arm_goto.py --execute that drives the MuJoCo desk (mj_scene) instead of the arm.

Same stdin commands and result lines as arm_goto.py and sim_arm.py, so tidy_desk runs unchanged
with --physics. Unlike sim_arm (kinematic), everything here is physics: the joints are servos with
the real gains and torque limits, the fingers squeeze with a force set by the squeeze, and objects
slip, roll, get pushed and fall where gravity takes them.

Extra commands:
  render PATH [CAMERA]   save one camera frame (default video2) - the robot's eyes
  world                  write the world JSON now (objects where the physics put them)
  place ID X Y [YAW]     drop an object at X Y from 5 cm (the scatterer)

With --record DIR the presentation and overhead cameras are written as videos in simulated time
(waiting for Claude takes no video time), with timeline.jsonl marking what happened when.

    MUJOCO_GL=egl python3 mj_arm.py --world WORLD.json --output-dir DIR [arm_goto options]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

import arm_goto
import mj_scene
import topdown_ik

TRACE_HELD = os.environ.get("MJ_TRACE_HELD")  # diagnostics file, see trace_held
SETTLE_S = 3.0          # s at most to settle after a move (arm_goto waits 3 s too)
GRIP_WAIT = 0.3         # s per closing step before reading the finger (arm_goto read())
OPEN_WAIT = 0.25        # s per opening step
HOLD_S = 0.3            # s held after a blocked probing step
REST_S = 1.0            # s at most for the joints to come to rest after a move counts as done
REST_ERROR = 0.004      # rad per joint: at rest


class MjArm:
    def __init__(self, args, emit):
        self.a, self.emit = args, emit
        self.world_path = Path(args.world)
        self.world = json.loads(self.world_path.read_text())
        self.limits = arm_goto.driver_limits(args.driver_config)
        self.model = mj_scene.build(self.world).compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_q = [m.joint(j).qposadr[0] for j in mj_scene.ARM_JOINTS]
        self.arm_v = [m.joint(j).dofadr[0] for j in mj_scene.ARM_JOINTS]
        self.finger_q = [m.joint(j).qposadr[0] for j in ("gripper_joint1", "gripper_joint2")]
        self.arm_u = [m.actuator(j).id for j in mj_scene.ARM_JOINTS]
        self.finger_u = [m.actuator(j).id for j in ("gripper_joint1", "gripper_joint2")]
        self.data.ctrl[:] = 0.0
        self.armed = False
        self.fault = None
        self.squeeze_target = 0.0
        self.cameras = None
        self.recorders = {}
        self.timeline = None
        self.frame_dt = 1.0 / args.fps
        self.next_frame = 0.0
        self.peak_torque = [0.0] * 6
        self.next_trace, self.held_ids, self.traced_hits, self.last_cmd = 0.0, set(), set(), ""
        self.traced_moving = set()
        self.settle(1.0)  # objects come to rest on the desk
        self.cameras = mj_scene.Cameras(self.model)  # ~30 s for the GPU context: before the first q= line
        if args.record:
            self.start_recording(Path(args.record))

    # --- physics ------------------------------------------------------
    def q(self):
        return [float(self.data.qpos[i]) for i in self.arm_q]

    def gripper(self):
        return mj_scene.finger_rad(float(np.mean([self.data.qpos[i] for i in self.finger_q])))

    def step(self, seconds):
        n = max(1, int(round(seconds / self.model.opt.timestep)))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if TRACE_HELD and self.data.time >= self.next_trace:
                self.trace_held()
            if self.recorders and self.data.time >= self.next_frame:
                self.frame()
                self.next_frame += self.frame_dt

    def trace_held(self):
        """Diagnostics (MJ_TRACE_HELD=file): what a held object touches besides the fingers, and when it leaves them."""
        self.next_trace = self.data.time + 0.02
        held_now = set()
        for o in self.world["objects"]:
            if self.between_fingers(o["id"]):
                held_now.add(o["id"])
        left, right = self.model.body("gripper_left").id, self.model.body("gripper_right").id
        tip = topdown_ik.fk(self.q()).translation
        lines = []
        for oid in held_now | self.held_ids:
            bid = self.model.body(oid).id
            for c in self.data.contact[:self.data.ncon]:
                b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
                if bid in (b1, b2):
                    other = b2 if b1 == bid else b1
                    name = self.model.body(other).name
                    if other not in (left, right) and (oid, name) not in self.traced_hits:
                        self.traced_hits.add((oid, name))
                        lines.append(f"t {self.data.time:.2f} {oid} touches {name} at {np.round(c.pos, 3).tolist()} "
                                     f"tip {np.round(tip, 3).tolist()} cmd {self.last_cmd}")
        for oid in self.held_ids - held_now:
            z = float(self.data.body(oid).xpos[2])
            lines.append(f"t {self.data.time:.2f} {oid} LEFT the fingers, object z {z:.3f} tip {np.round(tip, 3).tolist()} "
                         f"grip {self.gripper():.3f} cmd {self.last_cmd}")
        for oid in held_now - self.held_ids:
            lines.append(f"t {self.data.time:.2f} {oid} BETWEEN the fingers, object z {float(self.data.body(oid).xpos[2]):.3f} "
                         f"tip {np.round(tip, 3).tolist()} grip {self.gripper():.3f} cmd {self.last_cmd}")
        # Anything not in the fingers that starts moving fast: what hit it (the arm's links, another thing).
        arm_bodies = {self.model.body(n).id for n in ("gripper_left", "gripper_right")}
        for o in self.world["objects"]:
            if o["id"] in held_now:
                continue
            j = self.model.joint(o["id"])
            v = float(np.linalg.norm(self.data.qvel[j.dofadr[0]:j.dofadr[0] + 3]))
            moving = v > 0.05
            if moving and o["id"] not in self.traced_moving:
                bid = self.model.body(o["id"]).id
                hits = sorted({self.model.body(self.model.geom_bodyid[c.geom2 if self.model.geom_bodyid[c.geom1] == bid
                                                                         else c.geom1]).name
                               for c in self.data.contact[:self.data.ncon]
                               if bid in (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2])})
                lines.append(f"t {self.data.time:.2f} {o['id']} MOVING {v:.2f} m/s at "
                             f"{np.round(self.data.body(o['id']).xpos, 3).tolist()} touching {hits} "
                             f"tip {np.round(tip, 3).tolist()} cmd {self.last_cmd}")
            if moving:
                self.traced_moving.add(o["id"])
            else:
                self.traced_moving.discard(o["id"])
        if held_now - self.held_ids:
            self.traced_hits = {h for h in self.traced_hits if h[0] not in held_now - self.held_ids}
        self.held_ids = held_now
        if lines:
            with open(TRACE_HELD, "a") as f:
                f.write("\n".join(lines) + "\n")

    def settle(self, seconds):
        self.step(seconds)

    def set_arm(self, q):
        for u, v in zip(self.arm_u, q):
            self.data.ctrl[u] = v

    def set_fingers(self, rad):
        for u in self.finger_u:
            self.data.ctrl[u] = mj_scene.finger_q(max(0.0, rad))

    def gravity_torque(self):
        """Gravity load on each arm joint now (N*m): what the real motors hold besides the move."""
        return [abs(float(self.data.qfrc_gravcomp[i])) for i in self.arm_v]

    # --- recording ----------------------------------------------------
    def start_recording(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        self.cameras = self.cameras or mj_scene.Cameras(self.model)
        for cam in self.a.record_cams.split(","):
            self.recorders[cam] = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{mj_scene.IMAGE[0]}x{mj_scene.IMAGE[1]}", "-r", str(self.a.fps), "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(folder / f"{cam}.mp4")],
                stdin=subprocess.PIPE)
        self.timeline = (folder / "timeline.jsonl").open("w", buffering=1)
        self.frames = 0
        self.next_frame = self.data.time

    def frame(self):
        for cam, proc in self.recorders.items():
            proc.stdin.write(self.cameras.render(self.data, cam).tobytes())
        self.frames += 1

    def mark(self, event, **fields):
        if self.timeline:
            self.timeline.write(json.dumps(dict(frame=self.frames, sim_time=round(self.data.time, 3),
                                                wall=time.time(), event=event, **fields), ensure_ascii=False) + "\n")

    def stop_recording(self):
        for proc in self.recorders.values():
            proc.stdin.close()
            proc.wait(timeout=120)
        self.recorders = {}
        if self.timeline:
            self.timeline.close()
            self.timeline = None

    # --- world file ---------------------------------------------------
    def save_world(self):
        for o in self.world["objects"]:
            body = self.data.body(o["id"])
            o["centre"] = [round(float(v), 5) for v in body.xpos[:2]]
            o["z"] = round(float(body.xpos[2]), 5)
            w, x, y, z = (float(v) for v in body.xquat)
            o["yaw"] = round(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)), 4)
            o["quat"] = [round(float(v), 5) for v in body.xquat]
            o["held"] = bool(self.between_fingers(o["id"]) and o["z"] > 0.01)
        tmp = self.world_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.world, ensure_ascii=False))
        tmp.replace(self.world_path)

    def between_fingers(self, body_name):
        """True when the object touches both finger pads."""
        bid = self.model.body(body_name).id
        left, right = self.model.body("gripper_left").id, self.model.body("gripper_right").id
        touched = set()
        for c in self.data.contact[:self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if bid in (b1, b2):
                touched |= {b1, b2} & {left, right}
        return len(touched) == 2

    # --- motion -------------------------------------------------------
    def move(self, target, correct):
        if self.fault:
            raise ValueError(f"faulted ({self.fault}); no further moves")
        current = self.q()
        points = arm_goto.plan(current, target, self.limits, self.a.max_speed)
        lowest = arm_goto.check_path(points, self.a.min_tcp_z)
        self.emit("goto_sent", start=current, target=target, duration_s=points[-1][0], lowest_tcp_z=lowest)
        print(f"移動開始: {points[-1][0]:.1f}秒、手先の最低高さ {lowest:.3f} m。何か入力すると停止。", flush=True)
        self.mark("move", target=target)
        dt = self.model.opt.timestep
        t, end = 0.0, points[-1][0]
        check_every = max(1, int(0.02 / dt))
        k = 0
        while t < end:
            self.set_arm(arm_goto.commanded_at(points, t))
            self.step(dt)
            t += dt
            k += 1
            if k % check_every == 0:
                expected = arm_goto.commanded_at(points, t)
                measured = self.q()
                worst = max(abs(m - e) for m, e in zip(measured, expected))
                for i, tau in enumerate(self.gravity_torque()):
                    self.peak_torque[i] = max(self.peak_torque[i], tau)
                if worst > self.a.max_tracking:
                    self.set_arm(measured)  # the driver holds where it is
                    self.step(0.2)
                    self.fault = f"tracking error {worst:.3f} rad exceeds {self.a.max_tracking}"
                    self.emit("fault", reason=self.fault, positions=self.q())
                    self.mark("fault", reason=self.fault)
                    self.save_world()
                    raise RuntimeError(self.fault)
        self.set_arm(target)
        waited, held, blocked = 0.0, 0.0, False
        while True:
            self.step(0.05)
            waited += 0.05
            if max(abs(m - t_) for m, t_ in zip(self.q(), target)) <= arm_goto.SETTLE:
                held += 0.05
                if held >= arm_goto.SETTLE_HOLD:
                    break
            else:
                held = 0.0
            if waited > SETTLE_S:
                if correct:
                    self.fault = "joints did not settle within 0.03 rad of the target"
                    self.save_world()
                    raise RuntimeError(self.fault)
                self.set_arm(self.q())  # a probing step pressed on something: hold there
                self.step(HOLD_S)
                self.emit("step_blocked", target=target, positions=self.q())
                blocked = True
                break
        if not blocked:
            # On the real arm time runs on while the client reads and thinks, so the last few
            # hundredths of a radian close by themselves; here time stops between commands, and a probe
            # step started from that lag read as contact (sim 2026-10-28: an 8 cm bottle "touched" 1 cm
            # above its cap). Let the joints come to rest first.
            for _ in range(int(REST_S / 0.05)):
                if max(abs(m - t_) for m, t_ in zip(self.q(), target)) <= REST_ERROR:
                    break
                self.step(0.05)
        measured = self.q()
        self.emit("goto_settled", target=target, command=target, positions=measured, blocked=blocked)
        self.save_world()
        print("到達: " + " ".join(f"{q:+.3f}" for q in measured), flush=True)

    # --- gripper ------------------------------------------------------
    def grip_open(self, target):
        if not 0.0 < target <= self.a.grip_open_max:
            raise ValueError(f"open target must be in (0, {self.a.grip_open_max}] rad")
        position = self.gripper()
        while True:  # in steps, like arm_goto.soft_open
            position += max(-arm_goto.GRIP_OPEN_STEP, min(arm_goto.GRIP_OPEN_STEP, target - position))
            self.set_fingers(position)
            self.step(OPEN_WAIT)
            if abs(target - position) < 1e-9:
                break
        self.mark("grip_open", target=target)
        self.save_world()
        print(f"グリッパ: {self.gripper():+.3f} rad（目標 {target:.2f}）", flush=True)

    def grip_close(self, squeeze):
        def read():
            self.step(GRIP_WAIT)
            return self.gripper()

        def command(target):
            self.set_fingers(target)

        result, final = arm_goto.soft_close(read, command, self.gripper(), squeeze)
        self.step(0.3)
        self.squeeze_target = final
        self.emit("grip_close", result=result, final_target=final, position=self.gripper())
        self.mark("grip_close", result=result)
        self.save_world()
        print(f"グリッパ: {result}、位置 {self.gripper():+.3f} rad、目標 {final:+.3f} rad", flush=True)

    def grip_resqueeze(self, target, squeeze):
        if not arm_goto.GRIP_SQUEEZE_RANGE[0] <= squeeze <= arm_goto.GRIP_SQUEEZE_RANGE[1]:
            raise ValueError(f"squeeze must be in {arm_goto.GRIP_SQUEEZE_RANGE} rad")
        new = arm_goto.resqueeze_target(self.gripper(), target, squeeze)
        if new is not None:
            self.set_fingers(new)
            self.step(GRIP_WAIT)
            target = new
        self.squeeze_target = target
        self.emit("grip_resqueeze", squeezed=new is not None, final_target=target, position=self.gripper())
        self.save_world()
        print(f"グリッパ: {'resqueezed' if new is not None else 'held'}、位置 {self.gripper():+.3f} rad、"
              f"目標 {target:+.3f} rad", flush=True)

    # --- extras -------------------------------------------------------
    def render(self, path, camera="video2"):
        import cv2
        self.cameras = self.cameras or mj_scene.Cameras(self.model)
        mujoco.mj_forward(self.model, self.data)
        cv2.imwrite(str(path), self.cameras.render(self.data, camera))
        self.mark("look", camera=camera, path=str(path))
        print(f"撮影: {path} {camera}", flush=True)

    def place(self, oid, x, y, yaw=None):
        """Drop an object at x, y from 5 cm (the scatterer's hand)."""
        j = self.model.joint(oid)
        adr = j.qposadr[0]
        yaw = float(np.random.default_rng().uniform(-math.pi, math.pi)) if yaw is None else yaw
        self.data.qpos[adr:adr + 7] = [x, y, 0.05, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.qvel[j.dofadr[0]:j.dofadr[0] + 6] = 0
        self.step(0.6)
        self.mark("scatter", object=oid, xy=[x, y])
        self.save_world()
        print(f"配置: {oid} ({x:+.3f}, {y:+.3f})", flush=True)

    # --- command loop -------------------------------------------------
    def run(self):
        print("commands: arm / goto / step / home / grip / render / place / world / finish / status / quit", flush=True)
        print("q=" + " ".join(f"{q:+.3f}" for q in self.q()), flush=True)
        try:
            for line in sys.stdin:
                words = line.split()
                if not words:
                    continue
                cmd = words[0]
                self.last_cmd = line.strip()[:60]
                try:
                    if cmd == "arm":
                        if any(abs(q) > arm_goto.FOLDED for q in self.q()):
                            raise ValueError("arm requires every joint within 0.1 rad of zero")
                        self.armed = True
                        self.emit("hold_checked", positions=self.q())
                        print("保持確認完了。goto で移動できます。", flush=True)
                    elif cmd in ("goto", "step", "home"):
                        if not self.armed:
                            raise ValueError("arm first")
                        target = [0.0] * 6 if cmd == "home" else arm_goto.parse_target(words, {})
                        self.move(target, correct=cmd != "step")
                    elif cmd == "grip":
                        if not self.armed or self.fault:
                            raise ValueError("arm first; no fault allowed")
                        if len(words) >= 2 and words[1] == "open":
                            self.grip_open(float(words[2]) if len(words) == 3 else arm_goto.GRIP_OPEN)
                        elif len(words) == 2 and words[1] == "state":
                            if TRACE_HELD:
                                held = [o["id"] for o in self.world["objects"]
                                        if self.between_fingers(o["id"]) and self.data.body(o["id"]).xpos[2] > 0.01]
                                with open(TRACE_HELD, "a") as f:
                                    f.write(f"t {self.data.time:.2f} grip state {self.gripper():.3f} target "
                                            f"{self.squeeze_target:.3f} truly held {held}\n")
                            print(f"グリッパ: state、位置 {self.gripper():+.3f} rad", flush=True)
                        elif len(words) in (2, 3, 4) and words[1] == "close":
                            squeeze = float(words[2]) if len(words) >= 3 else arm_goto.GRIP_SQUEEZE
                            if not arm_goto.GRIP_SQUEEZE_RANGE[0] <= squeeze <= arm_goto.GRIP_SQUEEZE_RANGE[1]:
                                raise ValueError(f"squeeze must be in {arm_goto.GRIP_SQUEEZE_RANGE} rad")
                            self.grip_close(squeeze)
                        elif len(words) == 4 and words[1] == "resqueeze":
                            self.grip_resqueeze(float(words[2]), float(words[3]))
                        else:
                            raise ValueError("use grip open [RAD], grip close [SQUEEZE] or grip resqueeze T S")
                    elif cmd == "render":
                        self.render(words[1], words[2] if len(words) > 2 else "video2")
                    elif cmd == "height":  # the depth camera: height above the desk per image pixel (.npy)
                        mujoco.mj_forward(self.model, self.data)
                        np.save(words[1], self.cameras.height_map(self.data, words[2] if len(words) > 2 else "video2"))
                        print(f"撮影: {words[1]} height", flush=True)
                    elif cmd == "snap":
                        folder = self.a.output_dir / "snaps"
                        folder.mkdir(exist_ok=True)
                        self.render(folder / f"{words[1]}_video2.jpg")
                    elif cmd == "place":
                        self.place(words[1], float(words[2]), float(words[3]),
                                   float(words[4]) if len(words) > 4 else None)
                    elif cmd == "world":
                        self.save_world()
                        print("世界: 保存", flush=True)
                    elif cmd == "mark":
                        self.mark("note", text=" ".join(words[1:]))
                        print("記録: " + " ".join(words[1:]), flush=True)
                    elif cmd == "finish":
                        if any(abs(q) > 0.04 for q in self.q()):
                            raise ValueError("use home first; finish needs every joint within 0.04 rad of zero")
                        self.armed = False
                        self.emit("normal_finish", peak_gravity_torque=self.peak_torque)
                        print("終了処理完了: safe_home -> disable", flush=True)
                    elif cmd == "status":
                        print("q=" + " ".join(f"{q:+.3f}" for q in self.q()), flush=True)
                    elif cmd in ("disable", "quit"):
                        self.emit("client_exit")
                        return 0
                    else:
                        raise ValueError(f"unknown command {cmd}")
                except ValueError as error:
                    print("指令拒否: " + str(error), flush=True)
                    self.emit("command_rejected", reason=str(error))
                except RuntimeError as error:
                    print(f"停止要求: {error}。新しい移動は禁止。", flush=True)
                    self.emit("fault_latched", reason=str(error))
        finally:
            self.save_world()
            self.stop_recording()
        return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--driver-config", required=True)
    p.add_argument("--min-tcp-z", type=float, default=0.08)
    p.add_argument("--max-speed", type=float, default=arm_goto.MAX_SPEED)
    p.add_argument("--max-tracking", type=float, default=arm_goto.MAX_TRACKING)
    p.add_argument("--grip-open-max", type=float, default=5.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record", help="folder for the videos and timeline.jsonl")
    p.add_argument("--record-cams", default="show,video2")
    p.add_argument("--fps", type=float, default=15.0)
    # accepted for drop-in compatibility with arm_goto.py
    for flag in ("--execute", "--gripper", "--correct", "--resume"):
        p.add_argument(flag, action="store_true")
    p.add_argument("--snap-cameras", default="/dev/video2")
    p.add_argument("--preflight")
    a = p.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    with (a.output_dir / "events.jsonl").open("x", buffering=1) as out:
        def emit(event, **fields):
            out.write(json.dumps(dict(event=event, wall_ns=time.time_ns(), **fields), ensure_ascii=False) + "\n")
        return MjArm(a, emit).run()


if __name__ == "__main__":
    raise SystemExit(main())
