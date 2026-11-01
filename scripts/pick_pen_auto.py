#!/usr/bin/env python3
"""Pick the purple marker and drop it in the box, retrying on a miss.

Drives scripts/arm_goto.py (which keeps every safety check) through its stdin,
the same way an operator would type commands. The arm hold trial driver must
already be running with scripts/config/arm_pick_trial.yaml.

Each attempt: capture video2 -> locate the pen -> hover -> open -> probe down.
The probe lowers the open gripper in PROBE_STEP moves along the approach axis
and stops once the tip lags its free-space track by PROBE_LAG (it touched
something), so no object or floor height has to be known exactly:
  - stopped near the floor: back up PROBE_LIFT and close
  - stopped well above the floor (finger on the pen): lift, shift sideways
    along the finger axis, probe again; if every shift stops at the same
    height, that height is the floor after all
  - close found nothing: re-detect from home and retry
then lift -> place -> open -> check video2 that the pen left the source.

    source scripts/ros2_env.sh
    python3 scripts/pick_pen_auto.py --preflight PREFLIGHT.json --out RUN_DIR --box-px 905 495
"""
import argparse
import json
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import arm_goto
import box_fetch
import handeye_solve
import plan_pen_pick
import topdown_ik

HERE = Path(__file__).resolve().parent
GRIP_WIDE = 1.5        # rad opening before the descent
TIP_REACH = 0.005      # m the real fingertips reach below the URDF gripper_end
PROBE_START = 0.012    # m above the pen top where the probe begins
PROBE_STEP = 0.003     # m per probe move along the approach axis near the desk
PROBE_COARSE = 0.008   # m per probe move higher up
PROBE_FINE_ZONE = 0.015  # m above the expected floor contact where the fine steps start
PROBE_DEPTH = 0.009    # m the deepest probe command goes below the expected floor contact
PROBE_LAG = 0.005      # m behind the free-space track -> contact
PROBE_LIFT = 0.003     # m backed off from the floor before closing
ON_OBJECT = 0.008      # m above the expected floor contact -> a finger is on the object


class Client:
    """arm_goto.py in a subprocess; send one command, wait for its result line."""

    def __init__(self, cmd, log):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=HERE)
        self.lines = queue.Queue()
        self.log = log
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.log.write(line)
            self.log.flush()
            self.lines.put(line.rstrip("\n"))

    def wait_ready(self, timeout=30):
        """Return once joint telemetry is flowing (first periodic `q=` line)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.lines.get(timeout=1).startswith("q="):
                    time.sleep(1.0)  # let the arm status arrive too
                    return
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError("arm_goto exited")
        raise RuntimeError("arm_goto never reported joint telemetry")

    def send(self, command, done_prefixes, timeout=180):
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=1)
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError("arm_goto exited")
                continue
            if line.startswith(("指令拒否", "停止要求")):
                raise RuntimeError(f"{command!r}: {line}")
            if line.startswith(done_prefixes):
                return line
        raise RuntimeError(f"{command!r}: no result within {timeout} s")


def capture(path):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "v4l2", "-input_format", "mjpeg",
                    "-video_size", "1280x960", "-framerate", "30", "-i", "/dev/video2", "-ss", "3",
                    "-frames:v", "1", str(path)], check=True, timeout=20)


BOX_RADIUS = 0.09  # m around the box centre where purple counts as "in the box"


# Box floor ~8 mm (user's estimate, not measured) under a ~18 mm marker.
FLOOR_Z = {"table": 0.0, "box": 0.008}
PEN_TOP = {"table": plan_pen_pick.PEN_TOP_Z, "box": 0.026}
# Sideways shifts (m) along the finger axis tried after a finger lands on the pen.
NUDGES = (0.0, 0.006, -0.006)  # +-12 mm hit the box wall on 2026-10-27


def box_filter(calib, box_xy, inside):
    """Reject blobs on the wrong side of the box footprint (inside=True keeps box contents)."""
    def reject(px):
        xy = handeye_solve.image_to_base(calib, px, plan_pen_pick.PEN_TOP_Z)[:2]
        return (math.dist(xy, box_xy) <= BOX_RADIUS) != inside
    return reject


def pen_seen(image, calib, box_xy, where):
    try:
        plan_pen_pick.find_pen(cv2.imread(str(image)), box_filter(calib, box_xy, where == "box"))
    except SystemExit:
        return False
    return True


def locate(image, calib, box_xy, source):
    centre, end1, end2 = plan_pen_pick.find_pen(cv2.imread(str(image)),
                                                box_filter(calib, box_xy, source == "box"))
    xy = lambda px: handeye_solve.image_to_base(calib, px, PEN_TOP[source])[:2]
    return xy(centre), xy(end2) - xy(end1)


def plan(c, axis, limits, grasp_z, **fit):
    g = plan_pen_pick.plan_grasp(c, axis, grasp_z, limits, **fit)
    if g is None:
        raise RuntimeError(f"pen at {np.round(c, 3)} is out of reach even tilted 40 deg")
    return g


def probe_steps(c, axis, limits, floor_z, top_z, **fit):
    """IK poses from PROBE_START above the pen top down past the expected floor contact.

    Returns (grasp plan, [(commanded tip z, q), ...]) along one straight approach line.
    """
    deepest = floor_z + TIP_REACH - PROBE_DEPTH
    g = plan(c, axis, limits, deepest, **fit)
    rot = (topdown_ik.pin.utils.rotate("z", g["yaw"])
           @ (topdown_ik.pitched(g["pitch"]) if g["pitch"] else topdown_ik.TOPDOWN)
           @ topdown_ik.pin.utils.rotate("x", g["roll"]))
    approach = rot[:, 0]  # points down (approach_z < 0)
    end = np.r_[c, deepest]
    length = (top_z + PROBE_START - deepest) / -approach[2]
    steps, q_prev = [], g["grasp"]
    # Fine steps only near the desk, where contact matters; coarser ones above (2026-10-28: a 4.5 cm
    # paper took 20 one-second steps).
    fine_until = (PROBE_FINE_ZONE + PROBE_DEPTH) / -approach[2]
    distances, s = [], 0.0
    while s < length:
        distances.append(s)
        s += PROBE_STEP if s < fine_until else PROBE_COARSE
    distances.append(length)  # the probe always starts PROBE_START above the object's top
    for s in distances:  # solved bottom-up from the grasp IK
        point = end - approach * s
        q, pos_err, ang_err = topdown_ik.solve(list(point), yaw=g["yaw"], seeds=[np.array(q_prev)],
                                               pitch=g["pitch"], roll=g["roll"])
        if (pos_err > 1e-3 or ang_err > 1e-2 or not all(lo <= v <= hi for v, (lo, hi) in zip(q, limits))
                or max(abs(a - b) for a, b in zip(q, q_prev)) > 0.1):
            raise RuntimeError(f"no smooth IK along the probe line at z {point[2]:.3f}")
        steps.append((float(point[2]), [float(v) for v in q]))
        q_prev = q
    return g, steps[::-1]


def probe(steps, move, tip_z, lag=None):
    """Lower step by step until the tip falls PROBE_LAG behind its free-space track.

    move(q) commands one settled step; tip_z() is the measured tip height after it.
    Returns (index of the last step sent, measured tip z there, free-space lag, contact?).
    """
    baseline = None
    for i, (z_cmd, q) in enumerate(steps):
        move(q)
        z = tip_z()
        if baseline is None:
            baseline = z - z_cmd  # sag/bias in free space at the start
        elif z - z_cmd - baseline > (PROBE_LAG if lag is None else lag):
            # A joint still closing in on its step reads as a lag too (2026-11-01 real: false contacts 12 mm
            # above the desk with the arm stretched): the same step once more, then read it again.
            move(q)
            z = tip_z()
            if z - z_cmd - baseline > PROBE_LAG:
                return i, z, baseline, True
    return len(steps) - 1, z, baseline, False


def back_off(steps, z_contact, baseline):
    """Index of the probe step whose free-space tip sits PROBE_LIFT above the contact."""
    want = z_contact + PROBE_LIFT
    return min(range(len(steps)), key=lambda i: abs(steps[i][0] + baseline - want))


def detect_box(image, calib, seed, outer=None):
    """(box geometry, seed pixel for the next frame) or (None, seed) when the outline is not found.

    With `outer` (m, known box size) only a rectangle of that size is accepted.
    """
    try:
        img = cv2.imread(str(image))
        corners = (box_fetch.find_box(img, seed) if outer is None
                   else box_fetch.find_box_sized(img, seed, calib, outer))
        box = box_fetch.box_geometry(calib, corners)
    except RuntimeError:
        return None, seed
    centre = np.r_[box["centre"], box_fetch.RIM_Z]
    pts = (np.array(calib["T_cam_base"]) @ np.r_[centre, 1.0])[None, :3]
    return box, handeye_solve.project(pts, calib["camera"])[0]


def pull_box(client, plan, a, fmt):
    """Grip the near wall and drag the box toward the base. Returns a result string."""
    q = dict(zip(plan["names"], plan["q"]))
    client.send(f"goto {fmt(q['hover'])}", ("到達",))
    client.send(f"grip open {a.grip_open}", ("グリッパ",), 30)
    client.send(f"step {fmt(q['grip'])}", ("到達",))
    tip_z = topdown_ik.fk(arm_goto_last_q(a.out)).translation[2]
    client.send("snap pull_grip", ("撮影",), 60)
    if tip_z > box_fetch.GRIP_Z + 0.012:
        client.send(f"goto {fmt(q['hover'])}", ("到達",))
        return f"finger_on_rim_at_{tip_z * 1000:.0f}mm"
    close = client.send("grip close", ("グリッパ",), 120)
    if "closed_empty" in close:
        client.send(f"grip open {a.grip_open}", ("グリッパ",), 30)
        client.send(f"goto {fmt(q['hover'])}", ("到達",))
        return "missed_the_wall"
    print("[pull] holding the wall, pulling", flush=True)
    for name in plan["names"]:
        if name.startswith("pull"):
            client.send(f"step {fmt(q[name])}", ("到達",))
    client.send("snap pull_done", ("撮影",), 60)
    client.send(f"grip open {a.grip_open}", ("グリッパ",), 30)
    client.send(f"goto {fmt(q['release_lift'])}", ("到達",))
    return "pulled"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preflight", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--calib", default=str(HERE / "calib/video2.json"))
    p.add_argument("--box-px", type=float, nargs=2, required=True)
    p.add_argument("--driver-config", default=str(HERE / "config/arm_pick_trial.yaml"))
    p.add_argument("--source", choices=("table", "box"), default="table", help="where the pen is now")
    p.add_argument("--dest", choices=("box", "table"), default="box", help="where to put it")
    p.add_argument("--place-xy", type=float, nargs=2, default=(0.30, 0.0), help="table spot for --dest table")
    p.add_argument("--floor-z", type=float, help="expected support height under the pen (default per --source)")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--grip-open", type=float, help="opening before the descent (default 1.5 table, 1.2 box)")
    p.add_argument("--max-speed", type=float, default=0.15)
    p.add_argument("--fetch-beyond", type=float, default=0.40,
                   help="pull the box closer first when its centre is farther than this (m); 0 disables")
    a = p.parse_args()
    if a.source == a.dest:
        p.error("--source and --dest must differ")
    a.floor_z = a.floor_z if a.floor_z is not None else FLOOR_Z[a.source]
    a.grip_open = a.grip_open or (GRIP_WIDE if a.source == "table" else 1.2)
    a.out = a.out.resolve()
    a.preflight = str(Path(a.preflight).resolve())
    a.driver_config = str(Path(a.driver_config).resolve())
    a.out.mkdir(parents=True, exist_ok=False)
    calib = json.load(open(a.calib))
    limits = arm_goto.driver_limits(a.driver_config)
    box_xy = handeye_solve.image_to_base(calib, a.box_px, plan_pen_pick.BOX_RIM_Z)[:2]
    box_q = None
    for pitch in (0.0, 0.4, 0.6, 0.8):
        q, pos_err, _ = topdown_ik.solve([box_xy[0], box_xy[1], 0.13], yaw=math.atan2(box_xy[1], box_xy[0]),
                                         seeds=plan_pen_pick.SEEDS, pitch=pitch)
        if pos_err < 1e-3 and all(lo <= v <= hi for v, (lo, hi) in zip(q, limits)):
            box_q = [float(v) for v in q]
            break
    if box_q is None and a.dest == "box":
        raise SystemExit("box out of reach")
    place = None
    if a.dest == "table":
        place = {}
        for name, z in (("above", 0.10), ("down", 0.03)):
            q, pos_err, ang_err = topdown_ik.solve([a.place_xy[0], a.place_xy[1], z], seeds=plan_pen_pick.SEEDS)
            if pos_err > 1e-3 or not all(lo <= v <= hi for v, (lo, hi) in zip(q, limits)):
                raise SystemExit("place spot out of reach")
            place[name] = [float(v) for v in q]
    log = open(a.out / "client.log", "w")
    client = Client([sys.executable, "-u", "arm_goto.py", "--execute", "--gripper", "--correct",
                     "--max-speed", str(a.max_speed), "--driver-config", a.driver_config,
                     "--min-tcp-z", str(a.floor_z + TIP_REACH - PROBE_DEPTH - 0.002), "--snap-cameras", "/dev/video4,/dev/video0",
                     "--preflight", a.preflight, "--output-dir", str(a.out / "client")], log)
    fmt = lambda q: " ".join(f"{v:.4f}" for v in q)
    report = {"attempts": []}
    try:
        client.wait_ready()
        client.send("arm", ("保持確認完了",), 60)
        seed = np.array(a.box_px, float)
        if a.source == "box":
            image = a.out / "scene_0.jpg"
            capture(image)
            box, seed = detect_box(image, calib, seed)
            if box is not None:
                box_xy = box["centre"]
                far = float(np.linalg.norm(box_xy))
                print(f"[box] centre ({box_xy[0]:+.3f}, {box_xy[1]:+.3f}), {far * 100:.0f} cm from the base", flush=True)
                if a.fetch_beyond and far > a.fetch_beyond:
                    plan_ = box_fetch.plan_pull(box, limits)
                    print(f"[pull] gripping the wall at ({plan_['wall']['mid'][0]:+.3f}, "
                          f"{plan_['wall']['mid'][1]:+.3f}), pulling {plan_['distance'] * 100:.0f} cm", flush=True)
                    report["pull"] = pull_box(client, plan_, a, fmt)
                    print(f"[pull] {report['pull']}", flush=True)
                    if report["pull"] == "pulled":  # look for the box where it was dragged to
                        box_xy = box_xy - box_xy / far * plan_["distance"]
                        pts = (np.array(calib["T_cam_base"]) @ np.r_[box_xy, box_fetch.RIM_Z, 1.0])[None, :3]
                        seed = handeye_solve.project(pts, calib["camera"])[0]
                    client.send("home", ("到達",))  # clear view for re-detection
        for attempt in range(1, a.attempts + 1):
            image = a.out / f"scene_{attempt}.jpg"
            capture(image)
            if a.source == "box":
                box, seed = detect_box(image, calib, seed)
                if box is not None:
                    box_xy = box["centre"]
            if not pen_seen(image, calib, box_xy, a.source):
                report["result"] = f"no_pen_in_{a.source}"
                print(f"[{attempt}] no pen in the {a.source}", flush=True)
                break
            centre, axis = locate(image, calib, box_xy, a.source)
            entry = {"attempt": attempt, "pen_xy": centre.tolist(), "tries": []}
            report["attempts"].append(entry)
            floor_z, seated = a.floor_z, None
            for pass_no in range(2):  # second pass only after re-estimating the floor
                contacts = []
                for nudge in NUDGES:
                    g0 = plan(centre, axis, limits, floor_z)
                    finger = np.array(topdown_ik.fk(g0["grasp"]).rotation[:2, 1])
                    finger /= np.linalg.norm(finger)
                    poses, steps = probe_steps(centre + finger * nudge, axis, limits, floor_z, PEN_TOP[a.source])
                    print(f"[{attempt}] pen at ({centre[0]:+.3f}, {centre[1]:+.3f}); tilt "
                          f"{math.degrees(poses['pitch']):.0f} deg, shift {nudge * 1000:+.0f} mm, "
                          f"floor guess {floor_z * 1000:.0f} mm", flush=True)
                    client.send(f"goto {fmt(poses['hover'])}", ("到達",))
                    client.send(f"grip open {a.grip_open}", ("グリッパ",), 30)
                    print(f"[{attempt}] probing down", flush=True)
                    i, z, baseline, touched = probe(
                        steps, lambda q: client.send(f"step {fmt(q)}", ("到達",)),
                        lambda: topdown_ik.fk(arm_goto_last_q(a.out)).translation[2])
                    expected = floor_z + TIP_REACH
                    entry["tries"].append({"shift_mm": nudge * 1000, "pitch_deg": math.degrees(poses["pitch"]),
                                           "floor_guess": floor_z, "stop_tip_z": z, "contact": touched,
                                           "free_space_lag": baseline})
                    print(f"[{attempt}] {'contact' if touched else 'no contact'} at tip {z * 1000:.1f} mm "
                          f"(floor contact expected {expected * 1000:.1f} mm)", flush=True)
                    if touched:  # side views of what the fingers touched
                        label = (f"a{attempt}_{pass_no}_s{round(nudge * 1000):+d}"  # unique: ffmpeg never overwrites
                                 .replace("+", "p").replace("-", "m"))
                        entry["tries"][-1]["snap"] = client.send(f"snap {label}", ("撮影",), 60)
                    if not touched or z <= expected + ON_OBJECT:
                        j = back_off(steps, z, baseline) if touched else i
                        client.send(f"step {fmt(steps[j][1])}", ("到達",))
                        seated = poses
                        break
                    contacts.append(z)
                    client.send(f"goto {fmt(poses['hover'])}", ("到達",))
                if seated is not None or max(contacts) - min(contacts) > 0.005:
                    break
                # Every shift stopped at the same height: that is the floor, not the pen.
                floor_z = min(contacts) - TIP_REACH
                print(f"[{attempt}] every shift stopped near {min(contacts) * 1000:.0f} mm: "
                      f"floor re-estimated at {floor_z * 1000:.0f} mm", flush=True)
            if seated is None:
                entry["result"] = "stalled_every_shift"
                client.send("home", ("到達",))  # out of the camera view before re-detecting
                continue
            poses = seated
            close = client.send("grip close", ("グリッパ",), 120)
            entry["close"] = close
            client.send(f"goto {fmt(poses['hover'])}", ("到達",))
            if "closed_empty" in close:
                entry["result"] = "closed_empty"
                print(f"[{attempt}] nothing between the fingers, retry", flush=True)
                client.send(f"grip open {a.grip_open}", ("グリッパ",), 30)
                client.send("home", ("到達",))
                continue
            if a.dest == "box":
                client.send(f"goto {fmt(box_q)}", ("到達",))
                client.send("grip open 1.2", ("グリッパ",), 30)
            else:
                client.send(f"goto {fmt(place['above'])}", ("到達",))
                client.send(f"goto {fmt(place['down'])}", ("到達",))
                client.send("grip open 1.2", ("グリッパ",), 30)
                client.send(f"goto {fmt(place['above'])}", ("到達",))
            check = a.out / f"after_{attempt}.jpg"
            capture(check)
            if pen_seen(check, calib, box_xy, a.source):
                entry["result"] = f"pen_still_in_{a.source}"
                print(f"[{attempt}] the pen is still in the {a.source}, retry", flush=True)
                client.send("home", ("到達",))
                continue
            entry["result"] = f"moved_to_{a.dest}"
            print(f"[{attempt}] pen moved to the {a.dest}", flush=True)
            break
        client.send("home", ("到達",))
        client.send("finish", ("終了処理完了",), 60)
    except Exception as error:
        print(f"中断: {error}", flush=True)
        try:  # best effort: fold the arm and release torque through the same safe path
            client.send("home", ("到達",))
            client.send("finish", ("終了処理完了",), 60)
        except Exception as cleanup:
            print(f"後片付けもできません: {cleanup}。arm_goto.py --resume で戻すか、電源を切ってください。",
                  flush=True)
        raise
    finally:
        (a.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(json.dumps(report, ensure_ascii=False), flush=True)


def arm_goto_last_q(out):
    events = [json.loads(l) for l in open(out / "client/events.jsonl")]
    return [e for e in events if e["event"] == "goto_settled"][-1]["positions"]


if __name__ == "__main__":
    main()
