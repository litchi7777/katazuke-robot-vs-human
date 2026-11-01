#!/usr/bin/env python3
"""Plan a pick of a purple marker seen by the calibrated video2 camera (offline).

Finds the pen by colour, maps its centre and axis to robot XY with the
hand-eye calibration, and writes arm_goto poses: pen_hover (z 0.10),
pen_grasp (tip at --grasp-z) with the wrist turned so the fingers close across
the pen, and box_drop above the box pixel given on the command line.

    source scripts/ros2_env.sh
    python3 scripts/plan_pen_pick.py IMAGE CALIB_JSON --box-px 905 495 --out POSES_JSON
"""
import argparse
import json
import math

import cv2
import numpy as np

import arm_goto
import handeye_solve
import topdown_ik

SEEDS = [np.array([0, 1.8, 1.6, -1.4, 0, 0]), np.array([0, 1.6, 1.2, -1.1, 0, 0]),
         np.array([0, 1.4, 0.9, -0.9, 0, 0]), np.array([0, 1.2, 0.6, -0.7, 0, 0])]
PEN_TOP_Z = 0.015  # top surface of a ~1.5 cm marker lying on the table
BOX_RIM_Z = 0.05


def find_pen(img, exclude=None):
    """Largest purple blob (>= 800 px); `exclude(centre_px)` rejects e.g. blobs in the box."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (115, 60, 40), (165, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    blobs = [i for i in range(1, n) if stats[i, 4] >= 800 and not (exclude and exclude(cents[i]))]
    if not blobs:
        raise SystemExit("no purple object found")
    i = max(blobs, key=lambda k: stats[k, 4])
    ys, xs = np.nonzero(labels == i)
    pts = np.stack([xs, ys], 1).astype(float)
    centre = pts.mean(0)
    direction = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
    proj = (pts - centre) @ direction
    return centre, centre + direction * proj.min(), centre + direction * proj.max()


PITCHES = (0.0, 0.17, 0.35, 0.52, 0.7)  # rad: straight down first, then tilt up to 40 deg


def across_width(size, offset):
    """Object extent across the fingers when they are `offset` rad off perpendicular (size = length, width)."""
    length, width = size
    return width * abs(math.cos(offset)) + length * abs(math.sin(offset))


FINGER_TILT_MAX = 0.13   # |z| of the finger line: one finger at most ~3 mm lower than the other
TILT_DIRS = tuple(math.radians(d) for d in range(0, 360, 5))
ROLLS = tuple(math.radians(d) for d in (0, 5, -5, 10, -10, 15, -15, 20, -20, 25, -25, 30, -30))


def grasp_rotation(yaw, pitch, roll):
    return (topdown_ik.pin.utils.rotate("z", yaw) @ (topdown_ik.pitched(pitch) if pitch else topdown_ik.TOPDOWN)
            @ topdown_ik.pin.utils.rotate("x", roll))


def plan_grasp(xy, axis_xy, grasp_z, limits, backoff=0.09, size=None, max_across=None, max_align=math.radians(25),
               max_ik=40, accept=None):
    """Least-tilted grasp whose finger line stays (nearly) horizontal.

    Candidates are a tilt direction (yaw, every 5 deg) and a small wrist roll.
    A rolled wrist tilts the finger line, so one finger reaches the desk first
    (1 cm in simulation, 2026-10-27); the roll is kept to |finger z| <=
    FINGER_TILT_MAX. The misalignment between the fingers and the object's
    perpendicular is at most max_align, and with `size` (length, width in m)
    the object must fit within max_across between the fingers. For each tilt,
    candidates are tried best-fit first; the hover pose backs off 9 cm along
    the approach axis. `accept(candidate)` can veto a solved candidate (e.g. it
    would hit a known obstacle). Returns a dict (pitch, yaw, roll, align_err,
    across, hover, grasp) or None.
    """
    theta = math.atan2(axis_xy[1], axis_xy[0])
    perp = np.array([-math.sin(theta), math.cos(theta)])
    for pitch in PITCHES:
        candidates = []
        for yaw in (TILT_DIRS if pitch else TILT_DIRS[:36]):  # straight down: yaw and yaw+pi are the same
            for roll in (ROLLS if pitch else (0.0,)):
                R = grasp_rotation(yaw, pitch, roll)
                finger = R[:, 1]
                if abs(finger[2]) > FINGER_TILT_MAX:
                    continue
                h = finger[:2] / np.linalg.norm(finger[:2])
                align = math.acos(min(1.0, abs(float(h @ perp))))
                if align > max_align:
                    continue
                across = None if size is None else across_width(size, align)
                if across is not None and max_across is not None and across > max_across:
                    continue
                candidates.append(((across or 0.0) + 0.01 * abs(roll), align, yaw, roll, R))
        candidates.sort(key=lambda c: c[:2])
        for _, align, yaw, roll, R in candidates[:max_ik]:
            grasp = np.r_[xy, grasp_z]
            hover = grasp - R[:, 0] * backoff  # back along the approach axis
            out = {"pitch": pitch, "yaw": yaw, "roll": roll, "align_err": align,
                   "across": None if size is None else across_width(size, align)}
            for name, point in (("hover", hover), ("grasp", grasp)):
                q, pos_err, ang_err = topdown_ik.solve(list(point), yaw=yaw, seeds=SEEDS, pitch=pitch, roll=roll)
                if pos_err > 1e-3 or ang_err > 1e-2 or not all(lo <= v <= hi for v, (lo, hi) in zip(q, limits)):
                    break
                out[name] = [float(v) for v in q]
            else:
                if accept is None or accept(out):
                    return out
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("calib")
    p.add_argument("--box-px", type=float, nargs=2, required=True, help="box centre pixel in the image")
    p.add_argument("--driver-config", default="config/arm_pick_trial.yaml")
    p.add_argument("--grasp-z", type=float, default=0.007)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    calib = json.load(open(a.calib))
    centre, end1, end2 = find_pen(cv2.imread(a.image))
    xy = lambda px: handeye_solve.image_to_base(calib, px, PEN_TOP_Z)[:2]
    c, axis = xy(centre), xy(end2) - xy(end1)
    theta = math.atan2(axis[1], axis[0])
    yaw = (theta + math.pi / 2) % math.pi - math.pi / 2  # fingers close across the pen
    print(f"pen centre XY ({c[0]:+.4f}, {c[1]:+.4f}) length {np.linalg.norm(axis) * 1000:.0f} mm "
          f"axis {math.degrees(theta):+.1f} deg -> wrist yaw {math.degrees(yaw):+.1f} deg")
    limits = arm_goto.driver_limits(a.driver_config)
    poses = {}
    for name, z in (("pen_hover", 0.10), ("pen_grasp", a.grasp_z)):
        q, pos_err, ang_err = topdown_ik.solve([c[0], c[1], z], yaw=yaw, seeds=SEEDS)
        if pos_err > 1e-3 or ang_err > 1e-2:
            raise SystemExit(f"no IK for {name}")
        poses[name] = [round(float(v), 4) for v in q]
    bx, by = handeye_solve.image_to_base(calib, a.box_px, BOX_RIM_Z)[:2]
    for pitch in (0.0, 0.4, 0.6, 0.8):
        q, pos_err, _ = topdown_ik.solve([bx, by, 0.13], yaw=math.atan2(by, bx), seeds=SEEDS, pitch=pitch)
        if pos_err < 1e-3 and all(lo <= v <= hi for v, (lo, hi) in zip(q, limits)):
            poses["box_drop"] = [round(float(v), 4) for v in q]
            print(f"box XY ({bx:+.4f}, {by:+.4f}) drop pitch {pitch}")
            break
    else:
        raise SystemExit("box is out of reach")
    current = [0.0] * 6
    for name in ("pen_hover", "pen_grasp", "pen_hover", "box_drop"):
        lowest = arm_goto.check_path(arm_goto.plan(current, poses[name], limits), a.grasp_z - 0.002)
        current = poses[name]
        print(f"{name:9s} {poses[name]} lowest tip {lowest:.3f} m")
    json.dump(poses, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
