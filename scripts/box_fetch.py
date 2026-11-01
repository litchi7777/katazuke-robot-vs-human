#!/usr/bin/env python3
"""Find the open box in a video2 frame and plan pulling it closer by its near wall.

The gripper rail is 18.4 cm wide and sits 7-10 cm behind the fingertips, so a
tilted reach into a far box lands the rail on the rim (2026-10-27). Pulled
closer, the pen inside can be picked straight down with the rail well above
the rim.

    source scripts/ros2_env.sh
    python3 scripts/box_fetch.py IMAGE [--box-px U V]   # prints the box and the pull plan (offline)
"""
import argparse
import json
import itertools
import math
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin

import arm_goto
import handeye_solve
import plan_pen_pick
import topdown_ik

HERE = Path(__file__).resolve().parent
RIM_Z = 0.055          # m, rim height guess (only shifts the image->XY mapping by ~1 mm/cm)
WALL = 0.008           # m, wall thickness
GRIP_Z = 0.025         # m, URDF tip height while holding the wall
PULL_TO = 0.30         # m, radial distance of the gripped wall after the pull
WALL_TILT = 0.35       # rad; the rail middle stays ~57 mm above the tip
STEP = 0.02            # m between pull waypoints


def _lines(img, seed, half):
    x0, y0 = (int(max(0, v - half)) for v in seed)
    roi = cv2.cvtColor(img[y0:y0 + 2 * half, x0:x0 + 2 * half], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(roi, (5, 5), 0), 30, 90)
    found = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=70, maxLineGap=10)
    if found is None:
        return []
    return [l.astype(float) + [x0, y0, x0, y0] for l in found[:, 0]]


def find_box(img, seed, half=240):
    """Pixel corners of the box rim's outer edges, as 4 points in order around the box.

    Hough lines near `seed` are grouped by direction; the two groups closest to
    perpendicular with the most total length are the box sides, and in each the
    two outermost lines are the outer rim edges.
    """
    lines = _lines(img, seed, half)
    angle = lambda l: math.degrees(math.atan2(l[3] - l[1], l[2] - l[0])) % 180
    length = lambda l: math.hypot(l[2] - l[0], l[3] - l[1])
    groups = []
    for l in lines:
        for g in groups:
            if min(abs(angle(l) - g["a"]), 180 - abs(angle(l) - g["a"])) < 8:
                g["lines"].append(l)
                break
        else:
            groups.append({"a": angle(l), "lines": [l]})
    best = None
    for i, g in enumerate(groups):
        for h in groups[i + 1:]:
            d = abs(g["a"] - h["a"])
            if abs(min(d, 180 - d) - 90) > 12:
                continue
            score = sum(map(length, g["lines"] + h["lines"]))
            if best is None or score > best[0]:
                best = (score, g, h)
    if best is None:
        raise RuntimeError("no box outline near the seed pixel")
    sides = []
    for g in best[1:]:
        a = math.radians(g["a"])
        normal = np.array([-math.sin(a), math.cos(a)])
        offsets = [normal @ l[:2] for l in g["lines"]]
        sides.append((g["lines"][int(np.argmin(offsets))], g["lines"][int(np.argmax(offsets))]))
    if min(abs(np.array([-math.sin(math.radians(g["a"])), math.cos(math.radians(g["a"]))]) @ (s[1][:2] - s[0][:2]))
           for g, s in zip(best[1:], sides)) < 60:
        raise RuntimeError("box outline found only on one side")

    def cross(l1, l2):
        p, r = l1[:2], l1[2:] - l1[:2]
        q, s = l2[:2], l2[2:] - l2[:2]
        return p + np.cross(q - p, s) / np.cross(r, s) * r

    (a1, a2), (b1, b2) = sides
    return np.array([cross(a1, b1), cross(a1, b2), cross(a2, b2), cross(a2, b1)])


def _cross(l1, l2):
    p, r = l1[:2], l1[2:] - l1[:2]
    q, s = l2[:2], l2[2:] - l2[:2]
    return p + np.cross(q - p, s) / np.cross(r, s) * r


def find_box_sized(img, seed, calib, outer, tol=0.02, rim_z=RIM_Z, halves=(240, 320, 400)):
    """find_box_sized_once with a growing search window (a short edge can hide in a small one)."""
    for half in halves:
        try:
            return find_box_sized_once(img, seed, calib, outer, tol, rim_z, half)
        except RuntimeError:
            continue
    raise RuntimeError("no box of the expected size near the seed pixel")


def find_box_sized_once(img, seed, calib, outer, tol=0.02, rim_z=RIM_Z, half=240):
    """Like find_box, but only accepts a rectangle whose sides match the known outer size (m).

    Every pair of near-perpendicular line groups and every pair of lines within
    each group is tried; edges of nearby objects then cannot stretch the box.
    The best fit that contains the seed pixel wins.
    """
    lines = _lines(img, seed, half)
    angle = lambda l: math.degrees(math.atan2(l[3] - l[1], l[2] - l[0])) % 180
    groups = []
    for l in lines:
        for g in groups:
            if min(abs(angle(l) - g["a"]), 180 - abs(angle(l) - g["a"])) < 8:
                g["lines"].append(l)
                break
        else:
            groups.append({"a": angle(l), "lines": [l]})
    to_xy = lambda px: handeye_solve.image_to_base(calib, px, rim_z)[:2]
    best = None
    for i, g in enumerate(groups):
        for h in groups[i + 1:]:
            d = abs(g["a"] - h["a"])
            if abs(min(d, 180 - d) - 90) > 12:
                continue
            for a1, a2 in itertools.combinations(g["lines"][:12], 2):
                for b1, b2 in itertools.combinations(h["lines"][:12], 2):
                    try:
                        px = np.array([_cross(a1, b1), _cross(a1, b2), _cross(a2, b2), _cross(a2, b1)])
                    except (ZeroDivisionError, FloatingPointError):
                        continue
                    if not np.isfinite(px).all():
                        continue
                    xy = np.array([to_xy(c) for c in px])
                    sides = [np.linalg.norm(xy[k] - xy[(k + 1) % 4]) for k in range(4)]
                    err = max(abs(sd - outer) for sd in sides)
                    if err > tol:
                        continue
                    hull = cv2.convexHull(px.astype(np.float32))
                    if cv2.pointPolygonTest(hull, (float(seed[0]), float(seed[1])), False) < 0:
                        continue
                    if best is None or err < best[0]:
                        best = (err, px)
    if best is None:
        raise RuntimeError("no box of the expected size near the seed pixel")
    return best[1]


def box_geometry(calib, corners_px, rim_z=RIM_Z):
    """Base-frame box: corners, centre, and walls sorted nearest first.

    Each wall: mid (rim point on the wall centre line), along (unit), inward (unit, into the box).
    """
    pts = np.array([handeye_solve.image_to_base(calib, c, rim_z)[:2] for c in corners_px])
    centre = pts.mean(0)
    walls = []
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        along = (b - a) / np.linalg.norm(b - a)
        inward = np.array([along[1], -along[0]])
        if inward @ (centre - (a + b) / 2) < 0:
            inward = -inward
        mid = (a + b) / 2 + inward * WALL / 2
        walls.append({"mid": mid, "along": along, "inward": inward, "length": float(np.linalg.norm(b - a))})
    walls.sort(key=lambda w: np.linalg.norm(w["mid"]))
    return {"corners": pts, "centre": centre, "walls": walls}


def wall_frame(xy, inward, tilt):
    """Approach tilted `tilt` outward along the radial; fingers close across the wall."""
    radial = xy / np.linalg.norm(xy)
    app = np.r_[radial * math.sin(tilt), -math.cos(tilt)]
    y = np.r_[inward, 0.0]
    y -= app * (app @ y)
    y /= np.linalg.norm(y)
    return np.column_stack([app, y, np.cross(app, y)])


def solve_pose(position, rot, seeds):
    """Damped least squares IK for a full gripper_end pose; returns (q6, pos_err, ang_err)."""
    target = pin.SE3(rot, np.asarray(position, float))
    m, d, fid = topdown_ik._model, topdown_ik._data, topdown_ik._fid
    best = None
    for seed in seeds:
        q6 = np.clip(np.array(seed, float), topdown_ik._lo, topdown_ik._hi)
        for _ in range(400):
            q = np.zeros(m.nq)
            q[:6] = q6
            pin.framesForwardKinematics(m, d, q)
            err = pin.log6(d.oMf[fid].actInv(target)).vector
            if np.linalg.norm(err) < 1e-4:
                break
            pin.computeJointJacobians(m, d, q)
            J = pin.getFrameJacobian(m, d, fid, pin.LOCAL)[:, :6]
            q6 = np.clip(q6 + 0.5 * J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err),
                         topdown_ik._lo, topdown_ik._hi)
        T = topdown_ik.fk(q6)
        pos_err = float(np.linalg.norm(T.translation - target.translation))
        ang_err = float(np.linalg.norm(pin.log3(T.rotation.T @ rot)))
        if best is None or pos_err + 0.05 * ang_err < best[1] + 0.05 * best[2]:
            best = (q6, pos_err, ang_err)
    return best


def plan_pull(box, limits, pull_to=PULL_TO):
    """Joint waypoints: hover over the near wall, grip height, then a straight pull toward the base.

    Tries the two nearest walls and both finger directions; returns the first fully
    reachable plan as a dict, or raises.
    """
    for wall in box["walls"][:2]:
        xy = wall["mid"]
        radial = xy / np.linalg.norm(xy)
        distance = float(np.linalg.norm(xy) - pull_to)
        for sign in (1, -1):
            rot = wall_frame(xy, sign * wall["inward"], WALL_TILT)
            seeds = plan_pen_pick.SEEDS
            names, points = [], []
            hover = np.r_[xy, GRIP_Z] - rot[:, 0] * 0.09
            names.append("hover"), points.append(hover)
            names.append("grip"), points.append(np.r_[xy, GRIP_Z])
            n = max(1, math.ceil(distance / STEP))
            for k in range(1, n + 1):
                names.append(f"pull{k}"), points.append(np.r_[xy - radial * distance * k / n, GRIP_Z])
            end = points[-1]
            names.append("release_lift"), points.append(end - rot[:, 0] * 0.09)
            qs, prev, ok = [], None, True
            for name, point in zip(names, points):
                q, pos_err, ang_err = solve_pose(point, rot, [prev] if prev is not None else seeds)
                # Consecutive pull waypoints must stay on one IK branch (small joint steps).
                jump = max(abs(a - b) for a, b in zip(q, prev)) if name.startswith("pull") else 0.0
                if (pos_err > 1e-3 or ang_err > 1e-2 or not all(lo <= v <= hi for v, (lo, hi) in zip(q, limits))
                        or jump > 0.2):
                    ok = False
                    break
                qs.append([float(v) for v in q])
                prev = q
            if ok:
                return {"wall": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in wall.items()},
                        "distance": distance, "names": names, "q": qs}
    raise RuntimeError("no reachable pull plan from the two nearest walls")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("--calib", default=str(HERE / "calib/video2.json"))
    p.add_argument("--box-px", type=float, nargs=2, default=(960, 585), help="search seed pixel")
    p.add_argument("--driver-config", default=str(HERE / "config/arm_pick_trial.yaml"))
    p.add_argument("--overlay")
    a = p.parse_args()
    img = cv2.imread(a.image)
    corners = find_box(img, a.box_px)
    box = box_geometry(json.load(open(a.calib)), corners)
    print("corners px", np.round(corners).tolist())
    print("corners XY", np.round(box["corners"], 3).tolist(), "centre", np.round(box["centre"], 3).tolist())
    plan = plan_pull(box, arm_goto.driver_limits(a.driver_config))
    print(f"pull wall at {np.round(plan['wall']['mid'], 3).tolist()} by {plan['distance'] * 1000:.0f} mm, "
          f"{len(plan['q'])} waypoints")
    lowest = min(arm_goto.check_path(arm_goto.plan(q0, q1, arm_goto.driver_limits(a.driver_config)), -1)
                 for q0, q1 in zip(plan["q"], plan["q"][1:]))
    print(f"lowest tip on the pull path {lowest:.3f} m")
    if a.overlay:
        cv2.polylines(img, [corners.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.imwrite(a.overlay, img)


if __name__ == "__main__":
    main()
