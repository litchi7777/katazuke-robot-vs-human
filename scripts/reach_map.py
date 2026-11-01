#!/usr/bin/env python3
"""Where on the desk can the gripper reach, for a given set of joint limits? (offline)

For each desk point a grasp is searched: tilt 0-60 deg toward any direction,
fingers level (roll 0). A solution counts only if it is inside the limits and
collision free: no robot link within SELF_GAP of a non-adjacent link (pairs as
in rebotarm_rs.srdf) and no link below the desk. Collision is checked on
points sampled over the URDF collision meshes, so a gap thinner than the
sampling (~3 mm) can be missed.

    source scripts/ros2_env.sh
    python3 scripts/reach_map.py --out OUT_DIR [--limits NAME=YAML_OR_urdf ...]
"""
import argparse
import itertools
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
from scipy.spatial import cKDTree

import arm_goto
import handeye_solve
import plan_pen_pick
import topdown_ik

HERE = Path(__file__).resolve().parent
MESHES = HERE.parent / "rebotarm_ros2/src/rebotarm_bringup/description/RS/meshes/shared"
LINKS = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_end"]
ADJACENT = {("base_link", "link1"), ("link1", "link2"), ("link2", "link3"), ("link3", "link4"),
            ("link4", "link5"), ("link5", "link6"), ("link6", "gripper_end")}
SELF_GAP = 0.003      # m closest allowed approach of two non-adjacent links
DESK_GAP = 0.0        # m every link point (except the base and the fingertip region) stays above the desk
GRASP_Z = 0.01        # m tip height of the grasp tested on the desk
PITCHES = tuple(math.radians(d) for d in (0, 10, 20, 30, 40, 50, 60))
TILT_DIRS = tuple(math.radians(d) for d in range(0, 360, 15))


def stl_points(path, n=1500, seed=0):
    """n points sampled uniformly over an STL surface."""
    data = Path(path).read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    tri = np.frombuffer(data[84:84 + count * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    v = tri["v"].reshape(-1, 3, 3).astype(float)
    area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(v), n, p=area / area.sum())
    r1, r2 = rng.uniform(size=(2, n))
    s = np.sqrt(r1)
    return (v[pick, 0] * (1 - s)[:, None] + v[pick, 1] * (s * (1 - r2))[:, None] + v[pick, 2] * (s * r2)[:, None])


class Body:
    """The arm's collision surfaces, posed by joint angles."""

    def __init__(self):
        self.model, self.data = topdown_ik._model, topdown_ik._data
        self.frames = {name: self.model.getFrameId(name) for name in LINKS}
        self.points = {name: stl_points(MESHES / f"{name}.STL", 1500 if name != "base_link" else 800)
                       for name in LINKS}

    def posed(self, q6):
        q = np.zeros(self.model.nq)
        q[:6] = q6
        pin.framesForwardKinematics(self.model, self.data, q)
        out = {}
        for name, fid in self.frames.items():
            T = self.data.oMf[fid]
            out[name] = self.points[name] @ T.rotation.T + T.translation
        return out

    def collisions(self, q6):
        """List of problems: 'self:a-b' (links too close) and 'desk:link'."""
        posed = self.posed(q6)
        problems = []
        for name, pts in posed.items():
            if name == "base_link":
                continue
            low = pts[:, 2] < DESK_GAP
            if name == "gripper_end":
                low &= pts[:, 2] < -0.003  # the gripper body itself, not the probed fingertips
            if low.any():
                problems.append(f"desk:{name}")
        trees = {name: cKDTree(pts) for name, pts in posed.items()}
        for a, b in itertools.combinations(LINKS, 2):
            if (a, b) in ADJACENT or (b, a) in ADJACENT:
                continue
            d, _ = trees[b].query(posed[a], k=1, distance_upper_bound=SELF_GAP)
            if np.isfinite(d).any():
                problems.append(f"self:{a}-{b}")
        return problems


def load_limits(spec):
    if spec == "urdf":
        return list(zip(topdown_ik._lo, topdown_ik._hi))
    return arm_goto.driver_limits(spec)


def reachable(body, xy, limits, cache):
    """Least tilt (deg) of a collision-free level-finger grasp at xy, or None."""
    radial = math.atan2(xy[1], xy[0])
    seeds = plan_pen_pick.SEEDS + [np.array([radial, 1.6, 1.2, -1.1, 0, 0])]
    for pitch in PITCHES:
        dirs = sorted(TILT_DIRS, key=lambda t: abs((t - radial + math.pi) % (2 * math.pi) - math.pi))[:8] \
            if pitch else (radial,)
        for yaw in dirs:
            q, pos_err, ang_err = topdown_ik.solve([xy[0], xy[1], GRASP_Z], yaw=yaw, seeds=seeds, pitch=pitch)
            if pos_err > 1e-3 or ang_err > 1e-2:
                continue
            if not all(lo - 1e-6 <= v <= hi + 1e-6 for v, (lo, hi) in zip(q, limits)):
                continue
            if body.collisions(q):
                continue
            return math.degrees(pitch)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limits", action="append", help="NAME=driver YAML or NAME=urdf")
    p.add_argument("--step", type=float, default=0.03)
    p.add_argument("--image", help="desk photo (video2) to draw on")
    p.add_argument("--calib", default=str(HERE / "calib/video2.json"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    body = Body()
    home = body.collisions([0.0] * 6)
    print("folded pose problems:", home or "none")
    sets = dict(s.split("=", 1) for s in (a.limits or [f"current={HERE / 'config/arm_pick_trial.yaml'}"]))
    xs = np.arange(0.0, 0.72, a.step)
    ys = np.arange(-0.66, 0.67, a.step)
    results = {}
    for name, spec in sets.items():
        limits = load_limits(spec)
        grid = {}
        for x in xs:
            for y in ys:
                r = math.hypot(x, y)
                if r < 0.10 or r > 0.70:
                    continue
                grid[(round(x, 3), round(y, 3))] = reachable(body, (x, y), limits, None)
        ok = [k for k, v in grid.items() if v is not None]
        area = len(ok) * a.step ** 2
        top = sum(1 for k in ok if grid[k] == 0.0) * a.step ** 2
        far = max((math.hypot(*k) for k in ok), default=0.0)
        print(f"{name}: reachable {area:.3f} m^2 (straight down {top:.3f} m^2), farthest {far:.2f} m")
        results[name] = {"limits": limits, "area_m2": area, "straight_down_m2": top, "farthest_m": far,
                         "points": [[k[0], k[1], v] for k, v in grid.items()]}
    (a.out / "reach.json").write_text(json.dumps(results, indent=1))
    if a.image:
        calib = json.load(open(a.calib))
        colours = [(0, 200, 255), (255, 120, 0), (200, 0, 200), (0, 180, 0)]
        for name, colour in zip(results, colours):
            img = cv2.imread(a.image)
            overlay = img.copy()
            for x, y, tilt in results[name]["points"]:
                if tilt is None:
                    continue
                corners = np.array([[x - a.step / 2, y - a.step / 2, 0], [x + a.step / 2, y - a.step / 2, 0],
                                    [x + a.step / 2, y + a.step / 2, 0], [x - a.step / 2, y + a.step / 2, 0]])
                pts = (np.array(calib["T_cam_base"]) @ np.c_[corners, np.ones(4)].T).T[:, :3]
                px = handeye_solve.project(pts, calib["camera"])
                shade = (0, 200, 0) if tilt == 0 else (0, 200, 255) if tilt <= 30 else (0, 120, 255)
                cv2.fillPoly(overlay, [np.round(px).astype(np.int32)], shade)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.imwrite(str(a.out / f"reach_{name}.jpg"), img)


if __name__ == "__main__":
    main()
