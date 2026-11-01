#!/usr/bin/env python3
"""Kinematic desk world for rehearsing the tidy loop without the arm (2026-10-27).

The world is a JSON file shared by the simulated arm (sim_arm.py, which moves
objects) and the simulated overhead camera (render()). Geometry is in the
robot base frame (m):
  boxes     open containers: centre, yaw, outer [x, y], height, wall, floor
  fixtures  solid blocks that must not move: centre, yaw, size [x, y, z]
  objects   loose things: centre, yaw, size [length, width, height], z (bottom),
            soft (compressible), min_force/max_force (slips below, crushed above)

The gripper is the URDF gripper_end frame plus measured details: the real
fingertips reach TIP_REACH beyond it, the 18.4 cm rail and the motor sit behind,
and the finger gap follows two measurements on the real gripper (8 mm wall
closed at 0.19 rad, 18 mm pen at 0.85 rad, and fully open 100 mm at 5.9 rad).
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

import handeye_solve
import topdown_ik

TIP_REACH = 0.005
GAP_AT_ZERO = 0.0042     # m finger gap at 0 rad
GAP_PER_RAD = 0.0162     # m per rad (pen 18 mm at 0.85 rad, full opening 100 mm at 5.9 rad, 2026-10-28)
FINGER = (0.008, 0.016, 0.060)  # m thickness (along y), width (z), length (back from the tip)
RAIL = [(-0.1032, -0.0732), (-0.092, 0.092), (-0.034, 0.034)]
MOTOR = [(-0.1572, -0.0927), (-0.0314, 0.0314), (-0.041, 0.041)]
SOFT_CORE = 0.4          # a soft object resists only below this fraction of its height
GIVE = 0.0015            # m of penetration tolerated (finger pads, walls and the desk give a little)
GIVE_LOOSE = 0.003       # m for loose objects, which a finger also nudges aside
DESK_BGR = (150, 145, 146)
BOX_RIM_BGR = (150, 178, 196)
BOX_FLOOR_BGR = (165, 190, 205)


def gap(rad):
    return GAP_AT_ZERO + GAP_PER_RAD * rad


def rad_for_gap(width):
    return (width - GAP_AT_ZERO) / GAP_PER_RAD


def load(path):
    return json.loads(Path(path).read_text())


def save(world, path):
    Path(path).write_text(json.dumps(world, indent=1, ensure_ascii=False))


def _rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def footprint(centre, yaw, size):
    """4 corners (x, y) of a yawed rectangle."""
    hx, hy = size[0] / 2, size[1] / 2
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return local @ _rot(yaw).T + np.asarray(centre)


def _local(points_xy, centre, yaw):
    return (np.asarray(points_xy) - np.asarray(centre)) @ _rot(yaw)


def solids(world, ignore=()):
    """Axis-aligned-in-own-frame solids: (centre_xy, yaw, half_xy, z0, z1, owner)."""
    out = []
    for b in world["boxes"]:
        hx, hy, w = b["outer"][0] / 2, b["outer"][1] / 2, b["wall"]
        c, yaw = np.asarray(b["centre"]), b["yaw"]
        for off, half in (((hx - w / 2, 0), (w / 2, hy)), ((-(hx - w / 2), 0), (w / 2, hy)),
                          ((0, hy - w / 2), (hx, w / 2)), ((0, -(hy - w / 2)), (hx, w / 2))):
            out.append((c + _rot(yaw) @ off, yaw, half, 0.0, b["height"], b["name"]))
        out.append((c, yaw, (hx, hy), 0.0, b["floor"], b["name"] + ":floor"))
    for f in world["fixtures"]:
        out.append((np.asarray(f["centre"]), f["yaw"], (f["size"][0] / 2, f["size"][1] / 2), 0.0, f["size"][2],
                    f["name"]))
    for o in world["objects"]:
        if o["id"] in ignore or o.get("held"):
            continue
        h = o["size"][2] * (SOFT_CORE if o.get("soft") else 1.0)
        out.append((np.asarray(o["centre"]), o["yaw"], (o["size"][0] / 2, o["size"][1] / 2), o["z"], o["z"] + h,
                    o["id"]))
    return out


def _grid(bounds, n=(3, 5, 3)):
    axes = [np.linspace(lo, hi, k) for (lo, hi), k in zip(bounds, n)]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T


def gripper_points(opening):
    """Sample points of the gripper body in the gripper_end frame (fingers at this opening)."""
    g = gap(opening)
    t, wz, length = FINGER
    tip = TIP_REACH
    pts = [_grid([(tip - length, tip), (g / 2, g / 2 + t), (-wz / 2, wz / 2)], (7, 2, 3)),
           _grid([(tip - length, tip), (-g / 2 - t, -g / 2), (-wz / 2, wz / 2)], (7, 2, 3)),
           _grid(RAIL, (3, 13, 3)), _grid(MOTOR, (3, 3, 3))]
    return np.vstack(pts)


def collisions(world, q, opening, ignore=()):
    """Names of solids (and 'desk') that the gripper penetrates at joint pose q."""
    T = topdown_ik.fk(q)
    return collisions_at(world, (T.rotation @ gripper_points(opening).T).T + T.translation, ignore)


def collisions_at(world, pts, ignore=(), margin=0.0):
    """Names of solids (and 'desk') containing any of the world points pts; solids grown by margin (m)."""
    hit = set()
    if (pts[:, 2] < -GIVE).any():
        hit.add("desk")
    loose = {o["id"] for o in world["objects"]}
    for centre, yaw, half, z0, z1, owner in solids(world, ignore):
        give = GIVE_LOOSE if owner in loose else GIVE
        loc = _local(pts[:, :2], centre, yaw)
        inside = ((np.abs(loc[:, 0]) < half[0] - (give if owner in loose else 0.0) + margin)
                  & (np.abs(loc[:, 1]) < half[1] - (give if owner in loose else 0.0) + margin)
                  & (pts[:, 2] > z0 + GIVE) & (pts[:, 2] < z1 - GIVE))
        if inside.any():
            hit.add(owner)
    return hit


def between_fingers(world, q, opening):
    """Objects whose body lies between the open fingers (the closing sweep), nearest first."""
    T = topdown_ik.fk(q)
    R, p = T.rotation, T.translation
    g = gap(opening)
    found = []
    for o in world["objects"]:
        if o.get("held"):
            continue
        h = o["size"][2]
        # Sample the object's body; keep points inside the finger sweep box.
        local = _grid([(-o["size"][0] / 2, o["size"][0] / 2), (-o["size"][1] / 2, o["size"][1] / 2),
                       (0, h)], (9, 9, 4))
        world_pts = np.c_[local[:, :2] @ _rot(o["yaw"]).T + o["centre"], local[:, 2] + o["z"]]
        g_pts = (world_pts - p) @ R  # into the gripper frame
        sweep = ((g_pts[:, 0] > TIP_REACH - FINGER[2]) & (g_pts[:, 0] < TIP_REACH)
                 & (np.abs(g_pts[:, 1]) < g / 2) & (np.abs(g_pts[:, 2]) < FINGER[1] / 2 + 0.004))
        if sweep.any():
            span = g_pts[sweep, 1]
            found.append((o, float(span.max() - span.min()), int(sweep.sum())))
    return sorted(found, key=lambda f: -f[2])


def surface_below(world, xy, ignore=()):
    """Height and owner a dropped object lands on at xy."""
    best, owner = 0.0, "desk"
    for b in world["boxes"]:
        loc = _local([xy], b["centre"], b["yaw"])[0]
        inner = (b["outer"][0] / 2 - b["wall"], b["outer"][1] / 2 - b["wall"])
        if abs(loc[0]) < inner[0] and abs(loc[1]) < inner[1] and b["floor"] >= best:
            best, owner = b["floor"], b["name"]
    for f in world["fixtures"]:
        loc = _local([xy], f["centre"], f["yaw"])[0]
        if abs(loc[0]) < f["size"][0] / 2 and abs(loc[1]) < f["size"][1] / 2 and f["size"][2] > best:
            best, owner = f["size"][2], f["name"]
    return best, owner


def container_of(world, o):
    """Name of the box whose inner area holds the object's centre, or None."""
    for b in world["boxes"]:
        loc = _local([o["centre"]], b["centre"], b["yaw"])[0]
        inner = (b["outer"][0] / 2 - b["wall"], b["outer"][1] / 2 - b["wall"])
        if abs(loc[0]) < inner[0] and abs(loc[1]) < inner[1] and o["z"] <= b["height"]:
            return b["name"]
    return None


def _project(calib, xyz):
    pts = (np.array(calib["T_cam_base"]) @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    return handeye_solve.project(pts, calib["camera"])


def render(world, calib, path, size=(1280, 960), seed=0):
    """Overhead video2-like frame: desk, boxes, fixtures and objects (top faces), no arm."""
    rng = np.random.default_rng(seed)
    img = np.empty((size[1], size[0], 3), np.uint8)
    img[:] = DESK_BGR
    img = np.clip(img.astype(int) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)

    def poly(corners_xy, z, colour):
        px = _project(calib, np.c_[corners_xy, np.full(len(corners_xy), z)])
        cv2.fillPoly(img, [np.round(px).astype(np.int32)], colour, lineType=cv2.LINE_AA)

    items = []
    for b in world["boxes"]:
        items.append((b["height"], "box", b))
    for f in world["fixtures"]:
        items.append((f["size"][2], "fixture", f))
    for o in world["objects"]:
        items.append((o["z"] + o["size"][2], "object", o))
    for _, kind, it in sorted(items, key=lambda i: i[0]):
        if kind == "box":
            poly(footprint(it["centre"], it["yaw"], it["outer"]), it["height"], BOX_RIM_BGR)
            inner = [it["outer"][0] - 2 * it["wall"], it["outer"][1] - 2 * it["wall"]]
            poly(footprint(it["centre"], it["yaw"], inner), it["floor"], BOX_FLOOR_BGR)
            for o in world["objects"]:  # contents drawn after the floor
                if container_of(world, o) == it["name"]:
                    poly(footprint(o["centre"], o["yaw"], o["size"][:2]), o["z"] + o["size"][2], o["bgr"])
        elif kind == "fixture":
            poly(footprint(it["centre"], it["yaw"], it["size"][:2]), it["size"][2], it["bgr"])
        elif container_of(world, it) is None:
            poly(footprint(it["centre"], it["yaw"], it["size"][:2]), it["z"] + it["size"][2], it["bgr"])
    cv2.imwrite(str(path), img)
    return img


def bbox_px(calib, o):
    """Pixel bounding box of an object's top face and bottom face."""
    corners = footprint(o["centre"], o["yaw"], o["size"][:2])
    pts = np.vstack([_project(calib, np.c_[corners, np.full(4, z)]) for z in (o["z"], o["z"] + o["size"][2])])
    return [int(math.floor(pts[:, 0].min())), int(math.floor(pts[:, 1].min())),
            int(math.ceil(pts[:, 0].max())), int(math.ceil(pts[:, 1].max()))]
