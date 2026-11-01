#!/usr/bin/env python3
"""Random desk scenes for sim_world, and an oracle judge that answers like desk_judge.

    python3 sim_scenarios.py SEED OUT.json     # write one random world
"""
import json
import math
import sys

import numpy as np

import sim_world

# kind: size [length, width, height] (m), colour, judge answer, physics
CATALOG = {
    "pen":     dict(name="ペン", size=(0.140, 0.017, 0.017), bgr=(150, 40, 110), category="keep", force=3,
                    soft=False, min_force=2, max_force=5),
    "eraser":  dict(name="消しゴム", size=(0.050, 0.022, 0.012), bgr=(235, 235, 235), category="keep", force=2,
                    soft=False, min_force=1, max_force=4),
    "usb":     dict(name="USBメモリ", size=(0.055, 0.018, 0.008), bgr=(45, 45, 45), category="keep", force=3,
                    soft=False, min_force=2, max_force=5),
    "piece":   dict(name="将棋の駒", size=(0.030, 0.026, 0.008), bgr=(120, 170, 205), category="keep", force=3,
                    soft=False, min_force=2, max_force=5),
    "paper":   dict(name="丸めた紙", size=(0.065, 0.055, 0.045), bgr=(215, 215, 220), category="trash", force=2,
                    soft=True, min_force=1, max_force=5),
    "tissue":  dict(name="使用済みティッシュ", size=(0.070, 0.060, 0.030), bgr=(240, 240, 245), category="trash",
                    force=1, soft=True, min_force=1, max_force=5),
    # Kinds with a catch, for the learning experiment (mj_campaign): the judge's first guess may not hold.
    "can":     dict(name="空き缶", size=(0.120, 0.066, 0.066), bgr=(200, 200, 205), category="trash", force=2,
                    soft=False, min_force=3, max_force=5),
    "block":   dict(name="積み木", size=(0.045, 0.045, 0.045), bgr=(90, 160, 210), category="keep", force=2,
                    soft=False, min_force=3, max_force=5),
    "marker":  dict(name="太いマーカー", size=(0.135, 0.022, 0.022), bgr=(40, 40, 40), category="keep", force=3,
                    soft=False, min_force=3, max_force=5),
    "phone":   dict(name="スマートフォン", size=(0.150, 0.072, 0.009), bgr=(30, 30, 30), category="keep", force=3,
                    soft=False, min_force=3, max_force=4, should_grasp=False),
}


# The real desk's objects as photo-made 3D models (mj_assets/objects), for the digital twin.
TWIN_KINDS = {
    "purple_marker": dict(name="紫マーカー", size=(0.135, 0.019, 0.019), bgr=(140, 40, 110), category="keep", force=3,
                          soft=False, min_force=2, max_force=5),
    # The photo-made black marker renders as a hollow outline from above: the purple marker's shape, in black.
    "black_marker": dict(name="黒マーカー", size=(0.135, 0.019, 0.019), bgr=(30, 30, 30), category="keep", force=3,
                         soft=False, min_force=2, max_force=5, mesh="purple_marker", rgba=(0.08, 0.08, 0.09, 1)),
    # The "paper" model (height stretched x2.4 from a flat result) could not be lifted in the simulation,
    # so both balls use the "paper_b" model, turned differently (2026-10-27).
    "paper_a": dict(name="くしゃくしゃの紙", size=(0.075, 0.070, 0.038), bgr=(225, 225, 225), category="trash",
                    force=2, soft=True, min_force=1, max_force=5, mesh="paper_b"),
    "paper_b": dict(name="くしゃくしゃの紙", size=(0.075, 0.070, 0.038), bgr=(225, 225, 225), category="trash",
                    force=2, soft=True, min_force=1, max_force=5, mesh="paper_b"),
    "crushed_box": dict(name="つぶれた箱", size=(0.121, 0.078, 0.040), bgr=(230, 230, 230), category="trash",
                        force=2, soft=True, min_force=1, max_force=5),
    "shogi_piece": dict(name="将棋の駒", size=(0.030, 0.026, 0.008), bgr=(120, 170, 205), category="keep", force=3,
                        soft=False, min_force=2, max_force=5),
    "wood_block": dict(name="積み木", size=(0.035, 0.030, 0.020), bgr=(120, 170, 205), category="keep", force=2,
                       soft=False, min_force=2, max_force=5, mesh="block"),
    # Looks exactly like the wooden block but has a weight inside (180 g): only a full-strength grip lifts it.
    # Nothing in a photo tells, so only experience can (the "knowledge" experiment, 2026-10-27).
    "heavy_block": dict(name="積み木", size=(0.035, 0.030, 0.020), bgr=(120, 170, 205), category="keep", force=2,
                        soft=False, min_force=5, max_force=5, mesh="block", mass=0.18),
    "tissue_box": dict(name="ティッシュ箱", size=(0.120, 0.116, 0.100), bgr=(200, 230, 200), category="keep", force=3,
                       soft=False, min_force=3, max_force=5, should_grasp=False),
}
CATALOG.update(TWIN_KINDS)


# Precision devices stay on the desk for the human (house rule): the goal counts them right when untouched.
PRECISION = {"gso_black_mouse", "gso_white_mouse", "gso_game_console", "gso_camera", "gso_speaker"}


def _gso_kinds():
    """Scanned household objects (Google Scanned Objects, mj_assets/objects/gso_*/meta.json, import_gso.py).

    Objects the robot has never met: the judge has to read them from the picture. Kind = folder name.
    """
    from pathlib import Path
    kinds = {}
    for d in sorted((Path(__file__).resolve().parent / "mj_assets/objects").glob("gso_*/meta.json")):
        m = json.loads(d.read_text())
        kinds[d.parent.name] = dict(
            name=m["name_ja"], size=tuple(m["size_m"]), bgr=tuple(m["mean_bgr"]), category=m["category"],
            # light things need a light grip, heavy ones (mugs, bells) a firm one
            force=2 if m["mass_kg"] < 0.05 or m["soft"] else 4 if m["mass_kg"] > 0.25 else 3,
            soft=m["soft"], min_force=1 if m["soft"] else 2 if m["mass_kg"] < 0.2 else 3, max_force=5,
            mesh=d.parent.name, mass=m["mass_kg"], should_grasp=d.parent.name not in PRECISION,
            split=m["split"], graspable_guess=m["graspable_guess"])
    return kinds


GSO_KINDS = _gso_kinds()
CATALOG.update(GSO_KINDS)
GSO_TRAIN = [k for k, v in GSO_KINDS.items() if v["split"] == "train"]
GSO_TEST = [k for k, v in GSO_KINDS.items() if v["split"] == "test"]   # held out: never seen in training
# 2026-10-28 all_run_3 round 1 as the camera saw it (centre, long-axis direction), plus two extras.
TWIN_DESK = [("black_marker", (0.1547, -0.2258), (0.0485, -0.1359)), ("purple_marker", (0.2371, -0.0548), (0.1034, 0.11)),
             ("paper_a", (0.2469, -0.2623), (0.053, -0.0514)), ("paper_b", (0.3241, 0.1379), (-0.0053, -0.0764)),
             ("tissue_box", (0.2211, 0.2482), (0.99, -0.15)),
             ("crushed_box", (0.504, -0.004), (0.3624, 0.9320))]  # measured from the SAM mask, yaw 1.20 rad
TWIN_EXTRAS = [("shogi_piece", (0.30, -0.15), (1.0, 0.4)), ("wood_block", (0.33, 0.02), (0.3, 1.0))]


def twin_world(base, extras=True):
    """The real desk of 2026-10-28 rebuilt with the photo-made models; base is mj_scene.real_layout()."""
    world = dict(base, objects=[])
    for k, (kind, xy, axis) in enumerate(TWIN_DESK + (TWIN_EXTRAS if extras else []), 1):
        spec = CATALOG[kind]
        world["objects"].append({
            "id": f"{kind}{k}", "kind": kind, "name": spec["name"], "centre": list(xy),
            "yaw": math.atan2(axis[1], axis[0]), "size": list(spec["size"]), "z": 0.0, "bgr": spec["bgr"],
            "soft": spec["soft"], "min_force": spec["min_force"], "max_force": spec["max_force"],
            "category": spec["category"], "start": list(xy), "mesh": spec.get("mesh", kind),
            "rgba": spec.get("rgba"), "mass": spec.get("mass")})
    return world


def random_twin_world(base, seed, kinds, n=None):
    """The photo-made real objects scattered at random on the desk (for the Claude-judged experiment)."""
    rng = np.random.default_rng(seed)
    world = dict(base, objects=[])
    n = n or int(rng.integers(4, 7))
    count = {}
    for _ in range(400):
        if len(world["objects"]) >= n:
            break
        kind = kinds[int(rng.integers(len(kinds)))]
        spec = CATALOG[kind]
        r, ang = rng.uniform(0.20, 0.42), math.radians(rng.uniform(-40, 40))
        centre = [r * math.cos(ang), r * math.sin(ang)]
        radius = max(spec["size"][:2]) / 2
        if any(np.hypot(*np.subtract(centre, o["centre"])) < radius + max(o["size"][:2]) / 2 + 0.03
               for o in world["objects"]):
            continue
        count[kind] = count.get(kind, 0) + 1
        world["objects"].append({
            "id": f"{kind}{count[kind]}", "kind": kind, "name": spec["name"], "centre": centre,
            "yaw": float(rng.uniform(-math.pi, math.pi)), "size": list(spec["size"]), "z": 0.0, "bgr": spec["bgr"],
            "soft": spec["soft"], "min_force": spec["min_force"], "max_force": spec["max_force"],
            "category": spec["category"], "start": centre,
            "mesh": spec.get("mesh") or (kind if kind in TWIN_KINDS else None), "rgba": spec.get("rgba"), "mass": spec.get("mass")})
    return world


def random_gso_world(base, seed, kinds=None, n=None):
    """Scanned household objects scattered on the desk like random_twin_world (kinds default GSO_TRAIN).

    Like random_twin_world it only keeps objects apart, so base should have its boxes off the desk
    (mj_scene.real_layout()).
    """
    return random_twin_world(base, seed, list(kinds or GSO_TRAIN), n)


BASE_KINDS = ["pen", "eraser", "usb", "piece", "paper", "tissue"]  # the original scenes (tests, sim_campaign)
LEARNING_KINDS = ["pen", "eraser", "paper", "piece", "can", "block", "marker", "usb"]


def base_world(wide=False):
    if wide:  # boxes off to the sides, reachable once joint1 may turn +-1.7 rad
        return {
            "boxes": [
                {"name": "keep_box", "role": "keep", "centre": [0.15, 0.40], "yaw": 0.2, "outer": [0.124, 0.124],
                 "height": 0.055, "wall": 0.008, "floor": 0.008},
                {"name": "trash_box", "role": "trash", "centre": [0.12, -0.42], "yaw": -0.3, "outer": [0.124, 0.124],
                 "height": 0.055, "wall": 0.008, "floor": 0.008},
            ],
            "fixtures": [{"name": "tissue_box", "centre": [0.52, 0.30], "yaw": 0.4, "size": [0.12, 0.24, 0.09],
                          "bgr": (90, 160, 80)}],
            "objects": [],
        }
    return {
        "boxes": [
            {"name": "keep_box", "role": "keep", "centre": [0.30, 0.17], "yaw": 0.35, "outer": [0.124, 0.124],
             "height": 0.055, "wall": 0.008, "floor": 0.008},
            {"name": "trash_box", "role": "trash", "centre": [0.29, -0.18], "yaw": -0.2, "outer": [0.124, 0.124],
             "height": 0.055, "wall": 0.008, "floor": 0.008},
        ],
        "fixtures": [
            {"name": "tissue_box", "centre": [0.10, 0.33], "yaw": 0.1, "size": [0.12, 0.24, 0.09],
             "bgr": (90, 160, 80)},
        ],
        "objects": [],
    }


def _clear(world, centre, radius, gap=0.03, box_gap=0.10):
    if np.hypot(*centre) < 0.17:
        return False
    for b in world["boxes"]:
        if np.hypot(*(np.subtract(centre, b["centre"]))) < radius + box_gap:
            return False
    for f in world["fixtures"]:
        if np.hypot(*(np.subtract(centre, f["centre"]))) < radius + 0.15:
            return False
    for o in world["objects"]:
        if np.hypot(*(np.subtract(centre, o["centre"]))) < radius + max(o["size"][:2]) / 2 + gap:
            return False
    return True


def random_world(seed, n=None, kinds=None, r_range=(0.20, 0.42), angle_deg=36, hard=False, wide=False):
    """hard: objects may sit 1 cm apart and right next to the boxes, and up to 8 of them."""
    rng = np.random.default_rng(seed)
    world = base_world(wide)
    if wide:
        r_range, angle_deg = (0.20, 0.64 if wide == "far" else 0.55), 80
    kinds = kinds or BASE_KINDS + ["phone"] * (seed % 4 == 0)
    n = n or int(rng.integers(3, 7) if not hard else rng.integers(5, 9))
    gap, box_gap = (0.01, 0.075) if hard else (0.03, 0.10)
    count = {}
    for i in range(400):
        if len(world["objects"]) >= n:
            break
        kind = kinds[int(rng.integers(len(kinds)))]
        spec = CATALOG[kind]
        r = rng.uniform(*r_range)
        ang = math.radians(rng.uniform(-angle_deg, angle_deg))
        centre = [r * math.cos(ang), r * math.sin(ang)]
        if not _clear(world, centre, max(spec["size"][:2]) / 2, gap, box_gap):
            continue
        count[kind] = count.get(kind, 0) + 1
        world["objects"].append({
            "id": f"{kind}{count[kind]}", "kind": kind, "name": spec["name"], "centre": centre,
            "yaw": float(rng.uniform(-math.pi, math.pi)), "size": list(spec["size"]), "z": 0.0,
            "bgr": spec["bgr"], "soft": spec["soft"], "min_force": spec["min_force"],
            "max_force": spec["max_force"], "category": spec["category"], "start": centre,
        })
    return world


def oracle_judge(world, calib, noise_px=0.0, rng=None, force_error=0.0):
    """desk_judge-shaped answers from the world's ground truth (bbox jittered by noise_px).

    force_error: probability that the judged force is one level too low (tests slip recovery).
    """
    rng = rng or np.random.default_rng(0)
    objects = []
    for o in world["objects"]:
        if o.get("held"):
            continue
        spec = CATALOG[o["kind"]]
        bbox = sim_world.bbox_px(calib, o)
        if noise_px:
            bbox = [int(v + rng.normal(0, noise_px)) for v in bbox]
        grasp = spec.get("should_grasp", True)
        force = spec["force"] - (1 if rng.uniform() < force_error and spec["force"] > 1 else 0)
        objects.append({
            "name": o["name"], "bbox": bbox, "category": spec["category"], "category_reason": "oracle",
            "should_grasp": grasp, "grasp_reason": "oracle", "deformable": spec["soft"], "force": force, "force_reason": "oracle",
            "grasp_point": "中央", "size_cm": [v * 100 * rng.uniform(0.85, 1.15) for v in o["size"]],
            "confidence": 1.0,
            "ask_human": spec["category"] == "trash", "truth_id": o["id"],
        })
    for b in world["boxes"]:
        objects.append({"name": "木箱", "bbox": [0, 0, 1, 1], "category": "fixture", "category_reason": "oracle",
                        "should_grasp": False, "grasp_reason": "container", "force": 1, "force_reason": "",
                        "grasp_point": "", "size_cm": [12.4, 12.4, 5.5], "confidence": 1.0, "ask_human": False})
    return objects


if __name__ == "__main__":
    seed, out = int(sys.argv[1]), sys.argv[2]
    sim_world.save(random_world(seed), out)
    print(json.dumps([o["id"] for o in sim_world.load(out)["objects"]]))
