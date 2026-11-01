#!/usr/bin/env python3
"""Which things can the robot tidy at all? Each kind of the catalogue alone on the desk (simulation).

The battle mixes two difficulties: the thing itself and the clutter round it. Here every kind is put on an
empty desk by itself, in a few poses, and tidied with the same robot as the battle (judge fast, depth,
max-speed 0.35). One run per (kind, pose); results are appended to <out>/results.jsonl.

    source /opt/ros/humble/setup.bash
    MUJOCO_GL=egl python3 object_study.py --out ../outputs/study --kind gso_sponge --pose 0
    MUJOCO_GL=egl python3 object_study.py --out ../outputs/study --list   # every (kind, pose), one per line
"""
import argparse
import fcntl
import json
import math
from pathlib import Path

import battle
import sim_scenarios

# (x, y, yaw deg): in front and square; far out to the left and turned (the battle's reach is 0.18-0.46 m)
POSES = [(0.30, 0.00, 0.0), (0.40, 0.20, 45.0)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kind")
    p.add_argument("--pose", type=int, default=0)
    p.add_argument("--list", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--driver-config", default=str(battle.tidy_desk.HERE / "config/arm_pick_wide_trial.yaml"))
    a = p.parse_args()
    if a.list:
        for kind in sorted(sim_scenarios.CATALOG):
            for i in range(len(POSES)):
                print(kind, i)
        return
    x, y, yaw = POSES[a.pose]
    s = sim_scenarios.CATALOG[a.kind]
    # as battle.place, but for any kind of the catalogue (the real desk's twins too, not only GSO)
    world = battle.mj_scene.real_layout()
    world["objects"].append({
        "id": f"{a.kind}1", "kind": a.kind, "name": s["name"], "centre": [x, y], "yaw": math.radians(yaw),
        "size": list(s["size"]), "z": 0.0, "bgr": s["bgr"], "soft": s["soft"], "min_force": s["min_force"],
        "max_force": s["max_force"], "category": s["category"], "start": [x, y], "mesh": s.get("mesh"),
        "rgba": s.get("rgba"), "mass": s.get("mass")})
    fell = battle.settle(world)
    folder = a.out / f"{a.kind}_p{a.pose}"
    folder.mkdir(parents=True, exist_ok=True)
    row = {"kind": a.kind, "pose": a.pose, "name": s["name"], "size": s["size"], "mass": s.get("mass"),
           "soft": s["soft"], "category": s["category"], "precision": a.kind in sim_scenarios.PRECISION}
    if fell:
        row |= {"status": "rolled_off"}
    else:
        o = world["objects"][0]
        row["upright"] = abs(o["quat"][1]) < 0.2 and abs(o["quat"][2]) < 0.2
        try:
            row |= battle.run(world, folder, a)
        except Exception as e:  # a crash is a result too
            row |= {"status": "crashed", "error": repr(e)[:300]}
    with open(a.out / "results.jsonl", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
