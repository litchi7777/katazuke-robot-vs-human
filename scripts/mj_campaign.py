#!/usr/bin/env python3
"""How much smarter does the robot get from knowledge alone? The same physics scenes, three conditions.

  none        no notebook: the fixed rules (one force level up after a slip, give up otherwise)
  reflect     an empty notebook for every scene: it reflects on failures, but forgets between scenes
  accumulate  one notebook carried from scene to scene: lessons build up

Every scene is the same MuJoCo desk with the same objects in the same places for all three, so only
the knowledge differs. Writes results.json, results.csv and learning_curve.png.

    MUJOCO_GL=egl python3 mj_campaign.py --seeds 1-10 --out DIR [--judge oracle|fast] [--reflect offline|claude]
"""
import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import mj_scene
import sim_campaign
import sim_scenarios
import sim_world
import tidy_desk

CONDITIONS = ("none", "reflect", "accumulate")
LABELS_JA = {"none": "知識なし", "reflect": "その場で振り返る", "accumulate": "知識をためる"}


# Easy things first, then smaller and thinner ones (the user, 2026-10-27): like practising, the lessons
# from big objects are there before the hard ones come.
CURRICULUM = [
    ["block", "can", "paper", "marker"],
    ["block", "can", "paper", "marker", "pen", "eraser"],
    ["paper", "marker", "pen", "eraser", "usb", "piece", "can", "block"],
]


# The same curriculum with the photo-made models of the real desk's objects.
TWIN_CURRICULUM = [
    ["purple_marker", "black_marker", "paper_a", "paper_b"],
    ["purple_marker", "black_marker", "paper_a", "paper_b", "wood_block", "crushed_box"],
    ["purple_marker", "black_marker", "paper_a", "paper_b", "wood_block", "crushed_box", "shogi_piece", "tissue_box"],
]
OBJECTS = "primitives"


def scene(seed, k=1, total=1, n=None):
    """The k-th of `total` scenes: its kinds follow the curriculum stage for its place in the run."""
    curriculum = TWIN_CURRICULUM if OBJECTS.startswith("twin") else CURRICULUM
    stage = curriculum[min(len(curriculum) - 1, (k - 1) * len(curriculum) // max(1, total))]
    world = mj_scene.real_layout()
    if OBJECTS == "twin_heavy":
        # Wooden blocks that are secretly heavy, among the real desk's objects: a hidden property only
        # experience reveals. Every scene has at least one.
        w = sim_scenarios.random_twin_world(world, seed, ["heavy_block"] * 2 + stage[:4], n=n)
        if not any(o["kind"] == "heavy_block" for o in w["objects"]):
            w = sim_scenarios.random_twin_world(world, seed + 1000, ["heavy_block"] * 3 + stage[:4], n=n)
        return w
    if OBJECTS == "twin_can":
        # Heavy, smooth empty cans: the judge's grip is too weak for them every time (a systematic error
        # that knowledge can fix), mixed with the real desk's objects.
        return sim_scenarios.random_twin_world(world, seed, stage + ["can"] * len(stage), n=n)
    if OBJECTS == "twin":
        return sim_scenarios.random_twin_world(world, seed, stage, n=n)
    if OBJECTS in ("gso_train", "gso_test"):
        # Scanned household objects: the practice set, or the held-out set never seen while learning.
        kinds = sim_scenarios.GSO_TRAIN if OBJECTS == "gso_train" else sim_scenarios.GSO_TEST
        return sim_scenarios.random_gso_world(world, seed, kinds, n=n or 5)
    world["objects"] = sim_scenarios.random_world(seed, n=n, kinds=stage)["objects"]
    return world


def run_one(world, out, condition, notebook, a):
    path = out / "world.json"
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(world, ensure_ascii=False))
    args = ["--sim", str(path), "--physics", "--judge", a.judge, "--approve", "yes", "--out", str(out / "run"),
            "--rounds", str(a.rounds), "--reflect", a.reflect, "--driver-config", a.driver_config,
            "--judge-force-error", str(a.force_error), "--seed", str(a.seed_base), "--max-speed", "0.3",
            "--taught-drops"]
    if condition != "none":
        args += ["--notebook", str(notebook), "--run-name", out.name]
    if a.depth:
        args += ["--depth"]
    tidy_desk.main(args)
    report = json.loads((out / "run/report.json").read_text())
    rows = sim_campaign.score(sim_world.load(path), "yes")
    attempts = sum(len(o.get("attempts", [])) for o in report["objects"])
    placed = sum(1 for r in rows if r["ok"])
    goal = goal_score(sim_world.load(path))
    return {"condition": condition, "objects": len(rows), "placed": placed, "attempts": attempts,
            "attempts_per_object": round(attempts / max(1, len(rows)), 3),
            "first_try": sum(1 for o in report["objects"] if len(o.get("attempts", [])) == 1
                             and str(o.get("result", "")).startswith("moved_to_")),
            "status": report.get("status"), "lessons": (report.get("notebook") or {}).get("lessons", 0),
            "goal": goal["ok"], "goal_of": goal["of"], "outcomes": goal["outcomes"], "cleared": goal["cleared"],
            "discarded_keep": goal["discarded_keep"], "touched_precision": goal["touched_precision"]}


# The goal is a clear desk: every object in its box, whatever the gripper was expected to manage. Only
# precision devices stay for the human (house rule, 2026-10-28).
PRECISION = sim_scenarios.PRECISION


def goal_score(world):
    """How many objects ended where the goal wants them; per object (kind, outcome).

    ok: in its own box (precision devices: untouched). cleared: off the desk into either box (the house
    rules keep every box and bottle, so an "empty" packet kept is not against them). discarded_keep: a kept
    thing in the trash, the worst mistake. touched_precision: a precision device moved.
    """
    outcomes, ok, cleared, discarded_keep, touched_precision = [], 0, 0, 0, 0
    for o in world["objects"]:
        spec = sim_scenarios.CATALOG[o["kind"]]
        where = sim_world.container_of(world, o)
        moved = math.dist(o["centre"], o["start"]) > 0.01
        if o["kind"] in PRECISION:
            good = not moved
            touched_precision += moved
        else:
            good = where == ("trash_box" if spec["category"] == "trash" else "keep_box")
            discarded_keep += where == "trash_box" and spec["category"] != "trash"
        got = where or ("desk_moved" if moved else "desk")
        outcomes.append([o["kind"], got, good])
        ok += good
        cleared += good or where in ("trash_box", "keep_box")
    return {"ok": ok, "of": len(world["objects"]), "outcomes": outcomes, "cleared": cleared,
            "discarded_keep": discarded_keep, "touched_precision": touched_precision}


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = font_manager.FontProperties(fname=tidy_desk.desk_judge.FONT)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c in CONDITIONS:
        rs = [r for r in results if r["condition"] == c]
        xs = list(range(1, len(rs) + 1))
        axes[0].plot(xs, [r["placed"] / max(1, r["objects"]) for r in rs], marker="o", label=LABELS_JA[c])
        axes[1].plot(xs, [r["attempts_per_object"] for r in rs], marker="o", label=LABELS_JA[c])
    axes[0].set_title("片付けられた割合", fontproperties=font)
    axes[1].set_title("1個あたりの掴む回数", fontproperties=font)
    for ax in axes:
        ax.set_xlabel("シーン（散らかされた回数）", fontproperties=font)
        ax.legend(prop=font)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="1-8")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--judge", default="oracle", choices=("oracle", "fast"))
    p.add_argument("--reflect", default="offline", choices=("offline", "claude"))
    p.add_argument("--force-error", type=float, default=0.5, help="oracle: chance a judged force is one too low")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--objects", choices=("primitives", "twin", "twin_can", "twin_heavy", "gso_train", "gso_test"), default="primitives",
                   help="twin: the photo-made 3D models of the real desk's objects")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--depth", action="store_true", help="a depth camera at video2 (a RealSense)")
    p.add_argument("--notebook-from", type=Path, help="accumulate: start from a copy of this notebook")
    p.add_argument("--driver-config", default=str(tidy_desk.HERE / "config/arm_pick_wide_trial.yaml"))
    a = p.parse_args()
    global OBJECTS
    OBJECTS = a.objects
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = sim_campaign.parse_seeds(a.seeds)
    results = []
    for condition in a.conditions.split(","):
        shared = a.out / f"notebook_{condition}"
        shutil.rmtree(shared, ignore_errors=True)
        if a.notebook_from and condition == "accumulate":
            # Start from what was learned elsewhere (e.g. on the practice objects, tested on unseen ones).
            shutil.copytree(a.notebook_from, shared)
        for k, seed in enumerate(seeds, 1):
            notebook = shared if condition == "accumulate" else a.out / condition / f"s{seed}" / "notebook"
            r = run_one(scene(seed, k, len(seeds)), a.out / condition / f"s{seed}", condition, notebook, a)
            r.update(seed=seed, scene=k)
            results.append(r)
            print(f"[{LABELS_JA[condition]}] シーン{k}: {r['placed']}/{r['objects']} 個、掴む回数 {r['attempts']}"
                  f"（1個あたり {r['attempts_per_object']:.2f}）、教訓 {r['lessons']}", flush=True)
            (a.out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    with open(a.out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    plot(results, a.out)
    for c in a.conditions.split(","):
        rs = [r for r in results if r["condition"] == c]
        placed = sum(r["placed"] for r in rs) / max(1, sum(r["objects"] for r in rs))
        per = sum(r["attempts"] for r in rs) / max(1, sum(r["objects"] for r in rs))
        print(f"{LABELS_JA[c]}: 片付けた割合 {placed:.0%}、1個あたり {per:.2f} 回", flush=True)


if __name__ == "__main__":
    main()
