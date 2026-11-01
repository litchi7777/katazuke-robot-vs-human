#!/usr/bin/env python3
"""Does rehearsing help? The same "real" desks (GSO scans whose mass and friction the twin cannot know), two ways:

  A one-shot   tidy_desk as it is: no rehearsal, the judge's default tactic
  B rehearse   rehearse.py: twin from the photo, rehearse, execute, up to --real-attempts on the "real" desk

Neither has a depth camera (the real desk has none). Writes OUT/results.json and prints the table.

    source /opt/ros/humble/setup.bash
    export MUJOCO_GL=egl TIDY_SAM_GPU=GPU-99571e8f-28bd-4db3-dadb-aea80850be59 MJ_FINGER_KP=1200
    python3 rehearse_eval.py --seeds 1-6 --out OUT [--desks-at-once 2]
"""
import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import battle
import mj_campaign
import mj_objects
import sim_campaign
import sim_world

HERE = Path(__file__).resolve().parent
DRIVER = str(HERE / "config/arm_pick_wide_trial.yaml")
MASS_RANGE = (0.6, 1.8)        # x the catalogue's mass
FRICTION_RANGE = (0.6, 1.2)    # x the catalogue's friction
NEAR = 0.08                    # m: a claim belongs to the thing that started this near it


def desk(seed, objects=5):
    """A fixed "real" desk: mj_campaign's held-out scene for this seed, its physics changed by the seed."""
    mj_campaign.OBJECTS = "gso_test"
    world = mj_campaign.scene(seed, n=objects)
    rng = np.random.default_rng(1000 + seed)
    for o in world["objects"]:
        m = mj_objects.meta(o["mesh"])
        o["mass"] = round(float(m["mass_kg"] * rng.uniform(*MASS_RANGE)), 4)
        o["friction"] = round(float(m["friction"] * rng.uniform(*FRICTION_RANGE)), 3)
    battle.settle(world)
    return world


def claims_false(claims, start, final):
    """Things the robot said it boxed that did not end in a box. claims: (xy, result)."""
    wrong = []
    for xy, result in claims:
        if not str(result).startswith("moved_to_"):
            continue
        near = min(start["objects"], key=lambda o: math.dist(o["centre"], xy))
        now = next(o for o in final["objects"] if o["id"] == near["id"])
        if math.dist(near["centre"], xy) > NEAR or sim_world.container_of(final, now) is None:
            wrong.append(near["id"])
    return wrong


def run_a(world_path, out, seed):
    out.mkdir(parents=True, exist_ok=True)
    w = out / "world.json"
    w.write_text(world_path.read_text())
    start = time.time()
    with open(out / "tidy.log", "w") as log:
        subprocess.run([sys.executable, "-u", str(HERE / "tidy_desk.py"), "--sim", str(w), "--physics", "--judge", "fast",
                        "--approve", "yes", "--out", str(out / "run"), "--rounds", "2", "--reflect", "offline",
                        "--driver-config", DRIVER, "--seed", "0", "--max-speed", "0.35", "--taught-drops"],
                       stdout=log, stderr=subprocess.STDOUT, cwd=HERE, timeout=2400)
    report = json.loads((out / "run/report.json").read_text())
    claims = [(e["xy"], e.get("result")) for e in report["objects"]]
    return claims, sim_world.load(w), time.time() - start


def run_b(world_path, out, seed, attempts, reflect):
    start = time.time()
    with open(out.with_suffix(".log"), "w") as log:
        subprocess.run([sys.executable, "-u", str(HERE / "rehearse.py"), "--world", str(world_path), "--out", str(out), "--execute",
                        "--no-record", "--real-attempts", str(attempts), "--reflect", reflect, "--parallel", "3"],
                       stdout=log, stderr=subprocess.STDOUT, cwd=HERE, timeout=7200)
    log_path = out / "real_log.json"
    rows = json.loads(log_path.read_text()) if log_path.exists() else []
    claims = [(r["xy"], r["result"]) for a in rows for r in a["objects"]]
    return claims, sim_world.load(out / "desk.json"), time.time() - start, rows


def twin_verdicts(out, start, final):
    """Per plan entry of every attempt: what the twin said against what the desk did. (agree, twin ok real no,
    twin no real done by nobody, sims)"""
    rows = []
    for plan_path in sorted(out.glob("attempt_*/plan.json")):
        attempt = plan_path.parent
        for e in json.loads(plan_path.read_text())["objects"]:
            near = min(start["objects"], key=lambda o: math.dist(o["centre"], e["xy"]))
            rep = attempt / "execute/report.json"
            report = json.loads(rep.read_text()) if rep.exists() else {"objects": []}
            real = [x.get("result") for x in report["objects"] if math.dist(x["xy"], e["xy"]) < NEAR]
            rows.append({"attempt": attempt.name, "id": near["id"], "twin_ok": e["ok"], "sims": len(e["rehearsals"]),
                         "real_result": real[0] if real else None})
    return rows


def one_desk(seed, world, a):
    folder = a.out / f"desk_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "world.json"
    path.write_text(json.dumps(world, ensure_ascii=False))
    print(f"机 {seed}: " + "、".join(o["name"] for o in world["objects"]), flush=True)
    result = {"seed": seed, "objects": [o["kind"] for o in world["objects"]], "count": len(world["objects"])}
    if "A" in a.ways:
        claims, final, secs = run_a(path, folder / "A", seed)
        g = mj_campaign.goal_score(final)
        result["A"] = {"goal": g["ok"], "cleared": g["cleared"], "outcomes": g["outcomes"], "false_boxed": claims_false(claims, world, final),
                       "discarded_keep": g["discarded_keep"], "touched_precision": g["touched_precision"], "minutes": round(secs / 60, 1)}
        print(f"机 {seed} A: 箱に入った {g['cleared']}/{g['of']}、{secs / 60:.0f} 分", flush=True)
    if "B" in a.ways:
        claims, final, secs, rows = run_b(path, folder / "B", seed, a.real_attempts, a.reflect)
        g = mj_campaign.goal_score(final)
        result["B"] = {"goal": g["ok"], "cleared": g["cleared"], "outcomes": g["outcomes"], "false_boxed": claims_false(claims, world, final),
                       "discarded_keep": g["discarded_keep"], "touched_precision": g["touched_precision"], "minutes": round(secs / 60, 1),
                       "real_attempts": len(rows), "twin": twin_verdicts(folder / "B", world, final)}
        print(f"机 {seed} B: 箱に入った {g['cleared']}/{g['of']}、本番 {len(rows)} 回、{secs / 60:.0f} 分", flush=True)
    return result


def summary(results):
    total = sum(r["count"] for r in results)
    lines = [f"机 {len(results)} 個、物 {total} 個"]
    for way in ("A", "B"):
        rs = [r[way] for r in results if way in r]
        if rs:
            lines.append(f"{way}: 箱に入った {sum(r['cleared'] for r in rs)}、正しい箱 {sum(r['goal'] for r in rs)}、"
                         f"うその『入れた』 {sum(len(r['false_boxed']) for r in rs)}、"
                         f"残す物を捨てた {sum(r['discarded_keep'] for r in rs)}、{sum(r['minutes'] for r in rs):.0f} 分")
    twin = [t for r in results if "B" in r for t in r["B"]["twin"]]
    said_ok = [t for t in twin if t["twin_ok"] and t["real_result"] is not None]
    if said_ok:
        bad = [t for t in said_ok if not str(t["real_result"]).startswith("moved_to_")]
        lines.append(f"双子が成功と言って本番が失敗: {len(bad)}/{len(said_ok)}")
    # what the twin refused, against what plain one-shot did with the same thing on the same desk
    refused = agree = 0
    for r in results:
        if "A" not in r or "B" not in r:
            continue
        done = {o[0]: o[2] for o in r["A"]["outcomes"]}
        for t in r["B"]["twin"]:
            if not t["twin_ok"] and t["real_result"] is None:
                refused += 1
                agree += not done.get(t["id"].rstrip("0123456789"), False)
    if refused:
        lines.append(f"双子が失敗と言った物: {refused}、うち一発勝負でも失敗 {agree}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="1-6")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ways", default="AB")
    p.add_argument("--real-attempts", type=int, default=5)
    p.add_argument("--reflect", default="claude", choices=("claude", "offline"))
    p.add_argument("--desks-at-once", type=int, default=2)
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = sim_campaign.parse_seeds(a.seeds)
    worlds = {s: desk(s) for s in seeds}    # here, not in the threads: mj_scene.build is not thread safe
    results = []
    with ThreadPoolExecutor(max_workers=a.desks_at_once) as pool:
        for r in pool.map(lambda s: one_desk(s, worlds[s], a), seeds):
            results.append(r)
            (a.out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    lines = summary(results)
    (a.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
