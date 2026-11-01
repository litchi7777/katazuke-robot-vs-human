#!/usr/bin/env python3
"""Run tidy_desk.py on many random sim worlds and score the end state against the truth.

An object scores "ok" when it ends where the house rules want it: trash in the
trash box, kept things in the keep box, things the judge should not grasp
untouched. Crushed or faulted runs are counted separately.

    source scripts/ros2_env.sh
    python3 scripts/sim_campaign.py --seeds 0-19 --out DIR [--approve yes|no] [-- extra tidy_desk args]
"""
import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import sim_scenarios
import sim_world

HERE = Path(__file__).resolve().parent


def expected_place(o, approve):
    spec = sim_scenarios.CATALOG[o["kind"]]
    if not spec.get("should_grasp", True):
        return "untouched"
    if spec["category"] == "trash":
        return "trash_box" if approve == "yes" else "untouched"
    return "keep_box"


def score(world, approve):
    rows = []
    for o in world["objects"]:
        want = expected_place(o, approve)
        where = sim_world.container_of(world, o)
        moved = math.dist(o["centre"], o["start"]) > 0.01
        if want == "untouched":
            ok = not moved
            got = "moved" if moved else "untouched"
        else:
            ok = where == want
            got = where or ("desk_moved" if moved else "desk")
        if o.get("crushed"):
            ok, got = False, got + "+crushed"
        rows.append({"id": o["id"], "want": want, "got": got, "ok": ok})
    return rows


def parse_seeds(text):
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(v) for v in text.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--approve", choices=("yes", "no"), default="yes")
    p.add_argument("--hard", action="store_true", help="crowded scenes, objects next to the boxes")
    p.add_argument("--wide", nargs="?", const=True, default=False,
                   help="objects out to 0.55 m (or 0.64 m with --wide far) and +-80 deg, boxes to the sides")
    p.add_argument("extra", nargs="*", help="extra tidy_desk.py arguments after --")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    summary, per_kind, failures = [], Counter(), Counter()
    for seed in parse_seeds(a.seeds):
        run = a.out / f"seed{seed}"
        run.mkdir()
        world_path = run / "world.json"
        sim_world.save(sim_scenarios.random_world(seed, hard=a.hard, wide=a.wide), world_path)
        (run / "world_start.json").write_text(world_path.read_text())
        start = time.monotonic()
        done = subprocess.run([sys.executable, str(HERE / "tidy_desk.py"), "--sim", str(world_path), "--seed",
                               str(seed), "--judge", "oracle", "--approve", a.approve, "--out", str(run / "tidy"),
                               *a.extra], capture_output=True, text=True, timeout=1800)
        (run / "stdout.txt").write_text(done.stdout + done.stderr)
        report = json.loads((run / "tidy/report.json").read_text()) if (run / "tidy/report.json").exists() else {}
        rows = score(sim_world.load(world_path), a.approve)
        ok = sum(r["ok"] for r in rows)
        status = report.get("status", f"crashed rc={done.returncode}")
        summary.append({"seed": seed, "ok": ok, "n": len(rows), "status": status,
                        "seconds": round(time.monotonic() - start, 1), "rows": rows})
        for r in rows:
            kind = r["id"].rstrip("0123456789")
            per_kind[(kind, r["ok"])] += 1
            if not r["ok"]:
                failures[(kind, r["got"])] += 1
        print(f"seed {seed:3d}: {ok}/{len(rows)} ok  {status[:90]}", flush=True)
    total_ok = sum(s["ok"] for s in summary)
    total = sum(s["n"] for s in summary)
    clean = sum(s["status"] == "finished" for s in summary)
    result = {"objects_ok": total_ok, "objects": total, "rate": round(total_ok / max(1, total), 3),
              "runs_finished": clean, "runs": len(summary),
              "per_kind": {k: {"ok": per_kind[(k, True)], "fail": per_kind[(k, False)]}
                           for k in sorted({k for k, _ in per_kind})},
              "failures": {f"{k}:{g}": n for (k, g), n in failures.most_common()},
              "runs_detail": summary}
    (a.out / "summary.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "runs_detail"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
