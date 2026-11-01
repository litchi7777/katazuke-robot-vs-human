#!/usr/bin/env python3
"""絶対に片付けるロボット vs 絶対に散らかす男 (simulation).

Each round the scatterer (claude -p) reads the object catalogue and every earlier round, and lays out a desk
it expects the robot to fail on; the robot tidies it in the MuJoCo desk; the round's result is appended to
the history the scatterer reads next. Between rounds the robot's side is improved by hand (Claude Code):
fix the cause, re-run the same desk (--replay N) and the fixed benchmark.

    source /opt/ros/humble/setup.bash
    MUJOCO_GL=egl python3 battle.py --out ../outputs/battle            # next round
    MUJOCO_GL=egl python3 battle.py --out ../outputs/battle --replay 3 # round 3's desk again, after a fix
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

import desk_judge
import mj_campaign
import mj_scene
import sim_scenarios
import sim_world
import tidy_desk

R_MIN, R_MAX = 0.18, 0.46       # m from the base: what the arm can reach on the desk
ANGLE_MAX = 50.0                # deg either side of straight ahead
MAX_OBJECTS = 8
SETTLE_S = 3.0                  # s of physics before the robot looks
SETTLE_SPEED = 0.01             # m/s: still moving after that (rolling) -> taken off the desk
PUSH_STEP = 0.003               # m per nudge apart
PUSH_MAX = 0.15                 # m: a thing is moved at most this far to clear an overlap
TOUCH = 0.002                   # m: outlines may come this close (touching is allowed, overlapping is not)
MODEL = "sonnet"

SCHEMA = {"type": "object", "required": ["title", "trick", "objects"], "properties": {
    "title": {"type": "string"},
    "trick": {"type": "string"},
    "objects": {"type": "array", "minItems": 3, "maxItems": MAX_OBJECTS, "items": {
        "type": "object", "required": ["kind", "x", "y", "yaw_deg"],
        "properties": {"kind": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"},
                       "yaw_deg": {"type": "number"}}}}}}

PROMPT = """あなたは「絶対に散らかす男」です。片付けロボット（机の上の物をつまんで、ゴミはゴミ箱、残す物は保管箱へ入れる）を
失敗させる机を作ってください。ロボットは毎回あなたの手口を学んで改善されます。目的は、Claudeでできる片付けの限界を見つけることです。

ロボットについて分かっていること:
- 6軸アームとグリッパ（2本指、最大10cm開く、指先は細い）。真上から下ろしてつまむ。
- 土台は (0,0)。届く範囲は土台から {r_min}〜{r_max} m、正面(+x)から左右 {angle} 度以内。
- 上のカメラ1台と深度カメラで物を見つけ、Claude が物ごとに「ゴミ/残す」と力を判定する。
- 精密機器（マウス、ゲーム機）は掴まない決まり。それに触れている物も掴まない。
- ゴミ箱と保管箱は机の外の左右にある（置き場所は気にしなくてよい）。

置ける物（kind: 名前, 大きさ[長さ,幅,高さ cm], 重さ g, やわらかい, 区分）:
{catalog}

これまでの勝負（新しい順）:
{history}

ルール:
- 物は {n_min}〜{n_max} 個。同じ kind は2個まで。
- 位置 x, y は m（土台が原点、+x が正面、+y が左）。yaw_deg は物の長い方向の向き。
- 物どうしは触れてもよいが、重ねてはいけない（シミュレーションが壊れる）。
- 「物理的に絶対に無理」な机（全部が指に入らない、など）は作らない。人間なら片付けられる机で、ロボットの弱点を突くこと。
- ロボットが前の手口に勝ったら、別の手口か、もっと意地悪な組み合わせにする。負けた手口は、直ったか確かめるため時々また使う。

もう見つかった限界（ロボットには直せないと分かった手口。これはもう使わず、まだ見つかっていない弱点を探すこと）:
{limits}

title に机の名前（短く）、trick にどんな弱点を狙ったか（1〜2文）を日本語で書いてください。"""


def catalog_lines():
    lines = []
    for kind, s in sorted(sim_scenarios.GSO_KINDS.items()):
        size = " x ".join(f"{v * 100:.1f}" for v in s["size"])
        lines.append(f"- {kind}: {s['name']}, [{size}], {s['mass'] * 1000:.0f} g, "
                     f"{'やわらかい' if s['soft'] else '硬い'}, {s['category']}"
                     + ("（精密機器）" if kind in sim_scenarios.PRECISION else ""))
    return "\n".join(lines)


def history_lines(rounds):
    if not rounds:
        return "（まだない。最初の机です）"
    out = []
    for r in reversed(rounds[-12:]):
        lost = [f"{name}（{tidy_desk.RESULT_JA.get(res, tidy_desk.REASON_JA.get(res, res))}）"
                for name, res in r["results"] if not str(res).startswith("moved_to_")]
        out.append(f"- 第{r['round']}回「{r['title']}」狙い: {r['trick']} → ロボット {r['cleared']}/{r['goal_of']}"
                   + (f"、片付けられなかった物: {', '.join(lost)}" if lost else "、完全に片付けた")
                   + (f"。その後の改善: {r['fix']}" if r.get("fix") else ""))
    return "\n".join(out)


def ask_scatterer(rounds, folder, rejected=""):
    limits_path = folder.parent / "limits.txt"
    limits = limits_path.read_text().strip() if limits_path.exists() else ""
    prompt = PROMPT.format(r_min=R_MIN, r_max=R_MAX, angle=ANGLE_MAX, catalog=catalog_lines(),
                           history=history_lines(rounds), n_min=4, n_max=MAX_OBJECTS,
                           limits=limits or "（まだない）") + rejected
    return desk_judge._claude(prompt, SCHEMA, MODEL, folder, timeout=300)


def place(spec):
    """World objects from the scatterer's layout; None and a reason if it breaks a rule."""
    world = mj_scene.real_layout()
    count = {}
    for item in spec["objects"]:
        kind = item["kind"]
        if kind not in sim_scenarios.GSO_KINDS:
            return None, f"知らない物 {kind}"
        count[kind] = count.get(kind, 0) + 1
        if count[kind] > 2:
            return None, f"{kind} が3個以上"
        x, y = float(item["x"]), float(item["y"])
        r, ang = math.hypot(x, y), math.degrees(math.atan2(y, x))
        if not (R_MIN <= r <= R_MAX and abs(ang) <= ANGLE_MAX):
            return None, f"{kind} が届く範囲の外 (r {r:.2f} m, {ang:.0f} 度)"
        s = sim_scenarios.CATALOG[kind]
        world["objects"].append({
            "id": f"{kind}{count[kind]}", "kind": kind, "name": s["name"], "centre": [x, y],
            "yaw": math.radians(float(item["yaw_deg"])), "size": list(s["size"]), "z": 0.0, "bgr": s["bgr"],
            "soft": s["soft"], "min_force": s["min_force"], "max_force": s["max_force"],
            "category": s["category"], "start": [x, y], "mesh": s.get("mesh"), "rgba": s.get("rgba"),
            "mass": s.get("mass")})
    objs = world["objects"]
    for i, b in enumerate(objs):
        # Overlaps are pushed apart until they just touch (the scatterer means "touching", and its
        # footprints are guesses): the later thing takes the smallest move, in any direction, that clears
        # every earlier one and stays reachable.
        if not any(overlap(a, b) for a in objs[:i]):
            continue
        start = np.array(b["centre"], float)
        best = None
        for dist in np.arange(PUSH_STEP, PUSH_MAX + 1e-9, PUSH_STEP):
            for t in np.linspace(0, 2 * math.pi, 24, endpoint=False):
                c = start + dist * np.array([math.cos(t), math.sin(t)])
                r, ang = math.hypot(*c), math.degrees(math.atan2(c[1], c[0]))
                if not (R_MIN <= r <= R_MAX and abs(ang) <= ANGLE_MAX):
                    continue
                b["centre"] = list(c)
                if not any(overlap(a, b) for a in objs[:i]):
                    best = list(c)
                    break
            if best is not None:
                break
        if best is None:
            return None, f"{b['id']} を、ほかの物と重ならず届く範囲の中に置ける場所がない（物を詰めすぎ）"
        b["centre"] = b["start"] = best
    return world, None


def settle(world, seconds=SETTLE_S):
    """Let the desk come to rest before the robot looks (2026-11-01: a 19 cm ring toy fell over as placed
    and rolled off the desk, and was counted against the robot). Things keep the pose they settle in;
    what leaves the desk or is still rolling is taken out. Returns the names taken out."""
    import mujoco
    model = mj_scene.build(world).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    gone = []
    for o in list(world["objects"]):
        j = model.joint(o["id"])
        x, y, z, *quat = data.qpos[j.qposadr[0]:j.qposadr[0] + 7]
        speed = float(np.linalg.norm(data.qvel[j.dofadr[0]:j.dofadr[0] + 3]))
        if z < -0.02 or speed > SETTLE_SPEED or not (R_MIN - 0.02 <= math.hypot(x, y) <= R_MAX + 0.04):
            world["objects"].remove(o)
            gone.append(o["name"])
            continue
        o.setdefault("placed", list(o["start"]))
        o["centre"] = o["start"] = [round(float(x), 4), round(float(y), 4)]
        o["z"] = round(max(0.0, float(z) - 0.0005), 4)  # a thing that fell over rests higher than its base
        o["quat"] = [round(float(v), 5) for v in quat]
        w_, x_, y_, z_ = quat
        o["yaw"] = math.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
    return gone


def overlap(a, b):
    """The two footprint rectangles overlap by more than TOUCH (separating-axis test)."""
    def corners(o):
        c, (l, w) = np.array(o["centre"]), (o["size"][0] / 2 - TOUCH, o["size"][1] / 2 - TOUCH)
        u = np.array([math.cos(o["yaw"]), math.sin(o["yaw"])])
        v = np.array([-u[1], u[0]])
        return np.array([c + sx * l * u + sy * w * v for sx in (-1, 1) for sy in (-1, 1)]), (u, v)
    pa, axes_a = corners(a)
    pb, axes_b = corners(b)
    for axis in (*axes_a, *axes_b):
        da, db = pa @ axis, pb @ axis
        if da.max() < db.min() or db.max() < da.min():
            return False
    return True


def run(world, folder, a):
    path = folder / "world.json"
    path.write_text(json.dumps(world, ensure_ascii=False))
    args = ["--sim", str(path), "--physics", "--judge", "fast", "--approve", "yes", "--out", str(folder / "run"),
            "--rounds", "2", "--reflect", "offline", "--driver-config", a.driver_config, "--seed", "0",
            "--max-speed", "0.35", "--taught-drops", "--depth"]
    if a.record:
        args += ["--record", str(folder / "rec")]
    started = time.time()
    tidy_desk.main(args)
    report = json.loads((folder / "run/report.json").read_text())
    goal = mj_campaign.goal_score(sim_world.load(path))
    return {"goal": goal["ok"], "goal_of": goal["of"], "outcomes": goal["outcomes"], "cleared": goal["cleared"],
            "discarded_keep": goal["discarded_keep"], "touched_precision": goal["touched_precision"],
            "status": report.get("status"), "minutes": round((time.time() - started) / 60, 1),
            "results": [(o["name"], o.get("result")) for o in report["objects"]]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replay", type=int, help="run this round's desk again (after a fix) instead of a new one")
    p.add_argument("--record", action="store_true", help="presentation video frames (slower)")
    p.add_argument("--driver-config", default=str(tidy_desk.HERE / "config/arm_pick_wide_trial.yaml"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    hist_path = a.out / "history.json"
    rounds = json.loads(hist_path.read_text()) if hist_path.exists() else []
    if a.replay:
        base = next(r for r in rounds if r["round"] == a.replay)
        world = json.loads((a.out / f"round{a.replay:02d}/world.json").read_text())
        settled = all("placed" in o for o in world["objects"])
        for o in world["objects"]:
            # The pose it settled in the first time (height and turn kept): the same desk again. Settling
            # anew moved a cloth and a sponge on it enough to drop them from the replay.
            o["centre"] = list(o["start"])
            if not settled:  # a desk from before settling was saved: settle it now
                o.pop("quat", None)
                o["z"] = 0.0
        if not settled:
            fell = settle(world)
            if fell:
                print(f"置いたら倒れて転がった物を取り除いた: {'、'.join(fell)}")
        n = sum(1 for r in rounds if r.get("replay_of") == a.replay) + 1
        folder = a.out / f"round{a.replay:02d}_replay{n}"
        folder.mkdir()
        result = run(world, folder, a)
        entry = {**{k: base[k] for k in ("title", "trick")}, "round": f"{a.replay}再{n}", "replay_of": a.replay,
                 **result}
        print(f"第{a.replay}回の机をもう一度: {result['cleared']}/{result['goal_of']}（前は {base['cleared']}/{base['goal_of']}）")
    else:
        no = len([r for r in rounds if "replay_of" not in r]) + 1
        folder = a.out / f"round{no:02d}"
        folder.mkdir()
        rejected = ""
        for attempt in range(5):
            spec = ask_scatterer([r for r in rounds if "replay_of" not in r], folder, rejected)
            world, why = place(spec)
            if world is not None:
                fell = settle(world)
                if fell:
                    print(f"置いたら倒れて転がった物を取り除いた: {'、'.join(fell)}")
                if len(world["objects"]) >= 3:
                    break
                why = "置いた物がほとんど転がり落ちた"
            print(f"散らかす男の机がルール違反: {why}（作り直し）")
            rejected += f"\n- 前に作った机は使えなかった: {why}。物の大きさ（長さ x 幅）を守って、間をあけて置くこと。"
        else:
            raise SystemExit("5回ともルール違反")
        (folder / "scatterer.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1))
        print(f"第{no}回「{spec['title']}」狙い: {spec['trick']}")
        print("  置いた物: " + "、".join(o["name"] for o in world["objects"]))
        result = run(world, folder, a)
        entry = {"round": no, "title": spec["title"], "trick": spec["trick"], **result}
        print(f"結果: ロボット {result['cleared']}/{result['goal_of']}（{result['minutes']} 分）")
    for name, res in result["results"]:
        print(f"  {name}: {res}")
    rounds.append(entry)
    hist_path.write_text(json.dumps(rounds, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
