#!/usr/bin/env python3
"""Summon and rehearse: the object is summoned into the simulation, Claude thinks of a way to tidy it, the robot
tries it on the twin until it works, and only then on the real arm. A real failure goes back to Claude; the desk is
looked at again (the thing may have moved), summoned again, rehearsed again, up to 5 real attempts in all.

    one real attempt:  look -> twin of every object (summon.py) -> per object up to --sim-limit rehearsals, each from
                       the same twin desk, Claude choosing the next tactic from what really happened -> plan.json ->
                       real attempt with --execute (one grasp per planned thing), filmed

The simulation has no limit worth counting (--sim-limit is only there to stop a hopeless object); --real-attempts
limits the arm. An object with no successful rehearsal stays out of the plan, for a person.

    source /opt/ros/humble/setup.bash
    export MUJOCO_GL=egl TIDY_SAM_GPU=0 MJ_FINGER_KP=1200
    python3 rehearse.py --look LOOK_DIR --out OUT                # a look already taken: twin, rehearsals, plan
    python3 rehearse.py --world WORLD.json --out OUT [--execute] # a simulated "real" desk (WORLD itself is not changed)
    python3 rehearse.py --real --out OUT                         # camera only: photo, twin, rehearsals, plan
    python3 rehearse.py --real --out OUT --execute --preflight P.json   # REAL ARM: driver up, a person beside it

Without --execute nothing but the camera and the simulator is touched. A CAN preflight is good for 300 s only and
rehearsing takes longer: before every real attempt this waits until P.json is fresh, or runs --preflight-cmd.

Videos, numbered from 1 in order, in OUT/videos: sim_attempt_01_show.mp4, ... for every rehearsal and
real_attempt_01_video0.mkv / _video4.mkv for every real attempt (--no-record: none; the real arm refuses to start if a
camera does not record, unless --no-record is given).
"""
import argparse
import json
import math
import os
import shlex
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

import desk_judge
import desk_memory
import sim_world

HERE = Path(__file__).resolve().parent
DRIVER = str(HERE / "config/arm_pick_wide_trial.yaml")
BOXES = str(HERE / "config/desk_boxes.json")
MOVED_OTHER = 0.01     # m: a neighbour moved this far was disturbed
NEAR_BEFORE = 0.12     # m: a thing this near where a real attempt failed is that thing
CLAIM_NEAR = 0.03      # m: a thing this near where one was picked from is that one, not picked at all
RUN_TIMEOUT = 600      # s for one rehearsal, without video
RECORD_TIMEOUT = 1800  # s with video: rendering every frame is slow
REAL_CAMERAS = ("/dev/video0", "/dev/video4")
PREFLIGHT_AGE = 300    # s: arm_goto refuses an older CAN preflight
PREFLIGHT_WAIT = 1800  # s a person gets to take a new one
BOX_OF = {"trash": "trash_box", "keep": "keep_box", "unsure": "keep_box"}
PROPOSE_SCHEMA = {"type": "object", "required": ["tactic", "reason"], "properties": {
    "tactic": dict(desk_memory.TACTIC_SCHEMA, required=list(desk_memory.TACTIC_KEYS)),
    "reason": {"type": "string"}}}
PROPOSE_PROMPT = """You plan how a desk-tidying robot should pick up one object: a 6-axis arm with a two-finger gripper
that comes straight down from above. The fingers open at most 7.7 cm. The robot will first try it on a simulated copy.

The object: "{name}" (kind "{kind}"), about {l:.1f} x {w:.1f} x {h:.1f} cm (the height is an estimate from one photo),
{reach:.0f} cm from the robot's base, goes in the {box}. The judge says: deformable={soft}, force={force} ({force_reason}),
grasp point "{grasp_point}". Other things on the desk: {others}.
The photo of it: {crop}  (read it with the Read tool)

Choose the tactic. The robot can ONLY change these settings:
- force 1-5: grip strength (1 tissue, 2 crumpled paper, 3 pen, 4 firm/slippery, 5 strongest)
- opening "snug" (just wider than the object) or "wide" (fully open)
- soft true (squeeze whatever part the fingers reach; only for squashy things) or false (reach around it)
- turn 0 (fingers close across the short side) or 90 (finger line turned 90 degrees: across the long side)
- sweep_first true (first drag it toward the base, only works farther than 43 cm from the base) or false
- rim true (pinch its wall at the edge, one finger inside and one outside, instead of reaching around its middle) or false
{real}
Reply with the tactic and one short reason in Japanese (under 40 characters)."""
REAL_NOTE = """
What happened on the REAL arm so far on this desk (the copy can be too easy; take these seriously):
{lines}
"""
TRUTH_NOTE = """

This attempt was on the simulated copy, so here is what really happened (ground truth, more reliable than the result code):
{truth}
The copy's true size is {size} cm (length x width x height) and it weighs {mass:.0f} g: use these numbers, do not guess others.
Do not repeat a tactic already tried: the copy is the same each time, so it would end the same way.{real}"""


def say(text):
    print(text, flush=True)


# ---------------------------------------------------------------- pure logic (tested)
def right_box(category):
    return BOX_OF.get(category, "keep_box")


def truth_of(world, oid, category, start_world):
    """What the twin's physics says became of object oid: where, success, neighbours disturbed."""
    start = {o["id"]: o for o in start_world["objects"]}
    me = next(o for o in world["objects"] if o["id"] == oid)
    where = sim_world.container_of(world, me)
    disturbed = [o["id"] for o in world["objects"] if o["id"] != oid
                 and math.dist(o["centre"], start[o["id"]]["centre"]) > MOVED_OTHER]
    return {"where": where or "desk", "xy": [round(float(v), 3) for v in me["centre"]], "z": round(float(me["z"]), 3),
            "moved_m": round(math.dist(me["centre"], start[oid]["centre"]), 3), "disturbed": disturbed,
            "success": where == right_box(category)}


def result_code(truth, claimed):
    """The notebook's result for an attempt: the truth wins over what the robot believed."""
    if truth["success"]:
        return f"moved_to_{truth['where'].replace('_box', '')}"
    return "failed" if str(claimed).startswith(desk_memory.SUCCESS_PREFIX) else (claimed or "failed")


def robot_said(log):
    """What the robot itself logged during the grasp, word for word: how far it got (lifted, dropped on the way...)."""
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    said = [x.strip() for x in lines if x.startswith("    ") and not re.match(r"\s*(SAM|判定|作戦|掴む直前)", x)]
    return "\nThe robot's own log of this attempt (Japanese):\n" + "\n".join(said[:12]) if said else ""


def truth_text(truth, claimed):
    place = {"trash_box": "in the trash bin", "keep_box": "in the keep box", "desk": "still on the desk"}[truth["where"]]
    lines = [f"The object ended {place} at x={truth['xy'][0]:.3f}, y={truth['xy'][1]:.3f}, z={truth['z']:.3f} m, "
             f"{truth['moved_m'] * 100:.1f} cm from where it started."]
    if str(claimed).startswith(desk_memory.SUCCESS_PREFIX) and not truth["success"]:
        lines.append("The robot believed it had succeeded, but it had not.")
    if truth["disturbed"]:
        lines.append("It also pushed: " + ", ".join(truth["disturbed"]))
    return "\n".join(lines)


def unexplored(history):
    """Settings no tactic has used yet on this object, as partial tactics: facts for Claude, in no order of merit."""
    tried = [desk_memory.clamp_tactic(h["tactic"]) for h in history]
    return [c for c in ({"rim": True}, {"soft": True}, {"turn": 90}, {"opening": "wide"})
            if not any(all(t[k] == v for k, v in c.items()) for t in tried)]


def untried(tactic, history, verdict):
    """tactic, or the nearest setting change not tried yet: the twin ends the same way twice. None if all tried."""
    tried = [desk_memory.clamp_tactic(h["tactic"]) for h in history]
    t = desk_memory.clamp_tactic(tactic, desk_memory.default_tactic(verdict))
    if t not in tried:
        return t
    changes = [{"opening": "wide"}, {"force": min(5, t["force"] + 1)}, {"turn": 90 - t["turn"]}, {"opening": "snug"},
               {"force": min(5, t["force"] + 2)}, {"soft": not t["soft"]}, {"force": max(1, t["force"] - 1)}, {"rim": not t["rim"]}]
    for c in changes:
        for base in (t, {**t, "opening": "wide"}, {**t, "turn": 90 - t["turn"]}):
            n = {**base, **c}
            if n not in tried:
                return n
    return None


def order_plan(rows):
    """Plan entries, what may be done first first: rehearsed successes, nearest the base first (as tidy_desk), and
    a thing whose rehearsal pushed a neighbour after that neighbour."""
    good = sorted((r for r in rows if r["ok"]), key=lambda r: math.hypot(*r["xy"]))
    for r in list(good):
        if not r.get("disturbed"):
            continue
        after = [i for i, x in enumerate(good) if x["id"] in r["disturbed"]]
        if after and good.index(r) < max(after):
            good.remove(r)
            good.insert(max(after), r)
    return good + [r for r in rows if not r["ok"]]


def write_plan(path, rows, source):
    rows = order_plan(rows)
    plan = {"source": str(source), "objects": rows,
            "left_for_human": [r["name"] for r in rows if not r["ok"]]}
    Path(path).write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    return plan


# ---------------------------------------------------------------- the loop
def tidy(args, log, env=None, timeout=RUN_TIMEOUT):
    with open(log, "w") as f:
        return subprocess.run([sys.executable, "-u", str(HERE / "tidy_desk.py"), *args], stdout=f, stderr=subprocess.STDOUT,
                              timeout=timeout, env=env, cwd=HERE).returncode


def look(a, folder):
    """The look folder: given, or taken by tidy_desk --preview (camera only, or the simulated desk)."""
    if a.look:
        return Path(a.look).resolve()
    out = folder / "look"
    args = ["--preview", "--judge", "fast", "--out", str(out), "--boxes", BOXES, "--driver-config", a.driver_config]
    if a.world:
        args += ["--sim", str(a.desk), "--physics", "--approve", "yes"]
    say("見る: 写真を撮って、物を判定する")
    code = tidy(args, folder / "look.log")
    if code != 0:
        raise SystemExit(f"見る段階で失敗（{folder / 'look.log'}）")
    return out / "preview"


def propose(a, o, verdict, others, crop, real_lines):
    if a.reflect != "claude":
        return desk_memory.default_tactic(verdict), "判定どおり"
    prompt = PROPOSE_PROMPT.format(
        name=o["name"], kind=verdict.get("kind", "?"), l=o["size"][0] * 100, w=o["size"][1] * 100, h=o["size"][2] * 100,
        reach=math.hypot(*o["centre"]) * 100, box=right_box(o["category"]).replace("_", " "), soft=o["soft"],
        force=verdict.get("force"), force_reason=verdict.get("force_reason", ""), grasp_point=verdict.get("grasp_point", ""),
        others=others or "none", crop=crop, real=REAL_NOTE.format(lines=real_lines) if real_lines else "")
    try:
        r = desk_judge._claude(prompt, PROPOSE_SCHEMA, a.model, str(a.out), timeout=120)
        return desk_memory.clamp_tactic(r["tactic"], desk_memory.default_tactic(verdict)), str(r.get("reason", ""))[:60]
    except (RuntimeError, OSError, ValueError, KeyError):
        return desk_memory.default_tactic(verdict), "Claude が答えないので判定どおり"


def keep_sim_videos(a, rec, number):
    a.videos.mkdir(exist_ok=True)
    for f in sorted(rec.glob("*.mp4")) if rec.exists() else []:
        shutil.copy(f, a.videos / f"sim_attempt_{number:02d}_{f.name}")


def next_number(a):
    with a.counter_lock:
        a.sim_count += 1
        return a.sim_count


def rehearse_object(a, o, verdict, twin, nb, lock, crop, feedback):
    """Rehearsals on the twin, each from the same desk, until one works. Returns the plan entry."""
    pristine = twin / "world.json"
    start_world = json.loads(pristine.read_text())
    env = dict(os.environ, TIDY_EXTRA_OBJECTS=str(twin / "assets"))
    others = "、".join(x["name"] for x in start_world["objects"] if x["id"] != o["id"])
    size_cm = [round(v * 100, 1) for v in o["size"]]
    verdict = {**verdict, "size_cm": size_cm}   # the twin's true size: reflection must not invent another
    with lock:
        oid = nb.remember(o["name"], verdict.get("kind", o["name"]), o["category"], verdict, cv2.imread(str(crop)), size_cm)
    near = [f for f in feedback if f.get("xy") and math.dist(f["xy"], o["look_xy"]) < NEAR_BEFORE]
    real_lines = "\n".join(f"- attempt {f['attempt']}: {f['name']} at x={f['xy'][0]:.2f}, y={f['xy'][1]:.2f}, "
                           f"tactic {json.dumps(f['tactic'])}: {f['result']}" for f in feedback)
    real_note = REAL_NOTE.format(lines=real_lines) if real_lines else ""
    tactic, reason = propose(a, o, verdict, others, crop, real_lines)
    failed_real = [{"tactic": f["tactic"]} for f in near if f.get("tactic")]
    tactic = untried(tactic, failed_real, verdict) or tactic   # it failed on the arm: the twin was too easy for it
    say(f"【{o['name']}】Claude の作戦: {desk_memory.tactic_ja(tactic)}（{reason}）")
    history, done = [], None
    for i in range(1, a.sim_limit + 1):
        folder = twin.parent / "rehearsals" / o["id"] / f"try{i}"
        folder.mkdir(parents=True)
        world = folder / "world.json"
        shutil.copy(pristine, world)   # every rehearsal starts from the same twin desk
        plan = folder / "plan.json"
        plan.write_text(json.dumps({"objects": [{"name": o["name"], "xy": o["centre"], "tactic": tactic, "ok": True}]},
                                   ensure_ascii=False))
        args = ["--sim", str(world), "--physics", "--judge", "oracle", "--approve", "yes", "--plan", str(plan),
                "--out", str(folder / "run"), "--rounds", "1", "--attempts", "1", "--reflect", "offline",
                "--driver-config", a.driver_config, "--seed", "0", "--max-speed", "0.35", "--taught-drops"]
        number = next_number(a)
        if not a.no_record:
            args += ["--record", str(folder / "rec")]
        started = time.time()
        say(f"【{o['name']}】シミュレーション {number} 回目（この物の {i} 回目）: {desk_memory.tactic_ja(tactic)}")
        try:
            code = tidy(args, folder / "tidy.log", env, RUN_TIMEOUT if a.no_record else RECORD_TIMEOUT)
        except subprocess.TimeoutExpired:
            code = "timeout"
        rep = folder / "run/report.json"
        report = json.loads(rep.read_text()) if rep.exists() else {}
        claimed = next((e.get("result") for e in report.get("objects", [])), None)
        truth = truth_of(json.loads(world.read_text()), o["id"], o["category"], start_world)
        result = result_code(truth, claimed)
        h = {"try": i, "sim_attempt": number, "tactic": tactic, "claimed": claimed, "truth": truth, "result": result,
             "exit": code, "seconds": round(time.time() - started)}
        keep_sim_videos(a, folder / "rec", number)
        history.append(h)
        ja = desk_memory.RESULT_JA.get(result, result)
        place = {"trash_box": "ゴミ箱", "keep_box": "保管箱", "desk": "机の上"}[truth["where"]]
        say(f"【{o['name']}】→ {'成功' if truth['success'] else '失敗'}（{ja}、実際は{place}、{h['seconds']} 秒）")
        images = sorted(str(p) for p in (folder / "run").glob("r1_*.jpg"))[-3:]
        if truth["success"]:
            with lock:
                nb.record(oid, i, tactic, result, images, run="rehearsal", why=f"{i}回目のリハーサルで成功")
            done = tactic
            break
        with lock:
            obj = nb.object(oid)
            past = nb.history(oid, run="rehearsal")
        left = unexplored(history)
        left_note = "\nSettings not used yet on this object: " + (json.dumps(left) if left else "none") + "."
        extra = TRUTH_NOTE.format(truth=truth_text(truth, claimed) + robot_said(folder / "tidy.log") + left_note, size=" x ".join(f"{v:g}" for v in size_cm),
                                  mass=o["mass"] * 1000, real=real_note)
        if a.reflect == "claude":
            r = desk_memory.reflect(nb, obj, tactic, result, images, past, model=a.model, cwd=folder, extra=extra,
                                    reach_cm=math.hypot(*o["centre"]) * 100)
        else:
            r = desk_memory.reflect_offline(obj, tactic, result, past)
        h["why"] = r.get("why")
        with lock:
            ep = nb.record(oid, i, tactic, result, images, run="rehearsal", why=r.get("why"))
            nb.learn(obj, r, ep)
        say(f"【{o['name']}】振り返り: {r.get('why')}")
        tactic = untried(r.get("next_tactic"), history + failed_real, verdict)
        if tactic is None:
            say(f"【{o['name']}】試せる作戦が尽きた: {r.get('give_up_reason') or ''}")
            break
        if r.get("give_up") and left:   # the simulator costs nothing: what it never used gets its turn
            tactic = untried({**history[-1]["tactic"], **left[0]}, history + failed_real, verdict) or tactic
            say(f"【{o['name']}】Claude は無理だと言っているが、まだ使っていない設定があるので試す: {desk_memory.tactic_ja(tactic)}")
    last = history[-1]
    return {"id": o["id"], "name": o["name"], "kind": o["kind"], "category": o["category"], "xy": o["look_xy"],
            "twin_xy": o["centre"], "ok": done is not None, "tactic": done or last["tactic"], "rehearsals": history,
            "disturbed": last["truth"]["disturbed"]}


def rehearse_all(a, look_dir, folder, feedback):
    """Twin and rehearsals for the desk as this look shows it. Returns the plan, or None if nothing is left on it."""
    judged_all = json.loads((look_dir / "judge.json").read_text())["objects"]
    if not [j for j in judged_all if j.get("category") != "fixture" and j.get("xy")]:
        say("机の上に片付ける物はもうない")
        return None
    say("召喚: 机の双子をシミュレーションに作る")
    twin = folder / "twin"
    done = subprocess.run([sys.executable, "-u", str(HERE / "summon.py"), "--look", str(look_dir), "--out", str(twin)],
                          cwd=HERE, stdout=sys.stdout, stderr=sys.stderr)
    if done.returncode != 0:
        raise SystemExit("召喚に失敗")
    world = json.loads((twin / "world.json").read_text())
    judged = {tuple(j["xy"]): j for j in judged_all if j.get("xy")}
    img = cv2.imread(str(look_dir / "scene.jpg"))
    nb = desk_memory.Notebook(a.out / "notebook")
    lock = threading.Lock()
    crops = folder / "crops"
    crops.mkdir(exist_ok=True)
    jobs, rows = [], []
    for o in world["objects"]:
        v = judged[tuple(o["look_xy"])]
        verdict = {k: v[k] for k in ("force", "deformable", "size_cm", "category", "should_grasp", "kind", "force_reason",
                                     "grasp_point", "category_reason", "grasp_reason", "confidence", "ask_human") if k in v}
        x0, y0, x1, y1 = v["bbox"]
        crop = crops / f"{o['id']}.jpg"
        cv2.imwrite(str(crop), img[max(0, y0 - 25):y1 + 25, max(0, x0 - 25):x1 + 25])
        if not o.get("should_grasp", True):
            say(f"【{o['name']}】判定で掴まない物: 人に任せる")
            rows.append({"id": o["id"], "name": o["name"], "kind": o["kind"], "category": o["category"], "xy": o["look_xy"],
                         "ok": False, "tactic": None, "rehearsals": [], "reason": v.get("grasp_reason", "")})
            continue
        jobs.append((o, verdict, crop))
    say(f"リハーサル: {len(jobs)} 個、成功するまで（1個につき最大 {a.sim_limit} 回）")
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        futures = [pool.submit(rehearse_object, a, o, v, twin, nb, lock, c, feedback) for o, v, c in jobs]
        rows += [f.result() for f in futures]
    plan = write_plan(folder / "plan.json", rows, look_dir)
    shutil.copy(folder / "plan.json", a.out / "plan.json")
    say("計画:")
    n = 0
    for r in plan["objects"]:
        if r["ok"]:
            n += 1
            say(f"  {n}. {r['name']}: {desk_memory.tactic_ja(r['tactic'])}（シミュレーション {len(r['rehearsals'])} 回目で成功）")
    for r in plan["objects"]:
        if not r["ok"]:
            say(f"  × {r['name']}: シミュレーションで成功せず、人に任せる")
    return plan


def still_there(look_dir, rows):
    """Things the arm said it had boxed that the new look still finds where they were: failures after all."""
    seen = [j["xy"] for j in json.loads((look_dir / "judge.json").read_text())["objects"]
            if j.get("xy") and j.get("category") != "fixture"]
    return [{**r, "result": "the arm reported it boxed, but the camera still sees it there"} for r in rows
            if str(r["result"]).startswith(desk_memory.SUCCESS_PREFIX) and any(math.dist(r["xy"], xy) < CLAIM_NEAR for xy in seen)]


def fresh_preflight(a):
    """Before the arm moves: a CAN preflight younger than arm_goto accepts. Taken by --preflight-cmd, or a person."""
    if a.preflight_cmd:
        subprocess.run(a.preflight_cmd, shell=True, check=True)
    deadline = time.time() + PREFLIGHT_WAIT
    told = False
    while True:
        age = time.time() - a.preflight.stat().st_mtime if a.preflight.exists() else 1e9
        if age < PREFLIGHT_AGE - 60:    # a minute's margin for tidy_desk's own start
            return
        if not told:
            say(f"実機の前に CAN プリフライトを取り直してください: {a.preflight}（待っています）")
            told = True
        if time.time() > deadline:
            raise SystemExit("プリフライトが新しくならないので実機を動かさずに終わる")
        time.sleep(3)


class Filming:
    """ffmpeg on the side cameras for one real attempt, stopped by PID. Refuses if a camera does not start."""

    def __init__(self, a, number):
        a.videos.mkdir(exist_ok=True)
        self.procs = []
        for dev in REAL_CAMERAS:
            out = a.videos / f"real_attempt_{number:02d}_{Path(dev).name}.mkv"
            self.procs.append(subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "v4l2", "-input_format", "mjpeg", "-video_size",
                 "1280x720", "-framerate", "30", "-i", dev, "-c", "copy", str(out)],
                stdin=subprocess.DEVNULL))
        time.sleep(3)
        if any(p.poll() is not None for p in self.procs):
            self.stop()
            raise SystemExit("カメラの録画が始まらないので実機を動かさずに終わる")

    def stop(self):
        for p in self.procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in self.procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()


def real_attempt(a, number, folder):
    """One go of the plan: one grasp per planned thing. On the real arm with --real, else on the simulated desk."""
    out = folder / "execute"
    plan = ["--plan", str(folder / "plan.json")] if (folder / "plan.json").exists() else []
    args = [*plan, "--notebook", str(a.out / "notebook"), "--out", str(out),
            "--reflect", a.reflect, "--run-name", f"real_attempt_{number:02d}", "--driver-config", a.driver_config,
            "--rounds", "1", "--attempts", "1", *shlex.split(a.tidy_args)]
    env = {k: v for k, v in os.environ.items() if k != "TIDY_EXTRA_OBJECTS"}
    film = None
    if a.real:
        fresh_preflight(a)
        args += ["--preflight", str(a.preflight), "--boxes", BOXES]
        if not a.no_record:
            film = Filming(a, number)
            args += ["--snap-cameras", ""]    # ffmpeg holds both cameras now
        say(f"実機 {number} 回目: 開始")
    elif a.world:
        args += ["--sim", str(a.desk), "--physics", "--judge", "fast", "--approve", "yes", "--max-speed", "0.35", "--taught-drops"]
        if not a.no_record:
            args += ["--record", str(folder / "execute_rec")]
        say(f"本番 {number} 回目（シミュレーションの机）: 開始")
    else:
        raise SystemExit("--execute は --world か --real のときだけ")
    try:
        code = subprocess.run([sys.executable, "-u", str(HERE / "tidy_desk.py"), *args], cwd=HERE, env=env).returncode
    finally:
        if film:
            film.stop()
    say(f"本番 {number} 回目: 終了（コード {code}）")
    rec = folder / "execute_rec"
    for f in sorted(rec.glob("*.mp4")) if rec.exists() else []:
        shutil.copy(f, a.videos / f"desk_attempt_{number:02d}_{f.name}")
    return code


def outcomes(a, number, folder):
    """What the arm reported, per thing tried, for Claude and the record."""
    path = folder / "execute/report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    rows = []
    plan_path = folder / "plan.json"   # none on a first real try (--real-first)
    plan = json.loads(plan_path.read_text())["objects"] if plan_path.exists() else []
    for e in report.get("objects", []):
        match = [p for p in plan if p["ok"] and p.get("xy") and math.dist(p["xy"], e["xy"]) < NEAR_BEFORE]
        tactic = (e.get("attempts") or [{}])[-1].get("tactic") or (match[0]["tactic"] if match else None)
        rows.append({"attempt": number, "name": e["name"], "xy": e["xy"], "tactic": tactic, "result": e.get("result")})
    return rows, report.get("status")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--world", help="a simulated real desk (world JSON)")
    src.add_argument("--real", action="store_true", help="the real top camera")
    src.add_argument("--look", help="a look folder already taken (scene.jpg, judge.json, sam_*.json)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sim-limit", type=int, default=8, help="rehearsals per object before it is left for a person")
    p.add_argument("--real-attempts", type=int, default=5, help="real attempts in all (with --execute)")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--reflect", choices=("claude", "offline"), default="claude")
    p.add_argument("--model", default="sonnet")
    p.add_argument("--driver-config", default=DRIVER, help="joint and gripper limits; on the real arm, the running driver's")
    p.add_argument("--continue-from", type=Path, help="an earlier --out: its notebook, real results and attempt count")
    p.add_argument("--try-tactic", help="JSON tactic for the real arm, no rehearsal: a person decided to try it")
    p.add_argument("--no-sim", action="store_true", help="real trial and error only: reflection between real attempts")
    p.add_argument("--real-first", action="store_true", help="try the real arm first, rehearse only after a failure")
    p.add_argument("--execute", action="store_true", help="run the plan (THE REAL ARM with --real)")
    p.add_argument("--preflight", type=Path, help="--real --execute: the CAN preflight file")
    p.add_argument("--preflight-cmd", help="--real --execute: command that writes a fresh preflight before each attempt")
    p.add_argument("--no-record", action="store_true", help="no videos")
    p.add_argument("--tidy-args", default="", help="more tidy_desk options for the real attempts")
    a = p.parse_args()
    if a.execute and a.look:
        p.error("--execute needs --world or --real: a saved look cannot be looked at again")
    if a.execute and a.real and not a.preflight:
        p.error("--real --execute needs --preflight")
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    a.videos = a.out / "videos"
    a.sim_count, a.counter_lock = 0, threading.Lock()
    if a.world:   # the "real" desk changes as it is tidied; the file given stays as it is
        a.desk = a.out / "desk.json"
        shutil.copy(a.world, a.desk)
    start = time.time()
    feedback, log = [], []
    if a.continue_from:
        shutil.copytree(a.continue_from / "notebook", a.out / "notebook")
        log = json.loads((a.continue_from / "real_log.json").read_text())
        feedback = [r for entry in log for r in entry["objects"] if not str(r["result"]).startswith(desk_memory.SUCCESS_PREFIX)]
    for number in range(len(log) + 1, (a.real_attempts if a.execute else 1) + 1):
        folder = a.out / f"attempt_{number:02d}"
        folder.mkdir()
        if a.execute:
            say(f"===== 本番 {number}/{a.real_attempts} 回目の準備 =====")
        look_dir = look(a, folder)
        for r in still_there(look_dir, log[-1]["objects"] if log else []):
            say(f"  本番 {number - 1} 回目で入れたはずの {r['name']} がまだ机にある")
            feedback.append(r)
        if a.try_tactic:   # a person's decision: the real arm tries this, whatever the rehearsals said
            tactic = json.loads(a.try_tactic)
            things = [j for j in json.loads((look_dir / "judge.json").read_text())["objects"]
                      if j.get("xy") and j.get("category") != "fixture"]
            write_plan(folder / "plan.json", [{"name": j["name"], "xy": j["xy"], "tactic": tactic, "ok": True} for j in things],
                       str(look_dir))
            say(f"シミュレーションでは成功していない作戦を、人の判断で実機に試す: {desk_memory.tactic_ja(tactic)}")
        elif a.no_sim:
            say("シミュレーションなしで実機で試す（作戦は経験ノートの振り返りから）")
        elif a.real_first and not feedback:
            say("まず実機で試す（シミュレーションは失敗してから）")
        else:
            plan = rehearse_all(a, look_dir, folder, feedback)
            if plan is None:
                break
            if not any(r["ok"] for r in plan["objects"]):
                say("シミュレーションで成功した物がないので、本番はしない（人に任せる）")
                break
        if not a.execute:
            break
        code = real_attempt(a, number, folder)
        rows, status = outcomes(a, number, folder)
        for r in rows:
            ok = str(r["result"]).startswith(desk_memory.SUCCESS_PREFIX)
            say(f"  本番 {number} 回目: {r['name']} → {desk_memory.RESULT_JA.get(r['result'], r['result'])}")
            if not ok:
                feedback.append(r)
        log.append({"attempt": number, "exit": code, "status": status, "objects": rows, "plan": str(folder / "plan.json")})
        (a.out / "real_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))
        if code != 0 or not str(status).startswith("finished"):
            say(f"本番が正常に終わらなかったので止める（{status}）")
            break
    say(f"おわり: {time.time() - start:.0f} 秒、シミュレーション {a.sim_count} 回、本番 {len(log)} 回。計画: {a.out / 'plan.json'}")


if __name__ == "__main__":
    main()
