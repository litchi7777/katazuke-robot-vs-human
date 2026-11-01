#!/usr/bin/env python3
"""Tidy the desk: judge every object with claude -p, ask before discarding, sort into two boxes.

Each round:
  1. fold the arm out of view, capture video2, find the trash and keep boxes
  2. judge the frame (claude -p, or the oracle in simulation): category, whether
     to grasp at all, grip force 1-5, size, confidence
  3. ask the human on the local page (confirm_server.py) about everything that
     would be thrown away or is uncertain; nothing is discarded without "捨てる"
  4. for each approved object: probe down with the gripper opened to its size,
     close with the judged force, drop it in its box, fold, re-capture and check
     it left its spot (a slip retries once with one more force level)
Repeats until nothing actionable is left (or --rounds).

    source scripts/ros2_env.sh
    # rehearsal, no hardware:
    python3 scripts/tidy_desk.py --sim WORLD.json --judge oracle --approve yes --out DIR
    # real arm (driver running with arm_pick_trial.yaml, fresh preflight):
    python3 scripts/tidy_desk.py --preflight P.json --boxes scripts/config/desk_boxes.json --out DIR
"""
import argparse
import atexit
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import arm_goto
import confirm_server
import desk_judge
import desk_memory
import pick_pen_auto as ppa
import plan_pen_pick
import sim_scenarios
import sim_world
import topdown_ik

HERE = Path(__file__).resolve().parent
CONDA_SH = Path.home() / "miniforge3/etc/profile.d/conda.sh"
FORCE_SQUEEZE = desk_judge.FORCE_SQUEEZE
OPEN_MARGIN = 0.012     # m wider than the object across the fingers
OPEN_TALL_SHARE = 0.2   # ...plus this share of its height
OPEN_MIN = 0.4          # rad
RELEASE_OPEN = 3.0      # rad (about 5 cm) to let go when no grasp opening is known
CLEARANCE = 0.001       # m per side the object must clear the open fingers by (4 mm turned a 9.7 cm plate, held
                        # whole before, into a rim pinch that slipped: sim 2026-10-28)
CLOSE_LEAD = 0.008     # m: the fingers jump to the object's width plus this, then close in steps
SOFT_LEAD = 0.5        # share of a soft object's width the fingers jump to before stepping
SOFT_OPEN = 0.7         # fraction of a soft object's width the fingers open to
COARSE_STEP = 0.025     # m per move from the hover down to the fine probe (0.015 until 2026-11-01)
ROBOT_RADIUS = 0.16     # m around the base: the folded robot itself, never an obstacle to plan around
FLAT_CM = 0.5           # thinner rigid things (sheets, the checkerboard) cannot be pinched off the desk
TALL = 0.06             # m: taller objects are gripped wherever the probe stops (the rail meets their top)
TALL_GRIP = 0.055       # m below a tall object's top the fingertips go (they are ~7 cm long to the palm)
TALLEST = 0.25          # m: heights are believed up to this
TALL_TOP_BAND = 0.015   # m below a tall object's top a finger stop counts as on its side
RIM_GAP = 0.04          # m finger gap astride a rim (a wall up to ~2 cm with room for aiming errors)
RIM_WALL = 0.02         # m of rim between the fingers the planner allows for
RIM_INSET = float(os.environ.get("TIDY_RIM_INSET", 0.008))       # m inside the outer edge the pinch is centred (most walls are 3-15 mm)
RIM_STEP_DEG = 15       # deg between candidate pinch points round the rim
RIM_BAND = 0.01         # m below the top a depth reading still counts as the rim
RIM_ROUND = 1.2         # a rim point this much farther out than the rim's usual radius is a handle or a spout
RIM_DEPTH = 0.025       # m below the rim the fingertips go
DESK_LEVEL = 0.003        # m: depth readings this low on a thing's outline are the desk round it
DESK_HALO_TOP = 0.02      # m: ...for things at least this tall (a sheet or a card keeps every reading)
TOUCH_PX = 6              # px between outlines that count as touching (or one thing, like a mouse and its cable)
PARK_Z = 0.18            # m tip height the arm waits at over the keep box while the camera looks
RELEASE_SPREAD = 0.06    # m between spots in a box things are let go at
RELEASE_MARGIN = 0.05    # m the spots keep inside a box's walls
MISJUDGED_CM = 3.0       # cm and
MISJUDGED_RATIO = 1.4    # times between the judge's height and the depth's: ask before discarding
RIM_NO_WALL = 0.006     # m below the top a rim-pinch finger stopping counts as standing on a solid top
SOFT_WIDE = 0.095       # m: soft things wider than the open fingers everywhere are pinched by an edge
SOFT_EDGE_INSET = 0.012
PART_STEP = 0.01        # m between the grip heights tried for a narrow part
PART_SPLIT = 0.014      # m: a gap this wide between two pieces in the finger band lets a finger in
PART_MIN = 0.006        # m: a piece thinner than this across is not gripped
PART_ROOM = 0.008       # m at most each finger opens beyond the piece
PART_DEEPER = 0.5       # one cm deeper in the grip is worth this many cm farther from the centre
LONG_END_KEEP = 0.02    # m a grasp moved along a long thing stays this far from its end
PENDING = "pending"     # a page decision not given yet
PLAN_MATCH = 0.08       # m: a judged thing this near a plan entry's place is that entry
CONTAINER_WORDS = ("器", "ボウル", "椀", "カップ", "コップ", "マグ", "皿", "鉢", "ラメキン", "かご", "カゴ", "バケツ", "トレー")
HOLLOW_DROP = 0.012     # m the middle of a container lies below its rim
PART_SQUARE = math.radians(25)  # the piece's faces under the two finger halves may lean this much (not a wedge)  # m inside a cloth's edge the pinch is centred
ON_TOP_SHARE = 0.75     # a stop above this share of the object's height means a finger is on top of it
ON_TOP_MAX = 0.015      # m, cap of that margin above the floor contact
BACK_OFF = 0.12         # m toward the base from an object, where the arm waits while the camera looks again
SWEEP_Z = 0.008         # m URDF tip height while sweeping (real fingertips ~3 mm above the desk)
SWEEP_BEHIND = 0.025    # m beyond the object's far edge where the paddle comes down
SWEEP_TO = 0.48         # m from the base where a swept object should end up
DRAG_DISTANCES = (0.06, 0.10)  # m a crowded thing is pulled in to get the fingers round it
DRAG_OVERLAP = 0.01     # m two footprints (as circles round their long sides) may overlap after a pull
SWEEP_STEP = 0.02       # m per sweep move
SOFT_ON_TOP = 0.6       # a soft thing's probe stopping above this share of its height: a finger is on its top
SOFT_SHIFT = 0.012      # m a soft object's grasp point may move off its centre, away from walls
PLAN_MARGIN = 0.004     # m obstacles are grown by when planning (camera calibration error)
HELD_BELOW = 0.05       # m a held object may hang below the fingertips while carried
CARRY_Z = 0.20          # m tip height reached straight up before carrying to a box
APPROACH_Z = 0.30       # m tip height straight over an object before descending to it (user, 2026-10-28)
TRANSIT_HIGH = 0.30     # m higher carry tried when the 0.20 m path would touch something
LIFT_FIRST = 0.012      # m straight back along the approach before rejoining the probe line
ALIGN_GOOD = math.radians(10)  # fingers this far off a rigid object's sides: try across the other side
ALIGN_GAIN = math.radians(5)   # ...and take it only if it lines up this much better
RESQUEEZE_LIFT = 0.003  # m lifted before checking whether the held object turned and needs squeezing again
DROP_ABOVE_RIM = 0.07   # m tip height above the box rim when releasing
DROP_MIN_Z = 0.08       # m lowest release height (a bin beside the desk may have its rim below the desk)
DROP_HIGH = 0.25        # m above the rim: the highest straight-down release tried first
DROP_LOW = 0.05         # m above the rim: the lowest one
DROP_FLOOR = 0.03       # m above the desk top: never lower (the fingers pass over the desk edge)
PASS_OVER = 0.08        # m: a carry passing this close (seen from above) to a precision device goes round it
HANG_CLEAR = 0.03       # m a held object's bottom stays above a box rim (and the desk's other things) when carried
SOFT_GRIP_SHARE = 0.4   # a soft thing is squeezed this far up its height, not at its foot
SOFT_GRIP_MIN_H = 0.015  # m: lower soft things are squeezed where the fingers stop
CARRY_CLEAR = 0.03      # m a held thing's bottom is lifted above the tallest thing under the carry
CARRY_OVER = 0.10       # m the bottom of a held tall object is lifted above the desk before carrying
REFINE_GROW_PX = 30     # px the box is grown by when an object is outlined again after a failed grasp
REFINE_MAX_JUMP = 0.06  # m: a new outline farther than this from the old position is something else
OPEN_WIDE_SLACK = 1.25  # a lesson's "open fully" may try things the camera reads up to this much wider than the fingers
DEPTH_MAX_OBJECT = TALLEST + 0.02  # m: higher readings are not an object (the arm's own pixels are masked out)
CHANGED_MIN = 12.0      # LAB: the least change counted (an unchanged pale paper read 7 median, 11-01 real)
CHANGED_SHARE = 0.7     # a pixel counts as changed past this share of the thing's contrast with the desk
THIN_DIRECT = 0.03      # m: a thing lower than this on the desk is reached straight down, no contact sensing
DIRECT_ABOVE = 0.002    # m the fingertips stop above the desk for such a thing
ON_TOP_RAISED = 0.008   # m above the floor contact: every shift stopping there is a finger on the thing
WIDEN_ON_TOP = 0.008    # m per side the fingers open wider when every shift landed on the object's top
UNTOUCHABLE_CLEAR = 0.01  # m per side a precision device beside the target is grown by for the fingers
CABLE_WORDS = ("ケーブル", "コード", "ひも", "紐", "線", "チェーン", "cable", "cord")
CABLE_MAX_WIDTH_CM = 1.2   # a thing this thin...
CABLE_MIN_RATIO = 6.0      # ...and this many times as long is treated as a cable
UNEXPLAINED_SHARE = 0.3  # an unlisted patch standing on the desk this share of a thing's size may be it
FOUND_COLOUR = 25.0     # LAB: a changed patch this close to a thing's colour may be it, pushed
HELD_HALO_PX = 121      # px the arm's image is grown by (about 6 cm) when asking whether a thing left the desk
LEFT_MIN_MOVE = 0.015   # m: a change nearer than this to where a thing was is its own old place
RING_PX = 12            # px ring outside a box where the desk colour is read
BARE_DESK = 10.0        # LAB: an outline this close to the desk around it is bare desk
CHANGED = 25.0          # LAB distance at which a pixel counts as changed between before and after
CATEGORY_JA = {"trash": "ゴミ", "keep": "残す", "unsure": "わからない", "fixture": "固定"}
GIVE_UP_RADIUS = 0.03  # m: an object this close to a spot that failed is not tried again
RESULT_JA = {"moved_to_trash": "ゴミ箱に入れました", "moved_to_keep": "保管箱に入れました", "swept": "手前に引き寄せました"}
REASON_JA = {"too_flat": "平たすぎて掴めない（シート状）", "too_far": "遠すぎる（肩の負荷の上限）", "left_by_human": "人が「そのまま」と答えた", "no_trash_box": "ゴミ箱がない", "blocked": "箱や他の物が邪魔で指を下ろせない",
             "too_wide": "指の開きより幅が広い", "unreachable": "届かない／指を合わせられない",
             "stalled": "下まで下ろせない", "failed": "2回掴み損ねた", "lost": "見失った"}


class LiveCamera:
    """The overhead camera kept open: a photo is the next frame, not a fresh open and a 5 s exposure wait.

    2026-11-01 real run: ~20 photos at ~6 s each were 15 % of a 13.5 min tidy. Frames are read all the
    time so none is stale; save() takes the first frame started after it was called. On any failure
    it returns False and capture() falls back to ffmpeg.
    """

    SETTLE = 5.0   # s after opening before the first photo (auto exposure and white balance)
    FRESH = 0.1    # s after the call a frame must arrive to show the desk as it is now

    def __init__(self, device):
        self.frame, self.stamp, self.ok = None, 0.0, True
        self.opened = time.monotonic()
        self.lock = threading.Lock()
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            self.ok = False
            return
        self.cap = cap
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        atexit.register(self.close)  # a reader thread still inside cap.read() at exit aborts the process

    def close(self):
        self.ok = False
        self.thread.join(timeout=1.0)
        self.cap.release()

    def run(self):
        while self.ok:
            got, frame = self.cap.read()
            if not got:
                self.ok = False
                break
            with self.lock:
                self.frame, self.stamp = frame, time.monotonic()

    def save(self, path):
        time.sleep(max(0.0, self.opened + self.SETTLE - time.monotonic()))
        wanted, end = time.monotonic() + self.FRESH, time.monotonic() + 3.0
        while self.ok and time.monotonic() < end:
            with self.lock:
                frame, stamp = self.frame, self.stamp
            if frame is not None and stamp >= wanted and frame.shape[:2] == (960, 1280):
                return bool(cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]))
            time.sleep(0.01)
        return False


class Desk:
    """Everything one tidy run needs: arm client, camera, judge, boxes, report."""

    def __init__(self, a):
        self.a = a
        self.calib = json.load(open(a.calib))
        self.limits = arm_goto.driver_limits(a.driver_config)
        self.report = {"rounds": [], "objects": []}
        self.frames = 0
        self.camera = None if a.sim else LiveCamera("/dev/video2")
        rng = np.random.default_rng(a.seed + 7)
        angle = rng.uniform(0, 2 * math.pi)
        self.bias = np.array([math.cos(angle), math.sin(angle)]) * a.xy_bias_mm / 1000 if a.sim else np.zeros(2)
        boxes = json.loads(Path(a.boxes).read_text())
        # A box out of the overhead camera's view (e.g. a bin beside the desk) is given by position.
        self.fixed_boxes = {role: v for role, v in boxes.items() if "xy" in v}
        self.box_seeds = {role: np.array(v["seed_px"], float) for role, v in boxes.items() if "xy" not in v}
        self.box_outer = {role: v.get("outer", 0.124) for role, v in boxes.items()}
        self.boxes = {}
        self.given_up = []  # (xy, reason): spots not to try again this run
        self.round_items, self.decisions, self.finished = [], {}, {}
        self.standing = []  # (key, object) on the desk that this round leaves alone: obstacles all the same
        self.soft_hold = False
        self.hang = 0.0  # m the held object reaches below the fingertips (tall things are gripped near the top)
        self.released = {}  # box role -> things let go into it (each next one goes to another spot)
        self.held_reach = 0.0  # m the held thing may reach out sideways from the fingers
        self.at_box = None
        self.round_image = None
        self.blocked_by = []
        self.client = None
        self.approval_dir = a.out / "approval"
        self.approval_dir.mkdir()
        self.server = None
        self.nb = desk_memory.Notebook(a.notebook) if a.notebook else None
        self.run_name = a.run_name or a.out.name
        self.rng = np.random.default_rng(a.seed + 99)

    def apply_plan(self, objects):
        """--plan (rehearse.py): the things the plan may touch get their rank and tactic; the rest stay on the desk."""
        if not self.a.plan:
            return
        entries = [e for e in json.loads(Path(self.a.plan).read_text())["objects"] if e.get("ok")]
        for rank, e in enumerate(entries):
            near = [o for o in objects if o.get("xy") is not None and "plan_rank" not in o and o["category"] != "fixture"
                    and math.dist(o["xy"], e["xy"]) < PLAN_MATCH]
            if not near:
                print(f"    計画の {e['name']} が机に見当たらない", flush=True)
                continue
            o = min(near, key=lambda x: math.dist(x["xy"], e["xy"]))
            o["plan_rank"], o["plan_tactic"] = rank, e["tactic"]
        left = [o["name"] for o in objects if "plan_rank" not in o and o["category"] != "fixture"]
        if left:
            print("    計画にない物は触らない（人に任せる）: " + "、".join(left), flush=True)

    # --- camera and judge --------------------------------------------
    def capture(self, name):
        path = self.a.out / f"{name}.jpg"
        self.frames += 1
        if self.a.physics and self.client is None:  # --preview: no arm process, one look at the desk as it lies
            import mj_scene
            mj_scene.snapshot(self.a.sim, path)
        elif self.a.physics:
            self.send(f"render {path} video2", ("撮影",), 180)
        elif self.a.sim:
            sim_world.render(sim_world.load(self.a.sim), self.calib, path, seed=self.frames)
        elif self.camera is not None and self.camera.save(path):
            pass
        else:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "v4l2", "-input_format",
                            "mjpeg", "-video_size", "1280x960", "-framerate", "30", "-i", "/dev/video2", "-ss", "5",
                            "-frames:v", "1", str(path)], check=True, timeout=30)
        return path

    def judge(self, image, folder):
        img = cv2.imread(str(image))
        if self.a.judge == "claude":
            objects, meta = desk_judge.ask_image(image, folder, self.a.model)
        elif self.a.judge == "fast":
            advice = self.nb.prompt_advice() if self.nb else ""
            missed = None
            if self.a.depth:
                # Things the listing missed are found standing on the desk (depth) outside every listed box.
                pre = folder / "height_listing.npy"
                self.send(f"height {pre} video2", ("撮影",), 180)
                hm = np.load(pre)
                missed = lambda boxes: desk_judge.overlooked(hm, self.calib, boxes, self.arm_mask(hm.shape[:2]))
            objects, meta = desk_judge.ask_fast(image, folder, self.calib, self.a.model, advice=advice,
                                                recall=self.recall_verdict if self.nb else None, missed=missed)
            if any(o["name"] == "見落としていた物" for o in objects):
                print("    深度カメラで、一覧から漏れた物を見つけた: "
                      + "、".join(o.get("kind", o["name"]) for o in objects if o["name"] == "見落としていた物"),
                      flush=True)
            print(f"    判定 {meta['seconds']:.0f} 秒（物の一覧 {meta['detect_seconds']:.0f} 秒 + 物ごとの判定を並列）"
                  + (f"、知識ノートから思い出した物 {meta['recalled']} 件" if meta.get("recalled") else ""), flush=True)
        else:
            import sim_scenarios
            world = sim_world.load(self.a.sim)
            if self.a.physics:  # what already sits in a box beside the desk is out of the camera's view
                world["objects"] = [o for o in world["objects"] if sim_world.container_of(world, o) is None]
            objects = sim_scenarios.oracle_judge(world, self.calib, self.a.judge_noise,
                                                 np.random.default_rng(self.frames), self.a.judge_force_error)
            for o in objects:
                o.setdefault("kind", o["name"])
            meta = {"seconds": 0.0}
        desk_judge.locate_all(img, objects, self.calib, self.segment(image, [o["bbox"] for o in objects]))
        objects = [o for o in objects if not self.part_of_listed(o, objects)]
        if self.a.depth:
            self.measure_heights(objects, folder)
        if self.a.sim and self.a.judge_miss:  # the judge overlooks some objects
            rng = np.random.default_rng(1000 + self.frames)
            objects = [o for o in objects if o["category"] == "fixture" or rng.uniform() >= self.a.judge_miss]
        self.apply_bias(objects)
        desk_judge.draw_overlay(img, objects, folder / "overlay.jpg")
        (folder / "judge.json").write_text(json.dumps({**meta, "objects": desk_judge.public(objects)}, indent=1,
                                                      ensure_ascii=False))
        return objects

    @staticmethod
    def part_of_listed(o, objects):
        """A thing found by depth outside the listing whose outline lies mostly in a listed thing's box:
        an edge of that thing, not another one."""
        px = o.get("_pixels")
        if o["name"] != "見落としていた物" or px is None or len(px) == 0:
            return False
        for x in objects:
            if x is o or x["name"] == "見落としていた物":
                continue
            x0, y0, x1, y1 = x["bbox"]
            inside = (px[:, 0] >= x0) & (px[:, 0] <= x1) & (px[:, 1] >= y0) & (px[:, 1] <= y1)
            if inside.mean() > 0.5:
                return True
        return False

    def depth_fix(self, objects, image):
        """After a re-outline: measure again with the depth camera (same scene, nothing moved since)."""
        if self.a.depth:
            heights = [o["size_cm"][2] for o in objects]
            self.measure_heights(objects, Path(image).parent, quiet=True)  # a fallen bottle is lower now
            for o, h in zip(objects, heights):
                if o["size_cm"][2] < max(FLAT_CM, 0.3 * h):
                    # Mostly an outline that missed it (sim 2026-10-28: a fallen 13 cm mug read as flat).
                    o["size_cm"][2] = h

    def measure_heights(self, objects, folder, quiet=False):
        """A depth camera's height for each object replaces the judge's guess (a photo cannot tell
        a 8 mm shogi piece from a flat tag, 2026-10-27)."""
        path = folder / "height.npy"
        self.send(f"height {path} video2", ("撮影",), 180)
        hm = np.load(path)
        arm = self.arm_mask(hm.shape[:2])
        changed = []
        for o in objects:
            px = o.get("_pixels")
            if px is None or len(px) == 0 or o["category"] == "fixture":
                continue
            u, v = px[:, 0].clip(0, hm.shape[1] - 1), px[:, 1].clip(0, hm.shape[0] - 1)
            h = hm[v, u]
            # Not the desk, not the arm above it. The arm is masked by where it is, not by a height cap
            # (sim 2026-10-28: a 0.16 m cap read an 18.7 cm ring toy as 13.5 cm and the fingers came down on it).
            under_arm = arm[v, u] > 0
            if under_arm.mean() > 0.25:
                # Mostly hidden under the arm (it starts folded over the desk near the base): its visible
                # rim would read far too low (sim 2026-10-28: a 16 cm bell as 1.5 cm). Measured once the arm moves.
                continue
            ok = (h > 0.0) & (h < DEPTH_MAX_OBJECT) & ~under_arm
            if ok.sum() < 10:
                continue
            # The outline takes in a pixel or two of desk round a tall thing; seen past its top that desk is
            # far behind it (sim 2026-10-29: a 6.3 cm bottle 19.7 cm tall measured 13.4 cm long, and as an
            # obstacle it kept the fingers off three figures beside it). Desk-level readings are not the thing.
            if np.percentile(h[ok], 90) >= DESK_HALO_TOP and (ok & (h > DESK_LEVEL)).sum() >= 10:
                ok &= h > DESK_LEVEL
            top = float(np.percentile(h[ok], 90))
            # Its highest point, for the hover over it: a toy's antenna or a dome stands above the 90 %
            # top (sim 2026-10-28: a 11 cm wheeled toy read 8.3 cm; the descent to its hover hit it and
            # stopped the arm).
            o["peak_z"] = round(float(np.percentile(h[ok], 99.5)), 4)
            m = desk_judge.measure_3d(self.calib, px[ok], h[ok])
            top_band = ok & (h >= top - RIM_BAND)
            o["_rim_xy"] = desk_judge.pixels_to_base(self.calib, px[top_band], h[top_band])  # for a rim pinch
            o["_xyz"] = np.c_[desk_judge.pixels_to_base(self.calib, px[ok], h[ok]), h[ok]]  # for a part grasp
            if m is not None:  # where every visible point really is, not one plane's guess
                o["xy"] = [round(float(v), 4) for v in m["xy"]]
                o["axis_xy"] = [round(float(v), 4) for v in m["axis"]]
                o["size_m"] = [round(m["length"], 3), round(m["width"], 3)]
                o["reach_m"] = round(float(np.hypot(*o["xy"])), 3)
                o["size_from_depth"] = True
            old = float(o["size_cm"][2])
            o.setdefault("judged_height_cm", old)  # what the judge thought, before any depth reading
            o["size_cm"] = [o["size_cm"][0], o["size_cm"][1], round(top * 100, 1)]
            o["height_from_depth"] = True
            if abs(old - top * 100) >= 0.5:
                changed.append(f"{o['name']} {old:g}→{top * 100:.1f}cm")
        if changed and not quiet:
            print("    深度カメラで高さを測り直した: " + "、".join(changed), flush=True)

    def recall_verdict(self, name, crop, size_cm):
        """A remembered object's verdict (the judge is not asked again), or None."""
        obj, score = self.nb.recall(crop, size_cm, kind=name)  # the detector's name gates the match
        if obj is None or not obj.get("verdict"):
            return None
        print(f"    思い出した: {obj['name']}（似ている度 {score:.2f}）", flush=True)
        return {**obj["verdict"], "kind": obj.get("kind", obj["name"])}

    def segment(self, image, boxes):
        """SAM masks for the boxes (vision_seg.py in the tidyvision env), or None to use colour."""
        if (self.a.sim and not self.a.physics) or not self.a.sam or not boxes:
            return None
        folder = Path(image).parent
        (folder / "sam_boxes.json").write_text(json.dumps(boxes))
        start = time.monotonic()
        try:
            gpu = os.environ.get("TIDY_SAM_GPU", "0")  # a GPU index or UUID for SAM
            done = subprocess.run(["bash", "-lc", f"source {CONDA_SH} && CUDA_VISIBLE_DEVICES={gpu} conda run -n tidyvision "
                                   f"python {HERE / 'vision_seg.py'} {image} {folder / 'sam_boxes.json'} "
                                   f"{folder / 'sam_masks.json'}"], capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            print("    SAM が時間内に終わらないので色で探します", flush=True)
            return None
        if done.returncode != 0:
            print(f"    SAM が使えないので色で探します: {done.stderr[-200:]}", flush=True)
            return None
        print(f"    SAM で輪郭を切り出し（{time.monotonic() - start:.1f} 秒）", flush=True)
        return json.loads((folder / "sam_masks.json").read_text())

    def relocate(self, o, image, refine=False):
        """Find o again after a grasp attempt (a miss usually moves it).

        Its old pixels unchanged since the round's first frame -> it is still there. Otherwise look
        where the desk changed: pieces of change of about its size, outside the arm's shadow and away
        from where it was, are candidates; the one whose colour matches it best is outlined with SAM.
        """
        if self.round_image is None or o.get("_pixels") is None:
            desk_judge.locate_all(cv2.imread(str(image)), [o], self.calib, self.segment(image, [o["bbox"]]))
            self.depth_fix([o], image)
            return o.get("xy") is not None
        if self.still_there(o, image):
            if refine:
                # After a failed grasp it is often still there but nudged and turned (2026-10-27: a slipped
                # block landed 2 cm off and the next three grasps went to its old pose): outline it again.
                now = cv2.imread(str(image))
                h, w = now.shape[:2]
                x0, y0, x1, y1 = o["bbox"]
                grow = REFINE_GROW_PX
                box = [max(0, x0 - grow), max(0, y0 - grow), min(w - 1, x1 + grow), min(h - 1, y1 + grow)]
                saved = dict(o)
                desk_judge.locate_all(now, [o], self.calib, self.segment(image, [box]))
                self.depth_fix([o], image)
                if o.get("xy") is None or math.dist(o["xy"], saved["xy"]) > REFINE_MAX_JUMP:
                    o.clear()
                    o.update(saved)  # the outline found something else: keep what we knew
                elif o.get("_pixels") is not None and self.desk_contrast(o, now) < BARE_DESK:
                    # The new outline is bare desk (2026-11-01 real: SAM given a box round where paper had
                    # been outlined the empty desk there): it is not there, look where the desk changed.
                    o.clear()
                    o.update(saved)
                    return self.relocate_changed(o, image, max_colour=FOUND_COLOUR)
            return True
        if self.relocate_changed(o, image, max_colour=FOUND_COLOUR):
            return True
        # Not where it was and no patch of its colour elsewhere: a miss usually nudges a thing only a little
        # (2026-11-01 real: a tissue and a small box "lost" after one miss, both a few cm off), so outline
        # round its old place; anything there that is not bare desk is taken for it.
        return self.outline_near(o, image)

    def outline_near(self, o, image):
        """SAM round o's old box grown by REFINE_GROW_PX; True (o updated) if that finds something not bare
        desk within REFINE_MAX_JUMP of where o was."""
        if o.get("bbox") is None or o.get("xy") is None:
            return False
        now = cv2.imread(str(image))
        h, w = now.shape[:2]
        x0, y0, x1, y1 = o["bbox"]
        grow = REFINE_GROW_PX
        box = [max(0, x0 - grow), max(0, y0 - grow), min(w - 1, x1 + grow), min(h - 1, y1 + grow)]
        saved = dict(o)
        desk_judge.locate_all(now, [o], self.calib, self.segment(image, [box]))
        self.depth_fix([o], image)
        if (o.get("xy") is None or math.dist(o["xy"], saved["xy"]) > REFINE_MAX_JUMP or o.get("_pixels") is None
                or self.desk_contrast(o, now) < BARE_DESK):
            o.clear()
            o.update(saved)
            o["xy"] = None
            return False
        return True

    def relocate_changed(self, o, image, max_colour=None, arm_grow=0):
        """Find o where the desk changed since the round's first frame (see relocate).

        max_colour: a candidate must be this close (LAB) to o's colour, else o counts as not found."""
        now = cv2.imread(str(image))
        lab_now = cv2.cvtColor(now, cv2.COLOR_BGR2LAB).astype(float)
        lab_then = cv2.cvtColor(self.round_image, cv2.COLOR_BGR2LAB).astype(float)
        px = o["_pixels"]
        colour = lab_then[px[:, 1], px[:, 0]].mean(0)
        threshold = max(CHANGED_MIN, min(CHANGED, CHANGED_SHARE * self.desk_contrast(o, self.round_image)))
        changed = (np.linalg.norm(lab_now - lab_then, axis=2) > threshold).astype(np.uint8)
        arm = self.arm_shadow(image)
        if arm_grow:
            arm = cv2.dilate(arm, np.ones((arm_grow, arm_grow), np.uint8))
        changed[arm > 0] = 0
        changed[px[:, 1], px[:, 0]] = 0  # where it used to be
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(changed)
        # Its size in pixels: the outline's points are only a sample of a big thing (sim 2026-10-29: things
        # over ~12 000 px were never found again after a slip, their new place being "too big").
        area = o.get("_area") or len(px)
        best = None
        for i in range(1, n):
            if not 0.3 * area <= stats[i, 4] <= 3.0 * area:
                continue
            ys, xs = np.nonzero(labels == i)
            score = float(np.linalg.norm(lab_now[ys, xs].mean(0) - colour))
            if best is None or score < best[0]:
                x, y, w, h = stats[i, :4]
                best = (score, [int(x), int(y), int(x + w), int(y + h)])
        if best is None or (max_colour is not None and best[0] > max_colour):
            o["xy"] = None
            return False
        o["bbox"] = best[1]
        desk_judge.locate_all(now, [o], self.calib, self.segment(image, [o["bbox"]]))
        self.depth_fix([o], image)
        return o.get("xy") is not None

    def apply_bias(self, objects):
        """Simulation: shift everything the camera located by a fixed calibration error."""
        if not self.a.sim or self.a.physics or not any(self.bias):
            return
        for o in objects:
            if o.get("xy") is not None:
                o["xy"] = [o["xy"][0] + self.bias[0], o["xy"][1] + self.bias[1]]

    def find_boxes(self, image):
        for role, spec in self.fixed_boxes.items():
            half = np.array(spec.get("size", [0.20, 0.20])) / 2
            c = np.array(spec["xy"], float)
            corners = c + np.array([[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]])
            self.boxes[role] = {"centre": c, "corners": corners, "rim_z": float(spec["rim_z"]), "fixed": True,
                                "drop_q": spec.get("drop_q")}
        for role, seed in self.box_seeds.items():
            box, new_seed = ppa.detect_box(image, self.calib, seed, outer=self.box_outer[role])
            if box is not None:
                if self.a.sim and any(self.bias):
                    box["corners"] = box["corners"] + self.bias
                    box["centre"] = box["centre"] + self.bias
                self.boxes[role] = box
                self.box_seeds[role] = new_seed
        if "keep" not in self.boxes:
            raise RuntimeError("keep box not found")

    def container_of(self, xy):
        for role, box in self.boxes.items():
            if box.get("fixed"):
                continue  # beside the desk: nothing on the desk is "in" it
            corners = box["corners"]
            centre = corners.mean(0)
            inner = centre + (corners - centre) * 0.85  # inside the walls
            if cv2.pointPolygonTest(inner.astype(np.float32), (float(xy[0]), float(xy[1])), False) >= 0:
                return role
        return None

    # --- human approval -----------------------------------------------
    def true_choice(self, o):
        """Simulation: what a person would answer for o (discard / keep / leave), from the world's truth."""
        if o.get("xy") is None:
            return None
        world = sim_world.load(self.a.sim)
        near = min(world["objects"], key=lambda w: math.dist(w["centre"], o["xy"]), default=None)
        if near is None or math.dist(near["centre"], o["xy"]) > 0.06:
            return None
        if near.get("kind") in sim_scenarios.PRECISION:
            return "leave"
        return "discard" if near.get("category") == "trash" else "keep"

    def ask(self, round_no, items, image):
        """{key: discard|keep|leave}. Trash, unsure and low-confidence items go to the human."""
        img = cv2.imread(str(image))
        decisions, asked = {}, []
        for key, o in items:
            suggest = {"trash": "discard", "keep": "keep"}.get(o["category"], "leave")
            # Obvious cases go ahead without asking (the page still shows them and can change them until
            # the robot gets there): kept things to the keep box, clear trash (confidence >= --auto-discard)
            # to the trash bin. Unsure or low-confidence ones wait for the human.
            if o["category"] == "keep":
                preset = o["confidence"] >= 0.6
            elif o["category"] == "trash":
                preset = o["confidence"] >= self.a.auto_discard and not misjudged_size(o)
                if o["confidence"] >= self.a.auto_discard and not preset:
                    print(f"    {o['name']}: 見た目の高さ {o['judged_height_cm']:g}cm と深度の高さ "
                          f"{o['size_cm'][2]:g}cm が大きく違うので、捨てる前に人に確かめる", flush=True)
            else:
                preset = False
            if preset:
                decisions[key] = suggest  # shown on the page too, and can still be changed there
            x0, y0, x1, y1 = o["bbox"]
            pad = 30
            crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
            if crop.size:
                scale = 240 / max(crop.shape[:2])
                cv2.imwrite(str(self.approval_dir / f"{key}.jpg"), cv2.resize(crop, None, fx=scale, fy=scale))
            asked.append({"key": key, "name": o["name"], "category_ja": CATEGORY_JA[o["category"]],
                          "confidence": float(o["confidence"]), "reason": o["category_reason"],
                          "image": f"{key}.jpg", "suggest": suggest, "preset": preset})
        if not any(not item["preset"] for item in asked):
            self.publish(round_no, asked, decisions)
            return decisions
        if self.a.approve in ("yes", "no"):  # simulation: answer as the suggestion, or refuse discards
            where = dict(items)
            for item in asked:
                if item["preset"]:
                    continue
                answer = item["suggest"]
                if self.a.approve == "yes" and self.a.sim:
                    # A person at the page knows what it is (2026-10-28: "unsure" paper was left on the
                    # desk because the simulated person echoed the robot's "leave", and a honey dipper the
                    # judge called a cotton swab at 0.72 was discarded because they echoed "discard").
                    answer = self.true_choice(where[item["key"]]) or answer
                decisions[item["key"]] = answer if self.a.approve == "yes" or answer != "discard" else "leave"
            return decisions
        self.publish(round_no, asked, decisions)
        waiting = [item for item in asked if not item["preset"]]
        print(f"[確認] {len(waiting)} 件を確認ページで待っています（答えを待つ間に、ほかの物を先に片付ける）: "
              f"http://{self.a.host}:{self.a.port}/", flush=True)
        # Not blocking: the run loop takes the decided things first and waits only when nothing else is
        # left (the user, 2026-11-01: the arm stood still while two of eight waited for an answer).
        self.answer_deadline = time.monotonic() + self.a.approve_timeout
        for item in waiting:
            decisions[item["key"]] = PENDING
        return decisions

    def wait_answer(self, key):
        """Block until the page answers key (or the approval timeout): its choice, else leave it alone."""
        while time.monotonic() < self.answer_deadline:
            choice = self.latest_choice(key, PENDING)
            if choice != PENDING:
                return choice
            time.sleep(1.0)
        return "leave"

    def publish(self, round_no, items, decisions):
        """Show every candidate on the page; preset ones already carry their default choice."""
        if self.a.approve != "web":
            return
        response = self.approval_dir / "response.json"
        self.page_items = list(items) + list(getattr(self, "page_extra", []))
        (self.approval_dir / "request.json").write_text(
            json.dumps({"round": round_no, "items": self.page_items}, ensure_ascii=False))
        response.write_text(json.dumps({"decisions": {i["key"]: decisions[i["key"]] for i in items if i["preset"]}},
                                       ensure_ascii=False))

    def page_others(self, objects, todo, image):
        """Page cards for everything else on the desk: what a person must take, and what is already in a box."""
        img = cv2.imread(str(image))
        todo_keys = {id(o) for _, o in todo}
        cards = []
        for k, o in enumerate(objects, 1):
            if id(o) in todo_keys or (o.get("xy") is not None and np.hypot(*o["xy"]) < ROBOT_RADIUS):
                continue
            if o["category"] == "fixture":
                continue  # the robot itself
            key = f"x{k}"
            x0, y0, x1, y1 = o["bbox"]
            crop = img[max(0, y0 - 30):y1 + 30, max(0, x0 - 30):x1 + 30]
            if crop.size:
                scale = 240 / max(crop.shape[:2])
                cv2.imwrite(str(self.approval_dir / f"{key}.jpg"), cv2.resize(crop, None, fx=scale, fy=scale))
            info = ("人の手で片付けてください：" + o["grasp_reason"]) if not o["should_grasp"] else "今回は対象外"
            cards.append({"key": key, "name": o["name"], "category_ja": CATEGORY_JA[o["category"]],
                          "confidence": float(o["confidence"]), "reason": o["category_reason"],
                          "image": f"{key}.jpg", "suggest": "leave", "preset": True, "info": info})
        return cards

    def show_result(self, key, text):
        """Mark a card on the page as done (no more buttons)."""
        path = self.approval_dir / "request.json"
        if self.a.approve != "web" or not path.exists():
            return
        req = json.loads(path.read_text())
        for item in req["items"]:
            if item["key"] == key:
                item["result"] = text
        path.write_text(json.dumps(req, ensure_ascii=False))

    def latest_choice(self, key, default):
        """The page's current answer for key (it can be changed until the robot gets to it)."""
        response = self.approval_dir / "response.json"
        if self.a.approve == "web" and response.exists():
            return json.loads(response.read_text())["decisions"].get(key, default)
        return default

    # --- motion -------------------------------------------------------
    def send(self, command, prefixes, timeout=180):
        return self.client.send(command, prefixes, timeout)

    def release_open(self):
        """Opening to let go or back off after a miss: what the last grasp opened to, not all the way
        (the user, 2026-11-01: "そんな全力で開く必要あるか" after the fingers were driven past their stop)."""
        return float(min(self.a.grip_open_max, max(OPEN_MIN, getattr(self, "last_opening", None) or RELEASE_OPEN)))

    def say(self, text):
        """Print a progress line; in a recorded simulation also mark it on the video timeline (captions)."""
        print(text, flush=True)
        if self.a.physics and self.a.record and self.client is not None:
            self.send("mark " + " ".join(text.split()), ("記録",), 30)

    def tip_z(self):
        return topdown_ik.fk(ppa.arm_goto_last_q(self.a.out)).translation[2]

    def drop_pose(self, box):
        """Release pose above the box; never below DROP_MIN_Z (the arm stays above the desk edge).

        For a bin beside the desk the whole path from the folded pose is also checked against the
        robot's own links and the desk plane (reach_map.Body).
        """
        if box.get("drop_q") and self.a.taught_drops:  # taught on the real arm: use it as is
            return [float(v) for v in box["drop_q"]]
        bx, by = box["centre"]
        rim = sim_world_rim(box)
        body = drop_body() if box.get("fixed") else None
        # Every box the same way (the user, 2026-10-27): hand straight down over its centre, as high as
        # reachable between DROP_HIGH above the rim and DROP_LOW above it; tilted only if that fails.
        heights = [z for z in np.arange(rim + DROP_HIGH, rim + DROP_LOW - 1e-6, -0.01) if z >= DROP_FLOOR]
        for pitch in (0.0, 0.3, 0.5, 0.7):
            for z in heights:
                q, pos_err, _ = topdown_ik.solve([bx, by, z], yaw=math.atan2(by, bx), seeds=plan_pen_pick.SEEDS,
                                                 pitch=pitch)
                if pos_err > 1e-3 or not all(lo <= v <= hi for v, (lo, hi) in zip(q, self.limits)):
                    continue
                if body is not None:
                    path = arm_goto.plan([0.0] * 6, [float(v) for v in q], self.limits, self.a.max_speed)
                    if any(p for _, qq in path[::3] for p in body.collisions(qq) if p != "self:link2-link5"):
                        continue
                return [float(v) for v in q]
        if box.get("drop_q"):
            return [float(v) for v in box["drop_q"]]
        raise RuntimeError("box out of reach for the drop")

    def plan_pick(self, o, floor_z):
        """Grasp plan for o without moving: (None, plan) or (reason, None).

        reason is unreachable, too_wide or blocked; plan holds the first grasp
        g0, the opening, the sideways shifts and the sizes the probe needs.
        """
        c = np.array(o["xy"], float)
        if np.hypot(*c) > self.a.max_reach:
            return "too_far", None
        if o["size_cm"][2] < FLAT_CM and not o.get("deformable"):
            return "too_flat", None
        axis = np.array(o.get("axis_xy") or [1.0, 0.0], float)
        soft = bool(o.get("deformable"))
        if soft and rim_ring(o.get("_xyz"), floor_z + o["size_cm"][2] / 100):
            # A rim all round a lower middle is a bowl, whatever the word (sim 2026-10-28: a ramekin judged
            # a soft "ボール", ball or bowl, was pinched as cloth and the finger stopped on its rim 12 times).
            soft = False
            print("    深度で見ると縁のある器なので、硬い器としてつまむ", flush=True)
        judged = [v / 100 for v in o["size_cm"][:2]]
        judged = (max(judged), min(judged))
        # The image blob includes side faces and blur, so it overstates; trust the smaller of the two.
        # The depth camera's size puts every point where it really is (measure_3d): trust it alone
        # (sim 2026-10-28: a 10 cm mug judged 8 cm wide, and the fingers came down on its rim three times).
        if o.get("size_m") and o.get("size_from_depth"):
            size = tuple(o["size_m"])
        else:
            size = tuple(min(m, j) for m, j in zip(o["size_m"], judged)) if o.get("size_m") else judged
        if o.get("turn") == 90:  # the finger line turned a quarter: the long side now lies across it
            size = size[::-1]
        top_z = floor_z + float(np.clip(o["size_cm"][2] / 100, 0.005, TALLEST))
        tall_grip = top_z - floor_z > TALL
        if tall_grip:
            # Taller than the fingers: lowered to the desk the palm lands on its top and knocks it over
            # (sim 2026-10-28: a 10 cm box). Grip its upper part instead, above its centre of mass, so it
            # hangs straight; the probe line (and the hover) ends there.
            floor_z = top_z - TALL_GRIP
        max_across = sim_world.gap(self.a.grip_open_max) - 2 * CLEARANCE
        world = self.perceived_world(o, floor_z)

        def opening_for(across):
            # Soft things are squeezed, not reached around: open to part of them, not into the neighbours.
            # A tall thing is aimed at its top, where an error is largest and a finger landing on the cap
            # knocks it over (sim 2026-10-28: an 8 cm bottle): open wider the taller it is.
            margin = OPEN_MARGIN + OPEN_TALL_SHARE * (top_z - floor_z)
            span = SOFT_OPEN * size[1] + OPEN_MARGIN if soft else across + margin
            return float(np.clip(sim_world.rad_for_gap(span), OPEN_MIN, self.a.grip_open_max))

        def clear(g, opening=None):
            """No finger, rail or motor inside a known box wall or neighbour anywhere on the approach line.

            The orientation is fixed along the line, so the grasp-pose gripper is slid back toward the
            hover; obstacles are grown by the camera calibration error.
            """
            if opening is None:
                opening = opening_for(g["across"] if g.get("across") is not None else size[1])
            T = topdown_ik.fk(g["grasp"])
            pts = (T.rotation @ sim_world.gripper_points(opening).T).T + T.translation
            for d in np.linspace(0.0, 0.09, 10):
                hits = sim_world.collisions_at(world, pts - T.rotation[:, 0] * d, margin=PLAN_MARGIN)
                # The probe finds the desk / box floor itself; only walls and neighbours veto a grasp.
                if {h for h in hits if h != "desk" and not h.endswith(":floor")}:
                    return False
            return True

        # Soft things can be squeezed from any side, and off-centre away from walls if need be.
        fit = ({"max_align": math.pi / 2} if soft else {"size": size, "max_across": max_across}) | {"accept": clear}
        wide_ok = o.get("open_wide") and min(size) <= OPEN_WIDE_SLACK * max_across
        if (self.a.trust_fit or wide_ok) and not soft:
            # The human says it fits, or a lesson says to open fully: across the short side, fully open
            # (the camera overstates a tall or round thing's width, 2026-10-27: a 6.6 cm can read as too wide).
            fit = {"accept": clear}
        if soft and min(size) > SOFT_WIDE:
            # A folded towel or cloth wider than the fingers: squeezed in its middle the fingers only press
            # on its top (sim 2026-10-28, three times). Pinch its edge, one finger on the desk beside it.
            edge = self.plan_rim(o, floor_z, top_z, tall_grip, clear, inset=SOFT_EDGE_INSET)
            if edge is not None:
                return None, edge
        if o.get("rim_first") and not soft:  # the tactic says: pinch its wall at the edge, whatever it is
            rim = self.plan_rim(o, floor_z, top_z, tall_grip, clear)
            if rim is not None:
                return None, rim
            print("    縁をつまむ場所が見つからないので、ふつうに掴む", flush=True)
        g0 = None
        # A long rigid thing touching a neighbour at one end is taken nearer its other end (2026-11-01 real:
        # a marker against the Kimwipes box was "blocked" twice, centre only).
        along = np.asarray(axis, float) / (np.linalg.norm(axis) or 1.0)
        reach_along = size[0] / 2 - LONG_END_KEEP
        # Every 1.5 cm out to 2 cm short of either end (battle 2026-11-01: a 26 cm screwdriver whose tip lay by
        # a bottle was taken nowhere, the shifts stopping at 3 cm).
        long_shifts = [along * sign * d for d in np.arange(0.015, reach_along + 1e-9, 0.015) for sign in (1, -1)]
        soft_shifts = [np.array([math.cos(t), math.sin(t)]) * SOFT_SHIFT
                       for t in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
        tries = [(fit, [np.zeros(2)] + (soft_shifts if soft else long_shifts))]
        if soft and size[1] <= max_across:
            # A soft thing too is taken across its short side first: squeezed across its long side a sponge
            # was pushed away twice (battle 2026-11-01); across the short side it lifts even at force 1 (probe).
            tries.insert(0, ({"size": size, "max_across": max_across, "accept": clear}, [np.zeros(2)] + long_shifts))
        for try_fit, shifts in tries:
            for shift in shifts:
                try:
                    g = ppa.plan(c + shift, axis, self.limits, floor_z, **try_fit)
                    # The whole way down must be clear too, not only the grasp pose (a screwdriver's centre could
                    # be grasped, but the line down to it passed a bottle; its handle end was clear).
                    ppa.probe_steps(c + shift, axis, self.limits, floor_z, top_z, **try_fit)
                except RuntimeError:
                    continue
                g0, c, fit = g, c + shift, try_fit
                break
            if g0 is not None:
                break
        if g0 is None:
            try:
                ppa.plan(c, axis, self.limits, floor_z)
            except RuntimeError:
                return "unreachable", None
            try:
                ppa.plan(c, axis, self.limits, floor_z, **{k: v for k, v in fit.items() if k != "accept"})
            except RuntimeError:
                rim = None
                if not soft:
                    # A container is pinched by its wall; anything else by a narrow piece if it has one.
                    xyz = o.get("_xyz")
                    tries = (self.plan_rim, self.plan_part)
                    if xyz is not None and len(xyz) >= 30 and not hollow(xyz, top_z):
                        tries = tries[::-1]
                    elif (xyz is None or len(xyz) < 30) and not any(
                            w in f"{o.get('kind', '')}{o['name']}" for w in CONTAINER_WORDS):
                        # No depth to see a hollow: a rim pinch only for what the judge calls a container
                        # (2026-11-01 real: a 19 cm Kimwipes box was probed for a rim and was solid).
                        tries = (self.plan_part,)
                    for plan_wide in tries:
                        rim = plan_wide(o, floor_z, top_z, tall_grip, clear)
                        if rim is not None:
                            break
                return ("too_wide", None) if rim is None else (None, rim)
            return "blocked", None
        if not soft and "size" in fit and g0.get("align_err", 0.0) > ALIGN_GOOD:
            # The wrist could not square up with this side (joint 6 range): fingers on a box's corners turn
            # it as it comes off the desk and it slips out (sim 2026-10-28: 20 deg off, a block stayed on
            # the desk even at force 5). Across the other side, if that fits, they may line up.
            turned_axis, turned_size = np.array([-axis[1], axis[0]]), size[::-1]
            turned_fit = fit | {"size": turned_size}
            try:
                g1 = ppa.plan(c, turned_axis, self.limits, floor_z, **turned_fit)
            except RuntimeError:
                g1 = None
            if g1 is not None and g1["pitch"] <= g0["pitch"] and g1["align_err"] < g0["align_err"] - ALIGN_GAIN:
                print(f"    指が {math.degrees(g0['align_err']):.0f}° 斜めになるので、もう一方の辺を掴む"
                      f"（{math.degrees(g1['align_err']):.0f}°）", flush=True)
                axis, size, fit, g0 = turned_axis, turned_size, turned_fit, g1
        across = g0["across"] if g0.get("across") is not None else size[1]
        opening = self.a.grip_open_max if self.a.trust_fit or o.get("open_wide") else opening_for(across)
        room = (sim_world.gap(opening) - across) / 2  # clearance per side
        nudge = min(0.006, 0.6 * room)
        # A soft thing with straight sides (a sponge) is not squeezed in from its top: a finger opened to
        # part of it lands on its top. Opened round it instead, if that is clear of the neighbours.
        # Squared up with its sides like a rigid thing (single test 2026-10-28: across its short side a sponge
        # lifts at force 1; at a slant it turned out of the fingers).
        around = float(np.clip(sim_world.rad_for_gap(size[1] + OPEN_MARGIN), OPEN_MIN, self.a.grip_open_max))
        around_fit = {"size": size, "max_across": max_across, "accept": lambda g: clear(g, around)}
        if soft:
            try:
                ppa.plan(c, axis, self.limits, floor_z, **around_fit)
            except RuntimeError:
                around = None
        return None, {"c": c, "axis": axis, "soft": soft, "size": size, "top_z": top_z, "fit": fit, "g0": g0,
                      "floor_z": floor_z, "tall_grip": tall_grip,
                      "around": around if soft else None, "around_fit": around_fit,
                      "across": across, "opening": opening,
                      "nudges": (0.0,) if soft or nudge < 0.0015 else (0.0, nudge, -nudge)}

    def plan_rim(self, o, floor_z, top_z, tall_grip, clear, inset=RIM_INSET):
        """Fingers astride the wall of something too wide to reach around: a cup, bowl, plate or tape roll.

        The rim is the outline at the object's top; candidate pinch points go round it, nearest the
        robot first, each with the fingers closing across the rim (radially) a little inside the outer
        edge so the wall sits between them. A solid wide box has no wall: the inner finger then lands on
        its top and the probe reports it. None if no pinch point can be reached.
        """
        rim = o.get("_rim_xy")
        if rim is None and o.get("_pixels") is not None:  # no depth camera: the outline seen at the top height
            rim = desk_judge.pixels_to_base(self.calib, o["_pixels"], np.full(len(o["_pixels"]), top_z))
        if rim is None or len(rim) < 20:
            return None
        centre = rim.mean(0)
        d = rim - centre
        ang, rad = np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1])
        candidates = []
        for t in np.radians(np.arange(0, 360, RIM_STEP_DEG)):
            near = np.abs((ang - t + math.pi) % (2 * math.pi) - math.pi) < math.radians(RIM_STEP_DEG)
            if not near.any():
                continue
            n = np.array([math.cos(t), math.sin(t)])
            edge = centre + n * float(np.percentile(rad[near], 95))
            candidates.append((float(np.hypot(*edge)), edge - n * inset, np.array([-n[1], n[0]]),
                               math.dist(edge, centre)))
        # A mug's handle sticks out past the rim: pinching there lands a finger on the handle's top
        # (sim 2026-10-28: a cup pinched 7.7 cm out, where its rim is 4.8 cm out).
        usual = float(np.median([k[3] for k in candidates])) if candidates else 0.0
        candidates = [k for k in candidates if k[3] <= RIM_ROUND * usual]
        for _, c, tangent, _ in sorted(candidates, key=lambda k: k[0]):
            if np.hypot(*c) > self.a.max_reach:
                continue
            fit = {"size": (0.01, RIM_WALL), "max_across": RIM_GAP, "accept": clear}
            # Only the rim's top part: lower down a bowl's wall turns flat and the pinch slides off it
            # (sim 2026-10-28: a bowl pinched 2.6 cm up slipped three times).
            floor_z = max(floor_z, top_z - RIM_DEPTH)
            try:
                g0 = ppa.plan(c, tangent, self.limits, floor_z, **fit)
            except RuntimeError:
                continue
            print(f"    幅が広いので{'端' if o.get('deformable') else '縁'}をつまむ（中心から {math.dist(c, centre) * 100:.1f} cm）",
                  flush=True)
            return {"c": c, "axis": tangent, "soft": False, "size": (0.01, RIM_WALL), "top_z": top_z, "fit": fit,
                    "g0": g0, "floor_z": floor_z, "tall_grip": tall_grip, "rim": True, "across": RIM_WALL,
                    "opening": float(sim_world.rad_for_gap(RIM_GAP)), "nudges": (0.0,)}
        return None

    def plan_part(self, o, floor_z, top_z, tall_grip, clear):
        """Fingers around a narrow piece of something too wide to grip whole (a figure's fin, a bell's
        handle, a toy's arm), found in the depth camera's points. None if there is none or no depth."""
        xyz = o.get("_xyz")
        if xyz is None or len(xyz) < 30:
            return None
        max_across = sim_world.gap(self.a.grip_open_max) - 2 * CLEARANCE
        low = max(floor_z + 0.005, top_z - TALL_GRIP)  # the palm stays above the object's top
        for g in part_grasps(xyz, top_z, low, max_across)[:12]:
            if np.hypot(*g["c"]) > self.a.max_reach:
                continue
            fit = {"size": (0.01, g["across"]), "max_across": g["gap"], "accept": clear}
            try:
                g0 = ppa.plan(g["c"], g["axis"], self.limits, g["z"] - ppa.TIP_REACH, **fit)
            except RuntimeError:
                continue
            print(f"    幅が広いので、指の間に入る細い部分をつまむ（幅 {g['across'] * 100:.1f} cm、"
                  f"上から {(top_z - g['z']) * 100:.1f} cm）", flush=True)
            return {"c": g["c"], "axis": g["axis"], "soft": False, "size": (0.01, g["across"]), "top_z": top_z,
                    "fit": fit, "g0": g0, "floor_z": g["z"] - ppa.TIP_REACH, "tall_grip": tall_grip, "rim": True,
                    "part": True, "across": g["across"], "opening": float(sim_world.rad_for_gap(g["gap"])),
                    "nudges": (0.0,)}
        return None

    def pick(self, o, floor_z, force, log):
        """Probe down around the object and close.

        Returns 'held', 'missed', 'stalled', 'too_wide', 'blocked' or 'unreachable'.
        """
        fmt = lambda q: " ".join(f"{v:.4f}" for v in q)
        self.hang = 0.0
        self.held_reach = held_reach(o)
        stands_on = floor_z
        reason, plan = self.plan_pick(o, floor_z)
        self.rim_tactic = bool(plan and plan.get("rim") and not plan.get("part"))
        if reason:
            return reason
        c, axis, soft, top_z, fit, g0 = (plan[k] for k in ("c", "axis", "soft", "top_z", "fit", "g0"))
        floor_z = plan["floor_z"]  # a tall object's grip height, not the desk
        opening, nudges = plan["opening"], plan["nudges"]
        seated = None
        widened = False
        self.tip_z_at_seat = None
        if o.get("_widened_retry"):
            opening = float(min(self.a.grip_open_max, sim_world.rad_for_gap(sim_world.gap(opening) + 2 * WIDEN_ON_TOP)))
            widened, nudges = True, (0.0,)
        if soft and plan.get("around") and plan["around"] > opening + 0.05:
            # Round a soft thing from the start when that is clear of the neighbours: opened to part of it
            # a finger landed on a crumpled tissue's top and the grasp was tried again (2026-11-01 real).
            opening, widened = plan["around"], True
            fit = plan["around_fit"]
        for pass_no in range(2):
            contacts = []
            for nudge in nudges:
                finger = np.array(topdown_ik.fk(g0["grasp"]).rotation[:2, 1])
                finger /= np.linalg.norm(finger)
                poses = None
                # From over its highest point; a peak far above its top is mostly a neighbour's edge in its
                # outline (battle 2026-11-01: a screwdriver beside a 20 cm bottle "out of reach"): then its top.
                for start_z in dict.fromkeys((max(top_z, o.get("peak_z") or 0.0), top_z)):
                    try:
                        poses, steps = ppa.probe_steps(c + finger * nudge, axis, self.limits, floor_z, start_z, **fit)
                        break
                    except RuntimeError as error:
                        why = error
                if poses is None:
                    print(f"    下ろす道が作れない: {why}", flush=True)
                    return "unreachable"
                self.above_then(poses["hover"])  # straight over the object first, then down
                self.send(f"grip open {opening:.3f}", ("グリッパ",), 30)
                self.last_opening = opening
                steps = self.coarse_approach(poses, steps) + steps
                # A thin thing on the desk: straight down to just over the desk, no contact sensing (the user,
                # 2026-11-01: "机の高さまで下げて普通につかんだらいいでしょ" after a pen 43 cm out was grasped at
                # 17 mm on a false contact, the stretched arm lagging its track). Nothing here feels for contact: it trusts the height, so a thing judged thin wrongly is pressed on until the tracking guard stops the arm.
                direct = (floor_z == 0.0 and top_z - floor_z < THIN_DIRECT and not plan.get("rim")
                          and not plan["tall_grip"])
                if direct:
                    stop = floor_z + ppa.TIP_REACH + DIRECT_ABOVE
                    steps = [st for st in steps if st[0] >= stop] or steps[:1]
                i, z, baseline, touched = ppa.probe(
                    steps, lambda q: self.send(f"step {fmt(q)}", ("到達",)), self.tip_z, lag=1.0 if direct else None)
                expected = floor_z + ppa.TIP_REACH
                log.append({"shift_mm": nudge * 1000, "pitch_deg": round(math.degrees(poses["pitch"])),
                            "opening": round(opening, 3), "stop_tip_z": round(z, 4), "contact": touched})
                print(f"    下げて {'接触' if touched else '接触なし'}: 手先 {z * 1000:.0f} mm"
                      f"（床なら {expected * 1000:.0f} mm）ずらし {nudge * 1000:+.0f} mm", flush=True)
                # A finger on top of a thin object stops only its height above the floor.
                # 2026-10-28: a finger on a pen's flank stopped at 14 mm (floor 5 mm) and grasping there was right;
                # a finger on top of an object stops near its full height, so only that counts.
                on_object = min(ON_TOP_MAX, max(0.003, ON_TOP_SHARE * (top_z - floor_z)))
                # A tall object stops the rail on its top while the fingers are around its sides.
                tall = (top_z - floor_z) > TALL
                rim = plan.get("rim", False)  # the inner finger may stop on a bowl's curved inside
                # Gripping a tall thing's upper part, a finger meeting it below its top is on a widening side
                # (sim 2026-10-28: a ring-stacker toy), not on its top: close there.
                side = plan["tall_grip"] and z < top_z - TALL_TOP_BAND
                if (soft and touched and not widened and plan.get("around") and plan["around"] > opening + 0.05
                        and z > expected + SOFT_ON_TOP * (top_z - floor_z)):
                    # sim 2026-10-28: a sponge 7 cm wide, fingers opened to 6 cm, closed on its top three times
                    print("    指が柔らかい物の上に乗ったので、外側まで開いて掴み直す", flush=True)
                    self.retreat(steps, i, poses)
                    opening, fit, widened = plan["around"], plan["around_fit"], True
                    break
                xyz = o.get("_xyz")
                if (rim and not soft and touched and z > top_z + ppa.TIP_REACH - RIM_NO_WALL
                        and not (xyz is not None and len(xyz) >= 30 and hollow(xyz, top_z))):
                    # A finger meant to go inside stopped on the top of a thing the depth camera sees no
                    # hollow in: solid all across, no wall to pinch (sim 2026-10-29: an 11 cm puck pinched
                    # three times, each closing on nothing). A seen hollow means the finger hit the rim.
                    print("    縁の内側に指が入らない（中まで詰まっている）", flush=True)
                    self.retreat(steps, i, poses)
                    return "too_wide"
                if soft or tall or rim or side or not touched or z <= expected + on_object:
                    if touched and not soft and not tall:
                        i = ppa.back_off(steps, z, baseline)
                        self.send(f"step {fmt(steps[i][1])}", ("到達",))
                    elif soft and not rim and top_z - floor_z >= SOFT_GRIP_MIN_H:
                        # Squeezed at its foot a sponge bulges out from between the fingertips (battle 2026-11-01:
                        # twice lifted and left on the desk): up to part of its height first.
                        want = floor_z + ppa.TIP_REACH + SOFT_GRIP_SHARE * (top_z - floor_z)
                        k = min(range(i + 1), key=lambda n: abs(steps[n][0] + baseline - want))
                        if k < i:
                            i = k
                            self.send(f"step {fmt(steps[i][1])}", ("到達",))
                    seated, line = poses, (steps, i)
                    self.tip_z_at_seat = z
                    break
                contacts.append(z)
                self.retreat(steps, i, poses)
            if widened and seated is None and pass_no == 0:
                continue
            if (seated is not None or soft or max(contacts) - min(contacts) > 0.005 or floor_z == 0.0
                    or plan["tall_grip"]):
                break  # the desk height is known; only an unknown box floor is re-estimated
            on_top = abs(min(contacts) - (floor_z + ppa.TIP_REACH + (top_z - floor_z))) < 0.004
            # A judged height off by a few mm hid the same case (2026-11-01 real: a pen 43 cm out stopped all three
            # shifts at 17-18 mm, floor 5 mm, its judged top 25 mm): every shift stopping well above the floor
            # at one height is a finger on it as well.
            raised = min(contacts) > floor_z + ppa.TIP_REACH + ON_TOP_RAISED and max(contacts) - min(contacts) < 0.004
            if (on_top or raised) and not widened and pass_no == 0:
                wider = float(min(self.a.grip_open_max,
                                  sim_world.rad_for_gap(sim_world.gap(opening) + 2 * WIDEN_ON_TOP)))
                if wider > opening + 0.1:
                    print(f"    どこにずらしても机より高い所で止まるので、指を {2 * WIDEN_ON_TOP * 100:.1f} cm 広げて下ろし直す",
                          flush=True)
                    opening, widened = wider, True
                    nudges = (0.0,)
                    continue
            if on_top:
                # Every shift stopped on the object's own top: not a higher floor, but a finger landing on
                # it. Far out the camera's error passes the fingers' slack (2026-11-01 real: a pen 41 cm
                # out, three tries of three shifts each at its top height): once more, opened wider.
                wider = float(min(self.a.grip_open_max,
                                  sim_world.rad_for_gap(sim_world.gap(opening) + 2 * WIDEN_ON_TOP)))
                if not widened and pass_no == 0 and wider > opening + 0.1:
                    print(f"    どこにずらしても物の上で止まるので、指を {2 * WIDEN_ON_TOP * 100:.0f} cm 広げて下ろし直す",
                          flush=True)
                    opening, widened = wider, True
                    nudges = (0.0,)
                    continue
                break
            floor_z = min(contacts) - ppa.TIP_REACH  # every shift stopped at one height: that is the floor
            top_z = max(top_z, floor_z + 0.005)
            print(f"    どこも同じ高さで止まるので床を {floor_z * 1000:.0f} mm と推定し直し", flush=True)
        if seated is None:
            return "stalled"
        # Jump the fingers to just outside the object, then step (2026-11-01: stepping 0.05 rad per 0.3 s
        # from the fitted opening took 10-22 s a grasp on the real arm, 32 steps round a marker).
        lead = sim_world.rad_for_gap((SOFT_LEAD * plan["size"][1] if soft else plan["across"]) + CLOSE_LEAD)
        lead = f" {lead:.3f}" if OPEN_MIN / 2 < lead < opening - 0.1 and not plan.get("rim") else ""
        close = self.send(f"grip close {FORCE_SQUEEZE[force]:.3f}{lead}", ("グリッパ",), 120)
        self.last_close, self.last_squeeze = close, FORCE_SQUEEZE[force]
        self.hang = max(0.0, self.tip_z() - stands_on)
        self.retreat(*line, seated, held="closed_empty" not in close)
        # Crumpled paper squashes without stopping the fingers (2026-10-28: closed to 0.05 rad around a
        # sheet), so for soft things the camera, not the finger angle, decides whether it was taken.
        self.soft_hold = soft
        if "closed_empty" in close and not soft:
            self.send(f"grip open {opening:.3f}", ("グリッパ",), 30)
            stop_z = self.tip_z_at_seat if getattr(self, "tip_z_at_seat", None) is not None else 0.0
            if not o.get("_widened_retry") and stop_z > floor_z + ppa.TIP_REACH + ON_TOP_RAISED:
                # It closed on nothing after stopping well above the desk: a finger had come down on top of it
                # (2026-11-01 real: a pen 43 cm out, every shift stopped at 17-18 mm and the fingers closed above
                # it). Once more at once, opened wider.
                print(f"    机から {(stop_z - floor_z - ppa.TIP_REACH) * 1000:.0f} mm 上で止まって空振りしたので、"
                      f"指を広げてもう一度下ろす", flush=True)
                return self.pick({**o, "_widened_retry": True, "_original": o.get("_original", o)}, stands_on, force, log)
            return "missed"
        return "held"

    def coarse_approach(self, poses, steps):
        """Probe-line poses from the hover down to the fine probe, every COARSE_STEP.

        A neighbour taller than the target then stops the fingers with a small lag
        (seen as contact) instead of a long move that trips the tracking watchdog.
        """
        hover = topdown_ik.fk(poses["hover"])
        start = np.array([*topdown_ik.fk(steps[0][1]).translation])
        top = np.array(hover.translation)
        n = int(np.linalg.norm(top - start) // COARSE_STEP)
        out, prev = [], poses["hover"]
        for k in range(1, n + 1):
            point = top + (start - top) * k * COARSE_STEP / np.linalg.norm(top - start)
            q, pos_err, ang_err = desk_ik(point, hover.rotation, prev)
            if pos_err > 1e-3 or ang_err > 1e-2 or not all(lo <= v <= hi for v, (lo, hi) in zip(q, self.limits)):
                break
            out.append((float(point[2]), [float(v) for v in q]))
            prev = q
        return out

    def perceived_world(self, target, floor_z):
        """sim_world-shaped obstacles from what the camera and the judge saw (target excluded)."""
        boxes = []
        for role, box in self.boxes.items():
            if box.get("fixed"):
                continue  # beside the desk, never in the way of a grasp on it
            corners = box["corners"]
            d = corners[1] - corners[0]
            boxes.append({"name": f"{role}_box", "centre": corners.mean(0).tolist(), "yaw": math.atan2(d[1], d[0]),
                          "outer": [float(np.linalg.norm(corners[1] - corners[0])),
                                    float(np.linalg.norm(corners[2] - corners[1]))],
                          "height": box.get("rim_z", box_fetch_rim()), "wall": 0.008, "floor": 0.008})
        objects = []
        for k, x in self.round_items + self.standing:
            # Only what went into a box is gone: a thing that failed is still on the desk (sim 2026-10-29: a
            # mug knocked over by two failed grasps was no obstacle, and the fingers hit it on the next pick).
            if (x is target or x is (target or {}).get("_original") or x.get("xy") is None
                    or str(self.finished.get(k, "")).startswith("moved_to_")):
                continue
            if np.hypot(*x["xy"]) < ROBOT_RADIUS or self.container_of(x["xy"]) or self.box_like(x):
                continue  # the robot itself, things inside a box, and the boxes (modelled as open boxes above)
            if x["category"] == "fixture" and x["size_cm"][2] < 1.0:
                continue  # sheets (the calibration checkerboard): fingers may rest on them
            judged = sorted((x["size_cm"][0] / 100, x["size_cm"][1] / 100), reverse=True)
            length, width = ([min(m, j) for m, j in zip(x["size_m"], judged)] if x.get("size_m") else judged)
            ax = x.get("axis_xy") or [1.0, 0.0]
            if not x.get("should_grasp", True) and x["category"] != "fixture":
                # A precision device next to the target: the fingers keep well off it, not just off it.
                length, width = length + 2 * UNTOUCHABLE_CLEAR, width + 2 * UNTOUCHABLE_CLEAR
            objects.append({"id": k, "centre": list(x["xy"]), "yaw": math.atan2(ax[1], ax[0]),
                            "size": [length, width, max(0.005, x["size_cm"][2] / 100)], "z": 0.0,
                            "soft": bool(x.get("deformable"))})
        return {"boxes": boxes, "fixtures": [], "objects": objects}

    def box_like(self, x):
        """A judged object that is one of the detected boxes."""
        return any(cv2.pointPolygonTest(b["corners"].astype(np.float32), (float(x["xy"][0]), float(x["xy"][1])), False)
                   >= 0 for b in self.boxes.values())

    def retreat(self, steps, i, poses, held=False):
        """Leave along the probe line (straight up the approach), not a joint-space arc that can dip."""
        fmt = lambda q: " ".join(f"{v:.4f}" for v in q)

        def back_up(distance):
            q_now = ppa.arm_goto_last_q(self.a.out)
            T = topdown_ik.fk(q_now)
            q_up, pos_err, ang_err = desk_ik(T.translation - T.rotation[:, 0] * distance, T.rotation, q_now)
            if pos_err < 1e-3 and ang_err < 1e-2 and all(lo <= v <= hi for v, (lo, hi) in zip(q_up, self.limits)):
                self.send(f"step {fmt(q_up)}", ("到達",))

        if held:
            # Coming off the desk a box held a few degrees off turns square between the fingers and the
            # squeeze is gone (sim 2026-10-28: a 180 g block stayed on the desk at force 5). A few mm up,
            # squeeze again from where the fingers are now.
            back_up(RESQUEEZE_LIFT)
            target = float(self.last_close.split("目標")[1].split("rad")[0])
            reply = self.send(f"grip resqueeze {target:.3f} {self.last_squeeze:.3f}", ("グリッパ",), 30)
            if "resqueezed" in reply:
                print("    持ち上げ始めに指が進んだので握り直した", flush=True)
            self.last_close = reply
        # First straight back up from where the arm actually stopped (it may sit a little off the line).
        back_up(LIFT_FIRST)
        for k in list(range(i - 4, 0, -4)) + [0]:
            self.send(f"step {fmt(steps[k][1])}", ("到達",))
        self.send(f"goto {fmt(poses['hover'])}", ("到達",))

    def go_to_drop(self, dest):
        """From the lifted grasp straight to the box's release pose (via the folded pose only if needed)."""
        fmt = lambda q: " ".join(f"{v:.4f}" for v in q)
        q_drop = self.drop_pose(self.boxes[dest])
        self.at_box = dest
        # A tall thing gripped near its top hangs far below the fingers: its bottom must clear the rim
        # (sim 2026-10-28: a 16 cm bell held 10 cm up hit the keep box wall and stopped the arm).
        need = sim_world_rim(self.boxes[dest]) + self.hang + HANG_CLEAR
        high_needed = self.hang > 0.02 and topdown_ik.fk(q_drop).translation[2] < need
        if high_needed:
            print(f"    持った物が指の下に {self.hang * 100:.0f} cm 下がるので、箱の縁より高い所で離す", flush=True)
        # Each next thing goes to another spot in the box: let go at one point, a jar stood on a jar and
        # stuck out of the keep box (sim 2026-10-29).
        z = max(topdown_ik.fk(q_drop).translation[2], need if high_needed else 0.0)
        spots = (release_spots(self.boxes[dest], self.released.get(dest, 0), self.held_reach)
                 + ([None] if high_needed else []))
        for xy in spots:  # None: over the middle
            spot = self.high_release(self.boxes[dest], z, at=xy)
            if spot is not None:
                q_drop = spot
                break
        q_now = ppa.arm_goto_last_q(self.a.out)
        # First straight up to CARRY_Z, so the carry passes high above everything on the desk.
        carry_z = max(CARRY_Z, self.hang + CARRY_OVER)
        if topdown_ik.fk(q_now).translation[2] < carry_z:
            q_up = self.lift(q_now, carry_z)
            if q_up is not None:
                self.send(f"step {fmt(q_up)}", ("到達",))
                q_now = q_up
        above = self.raised(q_drop)
        # Something slipping out on the way must not land on a thing left for a person (sim 2026-10-28: a
        # figure dropped on the carry pushed a mouse 1.5 cm). Over one, go round by the folded pose.
        fragile = [x for _, x in self.standing if not x.get("should_grasp", True) and x.get("xy") is not None]
        target = above or q_drop
        under = self.passes_over(q_now, target, fragile)
        if under:
            # Pulled in while turning toward the box, then out over it: the fingertips stay near the base.
            # Folding all the way and back cost ~13 s a carry on the real arm (2026-11-01).
            for via in ([target[0], 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6):
                if not self.passes_over(q_now, via, fragile) and not self.passes_over(via, target, fragile):
                    print(f"    運ぶ道の下に {'、'.join(under)} があるので、腕を引いて回り込む", flush=True)
                    self.send(f"goto {fmt(via)}", ("到達",))
                    q_now = via
                    break
        # Lifted to CARRY_Z the carry passes over most of the desk (the user, 2026-10-28): straight to the box.
        # But a thing hanging 13 cm under the fingers swept three 19 cm bottles off the desk (battle
        # 2026-11-01): when the path meets something, lift higher, else pull in and go round, else go on.
        hang = max(HELD_BELOW, self.hang)
        if not self.path_clear(q_now, target, hang):
            first = list(self.blocked_by)
            tallest = max([x["size_cm"][2] / 100 for _, x in self.round_items + self.standing
                           if x["name"] in first] or [0.0])
            q_up = self.lift(q_now, tallest + hang + CARRY_CLEAR)
            ways = ([("持ち上げて", [q_up])] if q_up is not None else []) + [
                ("腕を引いて回り込んで", [[target[0], 0.0, 0.0, 0.0, 0.0, 0.0]]), ("腕をたたんで回り込んで", [[0.0] * 6])]
            for how, vias in ways:
                legs = [q_now] + vias + [target]
                if all(self.path_clear(a, b, hang) for a, b in zip(legs, legs[1:])):
                    print(f"    運ぶ道で {'、'.join(first)} に当たるので、{how}運ぶ", flush=True)
                    for v in vias:
                        self.send(f"{'step' if how == '持ち上げて' else 'goto'} {fmt(v)}", ("到達",))
                    q_now = vias[-1]
                    break
            else:
                print(f"    （経路チェック: {'、'.join(first)} の近くを通る。避ける道がないのでそのまま進む）", flush=True)
        if above is not None:  # over the box high, then straight down to the release pose
            self.send(f"goto {fmt(above)}", ("到達",))
            self.send(f"step {fmt(q_drop)}", ("到達",))
        else:
            self.send(f"goto {fmt(q_drop)}", ("到達",))

    def passes_over(self, q_from, q_to, things):
        """Names of `things` the fingertips pass within PASS_OVER of (seen from above) on the joint path."""
        near = []
        for _, q in arm_goto.plan(list(q_from), list(q_to), self.limits, self.a.max_speed)[::3]:
            tip = topdown_ik.fk(q).translation[:2]
            near += [x["name"] for x in things if math.dist(tip, x["xy"]) < PASS_OVER and x["name"] not in near]
        return near

    def lift(self, q, height):
        """Straight up from q toward `height`, as high as reachable; the hand may lean outward a little.

        Pointing straight down the arm reaches only 9-16 cm over most of the desk (wrist limit,
        2026-10-28), so a carry stayed that low and a held tall thing swept the desk (sim: a bell hanging
        9 cm under the fingers pushed four toys across it). Leaning 9-26 deg reaches 21-24 cm.
        """
        T = topdown_ik.fk(q)
        r = np.r_[T.translation[:2], 0.0]
        side = np.cross([0.0, 0.0, 1.0], r / (np.linalg.norm(r) or 1.0))  # horizontal, across the reach
        for z in np.arange(height, T.translation[2] + 0.02 - 1e-6, -0.02):
            for lean in (0.0, 0.15, -0.15, 0.3, -0.3, 0.45, -0.45):
                K = np.array([[0, -side[2], side[1]], [side[2], 0, -side[0]], [-side[1], side[0], 0]])
                R = np.eye(3) + math.sin(lean) * K + (1 - math.cos(lean)) * K @ K  # Rodrigues about `side`
                up, pos_err, ang_err = desk_ik(np.r_[T.translation[:2], z], R @ T.rotation, q)
                if pos_err < 1e-3 and ang_err < 1e-2 and all(lo <= v <= hi for v, (lo, hi) in zip(up, self.limits)):
                    return [float(v) for v in up]
        return None

    def high_release(self, box, need, at=None):
        """A release pose over the box (or over `at` in it) with the tip at least `need` high (tilted if
        needed), or None."""
        bx, by = box.get("centre", box.get("xy")) if at is None else at
        body = drop_body() if box.get("fixed") else None
        for pitch in (0.0, 0.3, 0.5, 0.7):
            for z in np.arange(need, need + 0.08, 0.01):
                q, pos_err, _ = topdown_ik.solve([bx, by, z], yaw=math.atan2(by, bx), seeds=plan_pen_pick.SEEDS,
                                                 pitch=pitch)
                if pos_err > 1e-3 or not all(lo <= v <= hi for v, (lo, hi) in zip(q, self.limits)):
                    continue
                if body is not None and any(p for p in body.collisions(q) if p != "self:link2-link5"):
                    continue
                return [float(v) for v in q]
        return None

    def raised(self, q, height=None):
        """The same gripper pose lifted straight up toward `height` (default CARRY_Z).

        With the hand kept pointing the same way the arm reaches only ~15 cm near the base (wrist limit,
        2026-10-28), so the highest reachable height on the way is used. None if no higher pose exists.
        """
        height = CARRY_Z if height is None else height
        T = topdown_ik.fk(q)
        prev = q
        best = None
        for z in np.arange(T.translation[2] + 0.02, height + 1e-6, 0.02):
            up, pos_err, ang_err = desk_ik(np.r_[T.translation[:2], z], T.rotation, prev)
            if pos_err > 1e-3 or ang_err > 1e-2 or not all(lo <= v <= hi for v, (lo, hi) in zip(up, self.limits)):
                break
            best, prev = [float(v) for v in up], up
        return best

    def above_then(self, q):
        """Go high over the target pose first, then straight down to it (never sweep low across the desk)."""
        fmt = lambda v: " ".join(f"{x:.4f}" for x in v)
        above = self.raised(q, APPROACH_Z)
        if above is not None:
            self.send(f"goto {fmt(above)}", ("到達",))
            self.send(f"step {fmt(q)}", ("到達",))
        else:
            self.send(f"goto {fmt(q)}", ("到達",))

    def release_here(self):
        """Open over the box and rise straight up; the next object is approached from here."""
        fmt = lambda v: " ".join(f"{x:.4f}" for x in v)
        self.send(f"grip open {self.release_open():.3f}", ("グリッパ",), 30)
        if self.at_box is not None:
            self.released[self.at_box] = self.released.get(self.at_box, 0) + 1
        up = self.raised(ppa.arm_goto_last_q(self.a.out))
        if up is not None:
            self.send(f"step {fmt(up)}", ("到達",))

    def park(self):
        """Over the keep box, out of the overhead camera's view of the desk. False if no box is known yet.

        The folded pose hangs over the desk near the base (sim 2026-10-28: its fingers hid a bowl and a
        cup there; the judge called the bowl a 13 cm ball and the cup maybe full of liquid).
        """
        fmt = lambda v: " ".join(f"{x:.4f}" for x in v)
        box = self.boxes.get("keep") or next(iter(self.boxes.values()), None)
        if box is None:
            return False
        try:
            q_drop = self.drop_pose(box)
        except RuntimeError:
            return False
        try:
            q_now = ppa.arm_goto_last_q(self.a.out)
        except (IndexError, FileNotFoundError):
            q_now = [0.0] * 6
        if topdown_ik.fk(q_now).translation[2] < CARRY_Z:
            q_up = self.lift(q_now, CARRY_Z)
            if q_up is not None:
                self.send(f"step {fmt(q_up)}", ("到達",))
        # High over it, not at the release pose: what was put in may stand above the rim (sim 2026-10-29: a
        # jar let go onto another jar stood 10 cm out of the keep box and the parking arm pressed on it).
        q_park = self.high_release(box, PARK_Z) or self.raised(q_drop) or q_drop
        self.send(f"goto {fmt(q_park)}", ("到達",))
        return True

    def clear_view(self, o, may_park=True):
        """Straight up from where the arm is, then back toward the base: the arm is then nearer the base
        than the object, so the overhead camera (looking from the far end) sees the object again.

        may_park=False: when that needs the park (over the keep box) or the folded pose, stay up and
        return False instead (a held thing hovered over the keep box looks as if it were kept)."""
        fmt = lambda v: " ".join(f"{x:.4f}" for x in v)
        q = ppa.arm_goto_last_q(self.a.out)
        up = self.raised(q)
        if up is not None:
            self.send(f"step {fmt(up)}", ("到達",))
            q = up
        c = np.array(o["xy"], float)
        r = float(np.linalg.norm(c))
        back = c * max(0.0, r - BACK_OFF) / r
        target = None
        for pitch in (0.0, 0.3):
            qb, pos_err, _ = topdown_ik.solve([back[0], back[1], CARRY_Z], yaw=math.atan2(c[1], c[0]),
                                              seeds=[np.array(q)] + plan_pen_pick.SEEDS, pitch=pitch)
            if pos_err < 1e-3 and all(lo <= v <= hi for v, (lo, hi) in zip(qb, self.limits)):
                target = [float(v) for v in qb]
                break
        if target is None or r - BACK_OFF < ROBOT_RADIUS:
            if not may_park:
                return False
            if not self.park():  # the folded pose would hide an object this near the base
                self.send("home", ("到達",))
        else:
            self.send(f"goto {fmt(target)}", ("到達",))
        return True

    def path_clear(self, q_from, q_to, held_below=HELD_BELOW):
        """The direct joint path keeps the gripper out of every box, fixture and object the camera saw."""
        world = self.perceived_world(None, 0.0)
        for role, box in self.boxes.items():
            if not box.get("fixed"):
                continue
        for k, x in self.round_items:  # fixtures judged by Claude (tissue box, boards...) as solid blocks
            if x["category"] == "fixture" and x.get("xy") is not None and x["size_cm"][2] >= 1.0 \
                    and np.hypot(*x["xy"]) >= ROBOT_RADIUS and not self.box_like(x):
                ax = x.get("axis_xy") or [1.0, 0.0]
                length, width = (x.get("size_m") or [x["size_cm"][0] / 100, x["size_cm"][1] / 100])[:2]
                world["fixtures"].append({"name": k, "centre": list(x["xy"]), "yaw": math.atan2(ax[1], ax[0]),
                                          "size": [length, width, x["size_cm"][2] / 100]})
        body = drop_body()
        names = {k: x["name"] for k, x in self.round_items + self.standing}
        for _, q in arm_goto.plan(list(q_from), list(q_to), self.limits, self.a.max_speed)[::3]:
            hits = {h for h in sim_world.collisions(world, q, 0.5) if h != "desk"}
            if hits:
                self.blocked_by = sorted(names.get(h, h) for h in hits)
                return False
            # Every arm link, and a hanging object (a block reaching HELD_BELOW under the fingertips).
            links = np.vstack([pts[::4] for name, pts in body.posed(q).items() if name not in ("base_link", "link1")])
            T = topdown_ik.fk(q)
            held = T.translation + np.outer(np.linspace(0.0, held_below, 6), [0.0, 0.0, -1.0])
            hits = {h for h in sim_world.collisions_at(world, np.vstack([links, held]), margin=PLAN_MARGIN)
                    if not h.endswith(":floor")}
            if hits:
                self.blocked_by = sorted(names.get(h, h) for h in hits)
                return False
        return True

    def place(self, dest):
        fmt = lambda q: " ".join(f"{v:.4f}" for v in q)
        self.send(f"goto {fmt(self.drop_pose(self.boxes[dest]))}", ("到達",))
        self.send(f"grip open {self.release_open():.3f}", ("グリッパ",), 30)
        self.send("home", ("到達",))

    def plan_sweep(self, o, drag=None):
        """Joint poses for dragging o toward the base with the closed fingers as a paddle, or None.

        The approach leans out along the radial line and the finger line is tangential, so the
        closed fingers (about 2 cm wide) sweep the object straight in, fingertips SWEEP_Z above the desk.
        drag: pull it in by this much only (out of a crowd), and only to a spot clear of the others.
        """
        c = np.array(o["xy"], float)
        radial = c / np.linalg.norm(c)
        judged = [v / 100 for v in o["size_cm"][:2]]
        half = max(judged) / 2 + SWEEP_BEHIND
        start = c + radial * half
        end = radial * (SWEEP_TO + half) if drag is None else start - radial * drag
        if drag is not None:
            moved = c - radial * drag
            if np.linalg.norm(moved) < ROBOT_RADIUS + max(judged) / 2:
                return None
            for k, x in self.round_items + self.standing:
                if (x is o or x.get("xy") is None or str(self.finished.get(k, "")).startswith("moved_to_")
                        or self.container_of(x["xy"])):
                    continue
                reach = (max(judged) + max(x["size_cm"][:2]) / 100) / 2 - DRAG_OVERLAP
                if math.dist(moved, x["xy"]) < reach:
                    return None
        yaw = math.atan2(radial[1], radial[0])
        world = self.perceived_world(o, 0.0)
        for pitch in (0.7, 0.87, 1.05):
            rot = (topdown_ik.pin.utils.rotate("z", yaw) @ topdown_ik.pitched(pitch))
            n = max(1, int(np.linalg.norm(start - end) // SWEEP_STEP))
            points = [np.r_[start, SWEEP_Z + 0.06]] + [np.r_[start + (end - start) * k / n, SWEEP_Z]
                                                       for k in range(n + 1)] + [np.r_[end, SWEEP_Z + 0.06]]
            qs, prev = [], None
            for point in points:
                q, pos_err, ang_err = desk_ik(point, rot, prev if prev is not None else
                                              np.array([-yaw, 2.0, 1.6, -0.5, 0.0, 0.0]))
                if (pos_err > 1e-3 or ang_err > 1e-2 or not all(lo <= v <= hi for v, (lo, hi) in zip(q, self.limits))
                        or (prev is not None and max(abs(a - b) for a, b in zip(q, prev)) > 0.4)):
                    break
                hits = {h for h in sim_world.collisions(world, q, 0.0) if h != "desk" and not h.endswith(":floor")}
                if hits:
                    break
                qs.append([float(v) for v in q])
                prev = q
            else:
                return {"pitch": pitch, "q": qs, "from": float(np.linalg.norm(c)),
                        "to": SWEEP_TO if drag is None else float(np.linalg.norm(c) - drag)}
        return None

    def plan_drag(self, o, d, drag):
        """Joint poses pushing o by `drag` along unit d with the closed fingers (finger line across d), or None
        if the arm cannot, or o would end up against another thing."""
        c = np.array(o["xy"], float)
        judged = [v / 100 for v in o["size_cm"][:2]]
        moved = c + d * drag
        if np.linalg.norm(moved) < ROBOT_RADIUS + max(judged) / 2 or np.linalg.norm(moved) > self.a.max_reach:
            self.drag_why.append("行き先が届く範囲の外")
            return None
        for k, x in self.round_items + self.standing:
            if (x is o or x.get("xy") is None or str(self.finished.get(k, "")).startswith("moved_to_")
                    or self.container_of(x["xy"])):
                continue
            if math.dist(moved, x["xy"]) < (max(judged) + max(x["size_cm"][:2]) / 100) / 2 - DRAG_OVERLAP:
                self.drag_why.append(f"行き先に{x['name']}")
                return None
        half = max(judged) / 2 + SWEEP_BEHIND
        start, end = c - d * half, c - d * half + d * drag
        world = self.perceived_world(o, 0.0)
        radial = c / np.linalg.norm(c)
        for yaw in (math.atan2(radial[1], radial[0]), math.atan2(d[1], d[0]) + math.pi / 2):
            for pitch in (0.7, 0.87, 1.05):
                rot = topdown_ik.pin.utils.rotate("z", yaw) @ topdown_ik.pitched(pitch)
                n = max(1, int(drag // SWEEP_STEP))
                points = ([np.r_[start, SWEEP_Z + 0.06]] + [np.r_[start + (end - start) * k / n, SWEEP_Z]
                                                             for k in range(n + 1)] + [np.r_[end, SWEEP_Z + 0.06]])
                qs, prev = [], None
                for point in points:
                    q, pos_err, ang_err = desk_ik(point, rot, prev if prev is not None else
                                                  np.array([math.atan2(c[1], c[0]), 2.0, 1.6, -0.5, 0.0, 0.0]))
                    if (pos_err > 1e-3 or ang_err > 1e-2
                            or not all(lo <= v <= hi for v, (lo, hi) in zip(q, self.limits))
                            or (prev is not None and max(abs(a - b) for a, b in zip(q, prev)) > 0.4)):
                        break
                    if {h for h in sim_world.collisions(world, q, 0.0) if h != "desk" and not h.endswith(":floor")}:
                        self.drag_why.append("腕がほかの物に当たる")
                        break
                    qs.append([float(v) for v in q])
                    prev = q
                else:
                    return {"q": qs}
        self.drag_why.append("腕の姿勢が作れない")
        return None

    def drag_clear(self, o):
        """Push a thing the fingers cannot come down beside out of its crowd (closed fingers as a paddle):
        in, out or to either side, to a spot clear of the others. True if it was moved. Battle 2026-11-01:
        "blocked" was the commonest failure in crowded desks."""
        fmt = lambda q: " ".join(f"{v:.4f}" for v in q)
        c = np.array(o["xy"], float)
        radial = c / np.linalg.norm(c)
        side = np.array([-radial[1], radial[0]])
        ways = [("手前に", -radial), ("奥へ", radial), ("左へ", side), ("右へ", -side)]
        self.drag_why = []
        for drag in DRAG_DISTANCES:
            for how, d in ways:
                plan = self.plan_drag(o, d, drag)
                if plan is not None:
                    break
            if plan is not None:
                break
        else:
            print(f"    ずらす道もない（{'、'.join(dict.fromkeys(self.drag_why)) or '理由不明'}）", flush=True)
            return False
        print(f"    周りが詰まっているので、閉じた指で {how}{drag * 100:.0f} cm ずらす", flush=True)
        self.send(f"grip open {self.release_open():.3f}", ("グリッパ",), 30)
        self.send(f"goto {fmt(plan['q'][0])}", ("到達",))
        self.send("grip close 0.02", ("グリッパ",), 120)
        for q in plan["q"][1:]:
            self.send(f"step {fmt(q)}", ("到達",))
        self.send(f"grip open {self.release_open():.3f}", ("グリッパ",), 30)
        return True

    def sweep_in(self, o, log_entry):
        fmt = lambda q: " ".join(f"{v:.4f}" for v in q)
        plan = self.plan_sweep(o)
        log_entry["sweep"] = None if plan is None else {k: plan[k] for k in ("pitch", "from", "to")}
        if plan is None:
            return "unreachable"
        print(f"  遠い（{plan['from'] * 100:.0f}cm）ので、閉じた指で{plan['to'] * 100:.0f}cm まで引き寄せる", flush=True)
        self.send(f"goto {fmt(plan['q'][0])}", ("到達",))
        self.send("grip close 0.02", ("グリッパ",), 120)  # empty: the fingers close fully into a paddle
        for q in plan["q"][1:]:
            self.send(f"step {fmt(q)}", ("到達",))
        return "swept"

    def put_back(self, xy, height):
        """Set a wrongly picked object down where it came from."""
        fmt = lambda q: " ".join(f"{v:.4f}" for v in q)
        for pitch in (0.0, 0.17, 0.35, 0.52):
            q, pos_err, _ = topdown_ik.solve([xy[0], xy[1], height + 0.02], yaw=math.atan2(xy[1], xy[0]),
                                             seeds=plan_pen_pick.SEEDS, pitch=pitch)
            if pos_err < 1e-3 and all(lo <= v <= hi for v, (lo, hi) in zip(q, self.limits)):
                self.send(f"goto {fmt(q)}", ("到達",))
                break
        self.send(f"grip open {self.release_open():.3f}", ("グリッパ",), 30)
        self.send("home", ("到達",))

    def tidy_one(self, key, o, dest, log_entry):
        """Pick o, check from the folded pose what actually left the desk, and place that.

        With the experience notebook the tactic for each attempt comes from what was learned before,
        and every failure is reflected on (Claude, or the offline rules) into a lesson and the next tactic.
        Returns moved_to_<box>, unreachable, too_wide, lost or failed.
        """
        source = self.container_of(o["xy"])
        floor_z = 0.0 if source is None else self.boxes[source].get("floor_z", 0.008)
        reach = float(np.hypot(*o["xy"]))
        self.current_reach = reach
        if source is None and self.a.max_reach < reach <= self.a.sweep_reach:
            # Too far to stay out there for a whole probe: drag it in quickly, pick it next round.
            result = self.sweep_in(o, log_entry)
            self.send("home", ("到達",))
            return result
        verdict = {k: o[k] for k in ("force", "deformable", "size_cm", "category", "should_grasp") if k in o}
        oid = self.remember(o) if self.nb else None
        self.lessons_used = []
        if self.nb:
            tactic, why = self.nb.plan_for(oid, o.get("kind", o["name"]), verdict)
            self.lessons_used = list(self.nb.last_lessons)
        else:
            tactic, why = desk_memory.default_tactic(verdict), "判定どおり"
        if o.get("plan_tactic"):  # rehearse.py tried this on the twin
            tactic, why = desk_memory.clamp_tactic(o["plan_tactic"], tactic), "リハーサルの作戦"
        elif Path(os.environ.get("TIDY_LEARNED", "/nonexistent")).is_file():   # a person's notes: {"word": {tactic settings}}
            for word, hint in json.loads(Path(os.environ["TIDY_LEARNED"]).read_text()).items():
                if word in f"{o['name']}{o.get('kind', '')}":
                    tactic, why = desk_memory.clamp_tactic({**tactic, **hint}, tactic), "人が渡した作戦"
        log_entry["object_id"] = oid
        self.say(f"    作戦: {desk_memory.tactic_ja(tactic)}（{why}）")
        failures, dragged = 0, False
        for attempt in range(1, self.a.attempts + 1):
            # Look right before every attempt: earlier picks or misses may have moved it.
            if self.round_image is not None:
                if attempt > 1:
                    self.clear_view(o)
                if not self.look_again(key, o, attempt):
                    return "lost"
            if tactic.get("sweep_first") and (source is not None or reach < SWEEP_TO - 0.05):
                tactic = {**tactic, "sweep_first": False}  # already near: pulling in would push it away
            if tactic.get("sweep_first") and source is None:
                self.sweep_in(o, log_entry)
                self.send("home", ("到達",))
                tactic = {**tactic, "sweep_first": False}
                if not self.look_again(key, o, attempt):
                    return "lost"
            if tactic.get("soft") and not o.get("deformable"):
                # "Squeeze wherever the fingers reach" skips the width check: only for squashy things
                # (2026-10-27: it lowered the fingers onto a rigid 12 cm tissue box and jammed).
                tactic = {**tactic, "soft": False}
            tried = desk_memory.apply_tactic(o, tactic)
            tried["_original"] = o  # the planner must not see the object itself as an obstacle
            force = int(tried["force"])
            tries = []
            log_entry.setdefault("attempts", []).append({"force": force, "tactic": tactic, "tries": tries})
            print(f"  [{attempt}] {o['name']} を力{force}で掴みにいく", flush=True)
            result = self.pick(tried, floor_z, force, tries)
            # Whether it is still held is judged by what the object is (paper squashes), not by the tactic.
            self.soft_hold = self.soft_hold or bool(o.get("deformable"))
            log_entry["attempts"][-1]["pick"] = result
            if result == "held" and self.soft_hold and self.round_image is not None:
                # The fingers close all the way on paper too, so whether a soft thing was taken is
                # seen, not felt: pull back out of the camera's way and look before the carry
                # (2026-11-01 real run: an empty grip was carried to the box and back).
                # Near the base the view is cleared only over the keep box: then the box check decides.
                look = self.capture(f"{key}_lifted{attempt}") if self.clear_view(o, may_park=False) else None
                if look is not None and not self.occluded(o, self.arm_shadow(look)) and not self.left_desk(o, look, depth=False):
                    print("    持ち上げたが、机に残っている（掴めていない）→ 運ばずにやり直す", flush=True)
                    self.send(f"grip open {self.release_open():.3f}", ("グリッパ",), 30)
                    result = "not_taken"
                    log_entry["attempts"][-1]["pick"] = result
            if result == "held":
                self.go_to_drop(dest)
                check = self.capture(f"{key}_check{attempt}")
                if not self.soft_hold and not self.holding():
                    self.say("    運ぶ途中で滑り落ちた")
                    result = "slipped"
                else:
                    hidden = self.arm_shadow(check)
                    gone = [(k, x) for k, x in self.round_items
                            if k not in self.finished and not self.occluded(x, hidden) and self.left_desk(x, check)]
                    log_entry["attempts"][-1]["gone"] = [k for k, _ in gone]
                    if [k for k, _ in gone] == [key]:
                        self.release_here()
                        self.learned(oid, attempt, tactic, f"moved_to_{dest}", key)
                        return f"moved_to_{dest}"
                    if not gone and self.soft_hold:
                        print("    指は閉じたが、机から消えていない（掴めていない）", flush=True)
                        self.send(f"grip open {self.release_open():.3f}", ("グリッパ",), 30)
                        result = "not_taken"
                    else:
                        involved = [key] + [k for k, _ in gone if k != key]
                        trash_ok = all(self.latest_choice(k, self.decisions.get(k)) == "discard" for k in involved)
                        box = "trash" if trash_ok and "trash" in self.boxes else "keep"
                        names = "、".join(x["name"] for k, x in self.round_items if k in involved)
                        print(f"    何を掴んだか確定できない（候補: {names}）→ {'ゴミ箱' if box == 'trash' else '保管箱'}へ",
                              flush=True)
                        if box != dest:
                            self.go_to_drop(box)
                        self.release_here()
                        # Only the target is counted as put away: things that merely shifted with it (on a cloth
                        # it lifted, 2026-11-01 battle) were "gone" too and four were reported boxed while still
                        # on the desk. The next round's photo shows what is really left.
                        if key in [k for k, _ in gone] or not gone:
                            self.learned(oid, attempt, tactic, f"moved_to_{box}", key)
                            return f"moved_to_{box}"
                        result = "not_taken"  # something else left the desk, not this one
            # A failure: without the notebook the old fixed rules, with it a reflection.
            failures += 1
            if not self.nb:
                if result == "blocked" and not dragged and self.round_image is not None:
                    dragged = True
                    if self.drag_clear(o):
                        continue  # the next attempt looks for it where it was pulled to
                if result in ("unreachable", "too_wide", "blocked", "too_far", "too_flat"):
                    return result
                if result in ("slipped", "not_taken"):
                    # Lifted and left behind is a too weak grip as much as a slip on the way (battle 2026-11-01:
                    # a 200 g lotion bottle judged a soft tube stayed on the desk twice at force 2).
                    tactic = {**tactic, "force": min(5, tactic["force"] + 1)}
                continue
            reflection = self.learned(oid, attempt, tactic, result, key)
            if reflection["give_up"]:
                self.say(f"    あきらめる: {reflection.get('give_up_reason') or reflection['why']}")
                return result if result in REASON_JA else "failed"
            tactic = reflection["next_tactic"]
            self.say(f"    次の作戦: {desk_memory.tactic_ja(tactic)}")
        return "failed"

    def remember(self, o):
        """The object's id in the notebook (matched by appearance, or new)."""
        x0, y0, x1, y1 = o["bbox"]
        pad = 25
        img = self.round_image if self.round_image is not None else cv2.imread(str(self.a.out / "preview/scene.jpg"))
        crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        mask = None
        if o.get("_pixels") is not None:
            mask = np.zeros(crop.shape[:2], np.uint8)
            px = o["_pixels"] - [max(0, x0 - pad), max(0, y0 - pad)]
            inside = (px[:, 0] >= 0) & (px[:, 0] < mask.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < mask.shape[0])
            mask[px[inside, 1], px[inside, 0]] = 1
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        size = [v * 100 for v in o["size_m"]] if o.get("size_m") else list(o["size_cm"][:2])
        verdict = {k: o[k] for k in ("category", "category_reason", "should_grasp", "grasp_reason", "deformable",
                                     "force", "force_reason", "grasp_point", "size_cm", "confidence", "ask_human")
                   if k in o}
        return self.nb.remember(o["name"], o.get("kind", o["name"]), o["category"], verdict, crop, size, mask)

    def learned(self, oid, attempt, tactic, result, key):
        """Record the attempt in the notebook; on a failure reflect on it and keep the lesson."""
        if not self.nb:
            return None
        if self.lessons_used:  # did the lessons behind the first plan help? (bad ones get retired)
            self.nb.credit(self.lessons_used, desk_memory.is_success(result))
            self.lessons_used = []
        images = [str(p) for p in (self.a.out / f"{key}_aim{attempt}.jpg", self.a.out / f"{key}_check{attempt}.jpg",
                                   self.a.out / f"{key}_look{attempt + 1}.jpg") if p.exists()]
        if desk_memory.is_success(result):
            self.nb.record(oid, attempt, tactic, result, images, run=self.run_name)
            return None
        obj = self.nb.object(oid)
        history = self.nb.history(oid, run=self.run_name)
        if self.a.reflect == "claude":
            print("    振り返り中（Claude）…", flush=True)
            r = desk_memory.reflect(self.nb, obj, tactic, result, images, history, model=self.a.model,
                                    cwd=self.a.out, reach_cm=self.current_reach * 100)
        else:
            r = desk_memory.reflect_offline(obj, tactic, result, history)
        episode = self.nb.record(oid, attempt, tactic, result, images, run=self.run_name, why=r["why"])
        lesson = self.nb.learn(obj, r, episode)
        if self.a.consolidate and lesson and self.a.reflect == "claude":
            kind = (r.get("lesson") or {}).get("kind") or obj.get("kind", "")
            if self.nb.consolidate(kind, model=self.a.model, cwd=self.a.out):
                self.say(f"    ノートを整理: 『{kind}』の教訓を3つ以内にまとめ直した")
        les = r.get("lesson") or {}
        self.say(f"    振り返り: {r['why']}")
        if lesson and les.get("text"):
            self.say(f"    教訓（{desk_memory.LEVELS_JA.get(les.get('level'), '')}）: {les['text']}")
        return r

    def where_now(self, key, o):
        """After the last try at o: find where it lies now (failed grasps tip and push things), quietly."""
        if self.round_image is None:
            return
        self.park()
        fresh = self.capture(f"{key}_after")
        before = dict(o)
        if self.occluded(o, self.arm_shadow(fresh)) or not self.relocate(o, fresh, refine=True):
            o.clear()
            o.update(before)
        elif math.dist(before["xy"], o["xy"]) > 0.01:
            print(f"    {o['name']} は今 ({o['xy'][0]:+.3f}, {o['xy'][1]:+.3f}) にある（障害物として覚えておく）", flush=True)

    def look_again(self, key, o, attempt):
        """Photograph the desk and find o where it is now; save the aim picture. False if lost."""
        fresh = self.capture(f"{key}_look{attempt}")
        before = dict(o)
        if self.occluded(o, self.arm_shadow(fresh)) and self.park():
            fresh = self.capture(f"{key}_look{attempt}")
        if self.occluded(o, self.arm_shadow(fresh)):
            print("    アームの陰で見えないので、前に見つけた位置で掴みにいく", flush=True)
        elif self.relocate(o, fresh, refine=attempt > 1):
            print(f"    掴む直前の位置: ({o['xy'][0]:+.3f}, {o['xy'][1]:+.3f})", flush=True)
        elif attempt == 1:  # first look: keep what the judge saw (it may only be hard to see)
            o.clear()
            o.update(before)
            print("    今は見つからないので、判定のときの位置で掴みにいく", flush=True)
        else:
            print("    見失った", flush=True)
            return False
        img = cv2.imread(str(fresh))
        if o.get("centre_px"):
            u, v = (int(x) for x in o["centre_px"])
            cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
            x0, y0, x1, y1 = o["bbox"]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.imwrite(str(self.a.out / f"{key}_aim{attempt}.jpg"), img)
        return True


    def arm_shadow(self, image):
        """Image mask of where the arm itself is in this frame (it may hide desk objects)."""
        return self.arm_mask(cv2.imread(str(image)).shape[:2])

    def arm_mask(self, shape):
        """Mask (of this image shape) of where the arm is drawn at its last commanded pose."""
        mask = np.zeros(shape, np.uint8)
        try:
            q = ppa.arm_goto_last_q(self.a.out)
        except (IndexError, FileNotFoundError):
            q = [0.0] * 6  # not moved yet: folded at its start pose, which overhangs the desk near the base
        pts = np.vstack([p[::3] for name, p in drop_body().posed(q).items() if name != "base_link"])
        T = topdown_ik.fk(q)  # the link model stops at the wrist: add the fingers, closed to fully open
        fingers = np.vstack([(T.rotation @ sim_world.gripper_points(w).T).T + T.translation
                             for w in np.linspace(0.0, self.a.grip_open_max, 6)])
        pts = np.vstack([pts, fingers])
        px = sim_world._project(self.calib, pts)
        inside = (px[:, 0] >= 0) & (px[:, 0] < mask.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < mask.shape[0])
        for u, v in px[inside].astype(int):
            cv2.circle(mask, (int(u), int(v)), 6, 255, -1)
        return cv2.dilate(mask, np.ones((25, 25), np.uint8))

    def occluded(self, o, mask):
        """More than a quarter of the object's pixels lie under the arm in this frame."""
        px = o.get("_pixels")
        if px is None or len(px) == 0:
            return False
        h, w = mask.shape
        u, v = px[:, 0].clip(0, w - 1), px[:, 1].clip(0, h - 1)
        return float((mask[v, u] > 0).mean()) > 0.25

    def holding(self):
        if getattr(self, "rim_tactic", False):
            return True   # a thin wall between the fingers reads as closed on nothing (2026-11-01: a cup carried fine)
        return self.fingers_apart()

    def fingers_apart(self):
        """True while the fingers stay open past their squeeze target, i.e. something is between them."""
        target = float(self.last_close.split("目標")[1].split("rad")[0])
        state = self.send("grip state", ("グリッパ: state",), 30)
        position = float(state.split("位置")[1].split("rad")[0])
        return position > target + 0.5 * self.last_squeeze

    def still_there(self, o, image):
        """The object's own pixels look the same as in the round's first frame.

        Comparing before and after works for any colour (2026-10-28: white card and grey paper are
        too close to the desk colour for a desk-colour test).
        """
        if o.get("_pixels") is None or self.round_image is None:
            return True
        px = o["_pixels"]
        before = cv2.cvtColor(self.round_image, cv2.COLOR_BGR2LAB)[px[:, 1], px[:, 0]].astype(float)
        after = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2LAB)[px[:, 1], px[:, 0]].astype(float)
        # A pale thing gone from a pale desk changes its pixels by less than CHANGED (2026-11-01 real:
        # crumpled paper pushed 8 cm away was "still there" and the next grasp went to bare desk).
        threshold = max(CHANGED_MIN, min(CHANGED, CHANGED_SHARE * self.desk_contrast(o, self.round_image)))
        changed = np.linalg.norm(after - before, axis=1) > threshold
        return float(changed.mean()) < 0.5

    def left_desk(self, o, image, depth=True):
        """o is neither where it was nor anywhere else the desk changed: it is in the fingers (or a box).

        Not at its old place alone is not enough: a missed grasp pushes paper (2026-11-01 real: 8 cm).
        Where it was found now is kept in o, for the next try.
        """
        if self.still_there(o, image):
            return False
        if not self.a.depth:
            # Without a depth camera (the real desk) a changed patch elsewhere is as often a shadow or the light
            # (2026-11-01 real: a tissue put in the bin was "pushed" to bare desk and grasped for in the air).
            return True
        if depth and self.unexplained_on_desk(o, image):
            return False  # something it could be stands somewhere unlisted (a cloth it drags changes its look)
        moved = dict(o)
        # A thing in the fingers shows round them, larger than the fingers for a tissue (2026-11-01 real: a
        # held tissue was taken for one pushed across the desk, "not taken" though it was): look away from the arm.
        if (self.relocate_changed(moved, image, max_colour=FOUND_COLOUR, arm_grow=HELD_HALO_PX)
                and math.dist(moved["xy"], o["xy"]) > LEFT_MIN_MOVE):
            print(f"    {o['name']} は押されて ({moved['xy'][0]:+.3f}, {moved['xy'][1]:+.3f}) に動いた", flush=True)
            o.clear()
            o.update(moved)
            return False
        return True

    def unexplained_on_desk(self, o, image):
        """Depth camera: a patch at least a third of o's size stands on the desk where no other thing of this
        round is known to be (battle 2026-11-01: a cloth and a sponge, dragged and crumpled, looked like
        nothing seen before and were reported put away while still on the desk)."""
        if not self.a.depth or o.get("_pixels") is None:
            return False
        path = Path(image).with_suffix(".height.npy")
        if not path.exists():
            self.send(f"height {path} video2", ("撮影",), 180)
        hm = np.load(path)
        others = [x["bbox"] for k, x in self.round_items + self.standing
                  if x is not o and x.get("bbox") and not str(self.finished.get(k, "")).startswith("moved_to_")]
        area = o.get("_area") or len(o["_pixels"])
        for x0, y0, x1, y1 in desk_judge.overlooked(hm, self.calib, others, self.arm_mask(hm.shape[:2])):
            if (x1 - x0) * (y1 - y0) >= UNEXPLAINED_SHARE * area:
                return True
        return False

    def desk_contrast(self, o, image):
        """LAB distance between o's mean colour and the desk in a ring just outside its box, in image."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
        h, w = lab.shape[:2]
        px = o["_pixels"]
        inside = lab[px[:, 1].clip(0, h - 1), px[:, 0].clip(0, w - 1)].mean(0)
        x0, y0, x1, y1 = (int(v) for v in o["bbox"])
        ring = np.zeros((h, w), bool)
        g = RING_PX
        ring[max(0, y0 - g):min(h, y1 + g), max(0, x0 - g):min(w, x1 + g)] = True
        ring[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = False
        if not ring.any():
            return CHANGED / CHANGED_SHARE
        return float(np.linalg.norm(inside - np.median(lab[ring], axis=0)))

    # --- preview (no arm) ----------------------------------------------
    def preview(self):
        """Judge the current desk and plan every grasp without moving: preview.jpg + preview.json."""
        from PIL import Image, ImageDraw, ImageFont
        folder = self.a.out / "preview"
        folder.mkdir()
        image = self.capture("preview/scene")
        for role, seed in self.box_seeds.items():
            box, _ = ppa.detect_box(image, self.calib, seed, outer=self.box_outer[role])
            if box is not None:
                self.boxes[role] = box
        objects = self.judge(image, folder)
        self.round_items = [(f"p{k}", o) for k, o in enumerate(objects, 1)]
        img = cv2.imread(str(image))
        for role, box in self.boxes.items():
            px = sim_world._project(self.calib, np.c_[box["corners"], np.full(4, box_fetch_rim())])
            cv2.polylines(img, [np.round(px).astype(np.int32)], True, (0, 0, 255) if role == "trash" else (0, 160, 0), 2)
        rows, labels = [], []
        for key, o in self.round_items:
            row = {"key": key, "name": o["name"], "category": CATEGORY_JA[o["category"]], "force": o["force"],
                   "should_grasp": o["should_grasp"], "reason": o["grasp_reason"]}
            if o["category"] == "fixture" or not o["should_grasp"]:
                row["plan"] = "掴まない"
            elif o.get("xy") is None:
                row["plan"] = "位置が分からない"
            elif self.container_of(o["xy"]):
                row["plan"] = "もう箱の中"
            else:
                source = self.container_of(o["xy"])
                reason, plan = self.plan_pick(o, 0.0 if source is None else 0.008)
                if reason:
                    row["plan"] = REASON_JA.get(reason, reason)
                else:
                    g = plan["g0"]
                    row["plan"] = (f"掴める（傾き {math.degrees(g['pitch']):.0f}°、開き {plan['opening']:.2f} rad、"
                                   f"指の間 {plan['across'] * 1000:.0f} mm）")
                    T = topdown_ik.fk(g["grasp"])
                    half = sim_world.gap(plan["opening"]) / 2
                    tips = np.array([T.translation + T.rotation[:, 1] * half, T.translation - T.rotation[:, 1] * half])
                    tips[:, 2] = 0.0
                    px = sim_world._project(self.calib, tips)
                    a_, b_ = (tuple(int(v) for v in p) for p in px)
                    cv2.line(img, a_, b_, (255, 120, 0), 2)
                    for p_ in (a_, b_):
                        cv2.circle(img, p_, 5, (255, 120, 0), -1)
            rows.append(row)
            labels.append((o, f"{key} {row['category']} {o['name']}：{row['plan'].split('（')[0]}"))
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw, font = ImageDraw.Draw(pil), ImageFont.truetype(desk_judge.FONT, 16)
        for o, text in labels:
            x0, y0 = o["bbox"][:2]
            draw.text((x0, max(0, y0 - 20)), text, font=font, fill=(20, 20, 20), stroke_width=2,
                      stroke_fill=(255, 255, 255))
        pil.save(folder / "preview.jpg", quality=92)
        (folder / "preview.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"箱: {', '.join(sorted(self.boxes)) or 'なし'}")
        for row in rows:
            print(f"{row['key']:4s} {row['category']:4s} {row['name']}: {row['plan']}")
        return rows

    # --- main loop ----------------------------------------------------
    def run(self):
        a = self.a
        if a.physics:
            os.environ.setdefault("MUJOCO_GL", "egl")
            cmd = [sys.executable, "-u", str(HERE / "mj_arm.py"), "--world", a.sim, "--seed", str(a.seed),
                   "--max-tracking", str(arm_goto.MAX_TRACKING)]
            if a.record:
                cmd += ["--record", str(Path(a.record).resolve())]
        elif a.sim:
            cmd = [sys.executable, "-u", str(HERE / "sim_arm.py"), "--world", a.sim, "--seed", str(a.seed)]
        else:
            cmd = [sys.executable, "-u", str(HERE / "arm_goto.py"), "--execute", "--gripper", "--correct",
                   "--preflight", str(Path(a.preflight).resolve()), "--snap-cameras", a.snap_cameras]
        cmd += ["--max-speed", str(a.max_speed), "--driver-config", str(Path(a.driver_config).resolve()),
                "--min-tcp-z", str(ppa.TIP_REACH - ppa.PROBE_DEPTH - 0.002),
                "--grip-open-max", str(a.grip_open_max), "--output-dir", str(a.out / "client")]
        if a.approve == "web":
            self.server = confirm_server.serve(self.approval_dir, a.host, a.port, notes=self.nb)
        self.log = open(a.out / "client.log", "w")
        self.client = ppa.Client(cmd, self.log)
        try:
            self.client.wait_ready(180 if a.physics else 30)  # the GPU renderer takes ~30 s to start
            self.send("arm", ("保持確認完了",), 60)
            for round_no in range(1, a.rounds + 1):
                folder = a.out / f"round{round_no}"
                folder.mkdir()
                if a.scatter and round_no > 1:
                    self.scatter(a.scatter)
                self.send("home", ("到達",))
                image = self.capture(f"round{round_no}/scene")
                self.find_boxes(image)
                if self.park():  # the folded arm hides the desk near the base: look from over the box
                    image = self.capture(f"round{round_no}/scene")
                self.round_image = cv2.imread(str(image))
                objects = self.judge(image, folder)
                self.apply_plan(objects)
                for o, other in attached_to_untouchable(objects):
                    if not cable_like(o):
                        # Only touching: taken with the fingers kept clear of it (battle 2026-11-01: boxes laid
                        # against a mouse were all left, a wall of "touching" things).
                        print(f"    {o['name']} は {other['name']} に触れているので、指を離して掴む", flush=True)
                        continue
                    # sim 2026-10-29: a mouse's cable judged as a thing of its own was taken, and the mouse with it.
                    o["should_grasp"] = False
                    o["grasp_reason"] = f"{other['name']}（掴まない物）とつながっている"
                    print(f"    {o['name']} は {other['name']} につながっているので掴まない", flush=True)
                todo = [(f"r{round_no}_{k}", o) for k, o in enumerate(objects, 1)
                        if o["should_grasp"] and o["category"] != "fixture" and o.get("xy") is not None
                        and self.container_of(o["xy"]) is None
                        and not any(math.dist(o["xy"], xy) < GIVE_UP_RADIUS for xy, _ in self.given_up)
                        and (not self.a.only or o["category"] == self.a.only)
                        and (not self.a.plan or "plan_rank" in o)
                        and not any(word in o["name"] for word in self.a.exclude)
                        and (not self.a.include or any(word in o["name"] for word in self.a.include))]
                skipped = [o["name"] for o in objects if not o["should_grasp"] and o["category"] != "fixture"]
                self.report.setdefault("for_human", [])
                for o in objects:  # the desk must end clear: what the robot cannot take is listed for a person
                    if not o["should_grasp"] and o["category"] != "fixture" and o["name"] not in self.report["for_human"]:
                        self.report["for_human"].append(o["name"])
                print(f"[ラウンド{round_no}] 判定 {len(objects)} 件、片付け候補 {len(todo)} 件"
                      + (f"、掴まない: {', '.join(skipped)}" if skipped else ""), flush=True)
                self.report["rounds"].append({"round": round_no, "judged": len(objects), "todo": len(todo),
                                              "not_grasped": skipped})
                if not todo and not (a.scatter and round_no < a.rounds):
                    break
                self.page_extra = self.page_others(objects, todo, image)
                decisions = self.ask(round_no, todo, image)
                todo.sort(key=lambda t: t[1].get("plan_rank", np.hypot(*t[1]["xy"])))
                self.round_items, self.decisions, self.finished = list(todo), decisions, {}
                # What is left for a person (a mouse, a phone) or not taken this round is still in the way
                # (sim 2026-10-28: a mouse the planner did not know about was pushed 3 cm).
                taken = {id(o) for _, o in todo}
                self.standing = [(f"r{round_no}_x{k}", o) for k, o in enumerate(objects, 1)
                                 if id(o) not in taken and o.get("xy") is not None]
                queue, deferred = list(todo), set()
                while queue:
                    key, o = queue.pop(0)
                    choice = self.latest_choice(key, decisions[key])
                    if choice == PENDING:
                        if any(self.latest_choice(k, decisions[k]) != PENDING for k, _ in queue):
                            queue.append((key, o))  # answer not in yet: the decided ones first
                            continue
                        print(f"[{key}] {o['name']}: 確認ページの答えを待っています", flush=True)
                        choice = self.wait_answer(key)
                    entry = {"key": key, "name": o["name"], "category": o["category"], "decision": choice,
                             "force": o["force"], "xy": list(o["xy"]), "truth_id": o.get("truth_id")}
                    self.report["objects"].append(entry)
                    if choice == "leave":
                        entry["result"] = "left_by_human"
                        self.given_up.append((entry["xy"], "left_by_human"))  # do not ask again this run
                        continue
                    if key in self.finished:  # already taken by mistake with a neighbour, and placed
                        entry["result"] = self.finished[key]
                        continue
                    dest = "trash" if choice == "discard" else "keep"
                    if dest not in self.boxes:
                        entry["result"] = "no_trash_box"
                        self.given_up.append((o["xy"], "no_trash_box"))
                        print(f"[{key}] {o['name']}: ゴミ箱が見つからないので、そのままにします", flush=True)
                        continue
                    self.say(f"[{key}] {o['name']} → {'ゴミ箱' if dest == 'trash' else '保管箱'}")
                    entry["result"] = self.tidy_one(key, o, dest, entry)
                    self.say(f"[{key}] 結果: {RESULT_JA.get(entry['result'], REASON_JA.get(entry['result'], entry['result']))}")
                    self.show_result(key, RESULT_JA.get(entry["result"], REASON_JA.get(entry["result"], entry["result"])))
                    if entry["result"] == "blocked" and key not in deferred and len(queue) > 0:
                        # Neighbours in the way: try again after the rest of this round is cleared.
                        deferred.add(key)
                        self.report["objects"].remove(entry)
                        queue.append((key, o))
                        print(f"[{key}] 周りを片付けてからもう一度", flush=True)
                        continue
                    self.finished[key] = entry["result"]
                    if not entry["result"].startswith("moved_to_") and entry["result"] != "lost":
                        self.where_now(key, o)  # failed grasps move things: where it lies is an obstacle now
                    if not entry["result"].startswith("moved_to_") and entry["result"] not in ("blocked", "swept"):
                        self.given_up.append((entry["xy"], entry["result"]))  # blocked: retried next round
            self.send("home", ("到達",))
            self.send("finish", ("終了処理完了",), 60)
            self.report["status"] = "finished"
            self.report["needs_human"] = [{"xy": xy, "reason": r} for xy, r in self.given_up]
            if self.given_up:
                print("片付けずに残したもの: " + ", ".join(f"({x:+.2f}, {y:+.2f}) {REASON_JA.get(r, r)}"
                                                for (x, y), r in self.given_up), flush=True)
            if self.report.get("for_human"):
                print("ロボットでは持てないので、人の手で片付けてほしいもの: " + "、".join(self.report["for_human"]),
                      flush=True)
        except Exception as error:
            self.report["status"] = f"aborted: {error}"
            print(f"中断: {error}", flush=True)
            try:
                self.send("home", ("到達",))
                self.send("finish", ("終了処理完了",), 60)
            except Exception as cleanup:
                print(f"後片付けもできません: {cleanup}", flush=True)
        finally:
            try:
                self.client.proc.stdin.write("quit\n")
                self.client.proc.stdin.flush()
                self.client.proc.wait(timeout=20)
            except Exception:
                self.client.proc.kill()
            if self.server:
                self.server.shutdown()
            self.log.close()
            if self.nb:
                self.report["notebook"] = self.nb.stats()
            (a.out / "report.json").write_text(json.dumps(self.report, indent=1, ensure_ascii=False, default=float))
        return self.report

    def scatter(self, n):
        """The scatterer (simulation): take up to n things back out of the boxes and throw them on the desk."""
        world = sim_world.load(self.a.sim)
        boxed = [o for o in world["objects"] if sim_world.container_of(world, o) is not None]
        self.rng.shuffle(boxed)
        thrown = []
        for o in boxed[:n]:
            for _ in range(100):
                r, ang = self.rng.uniform(0.22, 0.40), math.radians(self.rng.uniform(-35, 35))
                xy = [r * math.cos(ang), r * math.sin(ang)]
                if all(math.dist(xy, x["centre"]) > 0.08 for x in world["objects"] if x is not o):
                    break
            self.send(f"place {o['id']} {xy[0]:.3f} {xy[1]:.3f}", ("配置",), 60)
            o["centre"] = xy
            thrown.append(o["name"])
        if thrown:
            self.say(f"[散らかし役] {'、'.join(thrown)} を机にばらまいた")


def hollow(xyz, top):
    """True for a container seen from above: a ring (no readings in the middle) or a middle below the rim."""
    centre = xyz[:, :2].mean(0)
    r = np.hypot(*(xyz[:, :2] - centre).T)
    inner = r < 0.6 * np.percentile(r, 95)
    if inner.sum() < 0.15 * len(xyz):
        return True
    return float(np.median(xyz[inner, 2])) < top - HOLLOW_DROP


def cable_like(o):
    """A cable, cord or string: thin and long, or called so (it may be tied to what it touches)."""
    if any(w in f"{o.get('kind', '')}{o['name']}" for w in CABLE_WORDS):
        return True
    a, b = sorted(o.get("size_cm", [0, 0])[:2], reverse=True)
    return b < CABLE_MAX_WIDTH_CM and a >= CABLE_MIN_RATIO * max(b, 0.1)


def attached_to_untouchable(objects, gap_px=TOUCH_PX):
    """(object, untouchable) pairs: a thing to grasp whose outline comes within gap_px of a thing not to be
    grasped (a precision device, glassware, something tied down). Taking it would drag the other along."""
    from scipy.spatial import cKDTree
    keep_off = [x for x in objects if not x.get("should_grasp", True) and x.get("category") != "fixture"
                and x.get("_pixels") is not None and len(x["_pixels"])
                and (x.get("size_cm") or [0, 0, 1.0])[2] >= 1.0]  # a sheet left only for being flat may move
    pairs = []
    for o in objects:
        if not o.get("should_grasp", True) or o.get("category") == "fixture" or o.get("_pixels") is None:
            continue
        for other in keep_off:
            if cKDTree(np.asarray(other["_pixels"], float)).query(np.asarray(o["_pixels"], float))[0].min() <= gap_px:
                pairs.append((o, other))
                break
    return pairs


def held_reach(o):
    """How far (m) a held thing may stick out sideways from the fingers: half its longest side."""
    sides = list(o.get("size_m") or []) + [v / 100 for v in (o.get("size_cm") or [])[:2]]
    return max(sides) / 2 if sides else 0.0


def release_spots(box, n, reach=0.0):
    """Where in the box the n-th thing let go into it goes, best first: the middle for the first, then
    round it, RELEASE_SPREAD apart and inside the walls by RELEASE_MARGIN or the held thing's reach and
    1 cm (sim 2026-10-29: an 11 cm bowl let go 6 cm in grazed the wall). [] means the taught pose as is."""
    if n == 0:
        return []
    c = np.asarray(box.get("centre", box.get("xy")), float)
    corners = np.asarray(box["corners"], float) if box.get("corners") is not None else None
    half = (np.ptp(corners, axis=0) / 2) if corners is not None else np.array([0.10, 0.10])
    room = np.maximum(half - max(RELEASE_MARGIN, reach + 0.01), 0.0)
    ring = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
    spots = [c + np.minimum(room, RELEASE_SPREAD) * np.array(d) for d in ring]
    k = (n - 1) % len(ring)
    return [tuple(map(float, p)) for p in spots[k:] + spots[:k]]


def misjudged_size(o):
    """The depth camera's height is far from the judge's: the judge may have taken it for something else.

    sim 2026-10-29: a 12 cm squirrel plush judged "crumpled paper, 5 cm" with high confidence was thrown
    away; a thing to discard is worth a question whenever the photo and the depth disagree this much.
    """
    judged, measured = o.get("judged_height_cm"), o["size_cm"][2]
    if judged is None or not o.get("height_from_depth"):
        return False
    lo, hi = sorted((float(judged), float(measured)))
    return hi - lo >= MISJUDGED_CM and hi >= MISJUDGED_RATIO * max(lo, 0.5)


def rim_ring(xyz, top):
    """True when the highest points go all round a lower middle: a bowl's or cup's rim, not a soft heap."""
    if xyz is None or len(xyz) < 30 or not hollow(xyz, top):
        return False
    centre = xyz[:, :2].mean(0)
    band = xyz[xyz[:, 2] >= top - RIM_BAND]
    r = np.hypot(*(band[:, :2] - centre).T)
    if np.mean(r < 0.6 * np.percentile(np.hypot(*(xyz[:, :2] - centre).T), 95)) > 0.1:
        return False  # the highest part is in the middle: a head or a heap, not a rim
    ang = np.arctan2(*(band[:, :2] - centre).T[::-1])
    filled = np.unique(((ang + math.pi) / (2 * math.pi) * 12).astype(int) % 12)
    return len(filled) >= 10


def part_grasps(xyz, top, low, max_across, centres=150):
    """Narrow pieces of an object too wide to grip whole, best first.

    xyz are the depth camera's points on the object. For each grip height z (from just under the top
    down to `low`), what stands above z is what the fingers straddle: a strip as wide as a finger is laid
    through a point at each angle, and the piece it crosses there (split from other pieces by gaps a
    finger fits in) must fit between the open fingers. Returns dicts c, axis (along the fingers' pads),
    across, gap (finger opening), z (tip height), best first: near the object's middle and deep.
    """
    mid = xyz[:, :2].mean(0)
    band = sim_world.FINGER[1] / 2 + 0.002
    found = []
    for z in np.arange(top - PART_STEP, low - 1e-6, -PART_STEP):
        above = xyz[xyz[:, 2] >= z]
        if len(above) < 10:
            continue
        pts = above[:, :2]
        for c in pts[:: max(1, len(pts) // centres)]:
            for t in np.radians(np.arange(0, 180, RIM_STEP_DEG)):
                n = np.array([math.cos(t), math.sin(t)])
                a = np.array([-n[1], n[0]])
                rel = pts - c
                along = rel @ a
                inside = np.abs(along) <= band
                s = np.sort(rel[inside] @ n)
                if len(s) < 5:
                    continue
                cuts = np.flatnonzero(np.diff(s) > PART_SPLIT)
                starts, ends = np.r_[0, cuts + 1], np.r_[cuts, len(s) - 1]
                k = int(np.argmin(np.where((s[starts] <= 0) & (s[ends] >= 0), 0, np.minimum(abs(s[starts]), abs(s[ends])))))
                lo, hi = s[starts[k]], s[ends[k]]
                if ends[k] - starts[k] < 4 or not PART_MIN <= hi - lo <= max_across - 2 * 0.004:
                    continue
                # Faces square to the fingers: the piece is as wide under each half of the finger pads.
                # Across a box's corner it is a wedge and squeezes out (and the strip reads narrow there).
                piece = inside & (rel @ n >= lo - 1e-9) & (rel @ n <= hi + 1e-9)
                halves = [piece & (along < 0), piece & (along >= 0)]
                if min(h.sum() for h in halves) < 2:
                    continue
                ends_lo = [float((rel[h] @ n).min()) for h in halves]
                ends_hi = [float((rel[h] @ n).max()) for h in halves]
                lean = band * math.tan(PART_SQUARE) + 0.002  # one grid step of the depth image
                if abs(ends_lo[0] - ends_lo[1]) > lean or abs(ends_hi[0] - ends_hi[1]) > lean:
                    continue
                left = lo - s[ends[k - 1]] if k > 0 else np.inf
                right = s[starts[k + 1]] - hi if k + 1 < len(starts) else np.inf
                room = float(np.clip(min(left, right) - sim_world.FINGER[0] - 0.002, 0.003, PART_ROOM))
                centre = c + n * (lo + hi) / 2
                found.append({"c": centre, "axis": a, "across": float(hi - lo),
                              "gap": float(min(hi - lo + 2 * room, max_across)), "z": float(z),
                              "score": float(np.hypot(*(centre - mid)) + PART_DEEPER * (z - low) + 0.1 * (hi - lo))})
    found.sort(key=lambda g: g["score"])
    best = []
    for g in found:  # one per place and direction
        if all(math.dist(g["c"], b["c"]) > 0.008 or abs(g["axis"] @ b["axis"]) < 0.97 for b in best):
            best.append(g)
    return best


def desk_ik(point, rot, seed):
    import box_fetch
    return box_fetch.solve_pose(point, rot, [seed])


_BODY = []


def drop_body():
    if not _BODY:
        import reach_map
        _BODY.append(reach_map.Body())
    return _BODY[0]


def box_fetch_rim():
    import box_fetch
    return box_fetch.RIM_Z


def sim_world_rim(box):
    return box.get("rim_z", 0.055)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sim", help="world JSON: run against sim_arm/sim_world instead of the arm")
    p.add_argument("--physics", action="store_true", help="with --sim: the MuJoCo desk (mj_arm.py), seen by SAM")
    p.add_argument("--record", help="physics: folder for the presentation videos")
    p.add_argument("--scatter", type=int, default=0, help="physics: things thrown back on the desk each round")
    p.add_argument("--notebook", help="experience notebook folder (knowledge carried between runs); none = no learning")
    p.add_argument("--plan", help="plan.json from rehearse.py: only its things, in its order, with its tactics")
    p.add_argument("--snap-cameras", default="/dev/video4,/dev/video0",
                   help="real arm: side cameras grabbed during a pick; empty when something else records them")
    p.add_argument("--reflect", choices=("claude", "offline"), default="claude",
                   help="who reflects on a failed grasp: claude -p, or the fixed offline rules")
    p.add_argument("--run-name", help="name of this run in the notebook (default: the --out folder name)")
    p.add_argument("--attempts", type=int, default=3, help="grasp attempts per object")
    p.add_argument("--consolidate", action="store_true",
                   help="merge a kind's lessons with Claude when more than four pile up")
    p.add_argument("--depth", action="store_true",
                   help="physics: a depth camera at video2 measures each object's height (a RealSense)")
    p.add_argument("--taught-drops", action="store_true",
                   help="release at the hand-taught drop_q poses instead of straight down over each box")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preflight")
    p.add_argument("--boxes", help="JSON {trash: {seed_px: [u, v]}, keep: {...}} (sim: derived from the world)")
    p.add_argument("--judge", choices=("fast", "claude", "oracle"), default="fast",
                   help="fast: list objects, then judge each crop in parallel; claude: one call for the whole frame")
    p.add_argument("--judge-noise", type=float, default=4.0, help="oracle bbox jitter (px)")
    p.add_argument("--judge-force-error", type=float, default=0.0, help="oracle: chance of a too-low force")
    p.add_argument("--judge-miss", type=float, default=0.0, help="sim: chance the judge overlooks an object")
    p.add_argument("--xy-bias-mm", type=float, default=0.0, help="sim: camera calibration error (mm, random direction)")
    p.add_argument("--model", default="sonnet")
    p.add_argument("--approve", choices=("web", "yes", "no"), default="web")
    p.add_argument("--approve-timeout", type=float, default=900)
    p.add_argument("--auto-discard", type=float, default=0.9,
                   help="trash judged at least this confident is discarded without asking (2026-10-28 rule; 1.1 = always ask)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--only", choices=("trash", "keep"), help="handle only objects judged this category")
    p.add_argument("--exclude", action="append", default=[], help="skip objects whose name contains this word")
    p.add_argument("--include", action="append", default=[], help="handle only objects whose name contains this word")
    p.add_argument("--no-sam", dest="sam", action="store_false", help="find objects by colour, not SAM")
    p.add_argument("--trust-fit", action="store_true",
                   help="skip the width check (the human says it fits) and open the fingers fully")
    p.add_argument("--preview", action="store_true", help="judge and plan the current desk, never move the arm")
    p.add_argument("--calib", default=str(HERE / "calib/video2.json"))
    p.add_argument("--driver-config", default=str(HERE / "config/arm_pick_trial.yaml"))
    p.add_argument("--max-speed", type=float, default=0.15)
    p.add_argument("--grip-open-max", type=float, default=4.5,
                   help="rad; the fingers open 100 mm at 5.9 rad, locked at 4.5 (arm_goto.GRIP_OPEN_LIMIT)")
    p.add_argument("--max-reach", type=float, default=0.55,
                   help="m from the base; farther grasps hold joint2 above its 11 N*m rating (RS06)")
    p.add_argument("--sweep-reach", type=float, default=0.66,
                   help="m: objects between --max-reach and this are dragged in with the closed fingers first")
    a = p.parse_args(argv)
    if not a.sim and (a.judge == "oracle" or a.approve != "web"):
        p.error("--judge oracle and --approve yes/no are simulation only")
    if not a.sim and not a.preflight and not a.preview:
        p.error("the real arm needs --preflight")
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    if a.physics and not a.sim:
        p.error("--physics needs --sim WORLD.json")
    if a.physics and not a.boxes:
        a.boxes = str(HERE / "config/desk_boxes.json")
    if a.sim and not a.boxes:
        world, calib = sim_world.load(a.sim), json.load(open(a.calib))
        seeds = {}
        for b in world["boxes"]:
            px = sim_world._project(calib, np.array([[*b["centre"], b["height"]]]))[0]
            seeds[b["role"]] = {"seed_px": [float(px[0]), float(px[1])]}
        a.boxes = str(a.out / "boxes.json")
        Path(a.boxes).write_text(json.dumps(seeds))
    if not a.boxes:
        p.error("--boxes is required on the real arm")
    if a.preview:
        Desk(a).preview()
        return
    report = Desk(a).run()
    print(json.dumps({"status": report["status"], "objects": [(o["name"], o.get("result")) for o in report["objects"]]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
