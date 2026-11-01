#!/usr/bin/env python3
"""Ask `claude -p` what is on the desk and whether each thing is trash (offline, no arm).

One overhead video2 frame goes to Claude with a labelled pixel grid drawn on a
copy, so it can give rough boxes. Each box is then refined locally: the largest
blob that differs from the desk colour gives the pixel centre, which the
hand-eye calibration maps to robot XY.

    source scripts/ros2_env.sh
    python3 scripts/desk_judge.py --out DIR [--image IMG]   # writes judge.json and overlay.jpg
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import handeye_solve

HERE = Path(__file__).resolve().parent
SENT_WIDTH = 960       # px; Claude downsizes large images itself, so send a known size and scale back
TIMEOUT_S = 600        # one judgement took 225 s on 2026-10-27; 300 s timed out once
ATTEMPTS = 2
CATEGORIES = ("trash", "keep", "unsure", "fixture")
LABELS_JA = {"trash": "ゴミ", "keep": "残す", "unsure": "不明", "fixture": "固定"}
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
# Grip force level chosen by Claude -> gripper squeeze past first contact (rad, kp 8).
FORCE_SQUEEZE = {1: 0.03, 2: 0.05, 3: 0.08, 4: 0.11, 5: 0.14}
SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "kind": {"type": "string"},
            "bbox": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
            "category": {"type": "string", "enum": list(CATEGORIES)},
            "category_reason": {"type": "string"},
            "should_grasp": {"type": "boolean"},
            "grasp_reason": {"type": "string"},
            "deformable": {"type": "boolean"},
            "force": {"type": "integer", "minimum": 1, "maximum": 5},
            "force_reason": {"type": "string"},
            "grasp_point": {"type": "string"},
            "size_cm": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "ask_human": {"type": "boolean"},
        },
        "required": ["name", "kind", "bbox", "category", "category_reason", "should_grasp", "grasp_reason", "deformable", "force",
                     "force_reason", "grasp_point", "size_cm", "confidence", "ask_human"],
    }}},
    "required": ["objects"],
}
HOUSE_RULES = """House rules: the desk must end up completely clear - only the robot itself stays. Crumpled paper
and used tissue are trash (to the trash bin). Everything else is kept (to the keep box): pens,
erasers, USB memory, shogi pieces, wooden blocks, tissue boxes, boards, boxes, sheets. When unsure,
ask the human. Obvious trash is discarded without asking; anything you are not sure about waits for
the human's answer. Precision devices are never
grasped by the robot (should_grasp=false, left for the human): phones, tablets, laptops, cameras,
earphones, speakers, computer mice, watches, glasses, measuring instruments and similar fragile electronics
(USB memory sticks are fine to grasp)."""

PROMPT = """Read the image {image}. It is an overhead photo of a desk tidied by a 6-axis robot arm with a
two-finger parallel gripper (the arm's base is at the top centre). A pixel grid is drawn every
100 px with labels; the image is {w}x{h} px. You are the robot's judge: decide, for every loose
object, what the robot should do. Answer in Japanese for the text fields.

{rules}

For each separate object give:
- name
- kind: what sort of thing it is, one short general Japanese word (ペン, 紙くず, 消しゴム, 箱, 駒 ...)
- bbox = [x0, y0, x1, y1] in pixels of this image, tight around the object
- category: trash / keep / unsure / fixture
  (fixture = only the robot arm itself; everything else on the desk is trash, keep or unsure)
- category_reason: why
- should_grasp: whether the robot should try to pick it up itself. Say false only when grasping could
  break it (a precision device, see the house rules; glassware that looks fragile), it is flat (a sheet,
  a single paper, a calibration checkerboard, a mat: thinner than about 5 mm), it is tied to something
  (cable), or liquid or food can be seen in it - those are left for the human. Do not say false because of
  its size or weight: the robot measures it with a depth camera, pinches a rim, wall, edge or narrow part
  of things wider than its fingers, and reports itself what it cannot hold. A closed or opaque bottle,
  pot or jar is grasped (true) unless liquid is visible.
- grasp_reason: why (or why not)
- deformable: true for soft things that squash when gripped (crumpled paper, tissue, cloth);
  the robot then closes on whatever part it reaches instead of reaching around it
- force: grip force 1-5. 1 = barely closing (tissue, would crush), 2 = light (crumpled paper,
  soft things), 3 = normal (pen, shogi piece), 4 = firm (heavy or slippery), 5 = strongest
  (dense, heavy, must not slip). Fragile things get low force or should_grasp=false.
- force_reason: why
- grasp_point: where the fingers should go (e.g. "ペンの中央を横から")
- size_cm: [length, width, height] estimate
- confidence: 0-1 for the category
- ask_human: true when the human should confirm (unclear cases; obvious trash and obvious keep do not need it)
- confidence: be honest - 0.9 or more only when it is obviously trash or obviously still useful
Separate pieces that touch each other into separate objects when you can, but the parts of one
object (a magnifier's handle and lens, a mug and its handle, a toy's body and wheels) are one object.
Keep every *_reason and grasp_point short (one phrase, under 25 Japanese characters)."""


def capture(path):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "v4l2", "-input_format", "mjpeg",
                    "-video_size", "1280x960", "-framerate", "30", "-i", "/dev/video2", "-ss", "5",
                    "-frames:v", "1", str(path)], check=True, timeout=30)


def gridded(img):
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(100, w, 100):
        cv2.line(out, (x, 0), (x, h), (0, 200, 255), 1)
        cv2.putText(out, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 255), 1)
    for y in range(100, h, 100):
        cv2.line(out, (0, y), (w, y), (0, 200, 255), 1)
        cv2.putText(out, str(y), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 255), 1)
    return out


def ask_claude(image, w, h, model):
    cmd = ["claude", "-p", PROMPT.format(image=image, w=w, h=h, rules=HOUSE_RULES), "--allowedTools", "Read",
           "--output-format", "json", "--json-schema", json.dumps(SCHEMA)]
    if model:
        cmd += ["--model", model]
    start = time.monotonic()
    for attempt in range(1, ATTEMPTS + 1):
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, cwd=Path(image).parent)
        except subprocess.TimeoutExpired:
            error = f"no answer within {TIMEOUT_S} s"
            continue
        if done.returncode == 0:
            reply = json.loads(done.stdout)
            if reply.get("structured_output"):
                return reply["structured_output"]["objects"], time.monotonic() - start, reply.get("total_cost_usd")
            error = f"no structured output: {str(reply.get('result'))[:200]}"
        else:
            error = done.stderr[-500:]
    raise RuntimeError(f"claude -p failed after {ATTEMPTS} attempts: {error}")


def refine(img, bbox, desk_lab, pad=15):
    """Largest non-desk blob inside the (padded) box: pixel centre and long-axis ends, or None."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    lab = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(float)
    diff = np.linalg.norm(lab - desk_lab, axis=2)
    mask = (diff > 18).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, 4]))
    if stats[i, 4] < 60:
        return None
    ys, xs = np.nonzero(labels == i)
    pts = np.stack([xs, ys], 1).astype(float) + [x0, y0]
    centre = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    ends = []
    for direction in vt:  # long axis, then short axis
        proj = (pts - centre) @ direction
        ends.append((centre + direction * proj.min(), centre + direction * proj.max()))
    return {"centre": centre, "long": ends[0], "short": ends[1], "pixels": pts.astype(np.int32)}


def measure(calib, blob, z=0.01):
    """Robot-frame centre, long-axis vector and length/width (m) of a refined blob."""
    xy = lambda px: handeye_solve.image_to_base(calib, px, z)[:2]
    (l1, l2), (s1, s2) = blob["long"], blob["short"]
    return {"xy": xy(blob["centre"]), "axis": xy(l2) - xy(l1),
            "length": float(np.linalg.norm(xy(l2) - xy(l1))), "width": float(np.linalg.norm(xy(s2) - xy(s1)))}


def pixels_to_base(calib, pixels, heights):
    """Robot-frame XY of each image pixel at its own measured height (m), vectorised image_to_base."""
    fx, fy, cx, cy, k1, k2 = calib["camera"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    rays = cv2.undistortPoints(np.asarray(pixels, float).reshape(-1, 1, 2), K, np.array([k1, k2, 0, 0])).reshape(-1, 2)
    T_bc = np.linalg.inv(np.array(calib["T_cam_base"]))
    origin = T_bc[:3, 3]
    directions = (T_bc[:3, :3] @ np.c_[rays, np.ones(len(rays))].T).T
    t = (np.asarray(heights, float) - origin[2]) / directions[:, 2]
    return origin[:2] + t[:, None] * directions[:, :2]


def measure_3d(calib, pixels, heights):
    """Footprint of an object from a depth camera: every visible surface point placed where it really is.

    The slanted camera sees a tall object's top and one side; measured on one plane they smear toward
    the camera (sim 2026-10-28: an 8 cm bottle aimed 8 mm off, the finger landed on its cap). With each
    pixel's own height the centre is the middle of the top surface (a bottle's cap, a cup's rim) and the
    size is the spread of all points. None if there are too few readings.
    """
    heights = np.asarray(heights, float)
    if len(heights) < 10:
        return None
    xy = pixels_to_base(calib, pixels, heights)
    top = float(np.percentile(heights, 90))
    # The top surface only: the camera sees one side of a tall thing, which would pull the centre its way.
    upper = xy[heights >= top - max(0.004, 0.06 * top)]
    centre = upper.mean(0) if len(upper) >= 5 else xy.mean(0)
    d = xy - xy.mean(0)
    _, vecs = np.linalg.eigh(d.T @ d)
    long_axis, short_axis = vecs[:, 1], vecs[:, 0]
    # Nearly the whole spread: 2-98 % of a round thing is only 0.89 of its diameter (sim 2026-10-28: a
    # 10.2 cm mug read 9.0 cm and the fingers came down on its rim); 0.5-99.5 % is 0.96 of it.
    extent = lambda v: float(np.subtract(*np.percentile(d @ v, [99.5, 0.5])))
    return {"xy": centre, "axis": long_axis * extent(long_axis), "length": extent(long_axis),
            "width": extent(short_axis), "top": top}


def ask_image(scene, out, model="sonnet"):
    """claude -p on one overhead frame: objects with bboxes in full-resolution pixels."""
    img = cv2.imread(str(scene))
    scale = SENT_WIDTH / img.shape[1]
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    grid = Path(out) / "scene_grid.jpg"
    cv2.imwrite(str(grid), gridded(small))
    objects, seconds, cost = ask_claude(grid.resolve(), small.shape[1], small.shape[0], model or None)
    for o in objects:
        o["bbox_sent"] = o["bbox"]
        o["bbox"] = [int(round(v / scale)) for v in o["bbox"]]
    return objects, {"seconds": seconds, "cost_usd_equivalent": cost}


DETECT_SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"},
                       "bbox": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}},
        "required": ["name", "bbox"]}}},
    "required": ["objects"],
}
DETECT_PROMPT = """Read the image {image} ({w}x{h} px, pixel grid every 100 px). It is an overhead photo of
a desk; the robot arm's base is at the top centre. List every separate loose object on the desk except the
robot arm itself: name (short, Japanese) and bbox [x0, y0, x1, y1] in this image. Separate touching objects
when you can, but the parts of one object (a magnifier's handle and lens, a mug and its handle, a toy's body
and wheels) are one object with one bbox. Nothing else."""
CROP_SCHEMA = {"type": "object", "properties": {
    k: v for k, v in SCHEMA["properties"]["objects"]["items"]["properties"].items() if k not in ("bbox", "name")},
    "required": [k for k in SCHEMA["properties"]["objects"]["items"]["required"] if k not in ("bbox", "name")]}
CROP_PROMPT = """Read the image {image}. It is a crop of an overhead photo of a desk tidied by a 6-axis robot
arm with a two-finger parallel gripper; it shows one object, which a detector called "{name}", about
{length:.0f} x {width:.0f} cm seen from above. You are the robot's judge for this one object.
Answer in Japanese for the text fields, and keep every *_reason and grasp_point under 25 characters.

{rules}

Give: kind (what sort of thing it is, one short general Japanese word: ペン, 紙くず, 消しゴム, 箱, 駒 ...),
category (trash / keep / unsure / fixture), category_reason, should_grasp, grasp_reason,
deformable (squashes when gripped), force 1-5 (1 tissue, 2 crumpled paper, 3 pen or piece, 4 firm,
5 heavy) and force_reason, grasp_point, size_cm [length, width, height], confidence 0-1 (0.9 or more
only when obvious), ask_human (true for unclear cases).
should_grasp is false only for precision devices, fragile-looking glassware, flat sheets thinner than 5 mm,
things tied down, and things with liquid or food visible in them. Never false because of size or weight:
the robot measures size with a depth camera, pinches a rim, wall, edge or narrow part of wide things, and
reports itself what it cannot hold. A closed or opaque bottle, pot or jar is true.
{advice}"""
WORKERS = 8
OVERLOOK_STEP = 4          # px between desk samples searched for things the listing missed
OVERLOOK_R = (0.14, 0.60)  # m from the base: the desk the robot works on (its own base excluded)
OVERLOOK_ANGLE = 60.0      # deg either side of straight ahead
OVERLOOK_HEIGHT = 0.008    # m above the desk: something stands there
OVERLOOK_GROW = 15         # px a listed box is grown by (shadows and edges belong to it)
OVERLOOK_MIN_CELLS = 15    # samples (~240 px at 4 px): smaller is an edge or noise


def person_said():
    """What the person has told about things on this desk (a file, read each time): facts the photo cannot show."""
    path = Path(os.environ.get("TIDY_PERSON_NOTES", ""))
    return f"The person who owns this desk said: {path.read_text().strip()}\n\n" if path.is_file() else ""


def _claude(prompt, schema, model, cwd, timeout=TIMEOUT_S):
    cmd = ["claude", "-p", prompt, "--allowedTools", "Read", "--output-format", "json",
           "--json-schema", json.dumps(schema)] + (["--model", model] if model else [])
    for _ in range(ATTEMPTS):
        try:
            # No extended thinking: a crop verdict took 29 s with it and 15 s without (2026-11-01, same answer).
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd,
                                  env={**os.environ, "MAX_THINKING_TOKENS": "0"})
        except subprocess.TimeoutExpired:
            continue
        if done.returncode == 0:
            reply = json.loads(done.stdout)
            if reply.get("structured_output"):
                return reply["structured_output"]
    raise RuntimeError("claude -p gave no answer")


def ask_fast(scene, out, calib, model="sonnet", advice="", recall=None, missed=None):
    """Two stages: one short claude -p call lists objects and boxes, then every object's crop is
    judged by its own claude -p call, all in parallel (2026-10-28: 4-5 min -> well under a minute).

    advice: lessons from the experience notebook, added to every crop prompt.
    recall(name, crop_bgr, size_cm) -> a remembered verdict or None: a known object is not asked again.
    missed(listed boxes, full-image px) -> boxes of things the listing missed, judged like the rest.
    """
    from concurrent.futures import ThreadPoolExecutor
    out = Path(out)
    start = time.monotonic()
    img = cv2.imread(str(scene))
    scale = SENT_WIDTH / img.shape[1]
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    grid = out / "scene_grid.jpg"
    cv2.imwrite(str(grid), gridded(small))
    found = _claude(DETECT_PROMPT.format(image=grid.resolve(), w=small.shape[1], h=small.shape[0]),
                    DETECT_SCHEMA, model, out)["objects"]
    for bbox in (missed([[v / scale for v in o["bbox"]] for o in found]) if missed else []):
        found.append({"name": "見落としていた物", "bbox": [v * scale for v in bbox]})
    detect_s = time.monotonic() - start
    jobs = []
    for k, o in enumerate(found):
        bbox = [int(round(v / scale)) for v in o["bbox"]]
        x0, y0, x1, y1 = bbox
        pad = 25
        crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        if crop.size == 0:
            continue
        path = (out / f"crop_{k}.jpg").resolve()
        cv2.imwrite(str(path), crop)
        corners = [handeye_solve.image_to_base(calib, p, 0.0)[:2] for p in ((x0, y0), (x1, y0), (x1, y1))]
        size = sorted([np.linalg.norm(corners[1] - corners[0]) * 100, np.linalg.norm(corners[2] - corners[1]) * 100],
                      reverse=True)
        jobs.append((o["name"], bbox, path, size))

    def judge(job):
        name, bbox, path, size = job
        known = recall(name, cv2.imread(str(path)), size) if recall else None
        if known:
            return {"name": name, "bbox": bbox, **known, "recalled": True}
        try:
            verdict = _claude(person_said() + CROP_PROMPT.format(image=path, name=name, length=size[0], width=size[1],
                                                 rules=HOUSE_RULES, advice=advice), CROP_SCHEMA, model, out)
        except RuntimeError:
            verdict = {"kind": name, "category": "unsure", "category_reason": "判定できず", "should_grasp": False,
                       "grasp_reason": "判定できず", "deformable": False, "force": 2, "force_reason": "",
                       "grasp_point": "", "size_cm": [size[0], size[1], 2.0], "confidence": 0.0, "ask_human": True}
        return {"name": name, "bbox": bbox, **verdict}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        objects = list(pool.map(judge, jobs))
    return objects, {"seconds": time.monotonic() - start, "detect_seconds": detect_s, "objects": len(objects),
                     "recalled": sum(1 for o in objects if o.get("recalled"))}


def overlooked(height, calib, boxes, hide=None):
    """Boxes (full-image px) round things standing on the desk (depth camera) that lie in no listed box:
    what the one-shot listing missed (sim 2026-11-01: an 11 x 2 cm honey dipper next to a big box was
    never listed, so never tidied). Desk colour alone cannot tell (a pale stick on a pale desk; the
    real desk's grain and the robot's cable read as things).

    height: per-pixel height above the desk (m); hide: mask of pixels to ignore (the arm). Only the
    reachable desk is searched, the robot's own base excluded; sampled every OVERLOOK_STEP px.
    """
    h, w = height.shape[:2]
    step = OVERLOOK_STEP
    vs, us = np.mgrid[step // 2:h:step, step // 2:w:step]
    xy = np.array([handeye_solve.image_to_base(calib, (u, v), 0.0)[:2] for u, v in zip(us.ravel(), vs.ravel())])
    r, ang = np.hypot(xy[:, 0], xy[:, 1]), np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    desk = ((r > OVERLOOK_R[0]) & (r < OVERLOOK_R[1]) & (np.abs(ang) < OVERLOOK_ANGLE)).reshape(vs.shape)
    hs = height[vs, us]
    standing = desk & (hs > OVERLOOK_HEIGHT) & (hs < 0.4)
    if hide is not None:
        standing &= hide[vs, us] == 0
    for x0, y0, x1, y1 in boxes:
        g = OVERLOOK_GROW
        standing &= ~((us >= x0 - g) & (us <= x1 + g) & (vs >= y0 - g) & (vs <= y1 + g))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(standing.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, 4] < OVERLOOK_MIN_CELLS:
            continue
        x, y, bw, bh = stats[i, :4]
        out.append([int(x * step), int(y * step), int(min(w - 1, (x + bw) * step)), int(min(h - 1, (y + bh) * step))])
    return out


def blob_from_pixels(pts):
    """Centre and long/short axis ends of a set of object pixels (as refine() returns)."""
    pts = np.asarray(pts, float)
    centre = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    ends = []
    for direction in vt:
        proj = (pts - centre) @ direction
        ends.append((centre + direction * proj.min(), centre + direction * proj.max()))
    return {"centre": centre, "long": ends[0], "short": ends[1], "pixels": pts.astype(np.int32)}


def locate_all(img, objects, calib, masks=None):
    """Add centre_px, xy, axis_xy, size_m and reach_m to each judged object (in place).

    masks (from vision_seg.py, SAM) give each object's own pixels; without one the
    object is found by its colour difference from the desk.
    """
    desk_lab = np.median(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0)  # desk fills most of the frame
    for k, o in enumerate(objects):
        mask = masks[k] if masks is not None and k < len(masks) else None
        blob = blob_from_pixels(mask["pixels"]) if mask else refine(img, o["bbox"], desk_lab)
        o["centre_px"] = None if blob is None else [round(float(v), 1) for v in blob["centre"]]
        o["xy"] = None
        if blob is not None:
            o["_pixels"] = blob["pixels"]  # the object's own pixels, for before/after comparisons
            # SAM's pixels are a sample of at most 4000: the area is the whole mask's.
            o["_area"] = int(mask.get("area") or len(blob["pixels"])) if mask else len(blob["pixels"])
            # A tall object's outline seen from the slanted camera includes its sides and its top is
            # nearer the camera; measuring at a good part of its height keeps the footprint honest
            # (2026-10-28: a 12 cm tall box read 14.5 cm wide at desk level).
            height = float(np.clip(o.get("size_cm", [0, 0, 1.0])[2] / 100 * 0.8, 0.01, 0.15))
            m = measure(calib, blob, z=height)
            o["xy"] = [round(float(v), 4) for v in m["xy"]]
            o["axis_xy"] = [round(float(v), 4) for v in m["axis"]]
            o["size_m"] = [round(m["length"], 3), round(m["width"], 3)]
            o["size_from_depth"] = False  # one plane's guess, until the depth camera measures it again
        o["reach_m"] = None if o["xy"] is None else round(float(np.hypot(*o["xy"])), 3)
    return objects


def presence(img, pixels):
    """Fraction of an object's former pixels that still differ from the desk colour in img."""
    desk_lab = np.median(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[pixels[:, 1], pixels[:, 0]].astype(float)
    return float((np.linalg.norm(lab - desk_lab, axis=1) > 18).mean())


def public(objects):
    """Objects without the private (non-JSON) fields."""
    return [{k: v for k, v in o.items() if not k.startswith("_")} for o in objects]


def draw_overlay(img, objects, path):
    overlay = img.copy()
    colours = {"trash": (0, 0, 255), "keep": (0, 180, 0), "unsure": (0, 200, 255), "fixture": (160, 160, 160)}
    for o in objects:
        x0, y0, x1, y1 = o["bbox"]
        c = colours[o["category"]]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), c, 2)
        if o.get("centre_px"):
            cv2.circle(overlay, tuple(int(v) for v in o["centre_px"]), 4, c, -1)
    # Japanese labels need a CJK font, which cv2.putText cannot draw.
    pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw, font = ImageDraw.Draw(pil), ImageFont.truetype(FONT, 18)
    for k, o in enumerate(objects, 1):
        b, g, r = colours[o["category"]]
        x0, y0 = o["bbox"][:2]
        label = f"{k} {LABELS_JA[o['category']]}：{o['name']}" + (f" 力{o['force']}" if o["should_grasp"] else " ×")
        draw.text((x0, max(0, y0 - 24)), label, font=font, fill=(r, g, b), stroke_width=2, stroke_fill=(255, 255, 255))
    pil.save(path, quality=92)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--image", help="use this frame instead of capturing video2")
    p.add_argument("--calib", default=str(HERE / "calib/video2.json"))
    p.add_argument("--model", default="sonnet", help="claude -p model alias ('' = default)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    scene = a.out / "scene.jpg"
    if a.image:
        scene.write_bytes(Path(a.image).read_bytes())
    else:
        capture(scene)
    img = cv2.imread(str(scene))
    objects, meta = ask_image(scene, a.out, a.model)
    locate_all(img, objects, json.load(open(a.calib)))
    draw_overlay(img, objects, a.out / "overlay.jpg")
    (a.out / "judge.json").write_text(json.dumps({**meta, "objects": public(objects)}, indent=1, ensure_ascii=False))
    print(f"claude -p: {len(objects)} objects in {meta['seconds']:.0f} s")
    for k, o in enumerate(objects, 1):
        where = "" if o["xy"] is None else f" at ({o['xy'][0]:+.3f}, {o['xy'][1]:+.3f}) {o['reach_m']} m"
        grasp = f"掴む 力{o['force']}" if o["should_grasp"] else "掴まない"
        print(f"{k:2d} {LABELS_JA[o['category']]} {grasp} {o['name']}{where}\n"
              f"     分類: {o['category_reason']} / 掴む: {o['grasp_reason']} / 力: {o['force_reason']}"
              f" / 位置: {o['grasp_point']} / 確信 {o['confidence']:.2f}{' / 人に確認' if o['ask_human'] else ''}")


if __name__ == "__main__":
    main()
