#!/usr/bin/env python3
"""Summon the desk into the simulation: one Hunyuan3D model per object seen by the top camera.

Input is a look at the desk: a folder with scene.jpg, judge.json, sam_boxes.json, sam_masks.json
(what `tidy_desk.py --preview` writes in OUT/preview, or round1 of a real run). Output, in --out:

    assets/<kind>/{visual.obj,visual.mtl,texture.png,collision_N.obj,meta.json}   read through $TIDY_EXTRA_OBJECTS
    world.json        the twin desk, settled by physics
    twin_top.jpg      the twin as the top camera sees it, side by side with the photo
    summon.json       per object: how it was made, silhouette IoU, fallback or not

    source /opt/ros/humble/setup.bash
    MUJOCO_GL=egl /usr/bin/python3 summon.py --look LOOK_DIR --out OUT

Steps per object: RGBA crop from the mask -> Hunyuan3D shape (summon_worker.py, Hunyuan venv, model loaded
once) -> orientation -> scale to the footprint and height -> flat base -> texture from the photo -> CoACD.

Orientation is searched, not assumed. Tried on a real photo (2026-11-01): a marker lying diagonally in the
input came back as a rod along the model's Z (1.99 long, 0.28 wide), so Hunyuan's output axes do not follow the
image; every object gets the rotation whose silhouette best matches its mask. The camera is also not straight
above (16 deg off, more towards the desk's edges): the view axis goes to the camera ray at the object, not to z.

A failed generation (or a silhouette IoU below MIN_IOU) falls back to the extruded outline, logged in Japanese
and marked in summon.json and meta.json ("method").
"""
import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mj_assets/tools"))
HUNYUAN_PY = os.environ.get("HUNYUAN_PY", os.path.expanduser("~/Hunyuan3D-2.1/.venv/bin/python"))
GPU = os.environ.get("TIDY_SAM_GPU", "0")   # which GPU Hunyuan3D uses
SITE = Path.home() / ".cache/rebot_summon/site"   # coacd lives here (pip --target), not in a venv
GPU_LOCK = Path.home() / ".cache/rebot_summon/gpu.lock"   # one Hunyuan at a time
INPUT_SIDE = 512     # px, long side of the crop handed to Hunyuan
TEXTURE_PAD = 12     # px of photo kept round the mask
HOLE_COLOUR = 14.0   # Lab distance: a patch inside the outline this near the desk colour is a hole
HOLE_SHARE = 0.06    # of the outline's area, at least
RING_POINTS = 48     # round an extruded ring
MIN_IOU = 0.45       # below this the mesh does not look like the photo: extrude the outline instead
BASE_CUT = 0.05      # share of the height cut flat at the bottom (Hunyuan never saw the underside)
CUP_BASE_CUT = 0.015
MIN_HEIGHT = 0.008   # m
SEARCH_GRID = 48     # px, silhouette grid of the orientation search
VIEWS, ROLLS = 240, 36   # view directions on the sphere x turns about the view
RIGID_DENSITY, SOFT_DENSITY = 600.0, 60.0   # kg/m3 of the hull: a guess from the photo (a 15 cm marker: 15 g)
DENSITY = ((("ペン", "マーカー"), 320.0), (("金属", "缶", "はさみ", "ハサミ", "ハンマー", "ドライバー", "工具"), 1500.0), (("マグ", "カップ", "コップ", "器", "ボウル", "皿", "鉢"), 450.0),
           (("箱", "パッケージ", "ケース"), 150.0), (("ボトル", "瓶"), 400.0))
SHADOW_CHROMA = 7.0  # LAB a/b distance from the desk under which a darker pixel is the desk in shadow
SHADOW_LIGHT = (0.45, 0.97)  # lightness relative to the desk of that shadow
FRICTION_RIGID, FRICTION_SOFT = 0.5, 0.9
COACD_THRESHOLD = 0.04
MAX_PARTS = 16
CONTAINER_WORDS = ("器", "ボウル", "椀", "カップ", "コップ", "マグ", "皿", "鉢", "ラメキン", "かご", "カゴ", "バケツ", "トレー")


def say(text):
    print(text, flush=True)


# ------------------------------------------------------------------ alignment maths (numpy only, tested)
def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def camera_position(T_cam_base):
    T = np.asarray(T_cam_base, float)
    return -T[:3, :3].T @ T[:3, 3]


def view_axes(T_cam_base, point):
    """(right, up, toward) unit vectors in the base frame, for the camera ray through point: what the image's
    right and up and the direction back to the camera are at that object. Right-handed: right x up = toward."""
    T = np.asarray(T_cam_base, float)
    toward = camera_position(T) - np.asarray(point, float)
    toward /= np.linalg.norm(toward)
    right = T[0, :3] - toward * (T[0, :3] @ toward)
    right /= np.linalg.norm(right)
    return right, np.cross(toward, right), toward


def view_to_object(V_view, axes, yaw):
    """Points in (right, up, toward) coordinates -> base orientation -> object frame turned by -yaw about z."""
    right, up, toward = axes
    base = V_view[:, [0]] * right + V_view[:, [1]] * up + V_view[:, [2]] * toward
    return base @ rot_z(yaw)    # row-vector form of rot_z(-yaw) @ p


def scale_to(V, size):
    """Scale each axis so the extents are size (x, y, z); x and y centred on 0, the lowest point at z = 0."""
    V = np.asarray(V, float)
    lo, hi = V.min(0), V.max(0)
    s = np.asarray(size, float) / np.maximum(hi - lo, 1e-9)
    out = (V - (lo + hi) / 2) * s
    out[:, 2] -= out[:, 2].min()
    return out, s


def flat_base(V, share=BASE_CUT):
    """Cut the bottom share of the height flat (lowest point 0 again): the model's underside is a guess, and
    a rounded one would rock on the desk."""
    V = np.array(V, float)
    h = V[:, 2].max()
    V[:, 2] = np.maximum(V[:, 2], share * h) - share * h
    return V


def yaw_from_axis(axis_xy):
    return math.atan2(axis_xy[1], axis_xy[0])


def lying_frame(V):
    """Rotation (rows, for V @ R.T) that lays a lying thing level, from its principal axes. The silhouette says
    nothing about tilt towards the camera: the real marker came out up to tens of degrees off, and squeezing that to
    the height sheared it thin and stood it on one end (2026-11-01). An elongated thing gets its long axis on x; a
    flat one only has its thinnest axis turned to z, keeping its turn about z from the photo."""
    C = V - V.mean(0)
    w, e = np.linalg.eigh(C.T @ C)      # ascending
    z, x = e[:, 0], e[:, 2]
    z = z if z[2] >= 0 else -z
    if w[2] > 2.25 * w[1]:              # extent ratio about 1.5
        x = x if x[0] >= 0 else -x
        y = np.cross(z, x)
        return np.array([x, y / np.linalg.norm(y), z])
    v, c = np.cross(z, [0.0, 0.0, 1.0]), z[2]    # the smallest turn taking z to +z
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def rim_frame(V, band=0.15, rounds=6):
    """Rotation (rows, for V @ R.T) that levels a cup by its rim: a plane through its top band, taken again until it
    stays. Its principal axes would lay it on its side, and a mug seen from above comes out leaning (33 deg, 2026-11-01)."""
    R = np.eye(3)
    for _ in range(rounds):
        O = V @ R.T
        top = O[O[:, 2] > O[:, 2].max() - band * np.ptp(O[:, 2])]
        n = np.linalg.svd(top - top.mean(0), full_matrices=False)[2][2]
        n = n if n[2] >= 0 else -n
        v, c = np.cross(n, [0.0, 0.0, 1.0]), n[2]
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = (np.eye(3) + K + K @ K / (1 + c)) @ R
    return R


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z, phi = 1 - 2 * i / n, math.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.c_[r * np.cos(phi), r * np.sin(phi), z]


def rotation_grid(views=VIEWS, rolls=ROLLS):
    """Rotations R (n, 3, 3): model points p -> R @ p = (right, up, toward). Column-free: rows are the axes."""
    out = []
    for v in fibonacci_sphere(views):
        a = np.cross(v, [0.0, 0.0, 1.0] if abs(v[2]) < 0.9 else [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(v, a)
        for t in np.linspace(0, 2 * math.pi, rolls, endpoint=False):
            r = math.cos(t) * a + math.sin(t) * b
            out.append([r, np.cross(v, r), v])
    return np.array(out)


def occupancy(P, grid):
    """Silhouettes of point sets P (n_rot, n_pts, 2; y up), each fitted to the grid keeping its aspect: bool
    (n_rot, grid, grid), one pixel dilated so that points make a closed shape."""
    lo, hi = P.min(1, keepdims=True), P.max(1, keepdims=True)
    s = 0.9 * grid / np.maximum((hi - lo).max(2, keepdims=True), 1e-9)
    q = (P - (lo + hi) / 2) * s + grid / 2
    ix, iy = np.clip(q[..., 0].astype(int), 0, grid - 1), np.clip((grid - q[..., 1]).astype(int), 0, grid - 1)
    n = len(P)
    occ = np.zeros((n, grid, grid), bool)
    occ[np.arange(n)[:, None], iy, ix] = True
    grown = occ.copy()
    grown[:, 1:, :] |= occ[:, :-1, :]
    grown[:, :-1, :] |= occ[:, 1:, :]
    grown[:, :, 1:] |= occ[:, :, :-1]
    grown[:, :, :-1] |= occ[:, :, 1:]
    return grown


def target_grid(mask, grid):
    """The mask (bool image) as a silhouette on the search grid, fitted like occupancy()."""
    ys, xs = np.nonzero(mask)
    P = np.c_[xs, -ys].astype(float)[None]
    if P.shape[1] > 20000:
        P = P[:, np.random.default_rng(0).choice(P.shape[1], 20000, replace=False)]
    return occupancy(P, grid)[0]


def best_rotation(V, mask, grid=SEARCH_GRID, points=2500):
    """The rotation of the model whose silhouette best matches the mask: (R, IoU on the search grid)."""
    rng = np.random.default_rng(0)
    pts = V if len(V) <= points else V[rng.choice(len(V), points, replace=False)]
    pts = pts - (pts.min(0) + pts.max(0)) / 2
    target = target_grid(mask, grid)

    def score(Rs):
        ious = []
        for k in range(0, len(Rs), 400):
            occ = occupancy(np.einsum("nij,pj->npi", Rs[k:k + 400, :2], pts), grid)
            ious.append((occ & target).sum((1, 2)) / np.maximum((occ | target).sum((1, 2)), 1))
        return np.concatenate(ious)

    Rs = rotation_grid()
    iou = score(Rs)
    best = Rs[int(iou.argmax())]
    top = float(iou.max())
    # polish: small turns about the three axes round the best one
    for step in (math.radians(6), math.radians(2)):
        for _ in range(3):
            d = np.array([-step, 0.0, step])
            turns = np.array([_small(a, b, c) for a in d for b in d for c in d])
            cand = turns @ best
            sc = score(cand)
            if sc.max() > top:
                top, best = float(sc.max()), cand[int(sc.argmax())]
    return best, top


def _small(a, b, c):
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz @ ry @ rx


def density(o):
    """kg/m3 of the convex hull for what the judge says it is (a hull's volume, so hollow things count light)."""
    text = f"{o.get('name', '')}{o.get('kind', '')}"
    if o.get("deformable"):
        return SOFT_DENSITY
    for words, rho in DENSITY:
        if any(w in text for w in words):
            return rho
    return RIGID_DENSITY


def mass_guess(hull_volume, rho):
    """kg: volume times density; the twin only has the photo."""
    return float(np.clip(hull_volume * rho, 0.002, 0.8))


def shape_height(judged_h, L, W, soft):
    """m: a long thing is about as thick as it is wide (a judge's 2 cm marker is 1.5 cm wide)."""
    h = float(judged_h)
    if not soft and L > 3 * W:
        h = min(h, W)
    return float(np.clip(h, MIN_HEIGHT, 0.30))


def is_container(o):
    return any(w in f"{o.get('name', '')}{o.get('kind', '')}" for w in CONTAINER_WORDS)


# ------------------------------------------------------------------ the photo
def load_look(look):
    """(image, judged objects with their mask image) from a look folder; masks found by box, not by position
    in the list: tidy_desk drops things from the judge's list after SAM ran."""
    import cv2
    look = Path(look)
    img = cv2.imread(str(look / "scene.jpg"))
    objects = json.loads((look / "judge.json").read_text())["objects"]
    boxes = json.loads((look / "sam_boxes.json").read_text())
    masks = json.loads((look / "sam_masks.json").read_text())
    out = []
    for o in objects:
        if o.get("category") == "fixture" or o.get("xy") is None:
            continue
        k = next((i for i, b in enumerate(boxes) if [int(v) for v in b] == [int(v) for v in o["bbox"]]), None)
        m = masks[k] if k is not None and k < len(masks) else None
        if not m:
            say(f"  {o['name']}: 輪郭がないので飛ばす")
            continue
        px = np.array(m["pixels"])
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[px[:, 1], px[:, 0]] = 255
        # SAM's pixels are a sample of at most 4000: close the gaps between them
        side = 2 * int(math.sqrt(m.get("area", len(px)) / len(px))) + 3
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side)))
        out.append((o, without_shadow(img, mask > 0)))
    return img, out


def without_shadow(img, mask):
    """The mask less the desk in shadow beside the thing: SAM takes it in (a real 15 mm marker read 26 mm wide).
    Kept as it was if that would remove most of it (a grey thing on a grey desk)."""
    import cv2
    ys, xs = np.nonzero(mask)
    pad = 20
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
    lab = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(float)
    m = mask[y0:y1, x0:x1]
    ring = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if ring.sum() < 20:
        return mask
    desk = np.median(lab[ring], axis=0)
    chroma = np.linalg.norm(lab[..., 1:] - desk[1:], axis=-1)
    shadow = m & (chroma < SHADOW_CHROMA) & (lab[..., 0] < SHADOW_LIGHT[1] * desk[0]) & (lab[..., 0] > SHADOW_LIGHT[0] * desk[0])
    left = (m & ~shadow).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(left, connectivity=4)
    if n < 2:
        return mask
    keep = labels == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    if keep.sum() < 0.5 * m.sum():
        return mask
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = keep
    return out


def footprint(mask, calib, axis_xy, z):
    """(length, width, centre) of a mask on the plane at height z, along the judged long axis."""
    import cv2
    import handeye_solve
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    xy = np.array([handeye_solve.image_to_base(calib, p, z)[:2] for p in c])
    u = np.asarray(axis_xy[:2], float) / np.linalg.norm(axis_xy[:2])
    v = np.array([-u[1], u[0]])
    a, b = xy @ u, xy @ v
    return float(np.ptp(a)), float(np.ptp(b)), ((a.min() + a.max()) / 2 * u + (b.min() + b.max()) / 2 * v).tolist()


def rgba_crop(img, mask, path):
    """The object cut out of the photo, long side INPUT_SIDE, for Hunyuan."""
    import cv2
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bgr, alpha = img[y0:y1, x0:x1], (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    s = INPUT_SIDE / max(bgr.shape[:2])
    bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    alpha = cv2.resize(alpha, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(path), np.dstack([bgr, alpha]))


def run_hunyuan(jobs, folder):
    """All the shapes in one process. Returns True if the worker ran; a job's own failure is seen by its file."""
    path = folder / "jobs.json"
    path.write_text(json.dumps(jobs))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "AMENT_PREFIX_PATH")}
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=GPU)
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(GPU_LOCK, "w") as lock, open(folder / "hunyuan.log", "w") as log:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            proc = subprocess.Popen([HUNYUAN_PY, str(HERE / "summon_worker.py"), str(path)], stdout=subprocess.PIPE,
                                    stderr=log, text=True, env=env)
        except OSError:   # no Hunyuan3D here: every object falls back to its extruded outline
            say("  Hunyuan3D がないので、輪郭の押し出しで作る")
            return False
        done = 0
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if line.startswith("MODEL_LOADED"):
                say("  Hunyuan3D を読み込んだ")
            elif line.startswith(("SHAPE_DONE", "SHAPE_FAIL")):
                done += 1
                say(f"  3Dモデル生成 {done}/{len(jobs)}" + ("（失敗）" if line.startswith("SHAPE_FAIL") else ""))
        return proc.wait() == 0


# ------------------------------------------------------------------ one object
def project(calib, xyz):
    import sim_world
    return sim_world._project(calib, xyz)


def place_points(V, centre, yaw):
    return (V @ rot_z(yaw).T) + np.array([centre[0], centre[1], 0.0])


def silhouette(calib, V, F, centre, yaw, shape):
    import cv2
    px = np.round(project(calib, place_points(V, centre, yaw))).astype(np.int32)
    im = np.zeros(shape, np.uint8)
    cv2.fillPoly(im, [t.reshape(-1, 1, 2) for t in px[F]], 255)
    return im > 0


def fit_to_mask(calib, V, F, centre, yaw, mask):
    """Move and stretch the model until its silhouette in the photo covers the mask: the measured footprint is
    taken on one plane, and the slanted camera sees a tall thing's side too. Returns (V, centre, IoU)."""
    V, centre = np.array(V), np.array(centre, float)
    ys, xs = np.nonzero(mask)
    M = np.c_[xs, ys].astype(float)
    for _ in range(4):
        sil = silhouette(calib, V, F, centre, yaw, mask.shape)
        sy, sx = np.nonzero(sil)
        if len(sx) < 10:
            break
        S = np.c_[sx, sy].astype(float)
        p0 = project(calib, place_points(np.zeros((1, 3)), centre, yaw))[0]
        ax = project(calib, place_points(np.array([[0.01, 0, 0]]), centre, yaw))[0] - p0
        ay = project(calib, place_points(np.array([[0, 0.01, 0]]), centre, yaw))[0] - p0
        offset = np.zeros(2)
        for i, a in enumerate((ax, ay)):
            pxm = np.linalg.norm(a) / 0.01     # px per m along this axis
            u = a / np.linalg.norm(a)
            sm, ss = M @ u, S @ u
            V[:, i] *= float(np.clip(np.ptp(sm) / max(np.ptp(ss), 1.0), 0.8, 1.25))
            d = ((sm.min() + sm.max()) - (ss.min() + ss.max())) / 2 / pxm
            offset += d * np.array([math.cos(yaw), math.sin(yaw)] if i == 0 else [-math.sin(yaw), math.cos(yaw)])
        centre += offset
    sil = silhouette(calib, V, F, centre, yaw, mask.shape)
    return V, centre, float((sil & mask).sum() / max(1, (sil | mask).sum()))


def texture_from_photo(img, mask, calib, V, centre, yaw, d):
    """Planar UV: every vertex takes the pixel the camera sees it at. The crop's outside is filled with the
    object's own colours so that a vertex just off the mask does not turn desk-coloured."""
    import cv2
    px = project(calib, place_points(V, centre, yaw))
    h, w = img.shape[:2]
    ys, xs = np.nonzero(mask)
    x0, x1 = max(0, int(min(xs.min(), px[:, 0].min())) - TEXTURE_PAD), min(w, int(max(xs.max(), px[:, 0].max())) + TEXTURE_PAD + 1)
    y0, y1 = max(0, int(min(ys.min(), px[:, 1].min())) - TEXTURE_PAD), min(h, int(max(ys.max(), px[:, 1].max())) + TEXTURE_PAD + 1)
    crop, inner = img[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
    eroded = cv2.erode(inner.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)   # the outline's pixels are half desk
    if eroded.sum() < 20:
        eroded = inner
    # nearest object pixel for everything else
    _, labels = cv2.distanceTransformWithLabels((~eroded).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    table = np.zeros((labels.max() + 1, 3), np.uint8)
    table[labels[eroded]] = crop[eroded]
    crop[~eroded] = table[labels[~eroded]]
    # the simulated camera applies the real one's white balance again: take it out of the photo
    gains = json.loads((HERE / "mj_assets/camera_gains.json").read_text()).get("video2") or [1, 1, 1]
    cv2.imwrite(str(d / "texture.png"), np.clip(crop.astype(float) / np.array(gains), 0, 255).astype(np.uint8))
    th, tw = crop.shape[:2]
    uv = np.c_[(px[:, 0] - x0 + 0.5) / tw, 1 - (px[:, 1] - y0 + 0.5) / th]
    bgr = [int(v) for v in img[mask].mean(0)]
    return np.clip(uv, 0, 1), bgr


def outline_prism(mask, calib, centre, yaw, height):
    """The fallback: the mask's outline extruded, as (V, F) in the object frame."""
    import cv2
    import handeye_solve
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.5, True)[:, 0, :].astype(float)
    xy = np.array([handeye_solve.image_to_base(calib, p, height)[:2] for p in c])
    local = (xy - np.asarray(centre)) @ rot_z(yaw)[:2, :2]
    n = len(local)
    V = np.vstack([np.c_[local, np.zeros(n)], np.c_[local, np.full(n, height)],
                   [[*local.mean(0), 0.0]], [[*local.mean(0), height]]])
    area = np.sum(local[:, 0] * np.roll(local[:, 1], -1) - np.roll(local[:, 0], -1) * local[:, 1])
    order = range(n) if area > 0 else range(n - 1, -1, -1)
    order = list(order)
    F = []
    for a, b in zip(order, order[1:] + order[:1]):
        F += [[a, b, n + b], [a, n + b, n + a], [2 * n + 1 - 0, n + a, n + b], [2 * n, b, a]]
    return V, np.array(F)


def hole_of(img, mask):
    """The hole of a ring seen from above (a roll of tape), as a mask, or None: the biggest patch inside the outline
    that has the colour of the desk round it. The hole must lie well inside and be a fair part of the outline."""
    import cv2
    m = mask.astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer = max(cs, key=cv2.contourArea)
    filled = np.zeros(m.shape, np.uint8)
    cv2.drawContours(filled, [outer], -1, 1, -1)
    around = cv2.dilate(filled, np.ones((25, 25), np.uint8)) - cv2.dilate(filled, np.ones((9, 9), np.uint8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
    desk = np.median(lab[around > 0], axis=0)
    deskish = (np.linalg.norm(lab - desk, axis=2) < HOLE_COLOUR).astype(np.uint8)
    inside = cv2.erode(filled, np.ones((7, 7), np.uint8))
    count, labels, stats, cent = cv2.connectedComponentsWithStats(deskish & inside, connectivity=4)
    if count < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    share = stats[k, cv2.CC_STAT_AREA] / max(1.0, cv2.contourArea(outer))
    middle = np.array(cv2.minEnclosingCircle(outer)[0])
    radius = cv2.minEnclosingCircle(outer)[1]
    if share < HOLE_SHARE or np.linalg.norm(cent[k] - middle) > 0.35 * radius:
        return None
    return cv2.morphologyEx((labels == k).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def ring_prism(mask, hole, calib, centre, yaw, height):
    """An extruded ring: outline and hole round the same centre, as (V, F) in the object frame."""
    import cv2
    import handeye_solve

    def loop(m):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
        xy = np.array([handeye_solve.image_to_base(calib, p, height)[:2] for p in c])
        return (xy - np.asarray(centre)) @ rot_z(yaw)[:2, :2]

    out, inner = loop(mask), loop(hole)
    mid = inner.mean(0)
    n = RING_POINTS
    angles = np.linspace(-math.pi, math.pi, n, endpoint=False)

    def round_it(P):
        a = np.arctan2(*(P - mid).T[::-1])
        return np.array([P[np.argmin(np.abs(np.angle(np.exp(1j * (a - t)))))] for t in angles])

    o, i = round_it(out), round_it(inner)
    V = np.vstack([np.c_[o, np.zeros(n)], np.c_[o, np.full(n, height)], np.c_[i, np.zeros(n)], np.c_[i, np.full(n, height)]])
    F = []
    for a in range(n):
        b = (a + 1) % n
        F += [[a, b, n + b], [a, n + b, n + a],                                 # outer wall
              [2 * n + b, 2 * n + a, 3 * n + a], [2 * n + b, 3 * n + a, 3 * n + b],  # inner wall
              [n + a, n + b, 3 * n + b], [n + a, 3 * n + b, 3 * n + a],         # top
              [b, a, 2 * n + a], [b, 2 * n + a, 2 * n + b]]                     # bottom
    return V, np.array(F)


def convex_parts(V, F, name, d):
    """Convex collision parts from CoACD, or one hull if it cannot run. Returns (parts, method)."""
    import import_gso
    sys.path.insert(0, str(SITE))
    try:
        import coacd
        coacd.set_log_level("error")
        mesh = coacd.Mesh(np.asarray(V, float), np.asarray(F, int))
        parts = coacd.run_coacd(mesh, threshold=COACD_THRESHOLD, max_convex_hull=MAX_PARTS, preprocess_mode="auto",
                                resolution=2000, mcts_iterations=60, mcts_max_depth=2, seed=1)
        parts = [np.asarray(p[0], float) for p in parts]
        parts = [p for p in parts if import_gso._solid(p)]
        if not parts:
            raise ValueError("no solid part")
        if len(parts) > MAX_PARTS:
            parts = import_gso.merge_parts(parts)
        return parts, f"coacd {len(parts)} parts"
    except Exception as e:
        say(f"  {name}: 凸分解ができず凸包1個にした（{e!r}）")
        return [V], "single convex hull (coacd failed)"


def write_asset(d, V_vis, F_vis, uv, parts, meta):
    import import_gso
    d.mkdir(parents=True, exist_ok=True)
    import_gso.write_visual(d, V_vis, uv, F_vis, F_vis)
    hulls = [import_gso.thin_hull(p) for p in parts]
    vols = np.array([h[2] for h in hulls])
    for k, (P, faces, _) in enumerate(hulls):
        import_gso.write_hull(d / f"collision_{k}.obj", P, faces)
    meta = dict(meta, collision_parts=len(hulls), visual_faces=len(F_vis),
                collision_faces=sum(len(h[1]) for h in hulls),
                collision_mass_split=[round(float(v), 4) for v in vols / vols.sum()],
                size_m=[round(float(v), 4) for v in np.ptp(V_vis, axis=0)])
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta, float(vols.sum())


def make_object(k, o, mask, img, calib, folder, assets):
    """One asset. Returns the row for summon.json and the world object."""
    import import_gso
    name, kind = o["name"], f"twin_{k}"
    soft = bool(o.get("deformable"))
    judged_h = float(o["size_cm"][2]) / 100
    L, W, centre = footprint(mask, calib, o["axis_xy"], float(np.clip(0.8 * judged_h, 0.002, 0.15)))
    H = shape_height(judged_h, L, W, soft)
    for hint in filter(None, os.environ.get("TIDY_REAL_HEIGHTS", "").split(";")):   # x,y,h from a real finger stop
        x, y, h = (float(v) for v in hint.split(","))
        if math.dist((x, y), o["xy"]) < 0.08:
            say(f"  {name}: 実機で指が止まった高さ {h * 100:.1f} cm を使う（写真の推定は {H * 100:.1f} cm）")
            H = h
    if L < W:   # the outline without its shadow has the other side longer
        L, W = W, L
        o = dict(o, axis_xy=[-o["axis_xy"][1], o["axis_xy"][0]])
    yaw = yaw_from_axis(o["axis_xy"])
    axes = view_axes(calib["T_cam_base"], [*centre, H / 2])
    raw, vis = folder / f"{kind}_raw.obj", folder / f"{kind}_vis.obj"
    row = {"kind": kind, "name": name, "method": None, "iou": None, "fallback": False}
    V_vis = None
    if raw.exists() and vis.exists():
        Vr, _, Fr, _ = import_gso.read_obj(raw)
        Vv, _, Fv, _ = import_gso.read_obj(vis)
        R, row["search_iou"] = best_rotation(Vv, mask)
        c0 = (Vv.min(0) + Vv.max(0)) / 2
        Q = np.eye(3)
        cup = is_container(o)
        if cup:
            Q = rim_frame(view_to_object((Vv - c0) @ R.T, axes, yaw))
        elif max(L, W) > 2 * H:
            Q = lying_frame(view_to_object((Vv - c0) @ R.T, axes, yaw))
        to_obj = lambda P: view_to_object((P - c0) @ R.T, axes, yaw) @ Q.T   # noqa: E731
        O = to_obj(Vv)
        O, s = scale_to(O, (L, W, H))
        lo, hi = to_obj(Vv).min(0), to_obj(Vv).max(0)
        cut = CUP_BASE_CUT if cup else BASE_CUT     # a cup's inside floor lies low: do not cut through it
        place = lambda P: flat_base((to_obj(P) - (lo + hi) / 2) * s + np.array([0, 0, H / 2]), cut)    # noqa: E731
        V_vis, F_vis = place(Vv), Fv
        V_vis, centre2, iou = fit_to_mask(calib, V_vis, F_vis, centre, yaw, mask)
        row["iou"] = round(iou, 3)
        if iou >= MIN_IOU:
            stretch = np.ptp(V_vis, axis=0) / np.maximum(np.ptp(place(Vv), axis=0), 1e-9)
            V_col, F_col = place(Vr) * stretch, Fr
            lo2 = V_vis.min(0)
            V_col = V_col + np.array([0, 0, lo2[2] - V_col[:, 2].min()])
            centre, row["method"] = list(centre2), "hunyuan3d-2.1 shape, texture projected from the photo"
        else:
            say(f"  {name}: 形が写真と合わない（一致度 {iou:.2f}）→ 輪郭の押し出しに切り替え（フォールバック）")
            V_vis = None
    else:
        say(f"  {name}: Hunyuan3D の生成に失敗 → 輪郭の押し出しに切り替え（フォールバック）")
    if V_vis is None:
        hole = hole_of(img, mask)
        V_vis, F_vis = outline_prism(mask, calib, centre, yaw, H) if hole is None else ring_prism(mask, hole, calib, centre, yaw, H)
        V_col, F_col = V_vis, F_vis
        row.update(method="FALLBACK: outline extruded (Hunyuan3D failed or did not match)", fallback=True, ring=hole is not None)
        if hole is not None:
            say(f"  {name}: 真ん中に穴があるので、輪の形で作る")
    d = assets / kind
    d.mkdir(parents=True, exist_ok=True)
    uv, bgr = texture_from_photo(img, mask, calib, V_vis, centre, yaw, d)
    parts, how = convex_parts(V_col, F_col, name, d)
    hull_vol = float(sum(import_gso.hull(p)[2] for p in parts))
    mass = mass_guess(hull_vol, density(o))
    meta = {"name_ja": name, "name_en": str(o.get("kind", name)), "category": o["category"], "mass_kg": round(mass, 4),
            "friction": FRICTION_SOFT if soft else FRICTION_RIGID, "soft": soft, "mean_bgr": bgr,
            "force": int(o.get("force", 3)), "should_grasp": bool(o.get("should_grasp", True)),
            "method": row["method"], "collision_method": how, "mass_is_guess": True, "container": is_container(o),
            "license": "made from the desk photo with Hunyuan3D-2.1", "split": "summon", "graspable_guess": True}
    meta, _ = write_asset(d, V_vis, F_vis, uv, parts, meta)
    size = meta["size_m"]
    row.update(collision=how, size_m=size, mass_kg=meta["mass_kg"], yaw=round(yaw, 3), centre=[round(float(v), 4) for v in centre])
    world_obj = {"id": f"{kind}_1", "kind": kind, "name": name, "centre": [float(v) for v in centre], "yaw": yaw,
                 "size": size, "z": 0.0, "bgr": bgr, "soft": soft, "min_force": 1, "max_force": 5,
                 "category": o["category"], "start": [float(v) for v in centre], "mesh": kind, "rgba": None,
                 "mass": meta["mass_kg"], "look_xy": list(o["xy"]), "should_grasp": bool(o.get("should_grasp", True))}
    return row, world_obj


# ------------------------------------------------------------------ the twin desk
def settle(world, seconds=2.0):
    """Let the twin come to rest. Unlike battle.settle nothing is taken off for being far out on the desk (the
    real desk's things may lie anywhere); only what falls off or keeps rolling is reported."""
    import mujoco
    import mj_scene
    model = mj_scene.build(world).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    rows = {}
    for o in world["objects"]:
        j = model.joint(o["id"])
        x, y, z, *quat = data.qpos[j.qposadr[0]:j.qposadr[0] + 7]
        speed = float(np.linalg.norm(data.qvel[j.dofadr[0]:j.dofadr[0] + 3]))
        moved = math.dist([x, y], o["start"])
        tilt = math.degrees(2 * math.asin(min(1.0, math.hypot(quat[1], quat[2]))))
        rows[o["id"]] = {"moved_m": round(moved, 4), "speed": round(speed, 4), "fell": bool(z < -0.02),
                         "tilt_deg": round(tilt, 1), "z": round(float(z), 4)}
        if z < -0.02:
            continue
        o["centre"] = o["start"] = [round(float(x), 4), round(float(y), 4)]
        o["z"] = round(max(0.0, float(z) - 0.0005), 4)
        o["quat"] = [round(float(v), 5) for v in quat]
        w_, x_, y_, z_ = quat
        o["yaw"] = math.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
    return rows


def compare_top(world_path, look, out):
    """The twin from the top camera next to the photo, for the eye."""
    import cv2
    import mj_scene
    twin = out.with_name("twin_only.jpg")
    mj_scene.snapshot(world_path, twin, seconds=0.01)
    a, b = cv2.imread(str(Path(look) / "scene.jpg")), cv2.imread(str(twin))
    cv2.imwrite(str(out), np.hstack([a, b]), [cv2.IMWRITE_JPEG_QUALITY, 92])


def summon(look, out, seed=0):
    import mj_scene
    start = time.time()
    look, out = Path(look).resolve(), Path(out).resolve()
    assets = out / "assets"
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    assets.mkdir(exist_ok=True)
    os.environ["TIDY_EXTRA_OBJECTS"] = str(assets)   # mj_objects is imported below, after this
    calib = json.loads((HERE / "calib/video2.json").read_text())
    img, items = load_look(look)
    say(f"召喚: 写真から {len(items)} 個の物を切り出した")
    jobs = []
    for k, (o, mask) in enumerate(items, 1):
        kind = f"twin_{k}"
        rgba_crop(img, mask, work / f"{kind}.png")
        jobs.append({"image": str(work / f"{kind}.png"), "out": str(work / f"{kind}_raw.obj"),
                     "out_visual": str(work / f"{kind}_vis.obj"), "seed": seed})
    say("召喚: Hunyuan3D で3Dモデルを作る")
    if jobs and not run_hunyuan(jobs, work):
        say("  Hunyuan3D が途中で止まった（work/hunyuan.log を見ること）")
    rows, world = [], mj_scene.real_layout()
    for k, (o, mask) in enumerate(items, 1):
        row, wo = make_object(k, o, mask, img, calib, work, assets)
        rows.append(row)
        world["objects"].append(wo)
        say(f"  {o['name']}: {'輪郭の押し出し' if row['fallback'] else 'Hunyuan3D'}、一致度 {row['iou']}、"
            f"{row['collision']}、{row['size_m'][0] * 100:.1f}x{row['size_m'][1] * 100:.1f}x{row['size_m'][2] * 100:.1f} cm、"
            f"推定 {row['mass_kg'] * 1000:.0f} g")
    return finish(world, rows, look, out, assets, start)


def finish(world, rows, look, out, assets, start):
    # In a fresh process: mj_objects read $TIDY_EXTRA_OBJECTS when it was imported, maybe before this run set it.
    path = out / "world.json"
    path.write_text(json.dumps(world, ensure_ascii=False, indent=1))
    env = dict(os.environ, TIDY_EXTRA_OBJECTS=str(assets))
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--settle", str(path), "--look", str(look)], check=True, env=env)
    settled = json.loads((out / "settled.json").read_text())
    for r, o in zip(rows, world["objects"]):
        r["settle"] = settled.get(o["id"])
    (out / "summon.json").write_text(json.dumps({"seconds": round(time.time() - start, 1), "objects": rows},
                                                indent=1, ensure_ascii=False))
    say(f"召喚おわり: {len(rows)} 個、{time.time() - start:.0f} 秒")
    return path


def settle_main(path, look):
    world = json.loads(Path(path).read_text())
    rows = settle(world)
    Path(path).write_text(json.dumps(world, ensure_ascii=False, indent=1))
    Path(path).with_name("settled.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for oid, r in rows.items():
        if r["fell"] or r["speed"] > 0.01 or r["moved_m"] > 0.02 or r["tilt_deg"] > 8:
            say(f"  注意: {oid} が置いたあと動いた（{r['moved_m'] * 100:.1f} cm、傾き {r['tilt_deg']:.0f} 度）")
    compare_top(path, look, Path(path).with_name("twin_top.jpg"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--look", required=True, help="folder with scene.jpg, judge.json, sam_boxes.json, sam_masks.json")
    p.add_argument("--out", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--settle", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.settle:
        settle_main(a.settle, a.look)
        return
    summon(a.look, a.out, a.seed)


if __name__ == "__main__":
    main()
