#!/usr/bin/env python3
"""The five steps of the method as stills for the video, drawn from one real run's own files.

    python3 make_flow_steps.py ROUND_DIR TWIN_IMAGE OUTDIR
ROUND_DIR holds scene.jpg, judge.json, sam_masks.json and crop_N.jpg of a real run.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H = 1280, 720
BG, WHITE, GREY, CYAN, GREEN, RED = (14, 18, 26), (255, 255, 255), (150, 160, 175), (90, 220, 255), (70, 210, 120), (255, 95, 80)
TITLES = ["見つける", "判定する", "輪郭を切り出す", "掴む", "失敗したら練習"]
TEXT = [["Claude が写真を見て、", "机の上の物に名前と枠を付ける"],
        ["物を1個ずつ切り抜いて Claude に渡す。", "ゴミか残すか、握る力を決める"],
        ["SAM が枠の中の物の形だけを切り出す"],
        ["輪郭の中心と向きを計算して、", "そこへ指を下ろす"],
        ["写真から3Dモデルを作り、", "Claude が作戦をシミュレーションで試して、", "実機で再挑戦する"]]
ROW = 620        # the steps along the bottom
PANEL = (440, 96, 1240, 590)


def font(size):
    return ImageFont.truetype(FONT, size, index=0)      # index 0: Japanese glyphs


def base(step):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((48, 96), f"{step + 1}", font=font(84), fill=CYAN)
    d.text((48, 206), TITLES[step], font=font(38), fill=WHITE)
    y = 276
    for line in wrap(TEXT[step], d):
        d.text((48, y), line, font=font(23), fill=(215, 222, 232))
        y += 38
    cell = (W - 96) / 5
    for k, t in enumerate(TITLES):
        x = 48 + cell * k
        on = k == step
        d.rectangle((x, ROW, x + cell - 12, ROW + 5), fill=CYAN if on else (60, 68, 82))
        d.text((x, ROW + 16), f"{k + 1} {t}", font=font(21), fill=WHITE if on else GREY)
    return im


def wrap(lines, d, width=360):
    out = []
    for line in lines:
        cur = ""
        for ch in line:
            if d.textlength(cur + ch, font=font(23)) > width:
                out.append(cur)
                cur = ""
            cur += ch
        out.append(cur)
    return out


def fit(im, box):
    """im scaled to fill the box, cut at the sides or the ends."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    s = max(w / im.width, h / im.height)
    r = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, t = (r.width - w) // 2, (r.height - h) // 2
    return r.crop((l, t, l + w, t + h))


def desk_view(scene, objects):
    """The part of the scene round the things."""
    b = np.array([o["bbox"] for o in objects])
    x0, y0, x1, y1 = b[:, 0].min() - 70, b[:, 1].min() - 70, b[:, 2].max() + 70, b[:, 3].max() + 70
    aspect = (PANEL[2] - PANEL[0]) / (PANEL[3] - PANEL[1])
    w = max(x1 - x0, (y1 - y0) * aspect)
    h = w / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x = max(0, min(scene.width - w, cx - w / 2))
    y = max(0, min(scene.height - h, cy - h / 2))
    return (x, y, w / (PANEL[2] - PANEL[0]))


def view(im, v):
    x, y, s = v
    pw, ph = PANEL[2] - PANEL[0], PANEL[3] - PANEL[1]
    return im.crop((int(x), int(y), int(x + pw * s), int(y + ph * s))).resize((pw, ph), Image.LANCZOS)


def to_panel(p, v):
    return ((p[0] - v[0]) / v[2], (p[1] - v[1]) / v[2])


def main():
    round_dir, twin, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    scene = Image.open(round_dir / "scene.jpg").convert("RGB")
    objects = json.loads((round_dir / "judge.json").read_text())["objects"]
    masks = json.loads((round_dir / "sam_masks.json").read_text())
    v = desk_view(scene, objects)

    im, p = base(0), view(scene, v)
    d = ImageDraw.Draw(p)
    for o in objects:
        a, b = to_panel(o["bbox"][:2], v), to_panel(o["bbox"][2:], v)
        d.rectangle((*a, *b), outline=CYAN, width=3)
        t = o["kind"]
        tw = d.textlength(t, font=font(20))
        d.rectangle((a[0], a[1] - 30, a[0] + tw + 12, a[1]), fill=CYAN)
        d.text((a[0] + 6, a[1] - 29), t, font=font(20), fill=BG)
    im.paste(p, PANEL[:2])
    im.save(out / "step1.png")

    im = base(1)
    d = ImageDraw.Draw(im)
    gap = 14
    cw = (PANEL[2] - PANEL[0] - 2 * gap) // 3
    ch = (PANEL[3] - PANEL[1] - gap) // 2
    for k, o in enumerate(objects[:6]):
        x, y = PANEL[0] + (k % 3) * (cw + gap), PANEL[1] + (k // 3) * (ch + gap)
        crop = scene.crop(tuple(o["bbox"]))     # as the crop sent, without its padding and notes
        im.paste(fit(crop, (0, 0, cw, ch - 66)), (x, y))
        trash = o["category"] == "trash"
        d.rectangle((x, y + ch - 66, x + cw, y + ch), fill=(30, 36, 48))
        d.text((x + 10, y + ch - 62), o["kind"], font=font(21), fill=WHITE)
        d.text((x + 10, y + ch - 33), "ゴミ" if trash else "残す", font=font(21), fill=RED if trash else GREEN)
        d.text((x + 74, y + ch - 33), f"力 {o['force']}", font=font(21), fill=GREY)
    im.save(out / "step2.png")

    colours = [(90, 220, 255), (255, 190, 70), (120, 230, 130), (255, 110, 160), (180, 140, 255), (255, 240, 90)]
    arr = np.array(scene).astype(np.float32)
    dark = arr * 0.45
    edges = []
    for k, m in enumerate(masks):
        mask = np.zeros(arr.shape[:2], np.uint8)
        px = np.array(m["pixels"], np.int32)
        mask[px[:, 1], px[:, 0]] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        mask = cv2.erode(cv2.dilate(mask, kernel), kernel)
        on = mask > 0
        dark[on] = arr[on] * 0.6 + np.array(colours[k % 6], np.float32) * 0.4
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edges.append((cs, colours[k % 6]))
    canvas = dark.astype(np.uint8).copy()
    for cs, c in edges:
        cv2.drawContours(canvas, cs, -1, c, 2, cv2.LINE_AA)
    im = base(2)
    im.paste(view(Image.fromarray(canvas), v), PANEL[:2])
    im.save(out / "step3.png")

    im, p = base(3), view(scene, v)
    d = ImageDraw.Draw(p)
    for o, m in zip(objects, masks):
        pts = np.array(m["pixels"], np.float32)
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        proj = (pts - c) @ vt[0]
        a, b = to_panel(c + vt[0] * proj.min(), v), to_panel(c + vt[0] * proj.max(), v)
        q = to_panel(c, v)
        d.line((*a, *b), fill=(255, 190, 70), width=4)
        d.line((q[0] - 15, q[1], q[0] + 15, q[1]), fill=RED, width=5)
        d.line((q[0], q[1] - 15, q[0], q[1] + 15), fill=RED, width=5)
    d.rectangle((0, 0, 220, 46), fill=BG)
    d.text((14, 8), "＋ 中心", font=font(23), fill=RED)
    d.text((118, 8), "― 向き", font=font(23), fill=(255, 190, 70))
    im.paste(p, PANEL[:2])
    im.save(out / "step4.png")

    im = base(4)
    t = Image.open(twin).convert("RGB")
    pw, ph = PANEL[2] - PANEL[0], PANEL[3] - PANEL[1]
    s = min(pw / t.width, ph / t.height)
    r = t.resize((round(t.width * s), round(t.height * s)), Image.LANCZOS)
    im.paste(r, (PANEL[0] + (pw - r.width) // 2, PANEL[1] + (ph - r.height) // 2))
    im.save(out / "step5.png")


if __name__ == "__main__":
    main()
