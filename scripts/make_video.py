#!/usr/bin/env python3
"""Compose the presentation video from a recorded simulation (mj_arm.py --record).

Layout (1920x1080): the presentation camera large on the left, the robot's overhead camera top right,
and below it the robot's thoughts as they happened (plan, reflection, lesson, result), taken from the
timeline marks. A title card opens it; the playback speed is always shown in the corner.

    python3 make_video.py --rec REC_DIR --out OUT.mp4 [--speed 2] [--title ...] [--subtitle ...]
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H = 1920, 1080
MAIN = (1440, 1080)
SIDE = (480, 360)
PANEL = (480, 720)
KEEP_LINES = 9
COLOURS = {"作戦": (120, 200, 255), "次の作戦": (120, 200, 255), "振り返り": (255, 200, 90),
           "教訓": (255, 120, 120), "結果": (140, 230, 140), "散らかし役": (255, 120, 220)}


def font(size):
    return ImageFont.truetype(FONT, size)


def wrap(text, fnt, width, draw):
    lines, line = [], ""
    for ch in text:
        if draw.textlength(line + ch, font=fnt) > width:
            lines.append(line)
            line = ch
        else:
            line += ch
    return lines + [line] if line else lines


def card(title, subtitle):
    img = Image.new("RGB", (W, H), (8, 8, 12))
    d = ImageDraw.Draw(img)
    for i, (text, size, colour, y) in enumerate(((title, 78, (255, 255, 255), 420), (subtitle, 44, (255, 210, 80), 560))):
        f = font(size)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=colour)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def caption_of(text):
    """(kind, display text) for a timeline note line from tidy_desk.say()."""
    t = text.strip()
    for key in sorted(COLOURS, key=len, reverse=True):  # 次の作戦 before 作戦
        if t.startswith(key) or t.startswith(f"[{key}]"):
            rest = t[len(key) + (2 if t.startswith("[") else 0):]
            if rest.startswith("（"):  # 教訓（種類ごとのコツ）: ...
                label, rest = rest[1:].split("）", 1)
                return key, f"〈{label}〉" + rest.lstrip(":： ")
            return key, rest.lstrip(":： ")
    if "結果" in t:
        return "結果", t.split("結果", 1)[1].lstrip(":： ")
    if "→" in t:
        return "対象", t.split("]", 1)[-1].strip()
    return "", t


def panel(notes, speed):
    img = Image.new("RGB", PANEL, (18, 18, 24))
    d = ImageDraw.Draw(img)
    d.text((16, 12), "ロボットの考え", font=font(30), fill=(255, 255, 255))
    y = 60
    for k, (kind, text) in enumerate(notes[-KEEP_LINES:]):
        newest = k == len(notes[-KEEP_LINES:]) - 1
        colour = COLOURS.get(kind, (220, 220, 220))
        label = f"{kind}：" if kind and kind != "対象" else ""
        f = font(24 if newest else 21)
        for line in wrap(label + text, f, PANEL[0] - 32, d)[:3]:
            d.text((16, y), line, font=f, fill=colour if newest else tuple(int(c * 0.75) for c in colour))
            y += 32 if newest else 28
        y += 8
        if y > PANEL[1] - 40:
            break
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rec", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--speed", type=float, default=2.0)
    p.add_argument("--fps", type=float, default=15.0, help="recorded frame rate")
    p.add_argument("--title", default="絶対に片付けるロボット vs 絶対に散らかす男")
    p.add_argument("--subtitle", default="知識だけでロボットはどれだけ賢くなるのか？")
    p.add_argument("--title-seconds", type=float, default=3.0)
    a = p.parse_args()
    marks = [json.loads(l) for l in open(a.rec / "timeline.jsonl")]
    notes_at = {}
    for m in marks:
        if m["event"] == "note":
            notes_at.setdefault(m["frame"], []).append(caption_of(m["text"]))
        elif m["event"] == "scatter":
            notes_at.setdefault(m["frame"], []).append(("散らかし役", f"{m['object']} を机に投げた"))
    main_cap, side_cap = cv2.VideoCapture(str(a.rec / "show.mp4")), cv2.VideoCapture(str(a.rec / "video2.mp4"))
    out_fps = a.fps * a.speed
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{W}x{H}", "-r", str(out_fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "20", str(a.out)], stdin=subprocess.PIPE)
    title = card(a.title, a.subtitle)
    for _ in range(int(a.title_seconds * out_fps)):
        proc.stdin.write(title.tobytes())
    notes, frame = [], 0
    badge = f"{a.speed:g}倍速" if a.speed != 1 else ""
    while True:
        ok1, big = main_cap.read()
        ok2, small = side_cap.read()
        if not ok1:
            break
        notes += notes_at.get(frame, [])
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:, :MAIN[0]] = cv2.resize(big, MAIN)
        if ok2:
            canvas[:SIDE[1], MAIN[0]:] = cv2.resize(small, SIDE)
        canvas[SIDE[1]:, MAIN[0]:] = panel(notes, a.speed)
        if badge:
            img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((20, 20, 190, 76), 10, fill=(0, 0, 0))
            d.text((36, 24), badge, font=font(36), fill=(255, 210, 80))
            d.text((MAIN[0] + 12, 8), "ロボットの目（真上のカメラ）", font=font(20), fill=(255, 255, 255),
                   stroke_width=2, stroke_fill=(0, 0, 0))
            canvas = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        proc.stdin.write(canvas.tobytes())
        frame += 1
    proc.stdin.close()
    proc.wait()
    print(f"{a.out}: {frame / out_fps + a.title_seconds:.0f} 秒")


if __name__ == "__main__":
    main()
