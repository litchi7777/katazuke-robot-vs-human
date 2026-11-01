#!/usr/bin/env python3
"""The final video from the 2:41 draft: its title without the two cards after it, caption fixes, the five steps, the rest of the desk, the bottle's first try.

    python3 make_final_edit.py OUTPUT.mp4
Real footage is 2x up to the second tape; from the cup on it runs on at 5x, uncut, each stretch marked with its speed.
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent / "outputs"
DRAFT = ROOT / "final/PAI_final_video_draft.mp4"
STEPS = ROOT / "final/flow_steps"
WORK = ROOT / "final/work_edit"
TIDY = ROOT / "summon/real_20/videos/real_attempt_01_video0.mkv"
TAPE = ROOT / "summon/real_22/videos/real_attempt_01_video0.mkv"
BOTTLE = ROOT / "summon/real_13/videos/real_attempt_03_video0.mkv"
NOTES = ROOT / "summon/real_11/notebook/lessons.json"
BOTTLE2 = ROOT / "summon/real_14/videos/real_attempt_04_video0.mkv"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H, FPS = 1280, 720, 30
STEP_SECONDS = 3.4
FAST = 5                       # from the cup on
BAR_ROWS = "x=100:y=655:w=1080:h=53"            # where a one-line caption's text sits in the draft
SIM_ROWS = "x=30:y=606:w=1220:h=96"             # the simulation's two lines


def f(n):
    return n / FPS


def run(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def enc(path):
    return ["-an", "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", str(path)]


def caption(name, text, y=680, bar=False, badge=None):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if bar:
        d.rectangle((0, 640, W, H), fill=(0, 0, 0, 180))
    if badge:
        d.rounded_rectangle((1186, 10, 1270, 44), 6, fill=(0, 0, 0, 200))
        d.text((1228, 27), f"{badge}倍速", font=ImageFont.truetype(FONT, 21, index=0), fill=(255, 214, 64), anchor="mm")
    d.text((W / 2, y), text, font=ImageFont.truetype(FONT, 34, index=0), fill=(255, 255, 255), anchor="mm")
    path = WORK / f"{name}.png"
    im.save(path)
    return path


def draft(name, start, end):
    path = WORK / f"{name}.mp4"
    run("-ss", str(f(start)), "-i", str(DRAFT), "-t", str(f(end - start)), *enc(path))
    return path


def fixed(name, start, end, rows, text, y=680):
    """A stretch of the draft with its caption rubbed out and written again."""
    path = WORK / f"{name}.mp4"
    run("-ss", str(f(start)), "-t", str(f(end - start)), "-i", str(DRAFT), "-i", str(caption(name, text, y)),
        "-filter_complex", f"[0:v]delogo={rows}:show=0[v];[v][1:v]overlay", *enc(path))
    return path


def raw(name, source, start, end, text, speed=2):
    """Real footage, sped up, with the caption bar and the speed badge."""
    path = WORK / f"{name}.mp4"
    run("-ss", str(start), "-t", str(end - start), "-i", str(source), "-i", str(caption(name, text, bar=True, badge=speed)),
        "-filter_complex", f"[0:v]setpts=PTS/{speed},fps={FPS},scale={W}:{H},format=yuv420p,setsar=1[v];[v][1:v]overlay", *enc(path))
    return path


def still(name, image, seconds=STEP_SECONDS):
    path = WORK / f"{name}.mp4"
    run("-loop", "1", "-framerate", str(FPS), "-t", str(seconds), "-i", str(image), *enc(path))
    return path


def notebook():
    """The tape's lessons as Claude wrote them, first to last."""
    lessons = json.loads(NOTES.read_text())
    picked = [lessons[0], lessons[1], lessons[2], lessons[-1]]
    im = Image.new("RGB", (W, H), (14, 18, 26))
    d = ImageDraw.Draw(im)
    big, text = ImageFont.truetype(FONT, 40, index=0), ImageFont.truetype(FONT, 29, index=0)
    d.text((80, 84), "Claude の経験ノート", font=big, fill=(255, 255, 255))
    d.text((80, 142), "テープについて書いた教訓（原文）", font=ImageFont.truetype(FONT, 24, index=0), fill=(150, 160, 175))
    y = 214
    for k, l in enumerate(picked):
        last = k == len(picked) - 1
        d.text((80, y), l["created"][11:16], font=text, fill=(110, 220, 255))
        cut = l["text"].find("。") + 1     # a long lesson breaks between its two sentences
        whole = d.textlength(l["text"], font=text) <= 1010
        lines = [l["text"]] if whole else [l["text"][:cut], l["text"][cut:]]
        for line in lines:
            d.text((200, y), line, font=text, fill=(255, 110, 90) if last else (225, 230, 238))
            y += 44
        y += 26
    path = WORK / "e_note.png"
    im.save(path)
    return path


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    rest = "他の物も順番に箱へ"
    tape2, cup = "実機2回目：「縁をつまむ」を実機へ → 成功", "コップも縁をつまんで保管箱へ"
    tissue = "丸めたティッシュはゴミ箱へ"
    bottle = "失敗したケース：遠い位置の物体や、大きすぎる物体はつかめなかった"
    parts = [draft("a_open", 0, 106)]
    parts += [still(f"c_step{k}", STEPS / f"step{k}.png") for k in range(1, 6)]
    parts += [draft("d_marker", 392, 1623),
              fixed("e_sim", 1759, 2179, SIM_ROWS, "シミュレーション：Claude が「縁をつまむ」を選び、持ち上げられると確認", 654),
              still("e_note", notebook(), 5.0),
              fixed("f_tape2", 2179, 2959, BAR_ROWS, tape2),
              raw("g_cup", TIDY, 150, 193, cup, FAST), raw("h_tissue", TIDY, 193, 245, tissue, FAST),
              raw("i_box", TIDY, 245, 292, rest, FAST), raw("i_tissue", TIDY, 292, 345, rest, FAST),
              raw("i_tape", TAPE, 45, 92, rest, FAST),
              raw("j_bottle1", BOTTLE, 120, 178, bottle, FAST), raw("j_bottle2", BOTTLE2, 150, 198, bottle, FAST)]
    listing = WORK / "list.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    run("-f", "concat", "-safe", "0", "-i", str(listing), "-an", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart", sys.argv[1])


if __name__ == "__main__":
    main()
