#!/usr/bin/env python3
"""The submission video: difficulty goes up level by level until the robot stops, with the robot's own reports against
what really happened, big captions, zooms, and sound made here (no third-party audio).

    python3 make_final_video.py OUT.mp4

Real footage runs at 2x or slower, as the assignment allows. Everything the robot and Claude "say" is quoted from
their logs.
"""
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent / "outputs"
RUNS = ROOT / "summon"
WORK = ROOT / "final/work2"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H, FPS = 1280, 720, 30
RATE = 44100
PAPER = (14, 18, 26)
YELLOW, WHITE, RED, CYAN, ORANGE = (255, 214, 64), (255, 255, 255), (255, 84, 84), (110, 220, 255), (255, 170, 80)


def font(size):
    return ImageFont.truetype(FONT, size, index=0)


def blank():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def big(path, text, colour=YELLOW, size=92, y=None):
    """A shout across the picture."""
    im = blank()
    d = ImageDraw.Draw(im)
    f = font(size)
    tw = d.textlength(text, font=f)
    d.text(((W - tw) / 2, H * 0.62 if y is None else y), text, font=f, fill=colour, stroke_width=8, stroke_fill=(0, 0, 0))
    im.save(path)


def bubble(path, who, text, colour):
    """Who says what, in a box at the bottom."""
    im = blank()
    d = ImageDraw.Draw(im)
    f, fw = font(38), font(24)
    tw = d.textlength(text, font=f)
    x0, y0 = max(24, (W - tw) / 2 - 26), H - 136
    d.rounded_rectangle([x0, y0, min(W - 24, x0 + tw + 52), H - 34], 16, fill=(0, 0, 0, 215), outline=colour, width=4)
    d.text((x0 + 22, y0 + 4), who, font=fw, fill=colour)
    d.text((x0 + 26, y0 + 38), text, font=f, fill=WHITE)
    im.save(path)


def board(path, badge):
    """The speed, top right."""
    im = blank()
    d = ImageDraw.Draw(im)
    if badge:
        fb = font(24)
        w = d.textlength(badge, font=fb)
        d.rounded_rectangle([W - w - 40, 14, W - 16, 60], 10, fill=(0, 0, 0, 190))
        d.text((W - w - 28, 18), badge, font=fb, fill=YELLOW)
    im.save(path)


def card(path, lines, background=None):
    im = Image.new("RGB", (W, H), PAPER)
    if background is not None:
        bg = Image.open(background).convert("RGB").resize((W, H))
        im = Image.blend(bg, Image.new("RGB", (W, H), (0, 0, 0)), 0.72)
    d = ImageDraw.Draw(im)
    total = sum(int(s * 1.55) for _, s, _ in lines)
    y = (H - total) // 2
    for text, size, colour in lines:
        f = font(size)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=colour, stroke_width=3, stroke_fill=(0, 0, 0))
        y += int(size * 1.55)
    im.save(path)


def zoom_filter(zoom, seconds):
    cx, cy, z0, z1 = zoom
    n = max(1, int(seconds * FPS))
    return (f",scale={2 * W}:{2 * H},zoompan=z='{z0}+({z1}-{z0})*on/{n}':x='min(iw-iw/zoom,max(0,2*{cx}-iw/zoom/2))':"
            f"y='min(ih-ih/zoom,max(0,2*{cy}-ih/zoom/2))':d=1:s={W}x{H}:fps={FPS}")


class Cut:
    def __init__(self):
        self.parts = []
        self.sounds = []     # (seconds on the final timeline, name)
        self.clock = 0.0
        WORK.mkdir(parents=True, exist_ok=True)

    def _name(self, ext):
        return WORK / f"{len(self.parts):03d}_{self._k}.{ext}"

    def _overlays(self, seconds, says, badge):
        """Input files and filter chain for the scoreboard and captions."""
        inputs, chain, last = [], "", "[v]"
        self._k = 0
        items = [(0, seconds, "board", (badge,))] + says
        for n, item in enumerate(items):
            t0, t1, kind, arg = item
            self._k = n
            path = self._name("png")
            if kind == "board":
                board(path, *arg)
            elif kind == "big":
                big(path, *arg)
            elif kind == "robot":
                bubble(path, "ロボットの報告", arg, CYAN)
            elif kind == "human":
                bubble(path, "人間", arg, ORANGE)
            elif kind == "claude":
                bubble(path, "Claude の振り返り", arg, YELLOW)
            elif kind == "note":
                bubble(path, "", arg, WHITE)
            inputs += ["-i", str(path)]
            chain += f";{last}[{n + 1}:v]overlay=enable='between(t,{t0},{t1})':format=auto[o{n}]"
            last = f"[o{n}]"
            if kind != "board":
                self.sounds.append((self.clock + t0, {"big": "boom", "robot": "beep", "human": "pop"}.get(kind, "pop")))
        return inputs, chain, last

    def clip(self, source, start, end, says=(), speed=2.0, zoom=None, badge=None):
        """zoom = (cx, cy, from, to): a slow push in on that point of the 1280x720 frame."""
        seconds = (end - start) / speed
        badge = badge or {2.0: "2倍速", 1.0: "等速"}.get(speed, "スロー")
        graph = f"[0:v]setpts=PTS/{speed},fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2"
        if zoom:
            graph += zoom_filter(zoom, seconds)
        graph += "[v]"
        inputs, chain, last = self._overlays(seconds, list(says), badge)
        self._k = "out"
        out = self._name("mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(end - start), "-i", str(source), *inputs,
                        "-filter_complex", graph + chain + f";{last}format=yuv420p[e]", "-map", "[e]", "-an",
                        "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out)], check=True)
        self.parts.append(out)
        self.clock += seconds

    def still(self, image, seconds, says=(), zoom=None, badge=""):
        graph = f"[0:v]fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0e121a"
        if zoom:
            graph += zoom_filter(zoom, seconds)
        graph += "[v]"
        inputs, chain, last = self._overlays(seconds, list(says), badge)
        self._k = "out"
        out = self._name("mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", str(seconds), "-i", str(image), *inputs,
                        "-filter_complex", graph + chain + f";{last}format=yuv420p[e]", "-map", "[e]",
                        "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out)], check=True)
        self.parts.append(out)
        self.clock += seconds

    def card(self, lines, seconds, sound="boom", background=None):
        self._k = "card"
        path = self._name("png")
        card(path, lines, background)
        self.sounds.append((self.clock, sound))
        self.still(path, seconds)

    def level(self, name, thing, stars):
        self.card([(name, 44, YELLOW), (thing, 72, WHITE), (stars, 30, CYAN)], 1.6)

    def frame(self, source, at):
        path = WORK / f"frame_{len(self.parts):03d}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(at), "-i", str(source), "-frames:v", "1", str(path)], check=True)
        return path


# ---------------------------------------------------------------- sound, all made here
def tone(freq, seconds, volume=0.3, shape="sine", decay=6.0):
    t = np.arange(int(RATE * seconds)) / RATE
    wave_ = np.sin(2 * np.pi * freq * t) if shape == "sine" else np.sign(np.sin(2 * np.pi * freq * t))
    return volume * wave_ * np.exp(-decay * t)


def effect(name):
    if name == "ding":
        return np.concatenate([tone(880, 0.12, 0.5), tone(1319, 0.5, 0.5, decay=4)])
    if name == "buzz":
        return tone(110, 0.6, 0.35, "square", 1.5) + tone(117, 0.6, 0.35, "square", 1.5)
    if name == "boom":
        t = np.arange(int(RATE * 0.5)) / RATE
        return 0.7 * np.sin(2 * np.pi * (110 - 70 * t) * t) * np.exp(-6 * t)
    if name == "beep":
        return np.concatenate([tone(1400, 0.07, 0.22, "square", 3), np.zeros(1500), tone(1400, 0.07, 0.22, "square", 3)])
    return tone(660, 0.1, 0.3, decay=20)   # pop


def music(seconds):
    """A plain bass and pluck loop, quiet under everything."""
    beat = 60 / 118
    bass = [55.0, 55.0, 65.4, 49.0]
    tune = [440, 523, 659, 523, 587, 494, 659, 494]
    out = np.zeros(int(RATE * (seconds + 2)))
    for bar in range(int(seconds / (4 * beat)) + 1):
        for b in range(4):
            at = int(RATE * (bar * 4 + b) * beat)
            note = tone(bass[bar % 4], beat * 0.9, 0.16, "sine", 2)
            out[at:at + len(note)] += note[:len(out) - at]
        for e in range(8):
            at = int(RATE * (bar * 4 + e / 2) * beat)
            note = tone(tune[(e + bar) % 8] * (1 if bar % 2 else 1.5), beat * 0.45, 0.05, "square", 7)
            out[at:at + len(note)] += note[:len(out) - at]
    return out


def soundtrack(path, seconds, sounds):
    out = np.zeros(int(RATE * seconds))   # no music for now: music(seconds) is here if wanted again
    fade = int(RATE * 1.5)
    out[-fade:] *= np.linspace(1, 0, fade)
    for at, name in sounds:
        e, i = effect(name), int(RATE * at)
        n = min(len(e), len(out) - i)
        if n > 0:
            out[i:i + n] += e[:n]
    out = np.clip(out, -1, 1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes((out * 32000).astype(np.int16).tobytes())


# ---------------------------------------------------------------- the story
def main():
    v = lambda run, n: RUNS / run / f"videos/real_attempt_{n:02d}_video0.mkv"
    c = Cut()
    TAPE = (755, 385)

    # The messy desk and the arm setting off
    c.clip(v("real_20", 1), 0, 12)
    c.card([("Claude は机を", 58, WHITE), ("どこまで片付けられるか", 64, YELLOW)], 2.8)
    c.still(RUNS / "real_20/attempt_01/look/preview/overlay.jpg", 3.0,
            [(0, 3.0, "note", "写真を見て、ゴミか残すかを決める")])

    c.level("LEVEL 1", "マーカー", "★")
    c.clip(v("real_6", 1), 98, 112, zoom=(700, 390, 1.0, 1.3))
    c.sounds.append((c.clock, "ding"))
    c.clip(v("real_6", 1), 114, 120, [(0.5, 3, "big", ("一発",))])

    c.level("LEVEL 2", "丸めたティッシュ", "★★")
    c.clip(v("real_20", 1), 205, 219, [(0.5, 6.5, "note", "柔らかい物も、ゴミはゴミ箱へ")])
    c.clip(v("real_20", 1), 236, 246)

    c.level("LEVEL 3", "マスキングテープ", "★★★")
    c.clip(v("real_7", 1), 226, 244, [(1.5, 5, "big", ("空振り",))], zoom=(*TAPE, 1.4, 1.7))
    c.sounds.append((c.clock - 7.5, "buzz"))
    c.card([("指が穴の中に下りていた", 52, WHITE), ("", 16, WHITE), ("写真からテープを再現して、シミュレーションで練習", 32, CYAN)],
           3.6, "pop", c.frame(v("real_7", 1), 120))
    c.clip(RUNS / "real_11/videos/sim_attempt_01_show.mp4", 12, 28,
           [(0.5, 4, "claude", "縁をつまむ"), (4.5, 8, "big", ("持ち上がった", WHITE, 64))], badge="シミュレーション 2倍速")
    c.card([("でも放せなかった", 50, WHITE), ("指に掛かったまま、戻ってきた", 34, WHITE), ("", 14, WHITE), ("実機ではどうか", 46, YELLOW)], 3.0, "buzz")
    c.clip(v("real_12", 2), 96, 120, [(7, 12, "big", ("掴んだ",))], zoom=(*TAPE, 1.2, 1.5))
    c.sounds.append((c.clock + 0.3, "ding"))
    c.clip(v("real_12", 2), 130, 140, [(0.3, 5, "big", ("入った",))])

    c.level("LEVEL 4", "コップ", "★★★★")
    c.still(c.frame(v("real_18", 1), 40), 3.0, [(0, 3.0, "note", "最初の判定：ガラスなので触らない")], zoom=(640, 300, 1.0, 1.3))
    c.still(c.frame(v("real_18", 1), 45), 3.0, [(0, 3.0, "note", "プラスチックだと伝えると、掴みにいった")], zoom=(640, 300, 1.3, 1.3))
    c.clip(v("real_18", 1), 116, 132, [(1, 6, "note", "縁をつまんだ")], zoom=(640, 330, 1.2, 1.4))
    c.clip(v("real_18", 1), 144, 158, [(1.5, 7, "robot", "運ぶ途中で滑り落ちた")], zoom=(200, 480, 1.2, 1.8))
    c.clip(v("real_18", 1), 176, 190, [(0.2, 3.4, "note", "実際は落としていない"), (3.5, 7, "note", "箱の上から、持ったまま戻っていく")], zoom=(200, 480, 1.4, 1.4))
    c.clip(v("real_20", 1), 164, 178, [(0.5, 5, "note", "判定を直して、もう一度")])
    c.sounds.append((c.clock + 3.8, "ding"))
    c.clip(v("real_20", 1), 182, 193, [(3.8, 5.5, "big", ("今度は入った",))])

    c.level("LEVEL 5", "ペットボトル", "★★★★★")
    c.clip(v("real_14", 4), 160, 172, zoom=(860, 300, 1.2, 1.5))
    c.clip(v("real_14", 4), 186, 196, [(0.6, 5, "robot", "保管箱に入れました")])
    c.sounds.append((c.clock + 0.3, "buzz"))
    c.still(c.frame(v("real_14", 4), 196), 2.8, [(0.3, 2.8, "big", ("入ってない", RED))], zoom=(912, 304, 1.0, 1.2))

    c.level("LEVEL MAX", "ペンチ", "★★★★★★")
    c.clip(v("real_16", 2), 48, 58, [(0.3, 2.6, "robot", "指の開きより幅が広い"), (2.7, 5, "note", "一度も下りてこない")], zoom=(740, 500, 2.0, 2.2))
    c.card([("ペンチは試しもしなかった", 52, RED), ("持ち手がV字に開いていて、幅18cmと見て諦めた", 30, WHITE)], 3.0, "buzz")

    c.card([("総仕上げ", 44, YELLOW), ("6個まとめて", 72, WHITE), ("テープだけ2回目で", 30, CYAN)], 1.8)
    c.still(c.frame(v("real_20", 1), 30), 2.0, badge="")
    c.clip(v("real_20", 1), 266, 276)
    c.clip(v("real_22", 1), 68, 86, [(0.3, 4, "note", "最後のテープ"), (6.5, 9, "big", ("入った",))])
    c.sounds.append((c.clock, "ding"))
    c.still(RUNS.parent / "final/final_desk.jpg", 2.8, [(0.2, 2.8, "big", ("机が空になった",))], zoom=(480, 300, 1.0, 1.15))
    c.card([("できた", 34, YELLOW), ("マーカー　テープ　コップ　箱　ティッシュ※", 36, WHITE), ("", 10, WHITE),
            ("できなかった", 34, RED), ("ペットボトル　ペンチ", 36, WHITE), ("", 10, WHITE),
            ("※ 机からは消えた。ゴミ箱の中は未確認", 24, WHITE), ("実機の映像は2倍速以下、待ち時間はカット", 24, WHITE)], 4.2, "pop")

    # Outtakes
    c.card([("NG集", 72, ORANGE)], 1.0, "pop")
    c.clip(v("real_14", 4), 166, 170, speed=1.0, zoom=(860, 300, 1.5, 1.7), says=[(0.3, 4.5, "note", "丸い胴で指が滑る")])
    c.clip(v("real_18", 1), 190, 197, says=[(0.3, 3.4, "note", "持ったまま帰宅")])

    listing = WORK / "list.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in c.parts))
    silent = WORK / "silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(silent)], check=True)
    audio = WORK / "sound.wav"
    soundtrack(audio, c.clock, c.sounds)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(audio), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", sys.argv[1]], check=True)
    print(f"{sys.argv[1]}: {c.clock:.1f} s")


if __name__ == "__main__":
    main()
