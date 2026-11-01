#!/usr/bin/env python3
"""The submission video: real footage at 2x with cuts and Japanese captions, plus title and result cards.

    python3 make_final_video.py OUT.mp4
"""
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent / "outputs"
RUNS = ROOT / "summon"
WORK = ROOT / "final/work"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H = 1280, 720
INK, PAPER, ACCENT = (255, 255, 255), (18, 22, 28), (255, 200, 60)


def font(size):
    return ImageFont.truetype(FONT, size, index=0)


def card(path, lines):
    """A full frame: lines of (text, size, colour), centred."""
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    total = sum(int(s * 1.6) for _, s, _ in lines)
    y = (H - total) // 2
    for text, size, colour in lines:
        f = font(size)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=colour)
        y += int(size * 1.6)
    im.save(path)


def caption(path, text, badge="2倍速"):
    """A transparent frame: the caption band at the bottom, the speed badge in the corner."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font(34)
    lines = text.split("\n")
    band = 30 + 50 * len(lines)
    d.rectangle([0, H - band, W, H], fill=(0, 0, 0, 170))
    for i, line in enumerate(lines):
        d.text(((W - d.textlength(line, font=f)) / 2, H - band + 14 + 50 * i), line, font=f, fill=INK)
    if badge:
        fb = font(26)
        w = d.textlength(badge, font=fb)
        d.rounded_rectangle([W - w - 44, 18, W - 16, 66], 10, fill=(0, 0, 0, 170))
        d.text((W - w - 30, 22), badge, font=fb, fill=ACCENT)
    im.save(path)


def still(i, image, seconds, text=None):
    out = WORK / f"{i:02d}.mp4"
    scale = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x12161c,format=yuv420p"
    cmd = ["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", str(seconds), "-i", str(image)]
    if text:
        cap = WORK / f"{i:02d}.png"
        caption(cap, text, badge=None)
        cmd += ["-i", str(cap), "-filter_complex", f"[0:v]{scale}[a];[a][1:v]overlay=format=auto,format=yuv420p"]
    else:
        cmd += ["-vf", scale]
    subprocess.run(cmd + ["-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", str(out)], check=True)
    return out


def clip(i, source, start, end, text, speed=2.0, badge="2倍速"):
    out = WORK / f"{i:02d}.mp4"
    cap = WORK / f"{i:02d}.png"
    caption(cap, text, badge)
    graph = (f"[0:v]setpts=PTS/{speed},scale={W}:{H}:force_original_aspect_ratio=decrease,"
             f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps=30[a];[a][1:v]overlay=format=auto,format=yuv420p")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(end - start), "-i", str(source),
                    "-i", str(cap), "-filter_complex", graph, "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                    "-an", str(out)], check=True)
    return out


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    v = lambda run, n: RUNS / run / f"videos/real_attempt_{n:02d}_video0.mkv"
    parts = []

    def c(lines):
        path = WORK / f"card{len(parts):02d}.png"
        card(path, lines)
        return path

    parts.append(still(len(parts), c([("絶対に片付けるロボット", 64, INK), ("vs", 40, ACCENT), ("絶対に散らかす男", 64, INK)]), 3.5))
    parts.append(still(len(parts), c([("人間が机に置いた物を", 46, INK), ("Claude が見て、考えて、片付ける", 46, INK), ("", 20, INK),
                                      ("6軸アーム ＋ カメラ ＋ Claude", 34, ACCENT), ("実機の映像は2倍速。待ち時間はカット", 28, INK)]), 4.5))
    parts.append(still(len(parts), RUNS / "real_20/attempt_01/look/preview/overlay.jpg", 5,
                       "カメラの写真から、物の名前とゴミか残すかを判定"))
    parts.append(clip(len(parts), v("real_6", 1), 62, 98, "マーカー：一発で成功"))
    parts.append(clip(len(parts), v("real_6", 1), 108, 124, "保管箱へ"))
    parts.append(clip(len(parts), v("real_7", 1), 60, 90, "テープ：真ん中を狙って空振り"))
    parts.append(still(len(parts), c([("Claude の振り返り", 44, ACCENT), ("「穴の中を狙ってしまった」", 44, INK), ("", 20, INK),
                                      ("写真からテープを3Dで召喚して", 34, INK), ("シミュレーションで作戦を試す", 34, INK)]), 4.5))
    parts.append(clip(len(parts), RUNS / "real_11/videos/sim_attempt_01_show.mp4", 2, 30,
                      "シミュレーション：Claude が「縁をつまむ」を選び、持ち上げられると確認\n（この試行では途中で落下）", badge="シミュレーション 2倍速"))
    parts.append(clip(len(parts), v("real_12", 2), 80, 132, "実機2回目：人の判断で「縁をつまむ」を実機へ → 成功"))
    parts.append(clip(len(parts), v("real_20", 1), 150, 186, "コップ（プラスチック）も縁をつまんで保管箱へ"))
    parts.append(clip(len(parts), v("real_20", 1), 195, 232, "丸めたティッシュはゴミ箱へ"))
    parts.append(clip(len(parts), v("real_14", 4), None or 110, 140, "失敗も：ペットボトルは「入れた」と報告したが、机に残っていた"))
    parts.append(still(len(parts), RUNS.parent / "final/final_desk.jpg", 4, "最後の机：空になった。保管箱の中にテープ、コップ、マーカー"))
    parts.append(still(len(parts), c([("結果", 44, ACCENT), ("マーカー・テープ・コップ・箱：保管箱へ", 34, INK), ("ティッシュ・缶：机から消えた（ゴミ箱の中は未確認）", 34, INK),
                                      ("ペットボトル：失敗（ロボットの誤報告あり）", 34, INK), ("ペンチ：試す前に「幅が広い」と諦めた", 34, INK), ("", 16, INK),
                                      ("指の開き 7.7 cm の中で、Claude はどこまで片付けられるか", 30, ACCENT)]), 7))
    listing = WORK / "list.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                    sys.argv[1]], check=True)
    print(sys.argv[1])


if __name__ == "__main__":
    main()
