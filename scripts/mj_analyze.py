#!/usr/bin/env python3
"""Per-kind numbers from mj_campaign runs: how each kind of object fared in each condition.

The Claude judge names objects freely, so each report entry is matched to the world object nearest
to where it was first seen. For each kind: first-try force, first-try success and attempts per object.

    python3 mj_analyze.py DIR_none DIR_reflect DIR_accumulate [--kind heavy_block]
"""
import argparse
import json
import math
from pathlib import Path

LABELS_JA = {"none": "知識なし", "reflect": "その場で振り返る", "accumulate": "知識をためる"}


def rows(campaign_dir):
    out = []
    for scene in sorted(Path(campaign_dir).glob("*/s*")):
        report, world = scene / "run/report.json", scene / "world.json"
        if not report.exists() or not world.exists():
            continue
        objs = json.loads(world.read_text())["objects"]
        for e in json.loads(report.read_text())["objects"]:
            if not e.get("xy"):
                continue
            o = min(objs, key=lambda o: math.dist(o["start"], e["xy"]))
            att = e.get("attempts", [])
            out.append({"scene": scene.name, "kind": o["kind"], "result": e.get("result"),
                        "attempts": len(att), "first_force": att[0]["force"] if att else None,
                        "first_ok": len(att) == 1 and str(e.get("result", "")).startswith("moved_to_")})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--kind")
    p.add_argument("--plot", help="PNG of the kind's first-try force and success per condition")
    a = p.parse_args()
    if a.plot and a.kind:
        plot_kind(a.dirs, a.kind, a.plot)
    for d in a.dirs:
        cond = next((c for c in LABELS_JA if c in Path(d).name), Path(d).name)
        rs = [r for r in rows(d) if not a.kind or r["kind"] == a.kind]
        if not rs:
            continue
        n = len(rs)
        ok = sum(1 for r in rs if str(r["result"]).startswith("moved_to_"))
        first = sum(1 for r in rs if r["first_ok"])
        att = sum(r["attempts"] for r in rs)
        forces = [r["first_force"] for r in rs if r["first_force"]]
        by_scene = {}
        for r in rs:
            by_scene.setdefault(r["scene"], []).append(r["first_force"])
        print(f"{LABELS_JA.get(cond, cond)}: {n} 個, 片付け {ok}/{n}, 1回目で成功 {first}/{n}, "
              f"1個あたり {att / n:.2f} 回, 最初の力 平均 {sum(forces) / max(1, len(forces)):.1f}  "
              + " ".join(f"{s}:{v}" for s, v in sorted(by_scene.items())))



def plot_kind(dirs, kind, out):
    """Two panels: the first-try grip force for each object of `kind` in the order met, and first-try
    success / attempts per object, per condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    colours = {"none": "#888888", "reflect": "#e8a33d", "accumulate": "#2a7de1"}
    stats = []
    for d in dirs:
        cond = next((c for c in LABELS_JA if c in Path(d).name), Path(d).name)
        rs = [r for r in rows(d) if r["kind"] == kind]
        if not rs:
            continue
        xs = list(range(1, len(rs) + 1))
        ax1.plot(xs, [r["first_force"] for r in rs], marker="o", color=colours.get(cond), label=LABELS_JA.get(cond, cond))
        n = len(rs)
        stats.append((LABELS_JA.get(cond, cond), sum(r["first_ok"] for r in rs) / n,
                      sum(r["attempts"] for r in rs) / n, colours.get(cond)))
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.axhline(5, color="#c33", ls="--", lw=1)
    ax1.text(1, 5.08, "持ち上がるのは力5（見た目では分からない重さ）", fontproperties=font, fontsize=9, color="#c33")
    ax1.set_ylim(0.5, 5.6)
    ax1.set_xlabel("出会った順の『重い積み木』", fontproperties=font)
    ax1.set_ylabel("最初に選んだ握る力", fontproperties=font)
    ax1.legend(prop=font)
    ax1.grid(alpha=0.3)
    names = [s[0] for s in stats]
    ax2.bar(names, [s[1] * 100 for s in stats], color=[s[3] for s in stats])
    for k, s in enumerate(stats):
        ax2.text(k, s[1] * 100 + 2, f"{s[1]:.0%}\n（{s[2]:.2f}回/個）", ha="center", fontproperties=font, fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("1回目で成功した割合（%）", fontproperties=font)
    for label in ax2.get_xticklabels():
        label.set_fontproperties(font)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
