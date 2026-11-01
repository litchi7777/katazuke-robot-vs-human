#!/usr/bin/env python3
"""Re-open a round container whose scan closed its opening (a lid of scanned surface over the mouth).

Some GSO scans of bowls never saw the inside: the mesh is a solid puck with a dome where the mouth is
(gso_ramekin: vertices 0.7-5.7 cm from the axis at the rim height, no inner wall). In the desk it is then
an 11 cm solid lump that no rim pinch can hold, which a real ramekin is not. This rebuilds the object as a
solid of revolution of the scan's outer profile with a wall and a floor of the given thickness:
visual.obj (one texel of the scan's own texture, the colour nearest meta's mean colour), collision_0..N.obj
(a floor disc and wall sectors, all convex) and meta.json's size, faces and mass split. The texture is
kept. Only for nearly round things: the profile is the 95th percentile radius in each height band.

    python3 hollow_scan.py gso_ramekin --wall 0.004 --floor 0.006
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import import_gso

OBJECTS = Path(__file__).resolve().parents[1] / "objects"
BAND = 0.003      # m height of a profile band
SEGMENTS = 32     # around the axis, visual
SECTORS = 12      # around the axis, collision walls (a 30 deg chord sags 1.7 mm at 5 cm)


def outer_profile(V):
    """[(r, z)] from the bottom to the rim, about the bounding box's vertical axis."""
    c = (V[:, :2].min(0) + V[:, :2].max(0)) / 2
    r, z = np.hypot(*(V[:, :2] - c).T), V[:, 2]
    top = float(z.max())
    prof = []
    for z0 in np.arange(0.0, top, BAND):
        s = (z >= z0) & (z < z0 + BAND)
        if s.any():
            prof.append((float(np.percentile(r[s], 95)), min(z0 + BAND / 2, top)))
    prof[0] = (prof[0][0], 0.0)
    prof[-1] = (prof[-1][0], top)
    return prof


def section(outer, wall, floor):
    """Closed (r, z) outline: outside up, across the rim, inside down to the floor, floor to the axis."""
    inner = [(max(r - wall, 0.0), z) for r, z in outer if z >= floor]
    return [(0.0, 0.0)] + outer + inner[::-1] + [(0.0, floor)]


def revolve(sec, n):
    """Vertices and faces of the outline turned round the z axis; points on the axis are shared."""
    verts, rings = [], []
    for r, z in sec:
        if r < 1e-6:
            verts.append((0.0, 0.0, z))
            rings.append([len(verts) - 1] * n)
        else:
            rings.append(list(range(len(verts), len(verts) + n)))
            verts += [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z) for k in range(n)]
    faces = []
    for a, b in zip(rings, rings[1:]):
        for k in range(n):
            k1 = (k + 1) % n
            for f in ((a[k], a[k1], b[k1]), (a[k], b[k1], b[k])):
                if len(set(f)) == 3:
                    faces.append(f)
    return np.array(verts), np.array(faces)


def white_uv(tex, bgr):
    img = cv2.imread(str(tex))
    d = np.linalg.norm(img.astype(float) - np.array(bgr, float), axis=2)
    y, x = np.unravel_index(int(np.argmin(d)), d.shape)
    return (x + 0.5) / img.shape[1], 1.0 - (y + 0.5) / img.shape[0]


def collision_parts(outer, wall, floor):
    """A floor disc and SECTORS wall pieces, each a convex point set."""
    ring = lambda r, z, a0, a1, k=4: [(r * math.cos(a), r * math.sin(a), z) for a in np.linspace(a0, a1, k)]
    base = [p for r, z in outer if z <= floor for p in ring(r, z, 0, 2 * math.pi, 24)]
    r_floor = float(np.interp(floor, [z for _, z in outer], [r for r, _ in outer]))
    base += ring(r_floor, floor, 0, 2 * math.pi, 24)
    parts = [np.array(base)]
    for k in range(SECTORS):
        a0, a1 = 2 * math.pi * k / SECTORS, 2 * math.pi * (k + 1) / SECTORS
        pts = [p for r, z in outer for p in ring(r, z, a0, a1) + ring(max(r - wall, 0.0), max(z, floor), a0, a1)]
        parts.append(np.array(pts))
    return parts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind")
    p.add_argument("--wall", type=float, default=0.004, help="m wall thickness")
    p.add_argument("--floor", type=float, default=0.006, help="m floor thickness")
    a = p.parse_args()
    d = OBJECTS / a.kind
    meta = json.loads((d / "meta.json").read_text())
    V = import_gso.read_obj(d / "visual.obj")[0]
    outer = outer_profile(V)
    verts, faces = revolve(section(outer, a.wall, a.floor), SEGMENTS)
    u, v = white_uv(d / "texture.png", meta["mean_bgr"])
    with open(d / "visual.obj", "w") as f:
        f.write(f"# hollow_scan.py: solid of revolution of the scan's outer profile\nmtllib visual.mtl\nusemtl mat\n")
        f.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in verts)
        f.write(f"vt {u:.5f} {v:.5f}\n")
        f.writelines(f"f {i + 1}/1 {j + 1}/1 {k + 1}/1\n" for i, j, k in faces)
    for old in d.glob("collision_*.obj"):
        old.unlink()
    hulls = [import_gso.thin_hull(P) for P in collision_parts(outer, a.wall, a.floor)]
    for k, (P, fc, _) in enumerate(hulls):
        import_gso.write_hull(d / f"collision_{k}.obj", P, fc)
    vols = np.array([h[2] for h in hulls])
    meta.update({
        "size_m": [round(float(s), 4) for s in np.ptp(verts, axis=0)],
        "method": meta["method"].split("; ")[0] + f"; mouth re-opened (hollow_scan.py: wall {a.wall * 1000:.0f} mm, "
                                                  f"floor {a.floor * 1000:.0f} mm)",
        "visual_faces": int(len(faces)), "collision_parts": len(hulls),
        "collision_faces": int(sum(len(h[1]) for h in hulls)),
        "collision_mass_split": [round(float(x), 4) for x in vols / vols.sum()],
    })
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    print(f"{a.kind}: {len(verts)} verts {len(faces)} faces, {len(hulls)} collision parts, "
          f"rim r {outer[-1][0] * 100:.1f} cm, height {outer[-1][1] * 100:.1f} cm")


if __name__ == "__main__":
    main()
