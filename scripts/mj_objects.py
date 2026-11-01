#!/usr/bin/env python3
"""Realistic mesh models of the real desk objects (mj_assets/objects/<kind>/) for the MuJoCo desk.

Each kind has visual.obj (+ texture.png), a convex collision.obj (< 200 faces) and meta.json, in metres,
z up, origin at the bottom centre, long axis along +x, in the object's natural resting pose. They were
made from the camera photos (Hunyuan3D-2.1 image-to-3D, or the photographed silhouette for the black
marker, or procedurally for the shogi piece and the block).
The gso_* kinds are scanned household objects (Google Scanned Objects, CC-BY 4.0, GSO_LICENSE.txt, made by
mj_assets/tools/import_gso.py); non-convex ones (mugs, scissors) have several convex parts
collision_0.obj, collision_1.obj, ... instead of one collision.obj, the mass split by meta's
collision_mass_split (by volume).

    body = spec.worldbody.add_body(name="pen1", pos=[x, y, 0.0005], quat=q); body.add_freejoint(name="pen1")
    add_mesh_object(body, "purple_marker", spec=spec)          # textured visual + convex collision
    add_mesh_object(body, "paper", rgba=[1, 1, 1, 1], spec=spec)  # plain colour instead of the texture
    (spec= can be left out after bind_spec(spec) or a previous call with the same spec)

A drop-in for mj_scene._add_object's primitives: the body pos is the object's bottom centre on the desk.

    MUJOCO_GL=egl python3 mj_objects.py [OUT.png]   # test: every object on a desk, 2 s physics, contact sheet
    MUJOCO_GL=egl python3 mj_objects.py --gso [OUT.png]   # the same for the gso_* kinds (gso_preview.png)
"""
import json
import os
from pathlib import Path

import mujoco
import numpy as np

OBJ_DIR = Path(__file__).resolve().parent / "mj_assets/objects"
KINDS = sorted(p.name for p in OBJ_DIR.iterdir() if (p / "meta.json").exists()) if OBJ_DIR.exists() else []
# One more folder of kinds, outside the tracked assets: the twins summon.py makes for one desk.
EXTRA_DIR = Path(os.environ["TIDY_EXTRA_OBJECTS"]) if os.environ.get("TIDY_EXTRA_OBJECTS") else None
EXTRA_KINDS = sorted(p.parent.name for p in EXTRA_DIR.glob("*/meta.json")) if EXTRA_DIR else []
GSO_KINDS = [k for k in KINDS if k.startswith("gso_")]   # scanned household objects (import_gso.py)
# mj_scene / sim_world kind names -> mesh kind
ALIASES = {"marker": "purple_marker", "pen": "black_marker", "tissue": "paper", "piece": "shogi_piece",
           "box": "crushed_box"}


def kind_dir(kind):
    return EXTRA_DIR / kind if kind in EXTRA_KINDS else OBJ_DIR / kind


def meta(kind):
    return json.loads((kind_dir(kind) / "meta.json").read_text())


# Crumpled paper squashes, so fingers sink in: a soft contact zone `width` m deep starting at impedance
# dmin, and a rough surface.
SOFT_CONTACT = (0.7, 0.004, 1.2)
_BOUND = {"spec": None}


def bind_spec(spec):
    """Make spec the default for add_mesh_object calls without spec= (an MjsBody has no link to its MjSpec)."""
    _BOUND["spec"] = spec
    return spec


def _spec_of(body):
    spec = _BOUND["spec"]
    if spec is not None:
        try:
            if (body.name and spec.body(body.name) is body) or any(b is body for b in spec.bodies):
                return spec
        except Exception:
            pass
    raise ValueError("add_mesh_object: pass spec= (or call bind_spec(spec) once); "
                     "an MjsBody does not know its MjSpec")


def collision_files(kind):
    """The convex collision meshes of a kind: [collision.obj], or collision_0.obj, collision_1.obj, ..."""
    d = kind_dir(kind)
    if (d / "collision.obj").exists():
        return [d / "collision.obj"]
    return sorted(d.glob("collision_*.obj"), key=lambda p: int(p.stem.split("_")[1]))


def _assets(spec, kind, textured):
    """Mesh / texture / material assets for a kind, added once per spec."""
    d = kind_dir(kind)
    files = collision_files(kind)
    names = dict(vis=f"objmesh_{kind}_vis", tex=f"objtex_{kind}", mat=f"objmat_{kind}",
                 col=[f"objmesh_{kind}_col"] if len(files) == 1 else
                 [f"objmesh_{kind}_col{k}" for k in range(len(files))])
    if spec.mesh(names["col"][0]) is None:
        for name, f in zip(names["col"], files):
            m = spec.add_mesh(name=name, file=str(f))
            m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        v = spec.add_mesh(name=names["vis"], file=str(d / "visual.obj"))
        v.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL   # open / thin visual shells are fine
        if (d / "texture.png").exists():
            spec.add_texture(name=names["tex"], type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(d / "texture.png"))
            mat = spec.add_material(name=names["mat"], rgba=[1, 1, 1, 1], specular=0.15, shininess=0.2)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = names["tex"]
    if not textured or spec.material(names["mat"]) is None:
        names["mat"] = None
    return names


def add_mesh_object(spec_body, kind, rgba=None, spec=None, name=None, mass=None, friction=None):
    """Add the object's meshes to an MjsBody (normally a free body, pos = bottom centre on the desk).

    visual: group 1, contype/conaffinity 0, massless, textured (or plain `rgba` if given / no texture).
    collision: convex mesh(es), contype 1, conaffinity 3, condim 4, mass (meta.json, or `mass` kg) and
    friction from meta.json, group 3 (hidden); several parts share the mass by volume. Soft objects get the
    same squashy solref/solimp as mj_scene. Returns (visual, collision): one geom for a single collision.obj,
    a list of geoms for a multi-part object.
    """
    kind = ALIASES.get(kind, kind)
    spec = spec or _spec_of(spec_body)
    _BOUND["spec"] = spec
    m = meta(kind)
    n = _assets(spec, kind, textured=rgba is None)
    base = name or spec_body.name
    vis = spec_body.add_geom(name=f"{base}_vis" if base else "", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=n["vis"],
                             contype=0, conaffinity=0, group=1, density=0)
    vis.mass = 0
    if n["mat"]:
        vis.material = n["mat"]
    else:
        vis.rgba = list(rgba if rgba is not None else m.get("rgba", [0.8, 0.8, 0.8, 1]))
    total = float(mass) if mass else m["mass_kg"]
    split = m.get("collision_mass_split") or [1.0 / len(n["col"])] * len(n["col"])
    cols = []
    for k, (mesh, share) in enumerate(zip(n["col"], split)):
        # the first part keeps the "<name>_col" name other code looks up
        gname = (f"{base}_col" if k == 0 else f"{base}_col{k}") if base else ""
        cols.append(spec_body.add_geom(name=gname, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
                                       contype=1, conaffinity=3, condim=4, group=3, mass=total * share,
                                       friction=[friction or m["friction"], 0.01, 0.001], rgba=[0.9, 0.3, 0.3, 0.4]))
    if m.get("soft"):
        # Crumpled paper is rough: more friction than the photo-made meta says.
        import os
        soft = os.environ.get("MJ_SOFT", "")  # experiments: "dmin,width,friction"
        dmin, width, mu = (float(v) for v in soft.split(",")) if soft else SOFT_CONTACT
        for col in cols:
            col.solref = [0.02, 1.0]
            col.solimp = [dmin, 0.95, width, 0.5, 2]
            col.friction = [max(mu, m["friction"]), 0.02, 0.002]
    return vis, cols[0] if len(cols) == 1 else cols


# ------------------------------------------------------------------------------------------- test
SPACING = 0.18


def _test(out=OBJ_DIR / "preview.png", seconds=2.0, kinds=None, cols=3, tile=(640, 480), row=None):
    """Drop `kinds` (default: the photo-made ones) on a desk, run `seconds` of physics, save a contact sheet.

    row: objects per row on the desk (default all in one row); the gso set needs a grid.
    """
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from PIL import Image, ImageDraw

    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth, spec.visual.global_.offheight = max(640, tile[0]), max(480, tile[1])
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    w = spec.worldbody
    w.add_geom(name="desk", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1], rgba=[0.80, 0.70, 0.62, 1],
               contype=1, conaffinity=3, friction=[0.6, 0.005, 0.0001])
    w.add_light(pos=[0.3, -0.3, 1.2], dir=[-0.2, 0.2, -1], diffuse=[0.5, 0.5, 0.5], castshadow=True)
    kinds = kinds or [k for k in KINDS if k not in GSO_KINDS]
    row = row or len(kinds)
    spacing = max(SPACING, max(max(meta(k)["size_m"][:2]) for k in kinds) + 0.06)
    for i, k in enumerate(kinds):
        x, y = spacing * (i % row), -spacing * (i // row)
        b = w.add_body(name=k, pos=[x, y, 0.002])
        b.add_freejoint(name=k)
        add_mesh_object(b, k, spec=spec)
        # a 1 cm grid marker strip next to each object (scale reference): 5 cm long, 1 cm wide
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.002, 0.0005], pos=[x, y - spacing * 0.6, 0.0005],
                   rgba=[0.1, 0.1, 0.1, 1], contype=0, conaffinity=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q0 = data.qpos.copy()
    report = {}
    n = int(seconds / model.opt.timestep)
    for i in range(n):
        if i == n - int(0.5 / model.opt.timestep):
            q_late = data.qpos.copy()   # to tell contact jitter from still sliding / rolling
        mujoco.mj_step(model, data)
    for k in kinds:
        j = model.joint(k)
        a = j.qposadr[0]
        d = data.qpos[a:a + 3] - q0[a:a + 3]
        v = np.linalg.norm(data.qvel[j.dofadr[0]:j.dofadr[0] + 6])
        drift = float(np.linalg.norm(data.qpos[a:a + 3] - q_late[a:a + 3]))
        turn = 2 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(data.qpos[a + 3:a + 7], q_late[a + 3:a + 7]))))))
        report[k] = dict(moved_m=round(float(np.linalg.norm(d)), 4), z=round(float(data.qpos[a + 2]), 4),
                         speed=round(float(v), 4), drift_mm=round(drift * 1000, 2), turn_deg=round(turn, 2),
                         mass=round(float(model.body(k).mass[0]), 4))
        # resting = no longer moving over the last 0.5 s, on the desk (not sunk or thrown), near where it was put
        report[k]["ok"] = bool(drift < 0.001 and turn < 1.0 and -0.003 < report[k]["z"] < 0.01
                               and report[k]["moved_m"] < 0.02 and np.all(np.isfinite(data.qpos)))
        print(k, report[k])
    r = mujoco.Renderer(model, tile[1], tile[0])
    tiles = []
    for k in kinds:
        m = meta(k)
        L = max(m["size_m"])
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [data.qpos[model.joint(k).qposadr[0]], data.qpos[model.joint(k).qposadr[0] + 1] - 0.02,
                         m["size_m"][2] / 2]
        cam.distance = max(0.12, L * 2.0)
        cam.azimuth, cam.elevation = 120, -35
        r.update_scene(data, cam)
        img = Image.fromarray(r.render())
        size = f"{m['size_m'][0]*100:.1f}x{m['size_m'][1]*100:.1f}x{m['size_m'][2]*100:.1f} cm"
        moved = f"moved {report[k]['moved_m']*1000:.1f} mm"
        label = (f"{k}  {size}  bar=5cm  {moved}" if tile[0] >= 640 else
                 f"{k}\n{size} {m.get('category', '')} {m.get('split', '')}\n{moved}"
                 + ("" if report[k]["ok"] else "  NOT RESTING"))
        ImageDraw.Draw(img).text((8, 8), label, fill=(0, 0, 0) if report[k]["ok"] else (220, 0, 0))
        tiles.append(img)
    # one overview of all objects side by side, same scale
    cam = mujoco.MjvCamera()
    nrows = (len(kinds) + row - 1) // row
    cam.lookat[:] = [spacing * (min(row, len(kinds)) - 1) / 2, -spacing * (nrows - 1) / 2 - 0.02, 0.02]
    cam.distance = spacing * max(min(row, len(kinds)), nrows * 1.3) * 0.95
    cam.azimuth, cam.elevation = 90, -40 if nrows == 1 else -60
    r.update_scene(data, cam)
    tiles.append(Image.fromarray(r.render()))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (tile[0] * cols, tile[1] * rows), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tile[0], (i // cols) * tile[1]))
    sheet.save(out)
    print("saved", out, " not resting:", [k for k in kinds if not report[k]["ok"]] or "none")
    return report


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args[:1] == ["--gso"]:
        _test(Path(args[1]) if len(args) > 1 else OBJ_DIR / "gso_preview.png", kinds=GSO_KINDS, cols=6,
              tile=(400, 300), row=8)
    else:
        _test(Path(args[0]) if args else OBJ_DIR / "preview.png")
