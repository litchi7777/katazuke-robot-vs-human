#!/usr/bin/env python3
"""MuJoCo desk: the reBot arm from its URDF, the desk, the two boxes beside it and loose objects,
seen by the three calibrated cameras (video2 overhead, video0 and video4 from the sides).

The world is the sim_world JSON (base frame, metres); this module turns it into a physics model.
Objects are free bodies made of primitives (a pen is a capsule, crumpled paper a cluster of
spheres), so grasps can slip, roll and drop for real. The arm joints are position servos with the
real MIT gains and torque limits, with gravity compensated like the driver does.

    MUJOCO_GL=egl python3 mj_scene.py WORLD.json OUT_DIR    # render every camera once
"""
import json
import os
import math
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
DESC = HERE.parent / "rebotarm_ros2/src/rebotarm_bringup/description/RS"
CALIBS = {name: HERE / f"calib/{name}.json" for name in ("video2", "video0", "video4")}
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
KP = [50.0, 150.0, 150.0, 50.0, 50.0, 50.0]          # arm_pick_wide_trial.yaml mit_kp
KD = [3.0, 10.0, 10.0, 5.0, 4.0, 4.0]                # mit_kd
TORQUE = [36.0, 36.0, 36.0, 14.0, 14.0, 14.0]        # RS06 / RS05 peak torque, N*m
FINGER_KP = float(os.environ.get("MJ_FINGER_KP", 1200.0))  # N/m per finger: light things hold at level 1, a 180 g block needs 4-5 (2026-10-27)
FINGER_FORCE = 25.0     # N cap per finger
GAP_AT_ZERO = 0.0042    # m finger gap at 0 rad (sim_world)
GAP_PER_RAD = 0.0162    # m per rad
DESK = dict(x=(-0.13, 0.80), y=(-0.62, 0.62), top=0.0, rgba=(0.57, 0.57, 0.59, 1))
IMAGE = (1280, 960)
CANVAS = (1920, 1440)   # rendered undistorted, then warped to IMAGE with the lens distortion
TIMESTEP = 0.002
FLOOR_Z = -0.70         # m, the room floor below the desk top (desk ~70 cm high)
DESK_SHADE = 1.0        # material brightness on the desk photo
CAMERA_GAINS = HERE / "mj_assets/camera_gains.json"  # per camera B, G, R gains: real desk colour / rendered
ASSETS = HERE / "mj_assets"
MARKER = dict(id=0, size=0.040, border=0.010)  # the ArUco marker taped on link4 for the hand-eye calibration


def finger_q(rad):
    """Prismatic finger position (each finger, m) for a gripper motor angle.

    The finger pads meet at q = 0, so the gap is 2 q; the real gap is GAP_AT_ZERO + GAP_PER_RAD * rad
    (2026-10-27: subtracting GAP_AT_ZERO here made every grip 4 mm tighter than the real one).
    """
    return max(0.0, (GAP_AT_ZERO + GAP_PER_RAD * rad) / 2)


def finger_rad(q):
    return (2 * q - GAP_AT_ZERO) / GAP_PER_RAD


def _rgba(bgr, a=1.0):
    return [bgr[2] / 255, bgr[1] / 255, bgr[0] / 255, a]


def _quat_z(yaw):
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def arm_spec():
    urdf = (DESC / "urdf/ReBot_Arm_RS.urdf").read_text()
    urdf = urdf.replace("package://rebotarm_bringup/description/RS/", str(DESC.resolve()) + "/")
    urdf = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler discardvisual="false" fusestatic="false" '
                  r'balanceinertia="true" strippath="false"/></mujoco>', urdf, count=1)
    return mujoco.MjSpec.from_string(urdf)


def build(world, calibs=CALIBS, markers=False):
    """MjSpec of the arm in the world; compile() it for a model."""
    spec = arm_spec()
    _SPEC.append(spec)
    spec.option.timestep = TIMESTEP
    spec.option.noslip_iterations = 3  # contacts do not creep (without it light paper crept out of any grip)
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.visual.global_.offwidth, spec.visual.global_.offheight = CANVAS
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]

    for g in spec.geoms:
        mesh = g.meshname
        # The URDF collision mesh of each finger spans the gap behind the pads (its convex hull would
        # fill it), so the pads collide through their visual meshes instead.
        if g.parent.name in ("gripper_left", "gripper_right"):
            pad = mesh.endswith(("pla_left.STL", "pla_right.STL"))
            g.contype = g.conaffinity = 1 if pad else 0
            if pad:
                g.friction = [1.2, 0.01, 0.001]
                g.condim = 4
                g.group = 0
            elif g.contype == 0 and "shared" in mesh:
                g.group = 3  # hidden collision copy
        elif "shared" in mesh and g.contype:
            g.group = 3
        elif g.contype:
            g.group = 3
    # The arm never touches itself (the meshes' convex hulls overlap when folded); it touches the desk,
    # the boxes and the objects. Arm geoms: contype 2 / conaffinity 0; everything else: 1 / 3.
    for g in spec.geoms:
        if g.contype or g.conaffinity:
            g.contype, g.conaffinity = 2, 0
    for b in spec.bodies:
        if b.name and b.name != "world":
            b.gravcomp = 1.0
    _add_fingertips(spec)
    for name, kd in zip(ARM_JOINTS, KD):
        j = spec.joint(name)
        j.damping = [kd, 0.0, 0.0]
        j.armature = 0.02
        j.frictionloss = 0.05
    for j in ("gripper_joint1", "gripper_joint2"):
        spec.joint(j).damping = [20.0, 0.0, 0.0]
        spec.joint(j).armature = 0.01
    for name, kp, tau in zip(ARM_JOINTS, KP, TORQUE):
        a = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=kp)
        a.forcelimited = True
        a.forcerange = [-tau, tau]
        a.ctrlrange = spec.joint(name).range
        a.ctrllimited = True
    for j in ("gripper_joint1", "gripper_joint2"):
        a = spec.add_actuator(name=j, target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=FINGER_KP)
        a.forcelimited = True
        a.forcerange = [-FINGER_FORCE, FINGER_FORCE]
        a.ctrlrange = [0.0, 0.05]
        a.ctrllimited = True

    w = spec.worldbody
    # The desk is a photo of the real (light brown) desk: an empty patch seen by video4, mirrored into a
    # seamless tile. Each camera's own colour cast is applied afterwards (Cameras, CAMERA_GAINS).
    spec.add_texture(name="desk_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(ASSETS / "desk.png"))
    mat = spec.add_material(name="desk_mat", rgba=[DESK_SHADE] * 3 + [1], texrepeat=[1 / 0.44, 1 / 0.44],
                            texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "desk_tex"
    (x0, x1), (y0, y1) = DESK["x"], DESK["y"]
    w.add_geom(name="desk", contype=1, conaffinity=3, type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0) / 2, (y1 - y0) / 2, 0.02],
               pos=[(x0 + x1) / 2, (y0 + y1) / 2, DESK["top"] - 0.02], material="desk_mat",
               friction=[0.6, 0.005, 0.0001])
    for lx in (x0 + 0.04, x1 - 0.04):
        for ly in (y0 + 0.04, y1 - 0.04):
            w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, (DESK["top"] - 0.04 - FLOOR_Z) / 2],
                       pos=[lx, ly, (DESK["top"] - 0.04 + FLOOR_Z) / 2], rgba=[0.75, 0.75, 0.77, 1],
                       contype=1, conaffinity=3)
    w.add_geom(name="floor", contype=1, conaffinity=3, type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, FLOOR_Z],
               rgba=[0.35, 0.33, 0.31, 1])
    # Room light from a wide ceiling: no hard shadows (a sharp arm shadow read as "the object is gone").
    w.add_light(name="ceiling", pos=[0.3, 0.0, 1.6], dir=[0, 0, -1], diffuse=[0.55, 0.55, 0.55],
                castshadow=False)
    w.add_light(name="fill", pos=[0.9, 0.6, 1.0], dir=[-0.5, -0.3, -0.8], diffuse=[0.25, 0.25, 0.25],
                castshadow=False)

    for b in world["boxes"]:
        _add_box(w, b)
    for f in world.get("fixtures", []):
        w.add_geom(name=f["name"], contype=1, conaffinity=3, type=mujoco.mjtGeom.mjGEOM_BOX, size=[f["size"][0] / 2, f["size"][1] / 2,
                                                                          f["size"][2] / 2],
                   pos=[*f["centre"], f["size"][2] / 2], quat=_quat_z(f["yaw"]), rgba=_rgba(f["bgr"]))
    for o in world["objects"]:
        _add_object(w, o)
    cams = {name: json.loads(Path(path).read_text()) for name, path in calibs.items()}
    if markers:
        _add_marker(spec, cams["video2"]["T_link_marker"], cams["video2"]["marker_link"])
    for name, calib in cams.items():
        _add_camera(w, name, calib)
    # A presentation camera for the video: from the right and in front, above the desk, so that the arm, the
    # thing and both boxes are seen with their depth. Nothing but the video uses it.
    pos, look = np.array([0.6655, 0.7335, 0.8535]), np.array([0.05, 0.0, 0.05])
    f = (look - pos) / np.linalg.norm(look - pos)
    x = np.cross(f, [0, 0, 1])
    x /= np.linalg.norm(x)
    w.add_camera(name="show", pos=pos.tolist(), xyaxes=list(x) + list(np.cross(x, f)), fovy=45)
    return spec


TIP_REACH = 0.005  # m the real fingertips reach beyond the URDF gripper_end (sim_world.TIP_REACH)


def _add_fingertips(spec):
    """Pads on the finger ends that reach TIP_REACH beyond gripper_end, like the real fingers.

    The finger meshes stop at the gripper_end origin; the measured tips are 5 mm further. Each pad is
    the last 15 mm of the finger's inner face, in the finger body's frame (from the arm at q = 0).
    """
    model = arm_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    end = data.body("gripper_end")
    approach = end.xmat.reshape(3, 3)[:, 0]
    tip = end.xpos + approach * TIP_REACH
    for name, side in (("gripper_left", -1.0), ("gripper_right", 1.0)):
        body = data.body(name)
        R = body.xmat.reshape(3, 3)
        across = end.xmat.reshape(3, 3)[:, 1] * side
        # Pad centre: 7.5 mm back from the tip, 4 mm out from the finger's inner face.
        centre_world = tip - approach * 0.0075 + across * 0.004
        local = R.T @ (centre_world - body.xpos)
        half_world = np.abs(np.c_[approach * 0.0075, across * 0.004, np.cross(approach, across) * 0.013])
        half_local = np.abs(R.T @ half_world).sum(axis=1)
        g = spec.body(name).add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half_local.tolist(), pos=local.tolist(),
                                     rgba=[0.15, 0.15, 0.15, 1], friction=[1.2, 0.01, 0.001], condim=4)
        g.contype, g.conaffinity = 2, 0


def _add_box(w, b):
    """An open box: floor and four walls around centre, rim at b['height'], standing on a stand."""
    body = w.add_body(name=b["name"], pos=[*b["centre"], 0.0], quat=_quat_z(b["yaw"]))
    hx, hy = b["outer"][0] / 2, b["outer"][1] / 2
    t, floor, rim = b["wall"], b["floor"], b["height"]
    rgba = _rgba(b.get("bgr", (235, 235, 235)))
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, t / 2], pos=[0, 0, floor - t / 2], rgba=rgba,
                  contype=1, conaffinity=3)
    h = (rim - floor) / 2
    zc = floor + h
    for sx, sy, size in ((hx - t / 2, 0, [t / 2, hy, h]), (-hx + t / 2, 0, [t / 2, hy, h]),
                         (0, hy - t / 2, [hx, t / 2, h]), (0, -hy + t / 2, [hx, t / 2, h])):
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=[sx, sy, zc], rgba=rgba, contype=1, conaffinity=3)
    stand_top = floor - t
    if False:  # (kept for a basket reaching the floor; both now stand on a cabinet below their floor)
        for sx, sy, size in ((hx - t / 2, 0, [t / 2, hy, (stand_top - FLOOR_Z) / 2]),
                             (-hx + t / 2, 0, [t / 2, hy, (stand_top - FLOOR_Z) / 2]),
                             (0, hy - t / 2, [hx, t / 2, (stand_top - FLOOR_Z) / 2]),
                             (0, -hy + t / 2, [hx, t / 2, (stand_top - FLOOR_Z) / 2])):
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=[sx, sy, (stand_top + FLOOR_Z) / 2],
                          rgba=rgba, contype=1, conaffinity=3)
    elif stand_top > FLOOR_Z:  # a box on a low cabinet
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx + 0.03, hy + 0.03, (stand_top - FLOOR_Z) / 2],
                      pos=[0, 0, (stand_top + FLOOR_Z) / 2], rgba=[0.55, 0.52, 0.48, 1], contype=1, conaffinity=3)


# kind -> how the object is built: a list of primitives in its own frame (x along its length).
def _parts(o):
    l, wd, h = o["size"]
    kind = o.get("kind", "")
    if kind in ("pen", "marker"):
        r = wd / 2
        return [("capsule", [r, l / 2 - r], [0, 0, r], [0, 0.7071, 0, 0.7071])]
    if kind in ("paper", "tissue"):
        return [("crumple", [l, wd, h], [0, 0, 0], None)]
    if kind == "can":
        return [("cylinder", [wd / 2, l / 2], [0, 0, wd / 2], [0, 0.7071, 0, 0.7071])]
    return [("box", [l / 2, wd / 2, h / 2], [0, 0, h / 2], None)]


MASS = {"pen": 0.012, "marker": 0.030, "eraser": 0.015, "usb": 0.008, "piece": 0.005, "paper": 0.006,
        "tissue": 0.003, "can": 0.080, "box": 0.03, "block": 0.060}
FRICTION = {"can": 0.35, "marker": 0.5}  # smooth surfaces; everything else 0.9


def _add_object(w, o):
    z = o.get("z", 0.0)
    quat = o.get("quat") or _quat_z(o["yaw"])
    body = w.add_body(name=o["id"], pos=[*o["centre"], z + 0.0005], quat=quat)
    body.add_freejoint(name=o["id"])
    if o.get("mesh"):  # a 3D model made from a photo of the real object (mj_objects)
        import mj_objects
        # mass: heavier than it looks (heavy_block); split over the parts of a multi-part (GSO) model
        mj_objects.add_mesh_object(body, o["mesh"], rgba=list(o["rgba"]) if o.get("rgba") else None,
                                   spec=_SPEC[-1], mass=o.get("mass"), friction=o.get("friction"))
        return
    soft = o.get("soft")
    parts = _parts(o)
    mass = MASS.get(o.get("kind"), 0.015) / len(parts)
    for kind, size, pos, q in parts:
        if kind == "crumple":
            mesh = _crumpled_mesh(w.spec if hasattr(w, "spec") else None, o, size)
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, rgba=_rgba(o["bgr"]), mass=mass,
                              friction=[FRICTION.get(o.get("kind"), 0.9), 0.01, 0.001], condim=4,
                              contype=1, conaffinity=3)
            if soft:
                g.solref = [0.02, 0.6]
                g.solimp = [0.7, 0.95, 0.004, 0.5, 2]
            continue
        if o.get("kind") == "can" and kind == "cylinder":  # a drink can: silver, with a red label band
            band = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[size[0] * 1.01, size[1] * 0.55],
                                 pos=pos, rgba=[0.80, 0.08, 0.10, 1], contype=0, conaffinity=0, mass=0, group=1)
            band.quat = q
            for end in (-1, 1):  # the darker rims
                rim = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[size[0] * 0.93, 0.0012],
                                    pos=[end * size[1], 0, size[0]], rgba=[0.55, 0.56, 0.6, 1], contype=0,
                                    conaffinity=0, mass=0, group=1)
                rim.quat = q
        g = body.add_geom(type=getattr(mujoco.mjtGeom, f"mjGEOM_{kind.upper()}"), size=size, pos=pos,
                          rgba=_rgba(o["bgr"]), mass=mass, friction=[FRICTION.get(o.get("kind"), 0.9), 0.01, 0.001],
                          condim=4, contype=1, conaffinity=3)
        if q:
            g.quat = q
        if soft:
            g.solref = [0.02, 0.6]   # squashes a little under the fingers
            g.solimp = [0.7, 0.95, 0.004, 0.5, 2]


_SPEC = []  # the spec being built (meshes are added at spec level)


def _crumpled_mesh(_, o, size):
    """A crumpled ball of paper or tissue: a sphere whose surface is pushed in and out at random folds,
    squashed to the object's size (tissue flatter and softer looking than paper). Returns the mesh name."""
    spec = _SPEC[-1]
    name = f"crumple_{o['id']}"
    rng = np.random.default_rng(abs(hash(o["id"])) % 2**32)
    # icosphere, 3 subdivisions
    t = (1 + 5 ** 0.5) / 2
    verts = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
             [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
             [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
             [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    verts = [np.array(v, float) / np.linalg.norm(v) for v in verts]
    for _ in range(3):
        cache, new_faces = {}, []
        def mid(a, b):
            key = tuple(sorted((a, b)))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    v = np.array(verts)
    # folds: a few random planes push the surface in sharply (creases), plus fine noise
    radius = np.ones(len(v))
    for _ in range(14):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        radius -= 0.10 * np.clip(np.abs(v @ n) - rng.uniform(0.1, 0.6), -0.05, 0.3)
    radius += rng.normal(0, 0.03, len(v))
    v = v * radius[:, None]
    v = (v - v.min(0)) / (v.max(0) - v.min(0))  # unit box
    v = (v - [0.5, 0.5, 0.0]) * np.array(size)  # object size, bottom at z = 0
    spec.add_mesh(name=name, uservert=v.ravel().tolist(), userface=np.array(faces).ravel().tolist())
    return name


def _add_camera(w, name, calib):
    """A camera with the calibrated intrinsics at the calibrated pose.

    It renders a larger undistorted CANVAS with the same focal length; Lens.warp then applies the
    calibrated radial distortion, so a rendered frame matches the real camera pixel for pixel.
    """
    fx, fy, cx, cy = calib["camera"][:4]
    T = np.linalg.inv(np.array(calib["T_cam_base"]))
    R = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])  # OpenCV (z forward, y down) -> MuJoCo (z back, y up)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    cam = w.add_camera(name=name, pos=T[:3, 3].tolist(), quat=q.tolist())
    ccx, ccy = cx + (CANVAS[0] - IMAGE[0]) / 2, cy + (CANVAS[1] - IMAGE[1]) / 2
    cam.resolution = list(CANVAS)
    cam.sensor_size = [CANVAS[0] / 1000, CANVAS[1] / 1000]  # 1 mm per 1000 px: focal_length in px / 1000
    cam.focal_length = [fy / 1000, fy / 1000]  # square canvas pixels; Lens.warp applies fx vs fy
    # MuJoCo measures the principal point from the image centre with x to the left and y up (found with
    # mj_calib_check: the other sign put every marker twice the offset away, e.g. 26 px on video2).
    cam.principal_length = [(CANVAS[0] / 2 - ccx) / 1000, (CANVAS[1] / 2 - ccy) / 1000]


class Lens:
    """Pixel maps from the undistorted canvas to the real camera's distorted image."""

    def __init__(self, calib):
        import cv2
        fx, fy, cx, cy, k1, k2 = calib["camera"]
        u, v = np.meshgrid(np.arange(IMAGE[0], dtype=np.float32), np.arange(IMAGE[1], dtype=np.float32))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # Strong lenses (video0: k1 -0.40) need many more iterations than undistortPoints' default 5.
        pts = cv2.undistortPointsIter(np.stack([u.ravel(), v.ravel()], 1)[:, None, :].astype(np.float64), K,
                                      np.array([k1, k2, 0, 0]), None, None,
                                      (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)).reshape(-1, 2)
        ccx, ccy = cx + (CANVAS[0] - IMAGE[0]) / 2, cy + (CANVAS[1] - IMAGE[1]) / 2
        self.map_x = (fy * pts[:, 0] + ccx).reshape(IMAGE[1], IMAGE[0]).astype(np.float32)
        self.map_y = (fy * pts[:, 1] + ccy).reshape(IMAGE[1], IMAGE[0]).astype(np.float32)

    def warp(self, canvas):
        import cv2
        return cv2.remap(canvas, self.map_x, self.map_y, cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def _add_marker(spec, T_link_marker, link):
    """The paper ArUco marker (white border) on the link, as in the calibration photos."""
    import cv2
    ASSETS.mkdir(exist_ok=True)
    path = ASSETS / f"aruco_{MARKER['id']}.png"
    if not path.exists():
        px = 400
        inner = int(px * MARKER["size"] / (MARKER["size"] + 2 * MARKER["border"]))
        img = np.full((px, px), 255, np.uint8)
        o = (px - inner) // 2
        img[o:o + inner, o:o + inner] = cv2.aruco.drawMarker(
            cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50), MARKER["id"], inner)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    spec.add_texture(name="aruco", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(path))
    mat = spec.add_material(name="aruco_mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "aruco"
    T = np.array(T_link_marker)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, T[:3, :3].ravel())
    half = MARKER["size"] / 2 + MARKER["border"]
    # A box face carries the texture: +z face, image x right / y up like the ArUco marker frame.
    spec.body(link).add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, 0.0004],
                             pos=(T[:3, 3] - T[:3, 2] * 0.0004).tolist(), quat=q.tolist(),
                             material="aruco_mat", contype=0, conaffinity=0, group=1)


def real_layout(boxes_json=HERE / "config/desk_boxes.json"):
    """World with the two boxes where they really stand beside the desk (desk_boxes.json)."""
    spec = json.loads(Path(boxes_json).read_text())
    world = {"boxes": [], "fixtures": [], "objects": []}
    colours = {"keep": (240, 240, 240), "trash": (90, 40, 25)}  # white box, navy waste basket
    for role, b in spec.items():
        depth = 0.35 if role == "trash" else 0.10  # a tall waste basket; a shallow storage box
        world["boxes"].append({"name": f"{role}_box", "role": role, "centre": list(b["xy"]), "yaw": 0.0,
                               "outer": list(b["size"]), "wall": 0.006, "height": b["rim_z"],
                               "floor": b["rim_z"] - depth, "bgr": colours[role]})
    return world


class Cameras:
    """Renders any scene camera as the robot sees it: the calibrated ones with their lens distortion."""

    def __init__(self, model, calibs=CALIBS):
        self.model = model
        self.renderer = mujoco.Renderer(model, CANVAS[1], CANVAS[0])
        self.calib = {name: json.loads(Path(path).read_text()) for name, path in calibs.items()}
        self.lens = {name: Lens(c) for name, c in self.calib.items()}
        self.gains = json.loads(CAMERA_GAINS.read_text()) if CAMERA_GAINS.exists() else {}

    def render(self, data, camera):
        import cv2
        self.renderer.update_scene(data, camera=camera)
        rgb = self.renderer.render()
        img = self.lens[camera].warp(rgb) if camera in self.lens else cv2.resize(rgb, IMAGE, interpolation=cv2.INTER_AREA)
        img = img[:, :, ::-1]  # BGR for cv2
        if camera in self.gains:  # the real camera's white balance
            img = np.clip(img.astype(np.float32) * np.array(self.gains[camera], np.float32), 0, 255).astype(np.uint8)
        return img.copy()

    def height_map(self, data, camera):
        """Height above the desk (m) seen at every pixel of the camera's (distorted) image: a depth camera
        placed where this camera is (a RealSense next to it), registered pixel for pixel."""
        import cv2
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(data, camera=camera)
            depth = self.renderer.render().astype(np.float64)  # m along the view axis, canvas pixels
        finally:
            self.renderer.disable_depth_rendering()
        fx, fy, cx, cy = self.calib[camera]["camera"][:4]
        ccx, ccy = cx + (CANVAS[0] - IMAGE[0]) / 2, cy + (CANVAS[1] - IMAGE[1]) / 2
        v, u = np.mgrid[0:CANVAS[1], 0:CANVAS[0]]
        pts = np.stack([(u - ccx) / fy * depth, (v - ccy) / fy * depth, depth, np.ones_like(depth)], -1)
        T = np.linalg.inv(np.array(self.calib[camera]["T_cam_base"]))
        height = (pts.reshape(-1, 4) @ T[2]).reshape(depth.shape).astype(np.float32)
        lens = self.lens[camera]
        return cv2.remap(height, lens.map_x, lens.map_y, cv2.INTER_NEAREST, borderValue=0.0)


def snapshot(world_path, path, camera="video2", seconds=1.0):
    """One frame of a world file as the camera sees it, the arm folded (tidy_desk --preview on the MuJoCo desk)."""
    import cv2
    model = build(json.loads(Path(world_path).read_text())).compile()
    data = mujoco.MjData(model)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    cv2.imwrite(str(path), Cameras(model).render(data, camera), [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    import cv2
    world = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    model = build(world).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cameras = Cameras(model)
    for cam in list(CALIBS) + ["show"]:
        cv2.imwrite(str(out / f"{cam}.jpg"), cameras.render(data, cam))


if __name__ == "__main__":
    main()
