#!/usr/bin/env python3
"""Import Google Scanned Objects (GSO, CC-BY 4.0) as desk objects for the MuJoCo desk (mj_objects).

Source: the MuJoCo-ready GSO conversion https://github.com/kevinzakka/mujoco_scanned_objects
(models/<name>/model.obj + texture.png + model_collision_*.obj, metres, z up, resting on z = 0).
Fetch only the models listed in OBJECTS, e.g.

    git clone --filter=blob:none --sparse --depth 1 https://github.com/kevinzakka/mujoco_scanned_objects.git SRC
    git -C SRC sparse-checkout set --no-cone $(python3 import_gso.py --list | sed 's#.*#/models/&/#')
    MUJOCO_GL=egl python3 import_gso.py SRC/models [short_name ...]     # all, or only these

Writes scripts/mj_assets/objects/gso_<short>/ in the mj_objects convention (metres, z up, origin at the
bottom centre, long axis along +x, natural resting pose): visual.obj/.mtl + texture.png (<= 512 px,
256-colour PNG), collision_0..N.obj (<= 16 convex parts, <= 200 faces each), meta.json; and GSO_LICENSE.txt.
The source files are only parsed, never executed. Re-running overwrites the folders.

Needs mujoco (resting-pose check) and, for visual meshes above MAX_VIS_FACES, pymeshlab, which runs in
another interpreter (GSO_MESHLAB_PY, default the Hunyuan3D venv) through this same file's --decimate mode.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "objects"
MESHLAB_PY = os.environ.get("GSO_MESHLAB_PY", "python3")
LICENSE = "CC-BY 4.0 Google Scanned Objects"
MAX_VIS_FACES = 3000
MAX_PARTS = 16
MAX_PART_VERTS = 100      # a hull of <= 100 points has <= 196 faces
MAX_TEX = 512
FINGER_GAP = 0.07         # practical opening across the grasped side (the full stroke is ~9 cm)

# gso name, short name, Japanese name, English name, category, mass (kg), friction, soft, pose, split.
# pose "upright": as scanned (GSO models stand as they rest on a table); "flat": laid on its largest face
# (empty boxes, tins and sticks lie down on a desk). The physics drop then settles whatever is left.
# split "test": held out as never-seen objects; about a third, spread over the shape types. "battle": only
# for battle.py (added 2026-11-01), so the train/test benchmark desks stay the same.
OBJECTS = [
    # cups and dishes (mugs and bowls are wider than the fingers open: the robot should leave them)
    ("ACE_Coffee_Mug_Kristen_16_oz_cup", "red_mug", "赤いマグカップ", "red patterned mug", "keep", 0.35, 0.6, False, "upright", "train"),
    ("Cole_Hardware_Mug_Classic_Blue", "teal_cup", "青緑のカップ", "teal cup", "keep", 0.30, 0.6, False, "upright", "test"),
    ("Room_Essentials_Mug_White_Yellow", "yellow_mug", "黄色と白のマグカップ", "yellow and white mug", "keep", 0.32, 0.6, False, "upright", "train"),
    ("Cole_Hardware_Bowl_Scirocco_YellowBlue", "yellow_bowl", "黄色い小鉢", "yellow bowl", "keep", 0.30, 0.6, False, "upright", "train"),
    ("Now_Designs_Bowl_Akita_Black", "patterned_bowl", "柄入りのボウル", "black patterned bowl", "keep", 0.38, 0.6, False, "upright", "test"),
    ("Threshold_Ramekin_White_Porcelain", "ramekin", "白いココット皿", "white ramekin", "keep", 0.25, 0.6, False, "upright", "train"),
    ("Kotobuki_Saucer_Dragon_Fly", "small_plate", "とんぼ柄の小皿", "small dragonfly saucer", "keep", 0.12, 0.6, False, "upright", "train"),
    # bottles, jars and a tin
    ("Weston_No_22_Cajun_Jerky_Tonic_12_fl_oz_nLj64ZnGwDh", "soda_bottle_a", "ジュースの空き瓶", "empty soda bottle", "trash", 0.20, 0.5, False, "upright", "train"),
    ("Weston_No_33_Signature_Sausage_Tonic_12_fl_oz", "soda_bottle_b", "ジュースの空き瓶", "empty soda bottle", "trash", 0.20, 0.5, False, "upright", "test"),
    ("Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original", "medicine_bottle", "胃腸薬のボトル", "medicine bottle", "keep", 0.35, 0.5, False, "upright", "train"),
    ("CoQ10", "supplement_bottle", "サプリメントの瓶", "supplement bottle", "keep", 0.12, 0.5, False, "upright", "train"),
    ("Mastic_Gum", "supplement_jar", "白いサプリメントの瓶", "white supplement jar", "keep", 0.11, 0.5, False, "upright", "test"),
    ("Polar_Herring_Fillets_Smoked_Peppered_705_oz_total", "fish_tin", "魚の缶詰の空き缶", "empty fish tin", "trash", 0.04, 0.4, False, "flat", "train"),
    # food packaging, emptied (trash)
    ("Nestle_Nips_Hard_Candy_Peanut_Butter", "candy_box", "キャンディーの空き箱", "empty candy box", "trash", 0.025, 0.5, False, "flat", "train"),
    ("NESCAFE_NESCAFE_TC_STKS_DECAF_6_CT", "coffee_stick_box", "コーヒースティックの空き箱", "empty coffee stick box", "trash", 0.02, 0.5, False, "flat", "test"),
    ("Nestle_Raisinets_Milk_Chocolate_35_oz_992_g", "chocolate_box", "チョコレートの空き箱", "empty chocolate box", "trash", 0.025, 0.5, False, "flat", "train"),
    ("Nestle_Skinny_Cow_Dreamy_Clusters_Candy_Dark_Chocolate_6_pack_1_oz_pouches", "snack_box", "お菓子の空き箱", "empty snack box", "trash", 0.03, 0.5, False, "flat", "test"),
    # small boxes and packages that are kept
    ("Crayola_Crayons_24_count", "crayon_box", "クレヨンの箱", "box of crayons", "keep", 0.11, 0.5, False, "flat", "train"),
    ("YumYum_D3_Liquid", "vitamin_box", "ビタミン剤の小箱", "small vitamin box", "keep", 0.06, 0.5, False, "upright", "train"),
    ("Swiss_Miss_Hot_Cocoa_KCups_Milk_Chocolate_12_count", "cocoa_box", "ココアの箱", "box of cocoa capsules", "keep", 0.30, 0.5, False, "upright", "test"),
    # toys and figurines
    ("Android_Figure_Orange", "android_orange", "オレンジのドロイド人形", "orange android figure", "keep", 0.09, 0.6, False, "upright", "train"),
    ("Android_Figure_Panda", "android_panda", "パンダ柄のドロイド人形", "panda android figure", "keep", 0.09, 0.6, False, "upright", "test"),
    ("Nintendo_Mario_Action_Figure", "mario_figure", "マリオのフィギュア", "Mario figure", "keep", 0.08, 0.6, False, "upright", "train"),
    ("Nintendo_Yoshi_Action_Figure", "yoshi_figure", "ヨッシーのフィギュア", "Yoshi figure", "keep", 0.09, 0.6, False, "upright", "test"),
    ("Schleich_Lion_Action_Figure", "lion_figure", "ライオンのフィギュア", "lion figure", "keep", 0.07, 0.6, False, "upright", "train"),
    ("Weisshai_Great_White_Shark", "shark_figure", "サメのフィギュア", "shark figure", "keep", 0.09, 0.6, False, "upright", "test"),
    ("BUNNY_RACER", "bunny_car", "うさぎの木の車", "wooden bunny car", "keep", 0.15, 0.6, False, "upright", "train"),
    ("Toysmith_Windem_Up_Flippin_Animals_Dog", "windup_dog", "ぜんまいの犬のおもちゃ", "wind-up dog toy", "keep", 0.03, 0.6, False, "upright", "train"),
    ("Thomas_Friends_Woodan_Railway_Henry", "toy_train", "緑の機関車のおもちゃ", "green toy locomotive", "keep", 0.07, 0.6, False, "upright", "train"),
    ("PINEAPPLE_MARACA_6_PCSSET", "pineapple_maraca", "パイナップルのマラカス", "pineapple maraca", "keep", 0.06, 0.6, False, "upright", "train"),
    ("LADYBUG_BEAD", "ladybug_toy", "てんとう虫の木のおもちゃ", "wooden ladybug toy", "keep", 0.05, 0.6, False, "upright", "test"),
    ("STACKING_RING", "ring_stacker", "輪っか積みのおもちゃ", "stacking rings toy", "keep", 0.20, 0.6, False, "upright", "train"),
    ("Squirrel", "squirrel_plush", "リスのぬいぐるみ", "squirrel plush toy", "keep", 0.08, 0.9, True, "upright", "test"),
    # tools, stationery, electronics, household
    ("Diamond_Visions_Scissors_Red", "scissors", "赤いハサミ", "red scissors", "keep", 0.07, 0.5, False, "upright", "train"),
    ("Magnifying_Glassassrt", "magnifier", "虫眼鏡", "magnifying glass", "keep", 0.06, 0.5, False, "upright", "train"),
    ("OXO_Soft_Works_Can_Opener_SnapLock", "can_opener", "缶切り", "can opener", "keep", 0.23, 0.6, False, "upright", "train"),
    ("Cole_Hardware_Mini_Honey_Dipper", "honey_dipper", "ハニーディッパー", "honey dipper", "keep", 0.015, 0.5, False, "flat", "test"),
    ("Cole_Hardware_School_Bell_Solid_Brass_38", "hand_bell", "真鍮のハンドベル", "brass hand bell", "keep", 0.38, 0.5, False, "upright", "train"),
    ("3M_Vinyl_Tape_Green_1_x_36_yd", "green_tape", "緑のビニールテープ", "green vinyl tape", "keep", 0.06, 0.7, False, "upright", "train"),
    ("Shurtape_Tape_Purple_CP28", "purple_tape", "紫の布テープ", "purple tape roll", "keep", 0.25, 0.7, False, "upright", "test"),
    ("Razer_Abyssus_Ambidextrous_Gaming_Mouse", "black_mouse", "黒いマウス", "black computer mouse", "keep", 0.09, 0.5, False, "upright", "train"),
    ("Razer_Taipan_White_Ambidextrous_Gaming_Mouse", "white_mouse", "白いマウス", "white computer mouse", "keep", 0.10, 0.5, False, "upright", "test"),
    ("Kingston_DT4000MR_G2_Management_Ready_USB_64GB", "usb_stick", "USBメモリ", "USB stick", "keep", 0.01, 0.5, False, "upright", "train"),
    ("Crosley_Alarm_Clock_Vintage_Metal", "alarm_clock", "目覚まし時計", "alarm clock", "keep", 0.35, 0.5, False, "upright", "train"),
    ("Nintendo_2DS_Crimson_Red", "game_console", "携帯ゲーム機", "handheld game console", "keep", 0.26, 0.5, False, "upright", "train"),
    ("Home_Fashions_Washcloth_Olive_Green", "washcloth", "たたんだタオル", "folded washcloth", "keep", 0.06, 0.9, True, "upright", "train"),
    ("Big_O_Sponges_Assorted_Cellulose_12_pack", "sponge", "スポンジ", "sponge", "keep", 0.015, 0.9, True, "upright", "test"),
    # 2026-11-01, more kinds for the scatterer (battle.py): tools, pale dishes, small and odd toys, flat cases,
    # packaging, bottles, two more precision devices and cloth
    ("Craftsman_Grip_Screwdriver_Phillips_Cushion", "screwdriver", "ドライバー", "screwdriver", "keep", 0.12, 0.6, False, "flat", "battle"),
    ("Cole_Hardware_Hammer_Black", "hammer", "金づち", "hammer", "keep", 0.45, 0.6, False, "flat", "battle"),
    ("HeavyDuty_Flashlight", "flashlight", "懐中電灯", "flashlight", "keep", 0.25, 0.6, False, "flat", "battle"),
    ("OXO_Cookie_Spatula", "spatula", "フライ返し", "spatula", "keep", 0.06, 0.5, False, "flat", "battle"),
    ("Threshold_Porcelain_Spoon_Rest_White", "spoon_rest", "白いスプーン置き", "white spoon rest", "keep", 0.15, 0.5, False, "upright", "battle"),
    ("Threshold_Bead_Cereal_Bowl_White", "white_bowl", "白いシリアルボウル", "white cereal bowl", "keep", 0.35, 0.6, False, "upright", "battle"),
    ("Ecoforms_Cup_B4_SAN", "plant_cup", "植木鉢のカップ", "plant pot cup", "keep", 0.10, 0.6, False, "upright", "battle"),
    ("Great_Dinos_Triceratops_Toy", "triceratops", "トリケラトプスのおもちゃ", "triceratops toy", "keep", 0.08, 0.6, False, "upright", "battle"),
    ("Schleich_African_Black_Rhino", "rhino_figure", "サイのフィギュア", "rhino figure", "keep", 0.06, 0.6, False, "upright", "battle"),
    ("Lego_Friends_Mia", "lego_figure", "レゴの箱", "small Lego set box", "keep", 0.2, 0.6, False, "upright", "battle"),
    ("Kong_Puppy_Teething_Rubber_Small_Pink", "rubber_toy", "ゴムの犬用おもちゃ", "pink rubber dog toy", "keep", 0.05, 0.9, False, "upright", "battle"),
    ("Fisher_price_Classic_Toys_Buzzy_Bee", "bee_toy", "蜂のおもちゃ", "bee pull toy", "keep", 0.12, 0.6, False, "upright", "battle"),
    ("Google_Cardboard_Original_package", "cardboard_package", "段ボールの空き箱", "empty cardboard package", "trash", 0.05, 0.5, False, "flat", "battle"),
    ("Orbit_Bubblemint_Mini_Bottle_6_ct", "gum_bottle", "ガムのパック", "pack of gum bottles", "keep", 0.05, 0.5, False, "upright", "battle"),
    ("Epson_LabelWorks_LC4WBN9_Tape_reel_labels_047_x_295_Roll_Black_on_White", "label_tape", "ラベルテープ", "label tape cartridge", "keep", 0.03, 0.5, False, "flat", "battle"),
    ("Mario_Luigi_Dream_Team_Nintendo_3DS_Game", "game_case", "ゲームソフトのケース", "game case", "keep", 0.04, 0.5, False, "flat", "battle"),
    ("Mad_Gab_Refresh_Card_Game", "card_game", "カードゲームの箱", "card game box", "keep", 0.15, 0.5, False, "flat", "battle"),
    ("Marc_Anthony_True_Professional_Strictly_Curls_Curl_Defining_Lotion", "lotion_bottle", "ローションのボトル", "lotion bottle", "keep", 0.20, 0.5, False, "upright", "battle"),
    ("Nikon_1_AW1_w11275mm_Lens_Silver", "camera", "カメラのレンズ", "camera lens", "keep", 0.35, 0.5, False, "upright", "battle"),
    ("Logitech_Ultimate_Ears_Boom_Wireless_Speaker_Night_Black", "speaker", "ワイヤレススピーカー", "wireless speaker", "keep", 0.54, 0.5, False, "upright", "battle"),
    ("Cole_Hardware_Dishtowel_Red", "dishtowel", "赤いふきん", "red dish towel", "keep", 0.05, 0.9, True, "upright", "battle"),
]
FIELDS = ("gso", "short", "name_ja", "name_en", "category", "mass_kg", "friction", "soft", "pose", "split")


# ---------------------------------------------------------------- OBJ reading / writing
def read_obj(path):
    """v, vt and triangulated faces (vertex and texcoord indices) of an OBJ file; texcoords may be absent."""
    V, VT, F, FT = [], [], [], []
    for line in Path(path).read_text(errors="replace").splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            V.append([float(x) for x in p[1:4]])
        elif p[0] == "vt":
            VT.append([float(x) for x in p[1:3]])
        elif p[0] == "f":
            idx = [q.split("/") for q in p[1:]]
            vi = [int(q[0]) for q in idx]
            ti = [int(q[1]) if len(q) > 1 and q[1] else 0 for q in idx]
            for k in range(1, len(vi) - 1):   # fan-triangulate polygons
                F.append([vi[0], vi[k], vi[k + 1]])
                FT.append([ti[0], ti[k], ti[k + 1]])
    V, F, FT = np.array(V), np.array(F), np.array(FT)
    F = np.where(F < 0, F + len(V) + 1, F) - 1
    VT = np.array(VT) if VT else None
    FT = (np.where(FT < 0, FT + len(VT) + 1, FT) - 1) if VT is not None else None
    return V, VT, F, FT


def compact(V, F):
    """Drop vertices no face uses."""
    used, inv = np.unique(F, return_inverse=True)
    return V[used], inv.reshape(F.shape)


def write_visual(d, V, VT, F, FT):
    lines = ["# generated by import_gso.py from Google Scanned Objects (CC-BY 4.0); metres, z up",
             "mtllib visual.mtl", "usemtl mat"]
    lines += ["v %.5f %.5f %.5f" % tuple(v) for v in V]
    lines += ["vt %.4f %.4f" % tuple(t) for t in VT]
    lines += ["f %d/%d %d/%d %d/%d" % (a + 1, ta + 1, b + 1, tb + 1, c + 1, tc + 1)
              for (a, b, c), (ta, tb, tc) in zip(F, FT)]
    (d / "visual.obj").write_text("\n".join(lines) + "\n")
    (d / "visual.mtl").write_text("newmtl mat\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd texture.png\n")


def write_hull(path, P, faces):
    lines = ["# convex collision part (import_gso.py); metres, z up"]
    lines += ["v %.5f %.5f %.5f" % tuple(v) for v in P]
    lines += ["f %d %d %d" % (a + 1, b + 1, c + 1) for a, b, c in faces]
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- decimation (pymeshlab, other python)
def decimate_main(src, dst, n):
    """--decimate mode, run by MESHLAB_PY: texture-preserving quadric decimation to n faces."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(src)
    # Some scans are split at every texture seam; weld coincident vertices or the decimation stalls.
    pct = getattr(pymeshlab, "PercentageValue", None) or pymeshlab.Percentage
    ms.meshing_merge_close_vertices(threshold=pct(0.01))
    # Scans with many texture seams stop early while the seams are kept; later passes let them move.
    for k in range(6):
        if ms.current_mesh().face_number() <= int(n):
            break
        ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=int(n), preserveboundary=k == 0,
                                                                 optimalplacement=True)
    ms.save_current_mesh(dst, save_wedge_texcoord=True, save_vertex_normal=False, save_textures=False)


def decimate(src, n, tmp):
    dst = Path(tmp) / "decimated.obj"
    subprocess.run([MESHLAB_PY, "-E", str(Path(__file__).resolve()), "--decimate", str(src), str(dst), str(n)],
                   check=True, cwd=tmp, stdout=subprocess.DEVNULL)
    return read_obj(dst)


# ---------------------------------------------------------------- convex parts
def hull(P):
    """(points, faces with outward normals, volume) of the convex hull of P."""
    from scipy.spatial import ConvexHull
    h = ConvexHull(P, qhull_options="Qt")
    P2 = P[h.vertices]
    remap = {v: i for i, v in enumerate(h.vertices)}
    faces = np.array([[remap[v] for v in s] for s in h.simplices])
    c = P2.mean(0)
    for k, (a, b, cc) in enumerate(faces):   # orient outward
        if np.dot(np.cross(P2[b] - P2[a], P2[cc] - P2[a]), P2[a] - c) < 0:
            faces[k] = [a, cc, b]
    return P2, faces, h.volume


def _solid(P):
    """A part with some volume (flat or degenerate CoACD pieces would break the hull code)."""
    try:
        return len(P) >= 4 and hull(P)[2] > 1e-9
    except Exception:
        return False


def thin_hull(P, max_verts=MAX_PART_VERTS):
    """Hull of at most max_verts points (farthest point sampling of the hull's vertices) -> <= 200 faces."""
    P, faces, vol = hull(P)
    if len(P) > max_verts:
        idx = [int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))]
        dist = np.linalg.norm(P - P[idx[0]], axis=1)
        while len(idx) < max_verts:
            i = int(np.argmax(dist))
            idx.append(i)
            dist = np.minimum(dist, np.linalg.norm(P - P[i], axis=1))
        P, faces, vol = hull(P[idx])
    return P, faces, vol


def merge_parts(parts, max_parts=MAX_PARTS):
    """Greedily merge the two convex parts whose joint hull adds the least volume, until <= max_parts.

    GSO ships up to 32 convex pieces (CoACD); merging the cheapest pairs keeps holes and handles open.
    """
    parts = [hull(p)[0] for p in parts]
    vols = [hull(p)[2] for p in parts]
    cost = {}

    def pair_cost(i, j):
        return hull(np.vstack([parts[i], parts[j]]))[2] - vols[i] - vols[j]

    alive = list(range(len(parts)))
    for a in alive:
        for b in alive:
            if a < b:
                cost[a, b] = pair_cost(a, b)
    while len(alive) > max_parts:
        (a, b) = min(cost, key=cost.get)
        parts.append(hull(np.vstack([parts[a], parts[b]]))[0])
        vols.append(hull(parts[-1])[2])
        alive = [k for k in alive if k not in (a, b)]
        cost = {k: v for k, v in cost.items() if a not in k and b not in k}
        n = len(parts) - 1
        for k in alive:
            cost[k, n] = pair_cost(k, n)
        alive.append(n)
    return [parts[k] for k in alive]


# ---------------------------------------------------------------- pose
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def min_rect(xy):
    """(angle of the long side, long, short) of the minimum-area rectangle around 2D points."""
    from scipy.spatial import ConvexHull
    H = xy[ConvexHull(xy).vertices]
    best = None
    for k in range(len(H)):
        e = H[(k + 1) % len(H)] - H[k]
        a = np.arctan2(e[1], e[0])
        R = rot_z(-a)[:2, :2]
        Q = H @ R.T
        w, h = np.ptp(Q[:, 0]), np.ptp(Q[:, 1])
        if best is None or w * h < best[0]:
            best = (w * h, a if w >= h else a + np.pi / 2, max(w, h), min(w, h))
    return best[1:]


def yaw_align(V):
    """Rotation about z that puts the footprint's long side along +x."""
    return rot_z(-min_rect(V[:, :2])[0])


def settle(parts, mass, seconds=2.0):
    """Drop the collision parts on a plane (1 mm) and return the rotation they come to rest in."""
    import mujoco
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1], friction=[0.6, 0.005, 0.0001])
    zmin = min(p[:, 2].min() for p in parts)
    body = spec.worldbody.add_body(name="o", pos=[0, 0, 0.001 - zmin])
    body.add_freejoint()
    vols = np.array([hull(p)[2] for p in parts])
    for k, p in enumerate(parts):
        P, faces, _ = hull(p)
        m = spec.add_mesh(name=f"p{k}", uservert=P.ravel().tolist(), userface=faces.ravel().tolist())
        m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"p{k}", mass=mass * vols[k] / vols.sum(),
                      condim=4, friction=[0.6, 0.01, 0.001])
    model = spec.compile()
    data = mujoco.MjData(model)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    return R.reshape(3, 3), float(np.linalg.norm(data.qvel))


# ---------------------------------------------------------------- texture, colour, grasp guess
def texture(src, dst):
    from PIL import Image
    im = Image.open(src).convert("RGB")
    if max(im.size) > MAX_TEX:
        s = MAX_TEX / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
    # 256 colours: a third of the size of a full-colour PNG, invisible at desk-camera resolution
    im.quantize(256).save(dst, optimize=True)
    return np.asarray(im)


def mean_bgr(img, V, VT, F, FT):
    """Area-weighted mean texture colour of the surface (BGR, like sim_scenarios' bgr)."""
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    uv = VT[FT].mean(axis=1) % 1.0
    h, w = img.shape[:2]
    px = img[np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)]
    rgb = (px * area[:, None]).sum(0) / area.sum()
    return [int(round(c)) for c in rgb[::-1]]


def graspable_guess(V):
    """Could the fingers (<= 7 cm apart) close on some part of it, from above?

    Yes if the whole footprint is narrow, or a 2 cm slice across the long axis in the middle 40 % is
    (a blade; not a mug handle at the end), or the top 30 % of an object is (a bottle neck, a mouse hump).
    """
    _, L, W = min_rect(V[:, :2])
    if W <= FINGER_GAP:
        return True
    x0, x1 = V[:, 0].min(), V[:, 0].max()
    # Only for long things: a slice near the edge of a round body is a narrow chord the fingers slip off.
    for xc in [] if L < 1.5 * W else np.arange(x0 + 0.3 * (x1 - x0), x1 - 0.3 * (x1 - x0) + 1e-9, 0.005):
        s = V[np.abs(V[:, 0] - xc) < 0.01]
        if len(s) > 10 and np.ptp(s[:, 1]) <= FINGER_GAP:
            return True
    H = V[:, 2].max()
    top = V[V[:, 2] > 0.7 * H]
    if H > 0.03 and len(top) > 10 and min_rect(top[:, :2])[2] <= FINGER_GAP:
        return True
    return False


# ---------------------------------------------------------------- one object
def convert(src_models, row, tmp):
    o = dict(zip(FIELDS, row))
    src = Path(src_models) / o["gso"]
    d = OUT / f"gso_{o['short']}"
    V, VT, F, FT = read_obj(src / "model.obj")
    if len(F) > MAX_VIS_FACES:
        V, VT, F, FT = decimate(src / "model.obj", MAX_VIS_FACES, tmp)
    V, F = compact(V, F)   # meshlab keeps the collapsed vertices in the file
    VT, FT = compact(VT, FT)
    parts = [read_obj(p)[0] for p in sorted(src.glob("model_collision_*.obj"),
                                            key=lambda p: int(p.stem.rsplit("_", 1)[1]))]
    parts = [p for p in parts if _solid(p)]
    if len(parts) > MAX_PARTS:
        parts = merge_parts(parts)

    # Pose: yaw-align, lay down if asked, then let physics settle it and keep the rest pose.
    R = yaw_align(V)
    if o["pose"] == "flat":
        ext = np.ptp(V @ R.T, axis=0)
        k = int(np.argmin(ext))
        R = (rot_x(np.pi / 2) if k == 1 else rot_y(np.pi / 2) if k == 0 else np.eye(3)) @ R
    Rs, speed = settle([p @ R.T for p in parts], o["mass_kg"])
    tilt = np.degrees(np.arccos(np.clip(Rs[2, 2], -1, 1)))
    if tilt > 2.0:   # it rolled or tipped: keep the pose it came to rest in
        R = Rs @ R
    R = yaw_align(V @ R.T) @ R
    V, parts = V @ R.T, [p @ R.T for p in parts]
    zmin = min(p[:, 2].min() for p in parts)   # rest the collision shape on z = 0
    shift = np.array([*(-(V[:, :2].min(0) + V[:, :2].max(0)) / 2), -zmin])
    V, parts = V + shift, [p + shift for p in parts]

    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    write_visual(d, V, VT, F, FT)
    img = texture(src / "texture.png", d / "texture.png")
    hulls = [thin_hull(p) for p in parts]
    vols = np.array([h[2] for h in hulls])
    for k, (P, faces, _) in enumerate(hulls):
        write_hull(d / f"collision_{k}.obj", P, faces)
    size = np.ptp(V, axis=0)
    meta = {
        "name_ja": o["name_ja"], "name_en": o["name_en"], "gso_name": o["gso"], "license": LICENSE,
        "category": o["category"], "mass_kg": o["mass_kg"], "friction": o["friction"], "soft": o["soft"],
        "size_m": [round(float(v), 4) for v in size], "graspable_guess": graspable_guess(V),
        "split": o["split"], "mean_bgr": mean_bgr(img, V, VT, F, FT),
        "method": f"import_gso.py: GSO scan, pose {o['pose']}, physics-settled (tilt {tilt:.0f} deg)",
        "visual_faces": int(len(F)), "collision_parts": len(hulls),
        "collision_faces": int(sum(len(h[1]) for h in hulls)),
        "collision_mass_split": [round(float(v), 4) for v in vols / vols.sum()],
    }
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    kb = sum(p.stat().st_size for p in d.iterdir()) / 1024
    print(f"{d.name:24s} {size[0]*100:5.1f} x{size[1]*100:5.1f} x{size[2]*100:5.1f} cm  "
          f"faces {len(F):5d}  parts {len(hulls):2d}  tilt {tilt:4.0f}  grasp {meta['graspable_guess']!s:5s} "
          f"{kb:5.0f} KB")
    if not (0.03 <= size.max() <= 0.22):
        print(f"  WARNING {d.name}: longest side {size.max()*100:.1f} cm is outside 3-22 cm")
    return meta


def write_license():
    rows = sorted(OBJECTS, key=lambda r: r[1])
    text = [
        "The gso_* objects in this folder are converted from Google Scanned Objects.",
        "",
        "Google Scanned Objects, licensed under CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/).",
        "Downs, L., Francis, A., Koenig, N., Kinman, B., Hickman, R., Reymann, K., McHugh, T. B., Vanhoucke, V.",
        "\"Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items\", ICRA 2022.",
        "https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/",
        "https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research",
        "MuJoCo conversion used as the source: https://github.com/kevinzakka/mujoco_scanned_objects",
        "",
        "Changes: re-posed (resting pose, long axis along +x, origin at the bottom centre), visual mesh decimated",
        "to <= 3000 faces, texture downscaled to 512 px / 256 colours, convex collision parts merged to <= 16",
        "and simplified (scripts/mj_assets/tools/import_gso.py). Masses and frictions are our estimates.",
        "",
        "Included models (folder <- GSO model name):",
    ] + [f"  gso_{r[1]} <- {r[0]}" for r in rows]
    (OUT / "GSO_LICENSE.txt").write_text("\n".join(text) + "\n")


def main(argv):
    if argv[:1] == ["--decimate"]:
        decimate_main(*argv[1:4])
        return
    if argv[:1] == ["--list"]:
        print("\n".join(r[0] for r in OBJECTS))
        return
    if not argv:
        sys.exit(__doc__)
    src, only = argv[0], set(argv[1:])
    with tempfile.TemporaryDirectory() as tmp:
        for row in OBJECTS:
            if not only or row[1] in only:
                convert(src, row, tmp)
    write_license()


if __name__ == "__main__":
    main(sys.argv[1:])
