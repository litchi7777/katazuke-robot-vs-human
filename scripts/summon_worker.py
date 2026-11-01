#!/usr/bin/env python3
"""Hunyuan3D-2.1 shape generation for summon.py: load the model once, then one mesh per job.

Runs in the Hunyuan venv (not the robot's python), started by summon.py:

    CUDA_VISIBLE_DEVICES=GPU-99571e8f-28bd-4db3-dadb-aea80850be59 ~/Hunyuan3D-2.1/.venv/bin/python \
        summon_worker.py JOBS.json

JOBS.json: [{"image": "x_rgba.png", "out": "x_raw.obj", "out_visual": "x_vis.obj", "seed": 0}, ...]. Prints SHAPE_DONE / SHAPE_FAIL per job.
"""
import json
import os
import sys
import time
import traceback

jobs = json.load(open(sys.argv[1]))
sys.path.pop(0)   # this folder's modules (calib/, ...) must not shadow Hunyuan's
H = os.environ.get("HUNYUAN_DIR", os.path.expanduser("~/Hunyuan3D-2.1"))
os.chdir(H)
sys.path[:0] = [H + "/hy3dshape", H + "/hy3dpaint", H]
import pymeshlab                               # noqa: E402
import torch                                   # noqa: E402
import trimesh                                 # noqa: E402
from PIL import Image                          # noqa: E402
try:
    from torchvision_fix import apply_fix
    apply_fix()
except Exception as e:
    print("torchvision_fix skipped", e, flush=True)
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402

MAX_FACES = 40000     # enough for the collision decomposition
VISUAL_FACES = 3000   # as the GSO objects

t0 = time.time()
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
print("MODEL_LOADED", round(time.time() - t0, 1), flush=True)


def clean(mesh):
    """Without floaters, cut down to MAX_FACES. Pieces of some size stay: a real tissue came out in several, and the
    largest alone was a fifth of the thing."""
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        top = max(len(p.faces) for p in parts)
        mesh = trimesh.util.concatenate([p for p in parts if len(p.faces) >= 0.05 * top])
    if len(mesh.faces) > MAX_FACES:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=MAX_FACES, preserveboundary=True, preservetopology=True)
        m = ms.current_mesh()
        mesh = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=False)
    return mesh


for job in jobs:
    t = time.time()
    try:
        img = Image.open(job["image"]).convert("RGBA")
        mesh = pipe(image=img, num_inference_steps=50, octree_resolution=320, num_chunks=16000,
                    generator=torch.manual_seed(int(job.get("seed", 0))))[0]
        mesh = clean(mesh)
        mesh.export(job["out"])
        lo = pymeshlab.MeshSet()
        lo.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
        lo.meshing_decimation_quadric_edge_collapse(targetfacenum=VISUAL_FACES, preservetopology=True)
        m = lo.current_mesh()
        trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=False).export(job["out_visual"])
        print("SHAPE_DONE", job["out"], round(time.time() - t, 1), len(mesh.faces), flush=True)
    except Exception as e:
        traceback.print_exc()
        print("SHAPE_FAIL", job["out"], repr(e)[:200], flush=True)
    torch.cuda.empty_cache()
