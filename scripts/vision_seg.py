#!/usr/bin/env python3
"""Object outlines for desk_judge boxes, with SAM (run in the `tidyvision` conda env, GPU).

Colour difference from the desk (desk_judge.refine) is fooled by boxes and
shadows next to an object: on 2026-10-28 a crumpled paper's centre came out
2 cm toward the box beside it. SAM, prompted with each judged box, returns the
object's own mask instead.

    conda run -n tidyvision python scripts/vision_seg.py IMAGE BOXES_JSON OUT_JSON
BOXES_JSON: [[x0, y0, x1, y1], ...]; OUT_JSON: [{"pixels": [[u, v], ...]} | null, ...]
"""
import json
import sys

import numpy as np
from ultralytics import SAM

MODEL = "sam2.1_b.pt"
MAX_PIXELS = 4000  # subsample big masks; enough for centre, axis and presence checks


def main():
    image, boxes_path, out_path = sys.argv[1:4]
    boxes = json.load(open(boxes_path))
    out = [None] * len(boxes)
    if boxes:
        model = SAM(MODEL)
        result = model(image, bboxes=boxes, verbose=False, device=0)[0]
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        rng = np.random.default_rng(0)
        for k, mask in enumerate(masks[:len(boxes)]):
            ys, xs = np.nonzero(mask)
            if len(xs) < 30:
                continue
            pts = np.stack([xs, ys], 1)
            if len(pts) > MAX_PIXELS:
                pts = pts[rng.choice(len(pts), MAX_PIXELS, replace=False)]
            out[k] = {"pixels": pts.tolist(), "area": int(mask.sum())}
    json.dump(out, open(out_path, "w"))


if __name__ == "__main__":
    main()
