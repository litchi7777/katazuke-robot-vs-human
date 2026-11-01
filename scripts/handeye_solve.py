#!/usr/bin/env python3
"""Eye-to-hand calibration from wrist ArUco snaps (offline).

Unknowns, solved jointly by least squares on marker-corner reprojection:
  T_cam_base   camera pose relative to the robot base (6)
  T_l6_marker  marker pose on the wrist link6 (6), unknown taping position
  fx, fy, cx, cy, k1, k2  camera model (6), unless --intrinsics fixes them
Joint angles from each `snap` event give T_base_l6 through the URDF.

Output JSON is used by image_to_base(): pixel + table height -> robot XY.

    source scripts/ros2_env.sh
    python3 scripts/handeye_solve.py --events RUN/client/events.jsonl --camera video2 \
        --marker-size 0.040 --out handeye_video2.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

import topdown_ik

DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
LINK = "link6"  # overridden by --link


def detect(path, marker_id):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    corners, ids, _ = cv2.aruco.detectMarkers(img, DICT, parameters=cv2.aruco.DetectorParameters_create())
    if ids is None:
        return None, img.shape[::-1]
    hits = [c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if i == marker_id]
    return (hits[0] if len(hits) == 1 else None), img.shape[::-1]


def link_pose(q6, link=None):
    q = np.zeros(topdown_ik._model.nq)
    q[:6] = q6
    pin.framesForwardKinematics(topdown_ik._model, topdown_ik._data, q)
    T = topdown_ik._data.oMf[topdown_ik._model.getFrameId(link or LINK)]
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = T.rotation, T.translation
    return M


def se3(v):
    M = np.eye(4)
    M[:3, :3] = cv2.Rodrigues(np.asarray(v[:3], float))[0]
    M[:3, 3] = v[3:6]
    return M


def marker_corners(size):
    h = size / 2  # ArUco corner order: TL, TR, BR, BL in the marker's x-right/y-up frame
    return np.array([[-h, h, 0, 1], [h, h, 0, 1], [h, -h, 0, 1], [-h, -h, 0, 1]], float)


def project(points_cam, cam):
    fx, fy, cx, cy, k1, k2 = cam
    x, y = points_cam[:, 0] / points_cam[:, 2], points_cam[:, 1] / points_cam[:, 2]
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    return np.stack([fx * x * d + cx, fy * y * d + cy], 1)


def residuals(p, data, size, fixed_cam, fixed_lm=None):
    T_cb = se3(p[:6])
    if fixed_lm is not None:  # marker pose on the link already known: p = [T_cb(6), cam(6)?]
        T_lm, cam = fixed_lm, (fixed_cam if fixed_cam is not None else p[6:12])
    else:
        T_lm = se3(p[6:12])
        cam = fixed_cam if fixed_cam is not None else p[12:18]
    corners = marker_corners(size)
    out = []
    for T_bl, uv in data:
        pts = (T_cb @ T_bl @ T_lm @ corners.T).T[:, :3]
        out.append((project(pts, cam) - uv).ravel())
    return np.concatenate(out)


def vec(M):
    return np.r_[cv2.Rodrigues(M[:3, :3])[0].ravel(), M[:3, 3]]


def initial_guess(data, size, cam):
    """Per-view marker PnP + OpenCV eye-to-hand calibration -> T_cam_base, T_link6_marker."""
    fx, fy, cx, cy, k1, k2 = cam
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    obj = np.ascontiguousarray(marker_corners(size)[:, :3])
    R_bg, t_bg, R_tc, t_tc, pairs = [], [], [], [], []
    for T_bl, uv in data:
        ok, rv, tv = cv2.solvePnP(obj, np.ascontiguousarray(uv, float), K, np.array([k1, k2, 0, 0]),
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        T_cm = se3(np.r_[rv.ravel(), tv.ravel()])
        T_lb = np.linalg.inv(T_bl)  # eye-to-hand: feed base->gripper instead of gripper->base
        R_bg.append(T_lb[:3, :3]); t_bg.append(T_lb[:3, 3])
        R_tc.append(T_cm[:3, :3]); t_tc.append(T_cm[:3, 3])
        pairs.append((T_bl, T_cm))
    if len(pairs) < 3:
        raise SystemExit("fewer than 3 views with a solvable marker pose")
    R, t = cv2.calibrateHandEye(R_bg, t_bg, R_tc, t_tc, method=cv2.CALIB_HAND_EYE_PARK)
    T_bc = np.eye(4)
    T_bc[:3, :3], T_bc[:3, 3] = R, t.ravel()  # camera pose in the base frame
    T_cb = np.linalg.inv(T_bc)
    # Marker on link6 from each view; average translation, take the median view's rotation.
    T_lms = [np.linalg.inv(T_bl) @ T_bc @ T_cm for T_bl, T_cm in pairs]
    T_lm = T_lms[len(T_lms) // 2].copy()
    T_lm[:3, 3] = np.median([M[:3, 3] for M in T_lms], axis=0)
    return np.r_[vec(T_cb), vec(T_lm)]


def image_to_base(calib, uv, z):
    """Pixel -> robot base XY on the horizontal plane at height z (m)."""
    fx, fy, cx, cy, k1, k2 = calib["camera"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    ray = cv2.undistortPoints(np.array([[uv]], float), K, np.array([k1, k2, 0, 0])).ravel()
    T_bc = np.linalg.inv(np.array(calib["T_cam_base"]))
    origin = T_bc[:3, 3]
    direction = T_bc[:3, :3] @ np.array([ray[0], ray[1], 1.0])
    t = (z - origin[2]) / direction[2]
    return origin + t * direction


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", required=True)
    p.add_argument("--camera", default="video2")
    p.add_argument("--marker-id", type=int, default=0)
    p.add_argument("--marker-size", type=float, required=True, help="black square side (m)")
    p.add_argument("--intrinsics", help="camera_intrinsics.py JSON; otherwise the camera model is fitted")
    p.add_argument("--link", default="link6", help="URDF link the marker is taped to")
    p.add_argument("--marker-from", help="earlier calibration JSON whose T_link_marker is reused (fixed)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    data, labels, size_px = [], [], None
    for line in open(a.events):
        e = json.loads(line)
        if e["event"] != "snap" or a.camera not in e["files"]:
            continue
        uv, size_px = detect(e["files"][a.camera], a.marker_id)
        if uv is not None:
            data.append((link_pose(e["positions"], a.link), uv))
            labels.append(e["label"])
    if len(data) < 8:
        raise SystemExit(f"only {len(data)} snaps with marker {a.marker_id}; need >= 8")
    fixed = None
    if a.intrinsics:
        intr = json.loads(Path(a.intrinsics).read_text())
        K, dist = np.array(intr["K"]), intr["dist"]
        fixed = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], dist[0], dist[1]])
    fixed_lm = None
    if a.marker_from:
        earlier = json.loads(Path(a.marker_from).read_text())
        if earlier.get("marker_link") != a.link:
            raise SystemExit("--marker-from was solved for a different --link")
        fixed_lm = np.array(earlier["T_link_marker"])
    # Unknown focal length: try a few typical USB-camera values, keep the best fit.
    fit = None
    for f in ([fixed[0]] if fixed is not None else (0.6, 0.75, 0.9, 1.1)):
        cam0 = fixed if fixed is not None else np.array([f * size_px[0], f * size_px[0],
                                                         size_px[0] / 2, size_px[1] / 2, 0.0, 0.0])
        p0 = initial_guess(data, a.marker_size, cam0)
        if fixed_lm is not None:
            # Camera pose from PnP of the median view with the known marker pose.
            p0 = p0[:6]
        if fixed is None:
            p0 = np.r_[p0, cam0]
        trial = least_squares(residuals, p0, args=(data, a.marker_size, fixed, fixed_lm),
                              loss="soft_l1", f_scale=2.0)
        if fit is None or trial.cost < fit.cost:
            fit = trial
    res = residuals(fit.x, data, a.marker_size, fixed, fixed_lm).reshape(len(data), 4, 2)
    per_view = np.sqrt((res ** 2).sum(2).mean(1))
    n_pose = 6 if fixed_lm is not None else 12
    cam = fixed if fixed is not None else fit.x[n_pose:n_pose + 6]
    T_lm_out = fixed_lm if fixed_lm is not None else se3(fit.x[6:12])
    out = {"camera_name": a.camera, "camera": [float(v) for v in cam],
           "T_cam_base": se3(fit.x[:6]).tolist(), "marker_link": a.link, "T_link_marker": T_lm_out.tolist(), "marker_fixed": fixed_lm is not None,
           "marker_size_m": a.marker_size, "views": labels,
           "per_view_rms_px": [float(v) for v in per_view], "rms_px": float(np.sqrt((res ** 2).mean() * 2))}
    Path(a.out).write_text(json.dumps(out, indent=1))
    cam_pos = np.linalg.inv(se3(fit.x[:6]))[:3, 3]
    print(f"{len(data)} views, reprojection RMS {out['rms_px']:.2f} px (worst {per_view.max():.2f})")
    print("camera position in base frame (m):", np.round(cam_pos, 3).tolist())
    print(f"marker offset on {a.link} (m):", np.round(T_lm_out[:3, 3], 4).tolist(), " camera:", np.round(cam, 2).tolist())


if __name__ == "__main__":
    main()
