#!/usr/bin/env python3
"""Top-down grasp IK for the B601-RS (offline; never touches CAN).

The gripper approach axis is +x of the URDF frame `gripper_end`. "Top-down"
means that axis points along world -z, with the fingers closing along world y.
Solved by damped least squares on the full pose, from several seeds, within
the URDF joint limits.

    source scripts/ros2_env.sh
    python3 scripts/topdown_ik.py 0.45 0.0 0.10          # one target (m)
    python3 scripts/topdown_ik.py --map 0.10              # reachability map at z
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pinocchio as pin

URDF = (Path(__file__).resolve().parents[1]
        / "rebotarm_ros2/src/rebotarm_bringup/description/RS/urdf/ReBot_Arm_RS.urdf")
FRAME = "gripper_end"
# Columns: approach (gripper x) down, finger axis (gripper y) along world y.
TOPDOWN = np.array([[0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0]])

_model = pin.buildModelFromUrdf(str(URDF))
_data = _model.createData()
_fid = _model.getFrameId(FRAME)
_lo = _model.lowerPositionLimit[:6].copy()
_hi = _model.upperPositionLimit[:6].copy()


def fk(q6):
    q = np.zeros(_model.nq)
    q[:6] = q6
    pin.framesForwardKinematics(_model, _data, q)
    return _data.oMf[_fid].copy()


def pitched(pitch):
    """Approach tilted from straight down toward +x by `pitch` rad; fingers along world y."""
    a = np.array([math.sin(pitch), 0.0, -math.cos(pitch)])
    y = np.array([0.0, 1.0, 0.0])
    return np.column_stack([a, y, np.cross(a, y)])


def solve(position, yaw=0.0, seeds=None, iters=300, tol=1e-4, pitch=0.0, roll=0.0):
    """Return (q6, position_error_m, angle_error_rad) of the best seed, or None.

    yaw turns the whole grasp about world z, pitch tilts the approach from straight
    down toward the yaw direction, roll turns the fingers about the approach axis.
    """
    rot = pin.utils.rotate("z", yaw) @ (pitched(pitch) if pitch else TOPDOWN) @ pin.utils.rotate("x", roll)
    target = pin.SE3(rot, np.asarray(position, dtype=float))
    seeds = seeds or [np.array([0, 0.8, 0.8, 0, 0, 0]), np.array([0, 1.2, 1.4, 0.5, 0, 0]),
                      np.array([0, 0.5, 1.0, -0.5, 0, 0]), np.array([0, 1.0, 0.5, 1.0, 0, 0])]
    best = None
    toward = math.atan2(position[1], position[0])
    # joint1 turns the arm by -q1 about base z in this URDF; start from both signs so a target far to
    # the side (beyond ~60 deg) still converges.
    starts = [(seed, sign * toward) for seed in seeds for sign in (1.0, -1.0)]
    for seed, j1 in starts:
        q6 = np.clip(np.array(seed, float), _lo, _hi)
        q6[0] = j1
        for _ in range(iters):
            q = np.zeros(_model.nq)
            q[:6] = q6
            pin.framesForwardKinematics(_model, _data, q)
            err = pin.log6(_data.oMf[_fid].actInv(target)).vector
            if np.linalg.norm(err) < tol:
                break
            pin.computeJointJacobians(_model, _data, q)
            J = pin.getFrameJacobian(_model, _data, _fid, pin.LOCAL)[:, :6]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q6 = np.clip(q6 + 0.5 * dq, _lo, _hi)
        T = fk(q6)
        pos_err = float(np.linalg.norm(T.translation - target.translation))
        ang_err = float(np.linalg.norm(pin.log3(T.rotation.T @ target.rotation)))
        if best is None or pos_err + 0.05 * ang_err < best[1] + 0.05 * best[2]:
            best = (q6, pos_err, ang_err)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("xyz", nargs="*", type=float)
    p.add_argument("--map", type=float, help="print a reachability map at this height (m)")
    a = p.parse_args()
    if a.map is not None:
        print("x\\y  " + " ".join(f"{y:+.2f}" for y in np.arange(-0.15, 0.151, 0.05)))
        for x in np.arange(0.25, 0.651, 0.05):
            row = []
            for y in np.arange(-0.15, 0.151, 0.05):
                q, pe, ae = solve([x, y, a.map])
                row.append("  ok " if pe < 0.002 and ae < 0.02 else f"{pe*1000:4.0f}mm")
            print(f"{x:.2f} " + " ".join(row))
        return
    q, pe, ae = solve(a.xyz)
    print("q =", np.round(q, 4).tolist(), f"pos_err={pe*1000:.2f}mm ang_err={math.degrees(ae):.2f}deg")


if __name__ == "__main__":
    main()
