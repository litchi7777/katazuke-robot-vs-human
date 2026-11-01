"""Pure logic of summon.py and rehearse.py: alignment maths, plan file. No GPU, no arm."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summon     # noqa: E402

rehearse = pytest.importorskip("rehearse")   # needs the ROS python paths, like the tidy_desk tests

T_DOWN = np.eye(4)      # a camera 1 m above the origin, looking straight down: x right in the image is base +y
T_DOWN[:3, :3] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
T_DOWN[:3, 3] = [0, 0, 1]


def test_camera_position():
    assert summon.camera_position(T_DOWN) == pytest.approx([0, 0, 1])


def test_view_axes_right_handed_straight_down():
    right, up, toward = summon.view_axes(T_DOWN, [0, 0, 0])
    assert right == pytest.approx([0, 1, 0])
    assert up == pytest.approx([-1, 0, 0])      # image up is towards the robot on this desk
    assert toward == pytest.approx([0, 0, 1])
    assert np.cross(right, up) == pytest.approx(toward)


def test_view_axes_follow_the_ray_not_z():
    _, _, toward = summon.view_axes(T_DOWN, [0.5, 0, 0])
    assert toward == pytest.approx(np.array([-0.5, 0, 1]) / math.hypot(0.5, 1))


def test_real_calibration_is_16_degrees_off_vertical():
    calib = json.loads((summon.HERE / "calib/video2.json").read_text())
    z_cam = np.array(calib["T_cam_base"])[2, :3]
    assert math.degrees(math.acos(-z_cam[2])) == pytest.approx(16.4, abs=0.5)


def test_view_to_object_undoes_the_yaw():
    axes = (np.array([0, 1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, 1.0]))
    yaw = math.radians(30)
    d = np.array([math.cos(yaw), math.sin(yaw), 0.0])      # a point along the long axis, in base coordinates
    view = np.array([[d @ a for a in axes]])
    assert summon.view_to_object(view, axes, yaw)[0] == pytest.approx([1, 0, 0], abs=1e-9)


def test_scale_to_size_and_rest():
    V = np.random.default_rng(0).normal(size=(200, 3))
    out, _ = summon.scale_to(V, (0.1, 0.03, 0.02))
    assert np.ptp(out, axis=0) == pytest.approx([0.1, 0.03, 0.02])
    assert out[:, 2].min() == pytest.approx(0)
    assert (out[:, :2].min(0) + out[:, :2].max(0)) == pytest.approx([0, 0], abs=1e-12)


def test_flat_base():
    V = np.array([[0, 0, 0.0], [0, 0, 0.01], [0, 0, 0.1]])
    out = summon.flat_base(V, 0.05)
    assert out[:, 2] == pytest.approx([0, 0.005, 0.095])


def rod(direction, n=800):
    rng = np.random.default_rng(1)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    a = np.cross(d, [0.3, 0.2, 0.9])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    return np.outer(rng.uniform(-1, 1, n), d) + 0.1 * (np.outer(rng.uniform(-1, 1, n), a) + np.outer(rng.uniform(-1, 1, n), b))


def test_lying_frame_lays_a_tilted_rod_level():
    V = rod([1, 0.2, 0.7])
    R = summon.lying_frame(V)
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(R) == pytest.approx(1)
    O = V @ R.T
    assert np.ptp(O[:, 0]) > 5 * np.ptp(O[:, 2])
    assert np.ptp(O[:, 2]) < 0.45


def test_lying_frame_keeps_a_flat_things_turn():
    rng = np.random.default_rng(2)
    V = np.c_[rng.uniform(-1, 1, 500), rng.uniform(-0.9, 0.9, 500), rng.uniform(-0.05, 0.05, 500)]
    t = math.radians(12)
    Q = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    O = (V @ Q.T) @ summon.lying_frame(V @ Q.T).T
    assert np.ptp(O[:, 2]) == pytest.approx(0.1, abs=0.02)


def test_best_rotation_finds_a_rod_lying_diagonally():
    # model along Z (as Hunyuan gave the marker); the mask shows it diagonal, top-left to bottom-right
    V = rod([0, 0, 1])
    mask = np.zeros((200, 200), bool)
    for t in np.linspace(40, 160, 400):
        mask[int(t) - 6:int(t) + 6, int(t) - 6:int(t) + 6] = True
    R, iou = summon.best_rotation(V, mask)
    assert iou > 0.6
    axis = R @ np.array([0, 0, 1.0])     # (right, up, toward)
    assert abs(axis[0]) == pytest.approx(abs(axis[1]), abs=0.2)
    assert axis[0] * axis[1] < 0         # right goes with down


def test_rotation_grid_is_proper():
    Rs = summon.rotation_grid(12, 4)
    assert np.linalg.det(Rs) == pytest.approx(np.ones(len(Rs)))


def test_height_of_a_long_thing_is_its_width():
    assert summon.shape_height(0.02, 0.14, 0.015, False) == pytest.approx(0.015)
    assert summon.shape_height(0.03, 0.08, 0.06, True) == pytest.approx(0.03)
    assert summon.shape_height(0.001, 0.05, 0.04, False) == pytest.approx(summon.MIN_HEIGHT)


def test_a_marker_weighs_about_15_g():
    vol = 0.14 * 0.015 * 0.015 * 1.5
    assert 0.003 < summon.mass_guess(vol, summon.density({"name": "マーカーペン"})) < 0.03
    hull = 0.139 * 0.024 * 0.019
    assert 0.012 < summon.mass_guess(hull, summon.density({"name": "マーカーペン"})) < 0.03


def test_density_soft_is_light():
    assert summon.density({"name": "紙", "deformable": True}) == summon.SOFT_DENSITY


def world(where):
    return {"boxes": [{"name": "keep_box", "role": "keep", "centre": [-0.3, -0.25], "yaw": 0.0, "outer": [0.25, 0.25],
                       "wall": 0.006, "height": 0.055, "floor": -0.045}],
            "objects": [{"id": "a", "centre": where, "z": 0.0}, {"id": "b", "centre": [0.3, 0.1], "z": 0.0}]}


def test_truth_of_success_is_the_physics_not_the_claim():
    start = world([0.3, 0.0])
    t = rehearse.truth_of(world([-0.3, -0.25]), "a", "keep", start)
    assert t["success"] and t["where"] == "keep_box" and t["disturbed"] == []
    t = rehearse.truth_of(world([0.31, 0.0]), "a", "keep", start)
    assert not t["success"] and t["where"] == "desk"
    assert rehearse.result_code(t, "moved_to_keep") == "failed"     # the robot's belief does not count
    assert rehearse.result_code(t, "slipped") == "slipped"


def test_truth_wrong_box_is_a_failure():
    t = rehearse.truth_of(world([-0.3, -0.25]), "a", "trash", world([0.3, 0.0]))
    assert not t["success"]


def test_truth_sees_a_pushed_neighbour():
    w = world([-0.3, -0.25])
    w["objects"][1]["centre"] = [0.33, 0.1]
    assert rehearse.truth_of(w, "a", "keep", world([0.3, 0.0]))["disturbed"] == ["b"]


def test_truth_text_says_the_robot_was_wrong():
    t = rehearse.truth_of(world([0.31, 0.0]), "a", "keep", world([0.3, 0.0]))
    assert "believed" in rehearse.truth_text(t, "moved_to_keep")


def tactic(**k):
    return {"force": 3, "opening": "snug", "soft": False, "turn": 0, "sweep_first": False, "rim": False, **k}


def test_untried_keeps_a_new_tactic():
    assert rehearse.untried(tactic(force=4), [{"tactic": tactic()}], {}) == tactic(force=4)


def test_untried_changes_a_repeated_one():
    assert rehearse.untried(tactic(), [{"tactic": tactic()}], {}) == tactic(opening="wide")


def row(id, xy, ok=True, disturbed=()):
    return {"id": id, "name": id, "xy": xy, "ok": ok, "tactic": tactic(), "disturbed": list(disturbed)}


def test_plan_order_near_first_and_failures_last():
    rows = rehearse.order_plan([row("far", [0.4, 0.0]), row("bad", [0.2, 0.0], ok=False), row("near", [0.25, 0.0])])
    assert [r["id"] for r in rows] == ["near", "far", "bad"]


def test_plan_order_a_pusher_goes_after_what_it_pushes():
    rows = rehearse.order_plan([row("near", [0.25, 0.0], disturbed=["far"]), row("far", [0.4, 0.0])])
    assert [r["id"] for r in rows] == ["far", "near"]


def test_write_plan_lists_what_is_left(tmp_path):
    plan = rehearse.write_plan(tmp_path / "plan.json", [row("a", [0.3, 0.0]), row("b", [0.2, 0.1], ok=False)], "look")
    on_disk = json.loads((tmp_path / "plan.json").read_text())
    assert on_disk == plan and on_disk["left_for_human"] == ["b"]
    assert "force" in on_disk["objects"][0]["tactic"]


def test_rehearse_real_arm_needs_explicit_flags(monkeypatch):
    """--real --execute without a preflight must stop before anything starts."""
    monkeypatch.setattr(sys, "argv", ["rehearse.py", "--real", "--out", "/nonexistent/x", "--execute"])
    with pytest.raises(SystemExit):
        rehearse.main()


def test_plan_picks_the_things_by_place_and_leaves_the_rest(tmp_path):
    from types import SimpleNamespace as NS
    import tidy_desk
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"objects": [
        {"name": "b", "xy": [0.40, 0.0], "tactic": tactic(force=4), "ok": True},
        {"name": "a", "xy": [0.30, 0.0], "tactic": tactic(), "ok": True},
        {"name": "no", "xy": [0.30, 0.2], "tactic": tactic(), "ok": False}]}))
    objs = [{"name": "A", "xy": [0.31, 0.0], "category": "keep"}, {"name": "B", "xy": [0.41, 0.01], "category": "trash"},
            {"name": "C", "xy": [0.30, 0.2], "category": "keep"}, {"name": "box", "xy": [0.3, 0.0], "category": "fixture"}]
    tidy_desk.Desk.apply_plan(NS(a=NS(plan=str(plan))), objs)
    assert objs[0]["plan_rank"] == 1 and objs[1]["plan_rank"] == 0
    assert objs[1]["plan_tactic"]["force"] == 4
    assert "plan_rank" not in objs[2] and "plan_rank" not in objs[3]


def test_no_plan_changes_nothing():
    from types import SimpleNamespace as NS
    import tidy_desk
    objs = [{"name": "A", "xy": [0.31, 0.0], "category": "keep"}]
    tidy_desk.Desk.apply_plan(NS(a=NS(plan=None)), objs)
    assert objs == [{"name": "A", "xy": [0.31, 0.0], "category": "keep"}]


def attempt_args(tmp_path, **k):
    from types import SimpleNamespace as NS
    d = dict(out=tmp_path, reflect="claude", driver_config="cfg.yaml", tidy_args="--max-speed 0.2", real=False, world=None, no_record=False,
             preflight=tmp_path / "p.json", preflight_cmd=None, videos=tmp_path / "videos", desk=tmp_path / "desk.json")
    d.update(k)
    return NS(**d)


def test_real_attempt_command_on_the_real_branch(tmp_path, monkeypatch):
    """The branch that drives the arm, with the arm and the cameras stubbed out: what would be run."""
    seen = {}
    filmed = []

    class Film:
        def __init__(self, a, n):
            filmed.append(n)

        def stop(self):
            filmed.append("stopped")

    def run(cmd, **k):
        seen["cmd"] = cmd
        seen["env"] = k["env"]
        return type("R", (), {"returncode": 0})

    monkeypatch.setattr(rehearse, "Filming", Film)
    monkeypatch.setattr(rehearse, "fresh_preflight", lambda a: seen.setdefault("preflight", True))
    monkeypatch.setattr(rehearse.subprocess, "run", run)
    monkeypatch.setenv("TIDY_EXTRA_OBJECTS", "/x")
    assert rehearse.real_attempt(attempt_args(tmp_path, real=True), 2, tmp_path) == 0
    cmd = seen["cmd"]
    assert seen["preflight"] and filmed == [2, "stopped"]
    assert "--sim" not in cmd and cmd[cmd.index("--preflight") + 1].endswith("p.json")
    assert cmd[cmd.index("--snap-cameras") + 1] == ""       # ffmpeg holds both side cameras
    assert cmd[cmd.index("--attempts") + 1] == "1" and cmd[cmd.index("--rounds") + 1] == "1"
    assert cmd[cmd.index("--max-speed") + 1] == "0.2"
    assert "TIDY_EXTRA_OBJECTS" not in seen["env"]


def test_real_attempt_stops_filming_when_tidy_crashes(tmp_path, monkeypatch):
    filmed = []

    class Film:
        def __init__(self, a, n):
            pass

        def stop(self):
            filmed.append("stopped")

    def run(cmd, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(rehearse, "Filming", Film)
    monkeypatch.setattr(rehearse, "fresh_preflight", lambda a: None)
    monkeypatch.setattr(rehearse.subprocess, "run", run)
    with pytest.raises(KeyboardInterrupt):
        rehearse.real_attempt(attempt_args(tmp_path, real=True), 1, tmp_path)
    assert filmed == ["stopped"]


def test_old_preflight_never_lets_the_arm_start(tmp_path, monkeypatch):
    import os
    a = attempt_args(tmp_path, real=True)
    a.preflight.write_text("{}")
    os.utime(a.preflight, (1000, 1000))
    monkeypatch.setattr(rehearse, "PREFLIGHT_WAIT", -1)
    with pytest.raises(SystemExit):
        rehearse.fresh_preflight(a)


def test_fresh_preflight_passes(tmp_path):
    a = attempt_args(tmp_path, real=True)
    a.preflight.write_text("{}")
    rehearse.fresh_preflight(a)


def test_outcomes_read_the_report(tmp_path):
    (tmp_path / "execute").mkdir()
    (tmp_path / "plan.json").write_text(json.dumps({"objects": [row("a", [0.3, 0.0])]}))
    (tmp_path / "execute/report.json").write_text(json.dumps({"status": "finished", "objects": [
        {"name": "a", "xy": [0.3, 0.01], "result": "slipped", "attempts": [{"tactic": tactic(force=5)}]}]}))
    rows, status = rehearse.outcomes(None, 3, tmp_path)
    assert status == "finished"
    assert rows == [{"attempt": 3, "name": "a", "xy": [0.3, 0.01], "tactic": tactic(force=5), "result": "slipped"}]


def test_rim_frame_levels_a_leaning_cup():
    rng = np.random.default_rng(3)
    t = rng.uniform(0, 2 * math.pi, 1500)
    z = rng.uniform(0, 1, 1500)
    V = np.c_[0.5 * np.cos(t), 0.5 * np.sin(t), z]      # a tube, its rim at z = 1
    a = math.radians(25)
    Q = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    O = (V @ Q.T) @ summon.rim_frame(V @ Q.T).T
    assert np.ptp(O[:, 2]) == pytest.approx(1.0, abs=0.08)
