import contextlib
import io
import math
import json
import sys
import time
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint1_range_trial as trial


class PlanTests(unittest.TestCase):
    def test_single_axis_plan_has_no_targets_for_other_motors(self):
        points = trial.bounded_plan([-0.017], [-0.017], 0.05, -0.05, 0.05)
        self.assertTrue(all(len(positions) == 1 for _, positions in points))
        self.assertAlmostEqual(points[-1][1][0], 0.033)
    def test_preflight_rejects_error_or_missing_motor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            data = dict(status="stopped", error_frames=1, initial_readings=[])
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                trial.verified_preflight(path)
            data.update(status="ok", error_frames=0)
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                trial.verified_preflight(path)

    def test_stale_preflight_rejected_even_if_all_positions_succeeded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(dict(status="ok", error_frames=0, initial_readings=[
                dict(motor_id=i, parameter="0x7019", status="ok", value=0) for i in range(1, 8)])))
            self.assertEqual(trial.verified_preflight(path)["status"], "ok")
            with patch.object(trial.time, "time", return_value=path.stat().st_mtime + 301):
                with self.assertRaises(ValueError):
                    trial.verified_preflight(path)

    def test_only_joint1_target_changes_and_commanded_speed_is_bounded(self):
        anchor = [0, 0.01, 0.02, -0.01, 0.01, 0]
        points = trial.bounded_plan(anchor, anchor, 0.05, -0.05, 0.05)
        self.assertEqual(points[0][1], anchor)
        self.assertEqual(points[-1][1], [0.05, *anchor[1:]])
        for (t0, q0), (t1, q1) in zip(points, points[1:]):
            self.assertEqual(q1[1:], anchor[1:])
            self.assertLessEqual(abs(q1[0] - q0[0]) / (t1 - t0), trial.MAX_SPEED + 1e-9)
        self.assertGreaterEqual(points[-1][0], 3)

    def test_large_step_window_and_absolute_limit_violations_rejected(self):
        for current, anchor, offset, lower, upper in (
            ([0]*6, [0]*6, 0.1, -0.15, 0.15),
            ([0]*6, [0]*6, 0.051, -0.05, 0.05),
            ([2.79, 0, 0, 0, 0, 0], [2.79, 0, 0, 0, 0, 0], 0.05, -0.05, 0.05),
        ):
            with self.subTest(offset=offset, anchor=anchor):
                with self.assertRaises(ValueError):
                    trial.bounded_plan(current, anchor, offset, lower, upper)

    def test_nan_and_other_joint_drift_prevent_goal_creation(self):
        for current, offset in (([0]*6, math.nan), ([0, 0.041, 0, 0, 0, 0], 0.01)):
            with self.assertRaises(ValueError):
                trial.bounded_plan(current, [0]*6, offset, -0.05, 0.05)

    def test_no_ros_initialization_or_output_creation_by_default(self):
        with patch.object(trial, "execute") as execute, patch.object(Path, "mkdir") as mkdir, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(trial.main([]), 0)
        execute.assert_not_called()
        mkdir.assert_not_called()

    def test_non_finite_or_large_session_window_rejected(self):
        for args in (["--upper-offset", "nan"], ["--upper-offset", "0.2"],
                     ["--lower-offset", "0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trial.main(args)


class CancelTests(unittest.TestCase):
    def test_moveit_cancel_also_invokes_controller_stop_on_late_acceptance(self):
        stop = Mock()
        track = trial.TrackedMoveItGoal(lambda *a, **kw: None, stop)
        send, result = Future(), Future()
        handle = Mock(accepted=True)
        handle.get_result_async.return_value = result
        track.attach(send)
        track.cancel()
        send.set_result(handle)
        self.assertGreaterEqual(stop.call_count, 2)
        handle.cancel_goal_async.assert_called_once()
        result.set_result(SimpleNamespace(status=6, result=SimpleNamespace(error_code=SimpleNamespace(val=-7))))
        self.assertTrue(track.terminal)
        self.assertFalse(track.success)

    def test_moveit_success_requires_success_status_and_error_code(self):
        track = trial.TrackedMoveItGoal(lambda *a, **kw: None)
        future = Future()
        future.set_result(SimpleNamespace(status=4, result=SimpleNamespace(error_code=SimpleNamespace(val=1))))
        track._result(future)
        self.assertTrue(track.success)

    def setUp(self):
        self.events = []
        self.track = trial.TrackedGoal(lambda event, **fields: self.events.append((event, fields)))
        self.send = Future()
        self.result = Future()
        self.cancel = Future()
        self.handle = Mock(accepted=True)
        self.handle.get_result_async.return_value = self.result
        self.handle.cancel_goal_async.return_value = self.cancel

    def test_cancel_before_acceptance_cancels_late_accepted_goal(self):
        self.track.attach(self.send)
        self.track.cancel()
        self.assertFalse(self.track.terminal)
        self.send.set_result(self.handle)
        self.handle.cancel_goal_async.assert_called_once_with()
        self.assertFalse(self.track.terminal)

    def test_cancel_ack_is_not_terminal_and_repeated_cancel_sends_once(self):
        self.track.attach(self.send)
        self.send.set_result(self.handle)
        self.track.cancel()
        self.track.cancel()
        self.cancel.set_result(SimpleNamespace(return_code=0, goals_canceling=[object()]))
        self.assertFalse(self.track.terminal)
        self.handle.cancel_goal_async.assert_called_once_with()
        self.result.set_result(SimpleNamespace(status=5, result=SimpleNamespace(
            error_code=0, error_string="canceled")))
        self.assertTrue(self.track.terminal)
        self.assertFalse(self.track.success)

    def test_result_code_zero_does_not_mask_canceled_status(self):
        self.track.attach(self.send)
        self.send.set_result(self.handle)
        self.result.set_result(SimpleNamespace(status=5, result=SimpleNamespace(
            error_code=0, error_string="canceled")))
        self.assertTrue(self.track.terminal)
        self.assertFalse(self.track.success)

    def test_response_error_remains_unresolved(self):
        self.track.attach(self.send)
        self.send.set_exception(RuntimeError("unknown server acceptance"))
        self.track.cancel()
        self.assertFalse(self.track.terminal)

    def test_cancel_rejection_still_waits_for_terminal_result(self):
        self.track.attach(self.send)
        self.send.set_result(self.handle)
        self.track.cancel()
        self.cancel.set_result(SimpleNamespace(return_code=1, goals_canceling=[]))
        self.assertFalse(self.track.terminal)


class WatchdogTests(unittest.TestCase):
    def test_isolated_client_requires_single_axis_driver_and_not_gripper_telemetry(self):
        node = self.node()
        node.joint1_only = True
        node.positions = [0.01]
        node.gripper = None
        node.gripper_received = 0
        node.status.joint_names = ["joint1"]
        self.assertEqual(trial.fresh_snapshot(node), ([0.01], 0.0))
        node.status.joint_names = list(trial.JOINTS)
        with self.assertRaises(RuntimeError):
            trial.fresh_snapshot(node)

    def node(self):
        return SimpleNamespace(positions=[0]*6, gripper=1,
                               joint_received=time.monotonic(), gripper_received=time.monotonic(),
                               status=SimpleNamespace(enabled=True, control_loop_active=True,
                                                      error_codes=[], state_machine="IDLE"),
                               invalid_message="")

    def test_stale_telemetry_disabled_driver_or_error_prevent_motion(self):
        for change in (lambda node: setattr(node, "joint_received", 0),
                       lambda node: setattr(node, "gripper_received", 0),
                       lambda node: setattr(node.status, "enabled", False),
                       lambda node: setattr(node.status, "state_machine", "ERROR"),
                       lambda node: setattr(node.status, "error_codes", ["SHUTDOWN_FAILED"])):
            node = self.node()
            change(node)
            with self.assertRaises(RuntimeError):
                trial.fresh_snapshot(node)

    def test_drift_of_non_commanded_axes_or_gripper_is_detected(self):
        for q, grip in (([0, 0, 0.041, 0, 0, 0], 1), ([0]*6, 1.081),
                        ([0.081, 0, 0, 0, 0, 0], 1), ([math.nan]*6, 1)):
            with self.assertRaises(RuntimeError):
                trial.check_drift(q, grip, [0]*6, 1, -0.05, 0.05)

    def test_arm_hold_client_requires_six_joint_driver_and_no_gripper_telemetry(self):
        node = self.node()
        node.arm_hold = True
        node.gripper = None
        node.gripper_received = 0
        node.status.joint_names = list(trial.JOINTS)
        self.assertEqual(trial.fresh_snapshot(node), ([0] * 6, 0.0))
        node.status.joint_names = ["joint1"]
        with self.assertRaises(RuntimeError):
            trial.fresh_snapshot(node)

    def test_moving_joint_window_applies_to_its_index_and_others_are_watched(self):
        anchor = [0.0, 0.0, 0.002, 0.0, -0.072, 0.05]
        trial.check_drift([0.0, 0.0, 0.06, 0.0, -0.072, 0.05], 0, anchor, 0, -0.01, 0.05, 2)
        for q in ([0.0, 0.0, 0.09, 0.0, -0.072, 0.05],    # joint3 beyond window + 0.03
                  [0.041, 0.0, 0.03, 0.0, -0.072, 0.05]):  # joint1 drifted
            with self.assertRaises(RuntimeError):
                trial.check_drift(q, 0, anchor, 0, -0.01, 0.05, 2)


class ArmHoldPlanTests(unittest.TestCase):
    def test_only_the_selected_joint_changes_and_speed_is_bounded(self):
        anchor = [0.0, -0.015, 0.002, 0.001, -0.072, 0.05]
        points = trial.bounded_plan(anchor, anchor, 0.04, -0.01, 0.05, 2)
        for _, positions in points:
            self.assertEqual(trial.others(positions, 2), trial.others(anchor, 2))
        self.assertAlmostEqual(points[-1][1][2], 0.042)
        speeds = [abs(b[1][2] - a[1][2]) / (b[0] - a[0]) for a, b in zip(points, points[1:])]
        self.assertLessEqual(max(speeds), trial.MAX_SPEED + 1e-9)

    def test_step_window_drift_and_index_are_checked_for_the_selected_joint(self):
        anchor = [0.0] * 6
        for kwargs in (dict(offset=0.06, upper=0.08),            # one step > 0.05
                       dict(offset=-0.02),                       # below window
                       dict(current=[0.0, 0.05, 0, 0, 0, 0]),    # joint2 drifted
                       dict(index=6)):
            values = dict(current=anchor, offset=0.04, lower=-0.01, upper=0.05, index=2)
            values.update(kwargs)
            with self.assertRaises(ValueError):
                trial.bounded_plan(values["current"], anchor, values["offset"],
                                   values["lower"], values["upper"], values["index"])

    def test_cli_rejects_mixed_modes_and_reports_selected_joint(self):
        for argv in (["--arm-hold", "--joint1-only"], ["--move-joint", "joint3"],
                     ["--arm-hold", "--moveit-profile", "p.json"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trial.main(argv)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(trial.main(["--arm-hold", "--move-joint", "joint3",
                                         "--lower-offset", "-0.01"]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual((plan["joint"], plan["namespace"], plan["arm_hold"]),
                         ("joint3", "rebotarm_arm_hold_trial", True))


if __name__ == "__main__":
    unittest.main()
