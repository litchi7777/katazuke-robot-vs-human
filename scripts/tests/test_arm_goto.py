import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arm_goto

ROOT = Path(__file__).resolve().parents[2]
LIMITS = arm_goto.driver_limits(ROOT / "scripts/config/arm_move_trial.yaml")


class PlanTests(unittest.TestCase):
    def test_peak_speed_and_endpoints(self):
        target = [0.0, 1.754, 1.582, -1.399, 0.0, 0.0]
        points = arm_goto.plan([0.0] * 6, target, LIMITS)
        self.assertEqual(points[0][1], [0.0] * 6)
        self.assertEqual(points[-1][1], target)
        for (t0, q0), (t1, q1) in zip(points, points[1:]):
            self.assertLessEqual(max(abs(b - a) for a, b in zip(q0, q1)) / (t1 - t0),
                                 arm_goto.MAX_SPEED + 1e-9)

    def test_targets_outside_driver_limits_or_nan_are_rejected(self):
        for target in ([0, 3.0, 0, 0, 0, 0], [0, 0, 0, 0.5, 0, 0], [0, 0, 0, 0, 0, float("nan")], [0] * 5):
            with self.assertRaises(ValueError):
                arm_goto.plan([0.0] * 6, target, LIMITS)

    def test_commanded_position_interpolates_and_holds_the_end(self):
        points = [(0.0, [0.0] * 6), (1.0, [1.0] * 6)]
        self.assertEqual(arm_goto.commanded_at(points, -1), [0.0] * 6)
        self.assertAlmostEqual(arm_goto.commanded_at(points, 0.25)[2], 0.25)
        self.assertEqual(arm_goto.commanded_at(points, 5), [1.0] * 6)

    def test_named_and_numeric_targets(self):
        poses = {"ready": [0, 0.75, 0.55, 0, 0, 0]}
        self.assertEqual(arm_goto.parse_target(["goto", "ready"], poses), [0, 0.75, 0.55, 0, 0, 0])
        self.assertEqual(arm_goto.parse_target(["goto", *"0 1 2 3 4 5".split()], poses), [0, 1, 2, 3, 4, 5])
        for words in (["goto", "nowhere"], ["goto", "1", "2"]):
            with self.assertRaises(ValueError):
                arm_goto.parse_target(words, poses)

    def test_non_trial_driver_config_is_refused(self):
        with self.assertRaises(ValueError):
            arm_goto.driver_limits(ROOT / "scripts/config/joint1_range_trial.yaml")


class PathTests(unittest.TestCase):
    def test_saved_poses_keep_the_gripper_tip_above_the_board(self):
        # 0.07 m: the lowest calibration height used with --min-tcp-z 0.07.
        poses = json.loads((ROOT / "scripts/config/shogi_poses.json").read_text())
        for name, q in poses.items():
            lowest = arm_goto.check_path(arm_goto.plan([0.0] * 6, q, LIMITS), 0.07)
            self.assertGreaterEqual(lowest, 0.07, name)

    def test_path_below_minimum_height_is_rejected(self):
        # Shoulder and elbow fully raised with the wrist level put the tip near the table.
        points = arm_goto.plan([0.0] * 6, [0.0, 2.1, 2.0, -1.0, 0.0, 0.0], LIMITS)
        with self.assertRaises(ValueError):
            arm_goto.check_path(points, 0.30)

    def test_dry_run_reports_poses_without_ros(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(arm_goto.main(["--driver-config", str(ROOT / "scripts/config/arm_move_trial.yaml"),
                                            "--poses", str(ROOT / "scripts/config/shogi_poses.json")]), 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["status"], "prepared_only")
        self.assertIn("hover_x030", report["poses"])


class CorrectionTests(unittest.TestCase):
    def test_overshoots_by_the_residual_and_is_capped(self):
        target = [0.0, 1.9, 1.3, -0.96, 0.0, -0.72]
        measured = [0.0, 1.887, 1.311, -0.973, 0.0, -0.72]
        command = arm_goto.corrected_command(target, measured, LIMITS)
        self.assertAlmostEqual(command[1], 1.913)
        self.assertAlmostEqual(command[2], 1.289)
        far = arm_goto.corrected_command(target, [0.0, 1.8, 1.3, -0.96, 0.0, -0.72], LIMITS)
        self.assertAlmostEqual(far[1], 1.9 + arm_goto.CORRECT_MAX)

    def test_small_residual_needs_no_correction_and_limits_hold(self):
        self.assertIsNone(arm_goto.corrected_command([0.1] * 6, [0.099] * 6, LIMITS))
        command = arm_goto.corrected_command([0.0, 2.19, 0, 0, 0, 0], [0.0, 2.17, 0, 0, 0, 0], LIMITS)
        self.assertLessEqual(command[1], LIMITS[1][1])


class SoftCloseTests(unittest.TestCase):
    def run_close(self, contact_at, start=1.2):
        commands = []
        state = {"pos": start}

        def command(target):
            commands.append(target)
            state["pos"] = max(target, contact_at)  # the piece stops the fingers

        result = arm_goto.soft_close(lambda: state["pos"], command, start)
        return result, commands

    def test_contact_holds_only_a_small_squeeze_past_the_piece(self):
        (result, final), commands = self.run_close(contact_at=0.6)
        self.assertEqual(result, "contact")
        self.assertAlmostEqual(final, 0.6 - arm_goto.GRIP_SQUEEZE)
        # Never commanded more than one step plus the contact threshold past the piece.
        self.assertGreaterEqual(min(commands), 0.6 - arm_goto.GRIP_CONTACT - arm_goto.GRIP_STEP - 1e-9)
        self.assertTrue(all(b < a for a, b in zip(commands[:-2], commands[1:-1])))

    def test_squeeze_is_chosen_per_object(self):
        state = {"pos": 1.2}

        def command(target):
            state["pos"] = max(target, 0.6)

        result, final = arm_goto.soft_close(lambda: state["pos"], command, 1.2, squeeze=0.03)
        self.assertEqual(result, "contact")
        self.assertAlmostEqual(final, state["pos"] - 0.03 if state["pos"] > final else final)
        self.assertGreater(final, 0.6 - arm_goto.GRIP_SQUEEZE)

    def test_squeezes_again_when_the_object_turns_square(self):
        # First contact on the corners at 0.6; squeezed, it turns square and the fingers run on to 0.5.
        state = {"pos": 1.2, "stop": 0.6}
        commands = []

        def command(target):
            commands.append(target)
            if state["stop"] == 0.6 and target <= 0.6 - arm_goto.GRIP_SQUEEZE + 1e-9:
                state["stop"] = 0.5  # the squeeze turned it square
            state["pos"] = max(target, state["stop"])

        result, final = arm_goto.soft_close(lambda: state["pos"], command, 1.2)
        self.assertEqual(result, "contact")
        self.assertAlmostEqual(final, 0.5 - arm_goto.GRIP_SQUEEZE)

    def test_empty_gripper_closes_to_zero(self):
        (result, final), commands = self.run_close(contact_at=-1.0, start=0.3)
        self.assertEqual((result, final), ("closed_empty", 0.0))
        self.assertEqual(commands[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
