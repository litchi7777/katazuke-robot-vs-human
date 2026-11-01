import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arm_goto
import pick_pen_auto as p
import topdown_ik

ROOT = Path(__file__).resolve().parents[2]
LIMITS = arm_goto.driver_limits(ROOT / "scripts/config/arm_pick_trial.yaml")


class FakeArm:
    """Tip follows the command with a constant sag but cannot go below `surface`."""

    def __init__(self, surface, sag=0.002):
        self.surface, self.sag, self.z = surface, sag, None
        self.sent = []

    def move(self, steps):
        def send(q):
            self.sent.append(q)
            z_cmd = next(z for z, qq in steps if qq is q)
            self.z = max(z_cmd + self.sag, self.surface)
        return send


class ProbeTests(unittest.TestCase):
    def setUp(self):
        # The pen seen in the box on 2026-10-27: radial 0.48 m, needs a tilted grasp.
        self.g, self.steps = p.probe_steps(np.array([0.426, 0.213]), np.array([1.0, 0.6]), LIMITS,
                                           0.008, 0.026)

    def test_probe_line_is_smooth_and_descending(self):
        zs = [z for z, _ in self.steps]
        self.assertGreater(self.g["pitch"], 0.0)
        self.assertTrue(all(a > b for a, b in zip(zs, zs[1:])))
        self.assertAlmostEqual(zs[-1], 0.008 + p.TIP_REACH - p.PROBE_DEPTH, places=6)
        self.assertGreaterEqual(zs[0], 0.026 + p.PROBE_START - 1e-6)
        for (_, a), (_, b) in zip(self.steps, self.steps[1:]):
            self.assertLess(max(abs(x - y) for x, y in zip(a, b)), 0.03)
        self.assertGreater(arm_goto.check_path(arm_goto.plan(self.steps[0][1], self.steps[-1][1], LIMITS),
                                               zs[-1] - 0.002), zs[-1] - 0.002)

    def test_floor_contact_stops_within_one_lag(self):
        arm = FakeArm(surface=0.013)
        i, z, baseline, touched = p.probe(self.steps, arm.move(self.steps), lambda: arm.z)
        self.assertTrue(touched)
        self.assertAlmostEqual(z, 0.013)
        self.assertAlmostEqual(baseline, 0.002)
        self.assertGreater(self.steps[i][0] + baseline, 0.013 - p.PROBE_LAG - p.PROBE_STEP - 1e-9)
        j = p.back_off(self.steps, z, baseline)
        self.assertLess(abs(self.steps[j][0] + baseline - (z + p.PROBE_LIFT)), p.PROBE_STEP)

    def test_landing_on_the_pen_is_high(self):
        arm = FakeArm(surface=0.031)  # finger on top of an 18 mm pen on an 8 mm floor
        _, z, _, touched = p.probe(self.steps, arm.move(self.steps), lambda: arm.z)
        self.assertTrue(touched)
        self.assertGreater(z, 0.008 + p.TIP_REACH + p.ON_OBJECT)

    def test_no_contact_reaches_the_deepest_step(self):
        arm = FakeArm(surface=-1.0)
        i, _, _, touched = p.probe(self.steps, arm.move(self.steps), lambda: arm.z)
        self.assertFalse(touched)
        self.assertEqual(i, len(self.steps) - 1)

    def test_table_pen_probe_is_top_down(self):
        g, steps = p.probe_steps(np.array([0.30, 0.0]), np.array([1.0, 0.3]), LIMITS, 0.0, 0.015)
        self.assertEqual(g["pitch"], 0.0)
        self.assertAlmostEqual(steps[-1][0], p.TIP_REACH - p.PROBE_DEPTH, places=6)
        self.assertAlmostEqual(topdown_ik.fk(steps[-1][1]).translation[0], 0.30, places=3)


if __name__ == "__main__":
    unittest.main()
