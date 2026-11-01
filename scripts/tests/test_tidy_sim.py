import json
import math
import sys
import tempfile
import unittest
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arm_goto
import confirm_server
import desk_judge
import plan_pen_pick
import sim_campaign
import sim_scenarios
import sim_world
import tidy_desk
import topdown_ik

ROOT = Path(__file__).resolve().parents[2]
LIMITS = arm_goto.driver_limits(ROOT / "scripts/config/arm_pick_trial.yaml")


class SimWorldTests(unittest.TestCase):
    def test_gap_model_matches_the_two_measurements(self):
        self.assertLess(abs(sim_world.gap(0.186) - 0.008), 0.0015)  # box wall, 2026-10-27
        self.assertLess(abs(sim_world.gap(0.849) - 0.018), 0.0005)  # marker pen
        self.assertLess(abs(sim_world.gap(5.9) - 0.100), 0.001)     # fully open, ruler 2026-10-28
        self.assertAlmostEqual(sim_world.rad_for_gap(sim_world.gap(1.2)), 1.2)

    def test_folded_arm_touches_nothing_and_a_low_tip_touches_the_desk(self):
        world = sim_scenarios.base_world()
        self.assertEqual(sim_world.collisions(world, [0.0] * 6, 1.0), set())
        q, err, _ = topdown_ik.solve([0.30, 0.0, -0.002], seeds=plan_pen_pick.SEEDS)
        self.assertLess(err, 1e-3)
        self.assertIn("desk", sim_world.collisions(world, q, 1.0))

    def test_a_dropped_object_lands_on_the_box_floor(self):
        world = sim_scenarios.base_world()
        box = world["boxes"][0]
        z, owner = sim_world.surface_below(world, box["centre"])
        self.assertEqual((z, owner), (box["floor"], box["name"]))


class GraspPlanTests(unittest.TestCase):
    def test_finger_line_stays_level_and_the_object_fits(self):
        for xy, axis in (((0.30, 0.0), (1.0, 0.3)), ((0.38, 0.12), (0.2, 1.0)), ((0.426, 0.213), (1.0, 0.6))):
            g = plan_pen_pick.plan_grasp(np.array(xy), np.array(axis), 0.0, LIMITS, size=(0.05, 0.022),
                                         max_across=0.0274)
            self.assertIsNotNone(g, xy)
            R = topdown_ik.fk(g["grasp"]).rotation
            self.assertLessEqual(abs(R[2, 1]), plan_pen_pick.FINGER_TILT_MAX + 1e-6)
            self.assertLessEqual(g["across"], 0.0274)

    def test_accept_can_veto_every_candidate(self):
        g = plan_pen_pick.plan_grasp(np.array([0.30, 0.0]), np.array([1.0, 0.0]), 0.0, LIMITS,
                                     accept=lambda c: False)
        self.assertIsNone(g)


class GripperOpenTests(unittest.TestCase):
    def test_opening_goes_in_small_steps_and_ends_on_target(self):
        sent = []
        arm_goto.soft_open(lambda t: sent.append(t) or t, 0.2, 5.8)
        self.assertAlmostEqual(sent[-1], 5.8)
        self.assertTrue(all(abs(b - a) <= arm_goto.GRIP_OPEN_STEP + 1e-9 for a, b in zip([0.2] + sent, sent)))
        sent.clear()
        arm_goto.soft_open(lambda t: sent.append(t) or t, 1.0, 1.0)
        self.assertEqual(sent, [1.0])


class PresenceTests(unittest.TestCase):
    def test_pixels_of_a_removed_object_read_as_desk(self):
        desk = np.full((100, 100, 3), (150, 145, 146), np.uint8)
        with_obj = desk.copy()
        with_obj[40:60, 40:60] = (40, 40, 200)
        pixels = np.array([(x, y) for y in range(40, 60) for x in range(40, 60)])
        self.assertGreater(desk_judge.presence(with_obj, pixels), 0.9)
        self.assertLess(desk_judge.presence(desk, pixels), 0.1)


class ConfirmPageTests(unittest.TestCase):
    def test_a_click_is_written_as_the_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "request.json").write_text(json.dumps({"round": 1, "items": [
                {"key": "r1_1", "name": "丸めた紙", "category_ja": "ゴミ", "confidence": 0.8, "reason": "紙くず",
                 "image": "r1_1.jpg", "suggest": "discard"}]}, ensure_ascii=False))
            server = confirm_server.serve(folder, "127.0.0.1", 0)
            port = server.server_address[1]
            try:
                page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
                self.assertIn("丸めた紙", page)
                data = urllib.parse.urlencode({"key": "r1_1", "d": "discard"}).encode()
                urllib.request.urlopen(f"http://127.0.0.1:{port}/decide", data)
                self.assertEqual(json.loads((folder / "response.json").read_text())["decisions"], {"r1_1": "discard"})
                bogus = urllib.parse.urlencode({"key": "nope", "d": "discard"}).encode()
                urllib.request.urlopen(f"http://127.0.0.1:{port}/decide", bogus)
                self.assertNotIn("nope", json.loads((folder / "response.json").read_text())["decisions"])
            finally:
                server.shutdown()


class TidySimTests(unittest.TestCase):
    def test_one_scene_ends_with_everything_in_its_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            world_path = Path(tmp) / "world.json"
            sim_world.save(sim_scenarios.random_world(1), world_path)
            tidy_desk.main(["--sim", str(world_path), "--judge", "oracle", "--approve", "yes",
                            "--out", str(Path(tmp) / "run"), "--grip-open-max", "2.4"])
            report = json.loads((Path(tmp) / "run/report.json").read_text())
            self.assertEqual(report["status"], "finished")
            rows = sim_campaign.score(sim_world.load(world_path), "yes")
            self.assertTrue(all(r["ok"] for r in rows), rows)

    def test_refused_discards_stay_on_the_desk(self):
        with tempfile.TemporaryDirectory() as tmp:
            world_path = Path(tmp) / "world.json"
            sim_world.save(sim_scenarios.random_world(1), world_path)
            tidy_desk.main(["--sim", str(world_path), "--judge", "oracle", "--approve", "no", "--auto-discard", "1.1",
                            "--out", str(Path(tmp) / "run"), "--grip-open-max", "2.4"])
            world = sim_world.load(world_path)
            for o in world["objects"]:
                if o["category"] == "trash":
                    self.assertLess(math.dist(o["centre"], o["start"]), 0.01, o["id"])
                    self.assertIsNone(sim_world.container_of(world, o))


if __name__ == "__main__":
    unittest.main()
