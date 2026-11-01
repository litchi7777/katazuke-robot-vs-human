import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import desk_judge  # noqa: E402
import sim_world  # noqa: E402

CALIB = json.loads((Path(__file__).resolve().parents[1] / "calib/video2.json").read_text())


def bottle_points(centre, radius, height):
    """Surface points of an upright cylinder the overhead camera can see: the top disc and its near side."""
    camera = np.linalg.inv(np.array(CALIB["T_cam_base"]))[:2, 3]
    pts = []
    for r in np.linspace(0, radius, 6):
        for a in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            pts.append([centre[0] + r * math.cos(a), centre[1] + r * math.sin(a), height])
    for z in np.linspace(0.005, height, 12):
        for a in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            normal = np.array([math.cos(a), math.sin(a)])
            if normal @ (camera - np.array(centre)) > 0:  # the side facing the camera
                pts.append([centre[0] + radius * normal[0], centre[1] + radius * normal[1], z])
    return np.array(pts)


class Measure3dTests(unittest.TestCase):
    def test_tall_bottle_centre_and_width(self):
        xyz = bottle_points((0.34, 0.15), 0.024, 0.084)
        px = sim_world._project(CALIB, xyz)
        m = desk_judge.measure_3d(CALIB, px, xyz[:, 2])
        self.assertLess(math.dist(m["xy"], (0.34, 0.15)), 0.002)
        self.assertAlmostEqual(m["width"], 0.048, delta=0.006)
        self.assertAlmostEqual(m["top"], 0.084, delta=0.002)

    def test_one_plane_guess_is_worse_for_a_tall_bottle(self):
        # What locate_all did before: every pixel on the plane at 0.8 of the height.
        xyz = bottle_points((0.34, 0.15), 0.024, 0.084)
        px = sim_world._project(CALIB, xyz)
        flat = desk_judge.pixels_to_base(CALIB, px, np.full(len(px), 0.8 * 0.084)).mean(0)
        m = desk_judge.measure_3d(CALIB, px, xyz[:, 2])
        self.assertLess(math.dist(m["xy"], (0.34, 0.15)), math.dist(flat, (0.34, 0.15)))

    def test_too_few_readings(self):
        self.assertIsNone(desk_judge.measure_3d(CALIB, np.zeros((3, 2)), np.zeros(3)))


if __name__ == "__main__":
    unittest.main()
