import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tidy_desk  # noqa: E402

MAX_ACROSS = 0.078


def top_view(height_at, half=0.08, step=0.002):
    """What an overhead depth camera sees: one point per grid cell at the surface height (0 = desk, unseen)."""
    pts = []
    for x in np.arange(-half, half, step):
        for y in np.arange(-half, half, step):
            h = height_at(x, y)
            if h > 0:
                pts.append([0.30 + x, y, h])
    return np.array(pts)


def bell(x, y):
    if abs(x) < 0.01 and abs(y) < 0.01:
        return 0.10  # handle 2 x 2 cm
    return 0.06 if math.hypot(x, y) < 0.045 else 0.0  # body 9 cm across


def box(x, y):
    return 0.03 if abs(x) < 0.06 and abs(y) < 0.05 else 0.0


def cup(x, y):
    r = math.hypot(x, y)
    if 0.040 <= r < 0.045:
        return 0.08  # wall
    return 0.005 if r < 0.040 else 0.0  # the inside floor


def plush(x, y):
    """A sitting soft toy: head high on one side, body and tail lower."""
    if math.hypot(x - 0.02, y) < 0.025:
        return 0.12
    return 0.06 if math.hypot(x, y) < 0.05 else 0.0


class PartGraspTests(unittest.TestCase):
    def test_rim_all_round_is_a_bowl_not_a_soft_heap(self):
        self.assertTrue(tidy_desk.rim_ring(top_view(cup), 0.08))
        self.assertFalse(tidy_desk.rim_ring(top_view(plush), 0.12))
        self.assertFalse(tidy_desk.rim_ring(top_view(box), 0.03))

    def test_bell_is_gripped_by_its_handle(self):
        xyz = top_view(bell)
        g = tidy_desk.part_grasps(xyz, 0.10, 0.10 - tidy_desk.TALL_GRIP, MAX_ACROSS)
        self.assertTrue(g)
        best = g[0]
        self.assertLess(best["across"], 0.03)
        self.assertLess(math.dist(best["c"], (0.30, 0.0)), 0.006)
        self.assertGreater(best["z"], 0.06)  # above the wide body

    def test_solid_wide_box_has_no_narrow_part(self):
        xyz = top_view(box)
        self.assertEqual(tidy_desk.part_grasps(xyz, 0.03, 0.005, MAX_ACROSS), [])
        self.assertFalse(tidy_desk.hollow(xyz, 0.03))

    def test_cup_wall_is_a_part_and_the_cup_is_hollow(self):
        xyz = top_view(cup)
        self.assertTrue(tidy_desk.hollow(xyz, 0.08))
        g = tidy_desk.part_grasps(xyz, 0.08, 0.08 - tidy_desk.TALL_GRIP, MAX_ACROSS)
        self.assertTrue(g)
        self.assertLess(g[0]["across"], 0.012)


if __name__ == "__main__":
    unittest.main()


def test_trash_whose_height_the_judge_got_wrong_is_asked_about():
    o = {"judged_height_cm": 5.0, "size_cm": [8, 8, 12.4], "height_from_depth": True}
    assert tidy_desk.misjudged_size(o)  # the plush taken for crumpled paper
    assert tidy_desk.misjudged_size({**o, "judged_height_cm": 8.0})  # the same plush, judged 8 cm another time
    assert not tidy_desk.misjudged_size({**o, "size_cm": [8, 8, 6.5]})  # a bit taller: same thing
    assert not tidy_desk.misjudged_size({**o, "judged_height_cm": 1.0, "size_cm": [8, 8, 2.5]})  # flat either way
    assert not tidy_desk.misjudged_size({"size_cm": [8, 8, 12.4]})  # no depth reading


def test_release_spots_spread_inside_the_box():
    box = {"centre": [-0.3, -0.25], "corners": [[-0.425, -0.375], [-0.175, -0.375], [-0.175, -0.125], [-0.425, -0.125]]}
    assert tidy_desk.release_spots(box, 0) == []  # the first one at the taught pose
    first, second = tidy_desk.release_spots(box, 1)[0], tidy_desk.release_spots(box, 2)[0]
    assert first != second
    for n in range(1, 12):
        for x, y in tidy_desk.release_spots(box, n):
            assert -0.425 + 0.05 - 1e-9 <= x <= -0.175 - 0.05 + 1e-9 and -0.375 + 0.05 - 1e-9 <= y <= -0.125 - 0.05 + 1e-9
    narrow = {"centre": [0.0, 0.0], "corners": [[-0.07, -0.15], [0.07, -0.15], [0.07, 0.15], [-0.07, 0.15]]}
    assert all(abs(x) <= 0.02 + 1e-9 for x, _ in tidy_desk.release_spots(narrow, 3))  # the 14 cm side


def test_release_spots_keep_a_wide_held_thing_off_the_walls():
    box = {"centre": [-0.3, -0.25], "corners": [[-0.425, -0.375], [-0.175, -0.375], [-0.175, -0.125], [-0.425, -0.125]]}
    reach = tidy_desk.held_reach({"size_m": [0.113, 0.11]})
    for x, y in tidy_desk.release_spots(box, 3, reach):
        assert min(x + 0.425, -0.175 - x, y + 0.375, -0.125 - y) >= reach + 0.01 - 1e-9


def test_a_thing_touching_a_precision_device_is_left_with_it():
    square = lambda x0, y0, n=20: [[x0 + i, y0 + j] for i in range(n) for j in range(n)]
    mouse = {"name": "マウス", "should_grasp": False, "category": "keep", "_pixels": square(100, 100)}
    cable = {"name": "ケーブル", "should_grasp": True, "category": "keep", "_pixels": square(122, 100)}  # 3 px off
    pen = {"name": "ペン", "should_grasp": True, "category": "keep", "_pixels": square(200, 100)}
    pairs = tidy_desk.attached_to_untouchable([mouse, cable, pen])
    assert [(o["name"], x["name"]) for o, x in pairs] == [("ケーブル", "マウス")]
