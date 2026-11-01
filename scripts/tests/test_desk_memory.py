import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import desk_memory as dm

DESK = (128, 128, 128)  # BGR grey desk
PX_PER_CM = 10


def scene(colour, size_cm, angle=0.0, shift=(0, 0), noise=6.0, gain=1.0, seed=0, blob=False):
    """Synthetic overhead crop: a coloured object on the grey desk, with camera noise."""
    rng = np.random.default_rng(seed)
    img = np.full((220, 220, 3), DESK, np.uint8)
    centre = (110 + shift[0], 110 + shift[1])
    length, width = (v * PX_PER_CM for v in size_cm)
    if blob:  # crumpled paper: a lumpy ellipse
        cv2.ellipse(img, centre, (int(length / 2), int(width / 2)), angle, 0, 360, colour, -1)
        for k in range(6):
            p = (centre[0] + int(rng.integers(-15, 15)), centre[1] + int(rng.integers(-15, 15)))
            cv2.circle(img, p, 8, tuple(int(c * 0.85) for c in colour), -1)
    else:
        box = cv2.boxPoints((centre, (length, width), angle)).astype(np.int32)
        cv2.fillPoly(img, [box], colour)
    img = img.astype(float) * gain + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


RED = (40, 40, 200)
BLUE = (200, 60, 30)
WHITE = (235, 238, 240)
VERDICT_PEN = {"category": "keep", "force": 3, "deformable": False, "size_cm": [14, 1.5, 1.5]}
VERDICT_PAPER = {"category": "trash", "force": 2, "deformable": True, "size_cm": [6, 5, 4]}


class AppearanceTests(unittest.TestCase):
    def test_same_object_twice_matches_and_others_do_not(self):
        a = dm.feature(scene(RED, (12, 4), angle=10, seed=1), [12, 4])
        b = dm.feature(scene(RED, (12, 4), angle=35, shift=(6, -4), noise=9, gain=0.93, seed=2), [12.6, 3.6])
        self.assertGreater(dm.similarity(a, b), dm.MATCH_THRESHOLD)
        other_colour = dm.feature(scene(BLUE, (12, 4), seed=3), [12, 4])
        self.assertLess(dm.similarity(a, other_colour), dm.MATCH_THRESHOLD)
        other_size = dm.feature(scene(RED, (5, 5), seed=4), [5, 5])
        self.assertLess(dm.similarity(a, other_size), dm.MATCH_THRESHOLD)

    def test_white_paper_matches_itself_and_not_a_pen(self):
        p1 = dm.feature(scene(WHITE, (6, 5), blob=True, seed=5), [6, 5])
        p2 = dm.feature(scene(WHITE, (6, 5), angle=40, blob=True, shift=(5, 5), gain=1.05, seed=6), [5.5, 4.6])
        pen = dm.feature(scene(BLUE, (14, 1.5), angle=20, seed=7), [14, 1.5])
        self.assertGreater(dm.similarity(p1, p2), dm.MATCH_THRESHOLD)
        self.assertLess(dm.similarity(p1, pen), 0.4)

    def test_mask_restricts_the_pixels(self):
        img = scene(RED, (12, 4), seed=8)
        mask = np.zeros(img.shape[:2], bool)
        mask[90:130, 50:170] = True
        self.assertGreater(dm.similarity(dm.feature(img, [12, 4], mask), dm.feature(img, [12, 4])), 0.8)

    def test_recall_and_remember(self):
        with tempfile.TemporaryDirectory() as d:
            nb = dm.Notebook(Path(d) / "kn")
            oid = nb.remember("赤いペン", "ペン", "keep", VERDICT_PEN, scene(RED, (12, 4), seed=1), [12, 4])
            again = nb.remember("赤ペン", "ペン", "keep", VERDICT_PEN,
                                scene(RED, (12, 4), angle=30, shift=(4, 3), seed=9), [12.4, 3.8])
            self.assertEqual(oid, again)
            self.assertEqual(nb.object(oid)["seen"], 2)
            self.assertTrue((Path(d) / "kn" / nb.object(oid)["crop"]).exists())
            found, score = nb.recall(scene(RED, (12, 4), angle=60, seed=10), [12, 4])
            self.assertEqual(found["id"], oid)
            self.assertGreater(score, dm.MATCH_THRESHOLD)
            none, best = nb.recall(scene(WHITE, (6, 5), blob=True, seed=11), [6, 5])
            self.assertIsNone(none)
            self.assertLess(best, dm.MATCH_THRESHOLD)
            self.assertNotEqual(nb.remember("紙くず", "紙くず", "trash", VERDICT_PAPER,
                                            scene(WHITE, (6, 5), blob=True, seed=11), [6, 5]), oid)


class TacticTests(unittest.TestCase):
    def test_clamp_repairs_garbage(self):
        t = dm.clamp_tactic({"force": "9", "opening": "FULL", "soft": "true", "turn": 270, "sweep_first": None,
                             "extra": 1})
        self.assertEqual(t, {"force": 5, "opening": "wide", "soft": True, "turn": 90, "sweep_first": False})
        self.assertEqual(dm.clamp_tactic("nonsense"), dm.default_tactic())
        self.assertEqual(dm.clamp_tactic({"force": -3, "turn": "sideways", "opening": 7})["force"], 1)
        self.assertEqual(dm.clamp_tactic({"turn": 180})["turn"], 0)
        self.assertEqual(dm.clamp_hint({"force": 4.4, "bogus": 1}), {"force": 4})
        self.assertIsNone(dm.clamp_hint({"bogus": 1}))

    def test_default_follows_the_judge(self):
        self.assertEqual(dm.default_tactic(VERDICT_PAPER),
                         {"force": 2, "opening": "snug", "soft": True, "turn": 0, "sweep_first": False})
        self.assertEqual(dm.tactic_ja(dm.default_tactic(VERDICT_PEN)), "力3・ぴったり開く・真上から")

    def test_apply_tactic_turns_the_axis(self):
        o = {"force": 3, "deformable": False, "axis_xy": [1.0, 0.0]}
        out = dm.apply_tactic(o, {"force": 4, "opening": "wide", "soft": False, "turn": 90, "sweep_first": True})
        self.assertEqual((out["force"], out["open_wide"], out["sweep_first"], out["turn"]), (4, True, True, 90))
        self.assertAlmostEqual(out["axis_xy"][1], 1.0)
        self.assertEqual(o["axis_xy"], [1.0, 0.0])


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "knowledge"
        self.nb = dm.Notebook(self.root)
        self.pen = self.nb.remember("ペン", "ペン", "keep", VERDICT_PEN, scene(BLUE, (14, 1.5)), [14, 1.5])

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_dir_is_created(self):
        self.assertTrue((self.root / "crops").is_dir())

    def test_plan_prefers_the_tactic_that_worked(self):
        t, src = self.nb.plan_for(None, "ペン", VERDICT_PEN)
        self.assertEqual((t, src), (dm.default_tactic(VERDICT_PEN), "初めての物：判定どおり"))
        firm = dict(dm.default_tactic(VERDICT_PEN), force=4, opening="wide")
        self.nb.record(self.pen, 1, dm.default_tactic(VERDICT_PEN), "slipped", run="r1")
        self.nb.record(self.pen, 2, firm, "moved_to_keep", run="r1")
        t, src = self.nb.plan_for(self.pen, "ペン", VERDICT_PEN)
        self.assertEqual(t, firm)
        self.assertEqual(src, "前回この物は 力4・広く開く・真上から で成功")
        obj = self.nb.object(self.pen)
        self.assertEqual((obj["attempts"], obj["successes"], obj["last_result"]), (2, 1, "moved_to_keep"))

    def test_plan_falls_back_object_kind_general(self):
        self.nb.add_lesson("general", "", "遠くの物は手前に引き寄せる", {"sweep_first": True})
        t, src = self.nb.plan_for(None, "紙くず", VERDICT_PAPER)
        self.assertTrue(t["sweep_first"])
        self.assertEqual(t["force"], 2)  # the default underneath stays the judge's
        self.assertTrue(src.startswith("全体の教訓："))
        self.nb.add_lesson("kind", "紙くず", "紙くずは広く開いて包む", {"opening": "wide"})
        t, src = self.nb.plan_for(None, "紙くず", VERDICT_PAPER)
        self.assertEqual((t["opening"], src), ("wide", "『紙くず』のコツ：紙くずは広く開いて包む"))
        self.nb.add_lesson("object", "ペン", "このペンは向きを変える", {"turn": 90}, object_id=self.pen)
        t, src = self.nb.plan_for(self.pen, "ペン", VERDICT_PEN)
        self.assertEqual((t["turn"], src), (90, "この物の教訓：このペンは向きを変える"))
        self.assertEqual(self.nb.lessons_for("ペン", self.pen)[0]["level"], "object")
        self.assertEqual(self.nb.lessons_for("紙くず")[0]["level"], "kind")

    def test_lessons_dedupe(self):
        a = self.nb.add_lesson("kind", "紙くず", "紙くずは滑りやすいので力を1段上げる", {"force": 3})
        b = self.nb.add_lesson("kind", "紙くず", "紙くずは滑りやすいので、力を1段上げる。", {"force": 4})
        c = self.nb.add_lesson("kind", "ペン", "紙くずは滑りやすいので力を1段上げる")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(self.nb.lessons), 2)
        self.assertEqual(self.nb.lessons[0]["tactic_hint"], {"force": 4})

    def test_files_reload_identically(self):
        self.nb.add_lesson("kind", "ペン", "ペンは力3で十分", {"force": 3})
        self.nb.record(self.pen, 1, dm.default_tactic(VERDICT_PEN), "missed", images=["a.jpg"], run="r1", why="ずれた")
        self.nb.record(self.pen, 2, dm.default_tactic(VERDICT_PEN), "moved_to_keep", run="r1")
        again = dm.Notebook(self.root)
        self.assertEqual(again.objects, self.nb.objects)
        self.assertEqual(again.lessons, self.nb.lessons)
        self.assertEqual(again.episodes, self.nb.episodes)
        self.assertEqual(json.loads((self.root / "episodes.jsonl").read_text().splitlines()[0])["why"], "ずれた")
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.endswith(".tmp")], [])

    def test_stats_show_the_learning_curve(self):
        tac = dm.default_tactic(VERDICT_PEN)
        for k, res in enumerate(("missed", "slipped", "moved_to_keep"), 1):
            self.nb.record(self.pen, k, tac, res, run="r1")
        self.nb.record(self.pen, 1, tac, "moved_to_keep", run="r2")
        s = self.nb.stats()
        self.assertEqual([(r["attempts"], r["successes"], r["attempts_per_success"]) for r in s["runs"]],
                         [(3, 1, 3.0), (1, 1, 1.0)])
        self.assertEqual((s["objects"], s["success_rate"], s["runs"][1]["first_try"]), (1, 0.5, 1))

    def test_prompt_advice_lists_lessons(self):
        self.assertEqual(self.nb.prompt_advice(), "")
        self.nb.add_lesson("general", "", "厚さ5mm未満の硬い物は人に任せる")
        self.nb.add_lesson("kind", "紙くず", "紙くずは広く開いて包む", {"opening": "wide"})
        text = self.nb.prompt_advice()
        self.assertIn("Experience notebook", text)
        self.assertIn("厚さ5mm未満", text)
        self.assertIn('kind "紙くず"', text)

    def test_notes_html_has_the_japanese_headers(self):
        self.nb.add_lesson("kind", "紙くず", "紙くずは<広く>開く", {"opening": "wide"})
        self.nb.record(self.pen, 1, dm.default_tactic(VERDICT_PEN), "moved_to_keep", run="r1")
        page = dm.notes_html(self.nb)
        for text in ("知っている物", "1 個", "教訓", "成功率", "この物だけ", "種類ごとのコツ", "全体の教訓",
                     "力3・ぴったり開く・真上から", "/notes/img/o001.jpg", "prefers-color-scheme"):
            self.assertIn(text, page)
        self.assertIn("&lt;広く&gt;", page)


class ReflectTests(unittest.TestCase):
    OBJ = {"id": "o001", "name": "消しゴム", "kind": "消しゴム", "verdict": {"force": 2, "deformable": False}}

    def run_until_done(self, results):
        tactic, history, out = dm.default_tactic(self.OBJ["verdict"]), [], []
        for k, res in enumerate(results, 1):
            r = dm.reflect_offline(self.OBJ, tactic, res, history)
            out.append(r)
            history.append({"attempt": k, "tactic": tactic, "result": res})
            tactic = r["next_tactic"]
            if r["give_up"]:
                break
        return out

    def test_slips_raise_the_force_then_give_up(self):
        out = self.run_until_done(["slipped"] * 10)
        forces = [r["next_tactic"]["force"] for r in out]
        self.assertEqual(forces[:2], [3, 4])
        self.assertEqual(out[0]["lesson"]["level"], "kind")
        self.assertEqual(out[0]["lesson"]["text"], "消しゴムは滑りやすいので力を1段上げる")
        self.assertTrue(out[-1]["give_up"])
        self.assertLessEqual(len(out), dm.GIVE_UP_AFTER)

    def test_other_failures(self):
        tac = dm.default_tactic(self.OBJ["verdict"])
        self.assertEqual(dm.reflect_offline(self.OBJ, tac, "missed", [])["next_tactic"]["turn"], 90)
        missed_turned = dm.reflect_offline(self.OBJ, dict(tac, turn=90), "missed", [{"tactic": tac, "result": "missed"}])
        self.assertEqual(missed_turned["next_tactic"]["opening"], "wide")
        wide = dm.reflect_offline(self.OBJ, tac, "too_wide", [])["next_tactic"]
        self.assertEqual(wide["turn"], 90)
        soft = dict(tac, soft=True)
        nt = dm.reflect_offline(self.OBJ, soft, "not_taken", [])["next_tactic"]
        self.assertEqual((nt["soft"], nt["force"]), (False, 3))
        blocked = dm.reflect_offline(self.OBJ, tac, "blocked", [])
        self.assertTrue(blocked["next_tactic"]["sweep_first"])
        self.assertTrue(dm.reflect_offline(self.OBJ, blocked["next_tactic"], "blocked",
                                           [{"tactic": tac, "result": "blocked"}])["give_up"])
        far = self.run_until_done(["too_far"] * 5)
        self.assertEqual(len(far), 2)
        self.assertTrue(far[0]["next_tactic"]["sweep_first"] and far[1]["give_up"])
        for r in far:
            self.assertEqual(set(r), {"why", "lesson", "next_tactic", "give_up", "give_up_reason", "source"})

    def test_reflect_clamps_claude_and_falls_back(self):
        import desk_judge
        reply = {"why": "持ち上げ途中で回転して落ちた" * 3, "give_up": False, "give_up_reason": "",
                 "next_tactic": {"force": 8, "opening": "huge", "soft": "no", "turn": 45, "sweep_first": 0},
                 "lesson": {"level": "kind", "kind": "", "text": "消しゴムは力4で", "tactic_hint": {"force": 4, "x": 1}}}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(desk_judge, "_claude", return_value=reply) as call:
            nb = dm.Notebook(d)
            r = dm.reflect(nb, self.OBJ, dm.default_tactic(), "slipped", ["/tmp/aim.jpg"], [])
            prompt = call.call_args[0][0]
            self.assertIn("ONLY", prompt)
            self.assertIn("/tmp/aim.jpg", prompt)
            self.assertIn("fell out of the fingers", prompt)
            self.assertEqual(r["next_tactic"], {"force": 5, "opening": "snug", "soft": False, "turn": 90,
                                                "sweep_first": False})
            self.assertLessEqual(len(r["why"]), 40)
            self.assertEqual((r["lesson"]["kind"], r["lesson"]["tactic_hint"]), ("消しゴム", {"force": 4}))
            lid = nb.learn(self.OBJ, r, source_episode=0)
            self.assertEqual(nb.lessons[0]["id"], lid)
        with mock.patch.object(desk_judge, "_claude", side_effect=RuntimeError("no answer")):
            r = dm.reflect(None, self.OBJ, dm.default_tactic(), "slipped", [], [])
            self.assertEqual(r["source"], "offline")


if __name__ == "__main__":
    unittest.main()


class ConsolidateTests(unittest.TestCase):
    def test_many_lessons_for_a_kind_are_merged_and_the_old_ones_retired(self):
        import desk_judge
        with tempfile.TemporaryDirectory() as tmp:
            nb = dm.Notebook(Path(tmp))
            texts = ["ペンは滑るので強めに握る", "細い円柱は指を広く開く", "転がる前に素早く閉じる",
                     "向きを変えると落ちにくい", "端ではなく真ん中を挟む", "最大の力でないと運搬中に落ちる"]
            for k, text in enumerate(texts):
                nb.add_lesson("kind", "ペン", text, {"force": 3 + k % 3})
            saved = desk_judge._claude
            desk_judge._claude = lambda *a, **k: {"lessons": [{"text": "ペンは力4でつかむ", "tactic_hint": {"force": 4}}]}
            try:
                ids = nb.consolidate("ペン")
            finally:
                desk_judge._claude = saved
            self.assertEqual(len(ids), 1)
            active = [les for les in nb.lessons if not les.get("retired")]
            self.assertEqual([les["text"] for les in active], ["ペンは力4でつかむ"])
            tactic, _ = nb.plan_for(None, "ペン", {"force": 2})
            self.assertEqual(tactic["force"], 4)

    def test_a_few_lessons_are_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = dm.Notebook(Path(tmp))
            nb.add_lesson("kind", "ペン", "ペンは力4", {"force": 4})
            self.assertEqual(nb.consolidate("ペン"), [])


class BestTacticTests(unittest.TestCase):
    def test_a_best_tactic_that_slips_again_is_dropped_but_a_miss_keeps_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = dm.Notebook(Path(tmp))
            crop = np.full((60, 60, 3), 128, np.uint8)
            crop[20:40, 20:40] = (40, 120, 200)
            oid = nb.remember("積み木", "箱", "keep", {"force": 3}, crop, [3.5, 3.0])
            lucky = dm.clamp_tactic({"force": 4})
            nb.record(oid, 1, lucky, "moved_to_keep")
            self.assertEqual(nb.object(oid)["best_tactic"]["force"], 4)
            nb.record(oid, 1, lucky, "missed")          # aim: keep it
            self.assertIsNotNone(nb.object(oid)["best_tactic"])
            nb.record(oid, 1, lucky, "slipped")         # the same tactic slipped: drop it
            self.assertIsNone(nb.object(oid)["best_tactic"])
            nb.add_lesson("kind", "箱", "重い箱は力5", {"force": 5})
            tactic, _ = nb.plan_for(oid, "箱", {"force": 3})
            self.assertEqual(tactic["force"], 5)
