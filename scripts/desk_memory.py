#!/usr/bin/env python3
"""The robot's experience notebook: what it has seen, how each grasp went, and the lessons it drew.

Nothing is trained. Every grasp attempt is written down; after a failure Claude (or the
rule-based stand-in) explains why and writes a short lesson; the next time the same object,
a similar kind of object, or anything at all comes along, the notebook is read back into the
plan (a grasp "tactic") and into the judge's prompt. Lives in a plain directory:

    knowledge/objects.json    known objects (appearance feature, crop, counters, best tactic)
    knowledge/lessons.json    lessons (this object / this kind / everything)
    knowledge/episodes.jsonl  one line per grasp attempt
    knowledge/crops/*.jpg     one picture per known object (for the notes page)

    nb = Notebook()
    oid = nb.remember(name, kind, category, verdict, crop_bgr, size_cm)
    tactic, why = nb.plan_for(oid, kind, verdict)
    ... try it ...
    r = reflect(nb, nb.object(oid), tactic, result, images, nb.history(oid, run))
    ep = nb.record(oid, attempt, tactic, result, images, run, why=r["why"])
    nb.learn(nb.object(oid), r, source_episode=ep)
"""
import difflib
import html
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE.parent / "knowledge"

# --- appearance -------------------------------------------------------
H_BINS, S_BINS, V_BINS = 12, 4, 3
HUE_SHIFT = 7.5           # OpenCV hue units: centres red in bin 0 instead of splitting it across 0/179
CENTRE_SHARE = 0.6        # without a mask, only the central 60% of the crop is the object
BORDER_SHARE = 0.08       # the crop's outer ring is desk: its median colour is the desk colour
DESK_LAB_DIST = 18.0      # LAB distance under which a pixel counts as desk (as desk_judge.presence)
MIN_OBJECT_SHARE = 0.05   # fewer non-desk pixels than this: the object is desk-coloured, keep them all
SIZE_SLACK_CM = 2.0       # judged/measured sizes wobble by this much between two looks (a 4.5 cm tall block: 5.7 vs 7.8 cm)
SIZE_SCALE_CM = (6.0, 5.0)  # length, width difference (beyond the slack) that costs a factor e
MATCH_THRESHOLD = 0.75
CROP_SIDE = 240           # px, saved crop thumbnails

# --- tactics ----------------------------------------------------------
OPENINGS = ("snug", "wide")
TURNS = (0, 90)
TACTIC_KEYS = ("force", "opening", "soft", "turn", "sweep_first", "rim")
SUCCESS_PREFIX = "moved_to_"
RETHINK_RESULTS = ("slipped", "not_taken", "too_wide", "too_flat", "stalled")  # failures that are not just aim
RETIRE_AFTER = 2          # failures after following a lesson (more than its successes) that retire it
GIVE_UP_AFTER = 4         # failed attempts on one object before reflection says stop
DEDUPE_RATIO = 0.85       # lesson texts this similar (same level and kind) are the same lesson
LEVELS = ("object", "kind", "general")
LEVELS_JA = {"object": "この物だけ", "kind": "種類ごとのコツ", "general": "全体の教訓"}
CATEGORY_JA = {"trash": "ゴミ", "keep": "残す", "unsure": "わからない", "fixture": "固定"}
RESULT_JA = {"moved_to_trash": "ゴミ箱へ", "moved_to_keep": "保管箱へ", "held": "掴めた", "missed": "空振り",
             "stalled": "指が途中で止まった", "too_wide": "幅が広すぎる", "blocked": "周りが邪魔",
             "unreachable": "届かない", "too_far": "遠すぎる", "too_flat": "平たすぎる", "slipped": "途中で落とした",
             "not_taken": "取れていない", "swept": "引き寄せた", "lost": "見失った", "failed": "失敗"}

# What each result code means physically (for Claude, who never saw the arm move).
RESULT_EXPLAIN = {
    "moved_to_trash": "success: it was carried to the trash bin and released there",
    "moved_to_keep": "success: it was carried to the keep box and released there",
    "held": "the fingers closed on something and it was lifted (not yet confirmed in the box)",
    "missed": "the fingers closed on nothing: the object was not between them when they shut "
              "(aim off, the finger line met it badly, or it slid away as the fingers closed)",
    "stalled": "on every try a finger stopped on something before reaching grasp height "
               "(it landed on top of the object or on a neighbour), so the gripper never closed",
    "too_wide": "planning only: no finger direction lets the object fit between the open fingers",
    "blocked": "planning only: every reachable approach would hit a box wall or a neighbouring object",
    "unreachable": "planning only: the arm cannot reach a top-down grasp at that spot",
    "too_far": "planning only: the object is beyond the safe reach of the shoulder",
    "too_flat": "planning only: a rigid thing thinner than 5 mm cannot be pinched off the desk",
    "slipped": "it was gripped and lifted, but fell out of the fingers while being carried to the box",
    "not_taken": "a soft object: the fingers closed, but the camera still sees it on the desk "
                 "(they squeezed air or only a loose corner)",
    "swept": "it was dragged toward the robot's base; it has not been grasped yet",
    "lost": "the camera can no longer find it",
    "failed": "three attempts went by without a confirmed result",
}

REFLECT_TIMEOUT_S = 240
TACTIC_SCHEMA = {"type": "object", "properties": {
    "force": {"type": "integer", "minimum": 1, "maximum": 5},
    "opening": {"type": "string", "enum": list(OPENINGS)},
    "soft": {"type": "boolean"},
    "turn": {"type": "integer", "enum": list(TURNS)},
    "sweep_first": {"type": "boolean"},
    "rim": {"type": "boolean"}}}
CONSOLIDATE_OVER = 4      # active lessons for one kind above which they are merged
CONSOLIDATE_SCHEMA = {"type": "object", "properties": {"lessons": {"type": "array", "items": {
    "type": "object", "properties": {"text": {"type": "string"}, "tactic_hint": TACTIC_SCHEMA},
    "required": ["text"]}}}, "required": ["lessons"]}
CONSOLIDATE_PROMPT = """A desk-tidying robot (two-finger gripper, picks from straight above) keeps an experience notebook.
These are its current lessons for "{kind}", learned one failure at a time; some may repeat or contradict
each other. helped/hurt count how often following a lesson led to success or failure.

{lessons}

Merge them into AT MOST three short, consistent lessons that keep what worked (favour lessons that helped,
drop ones that hurt or that only say something cannot be fixed). Each: text in Japanese (under 60
characters, no English setting names - write 力を上げる, 指を広く開く, 指の向きを90度回す, 届いた所をつまむ,
手前に引き寄せる) and an optional tactic_hint with only the settings the lesson changes."""
REFLECT_SCHEMA = {"type": "object", "properties": {
    "why": {"type": "string"},
    "lesson": {"type": "object", "properties": {
        "level": {"type": "string", "enum": list(LEVELS)},
        "kind": {"type": "string"},
        "text": {"type": "string"},
        "tactic_hint": TACTIC_SCHEMA},
        "required": ["level", "kind", "text", "tactic_hint"]},
    "next_tactic": dict(TACTIC_SCHEMA, required=list(TACTIC_KEYS)),
    "give_up": {"type": "boolean"},
    "give_up_reason": {"type": "string"}},
    "required": ["why", "lesson", "next_tactic", "give_up", "give_up_reason"]}

REFLECT_PROMPT = """You are the coach of a desk-tidying robot: a 6-axis arm with a two-finger parallel
gripper (fingers open up to about 7.7 cm) that picks each object from straight above and drops it in a
box. It keeps an experience notebook; your answer becomes a lesson in it. Answer every text field in
Japanese, short and concrete (why under 40 characters, lesson text under 60). The person reading them
does not read English: write the settings in Japanese words (力を上げる, 指を広く開く, 指の向きを90度回す,
手前に引き寄せる, 届いた所をつまむ), never the setting names force/opening/turn/soft/sweep_first.

The object: "{name}" (kind: "{kind}"), about {size} cm (length x width x height), judged {category}.
The judge said: deformable={deformable}, force={force} ({force_reason}), grasp point "{grasp_point}".

Tactic just tried: {tactic}
Result: {result} - {meaning}

Earlier attempts on this object:
{history}

What the notebook already says:
{lessons}

Look at the pictures with the Read tool (overhead camera; the red cross/box marks where it aimed):
{images}

The robot can ONLY change these tactic settings - nothing else (no new motions, no other tools, no help):
- force 1-5: grip strength (1 tissue, 2 crumpled paper, 3 pen, 4 firm/slippery, 5 strongest)
- opening "snug" (open just wider than the object) or "wide" (open the fingers fully)
- soft true (squeeze whatever part the fingers reach; for squashy things) or false (reach around it)
- turn 0 (fingers close across the short side) or 90 (finger line rotated 90 degrees: across the long side)
- sweep_first true (first drag the object toward the robot's base along the desk, then grasp) or false.
- rim true (pinch the wall of the thing at its edge, one finger inside it and one outside, instead of reaching around its middle) or false
  It only works for objects farther than about 43 cm from the base (it pulls them in to ~43 cm); for
  nearer ones it is skipped, and it cannot separate two things that touch.
The object is about {reach} cm from the robot's base.

Reply with:
- why: the most likely physical reason for this result
- lesson: level "object" (only this very object), "kind" (every "{kind}"), or "general" (any object);
  kind (use "{kind}" unless general); text (the rule to remember); tactic_hint (only the settings the
  lesson is about). Write a lesson ONLY for a cause that will repeat with this kind of object (it is
  heavy, slippery, too wide, squashes, sits against a wall...). When the failure looks like chance - the
  fingers landed a few millimetres off, an unlucky orientation, the object rolled - leave the lesson text
  EMPTY and just change next_tactic: a lesson from a chance failure undoes what was working
  (2026-10-27: three misses on one block gave "open wide", then "open snug", then "pinch").
- next_tactic: the full tactic for the next attempt (all five settings)
- give_up: true when no setting on the menu is likely to work (or after {give_up_after} failed attempts),
  with give_up_reason; otherwise false and an empty reason."""

ADVICE_HEADER = """Experience notebook - lessons this robot learned from its own earlier grasps (texts in
Japanese). Where one applies to this object, let it shape force, deformable and should_grasp:"""


# --- files ------------------------------------------------------------
def _atomic_write(path, data):
    """Write bytes or text next to path and rename over it, so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data.encode() if isinstance(data, str) else data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _now():
    # Milliseconds: lessons written within one second must still sort in the order they were written.
    return datetime.now().isoformat(timespec="milliseconds")


# --- appearance -------------------------------------------------------
def _object_pixels(crop_bgr, mask):
    """Boolean mask of the crop pixels that belong to the object."""
    h, w = crop_bgr.shape[:2]
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if m.shape == (h, w) and m.sum() >= 10:
            return m
    # No (usable) mask: the central part of the crop, minus whatever looks like the desk around it.
    m = np.zeros((h, w), bool)
    dy, dx = int(round(h * (1 - CENTRE_SHARE) / 2)), int(round(w * (1 - CENTRE_SHARE) / 2))
    m[dy:h - dy, dx:w - dx] = True
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(float)
    by, bx = max(1, int(h * BORDER_SHARE)), max(1, int(w * BORDER_SHARE))
    ring = np.ones((h, w), bool)
    ring[by:h - by, bx:w - bx] = False
    desk = np.median(lab[ring], axis=0)
    not_desk = np.linalg.norm(lab - desk, axis=2) > DESK_LAB_DIST
    if (m & not_desk).sum() >= MIN_OBJECT_SHARE * m.sum():
        m &= not_desk
    return m


def feature(crop_bgr, size_cm, mask=None):
    """HSV histogram of the object's pixels (12 hue x 4 sat x 3 val) + [length_cm, width_cm].

    Pixels with little saturation (white paper, grey plastic) have a meaningless hue, so they
    all go to hue bin 0: two photos of a white thing then fall in the same bins.
    """
    crop = cv2.GaussianBlur(np.ascontiguousarray(crop_bgr), (3, 3), 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[_object_pixels(crop, mask)].astype(float)
    hue = ((hsv[:, 0] + HUE_SHIFT) % 180 * H_BINS / 180).astype(int).clip(0, H_BINS - 1)
    sat = (hsv[:, 1] * S_BINS / 256).astype(int).clip(0, S_BINS - 1)
    val = (hsv[:, 2] * V_BINS / 256).astype(int).clip(0, V_BINS - 1)
    hue[sat == 0] = 0
    hist = np.bincount((hue * S_BINS + sat) * V_BINS + val, minlength=H_BINS * S_BINS * V_BINS).astype(float)
    hist /= max(hist.sum(), 1.0)
    dims = sorted((float(v) for v in list(size_cm)[:2]), reverse=True)
    return [round(float(v), 5) for v in hist] + [round(d, 2) for d in dims]


def similarity(f1, f2):
    """Histogram intersection x size factor, 0..1 (same object twice > 0.75, pen vs paper < 0.4)."""
    if f1 is None or f2 is None or len(f1) != len(f2) or len(f1) < 3:
        return 0.0
    a, b = np.asarray(f1, float), np.asarray(f2, float)
    hist = float(np.minimum(a[:-2], b[:-2]).sum())
    d = np.maximum(0.0, np.abs(a[-2:] - b[-2:]) - SIZE_SLACK_CM)
    return hist * math.exp(-d[0] / SIZE_SCALE_CM[0] - d[1] / SIZE_SCALE_CM[1])


# --- tactics ----------------------------------------------------------
def _as_bool(v, default):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1", "on", "はい"):
            return True
        if s in ("false", "no", "0", "off", "いいえ", ""):
            return False
    return default


def _as_force(v, default):
    try:
        return int(min(5, max(1, round(float(v)))))
    except (TypeError, ValueError):
        return default


def _as_opening(v, default):
    s = str(v).strip().lower()
    if s in OPENINGS:
        return s
    if s in ("full", "open", "max", "fully", "wide_open", "広く"):
        return "wide"
    if s in ("tight", "fit", "narrow", "close", "ぴったり"):
        return "snug"
    return default


def _as_turn(v, default):
    try:
        deg = float(v) % 180
    except (TypeError, ValueError):
        return default
    return 90 if 45 <= deg < 135 else 0


def default_tactic(verdict=None):
    """What the judge's verdict alone says: its force, soft for deformable things, nothing special."""
    verdict = verdict or {}
    return {"force": _as_force(verdict.get("force"), 3), "opening": "snug",
            "soft": _as_bool(verdict.get("deformable"), False), "turn": 0, "sweep_first": False, "rim": False}


def clamp_tactic(t, base=None):
    """A valid tactic from anything (Claude's reply, an old file); missing/garbage keys come from base."""
    fallback = default_tactic()
    if isinstance(base, dict) and base is not t:  # repair the base itself against the plain default first
        base = clamp_tactic(base, None)
    base = base if isinstance(base, dict) and all(k in base for k in TACTIC_KEYS) else fallback
    t = t if isinstance(t, dict) else {}
    return {"force": _as_force(t.get("force"), base["force"]),
            "opening": _as_opening(t.get("opening"), base["opening"]),
            "soft": _as_bool(t.get("soft"), base["soft"]),
            "turn": _as_turn(t.get("turn"), base["turn"]),
            "sweep_first": _as_bool(t.get("sweep_first"), base["sweep_first"]),
            "rim": _as_bool(t.get("rim"), base["rim"])}


def clamp_hint(h):
    """A partial tactic: only the valid keys that are present (None for nothing usable)."""
    if not isinstance(h, dict):
        return None
    full = clamp_tactic(h)
    out = {k: full[k] for k in TACTIC_KEYS if k in h and h[k] is not None}
    return out or None


def tactic_ja(t):
    """Tactic in plain Japanese words, e.g. 力3・ぴったり開く・真上から."""
    t = clamp_tactic(t)
    words = [f"力{t['force']}", "広く開く" if t["opening"] == "wide" else "ぴったり開く"]
    if t["soft"]:
        words.append("届いた所をつまむ")
    if t["rim"]:
        words.append("縁をつまむ")
    if t["turn"] == 90:
        words.append("指の向きを90度回す")
    words.append("手前に引き寄せてから" if t["sweep_first"] else "真上から")
    return "・".join(words)


def hint_ja(h):
    """Partial tactic in Japanese words (only the settings it sets)."""
    if not h:
        return ""
    words = []
    if "force" in h:
        words.append(f"力{h['force']}")
    if "opening" in h:
        words.append("広く開く" if h["opening"] == "wide" else "ぴったり開く")
    if "soft" in h:
        words.append("届いた所をつまむ" if h["soft"] else "回り込んで挟む")
    if "turn" in h:
        words.append("指の向きを90度回す" if h["turn"] == 90 else "いつもの向き")
    if "sweep_first" in h:
        words.append("手前に引き寄せてから" if h["sweep_first"] else "真上から")
    return "・".join(words)


def apply_tactic(o, tactic):
    """Copy of a judged object (tidy_desk's dict) with the tactic written into the fields plan_pick reads.

    force -> o["force"]; soft -> o["deformable"]; turn 90 -> o["axis_xy"] rotated 90 degrees and
    o["turn"] = 90 (plan_pick must then also swap its (length, width) size, see the report);
    opening wide -> o["open_wide"] = True; sweep_first -> o["sweep_first"] = True.
    """
    t = clamp_tactic(tactic)
    out = dict(o)
    out["force"] = t["force"]
    out["deformable"] = t["soft"]
    out["turn"] = t["turn"]
    out["open_wide"] = t["opening"] == "wide"
    out["sweep_first"] = t["sweep_first"]
    out["rim_first"] = t["rim"]
    if t["turn"] == 90:
        ax, ay = (o.get("axis_xy") or [1.0, 0.0])[:2]
        out["axis_xy"] = [-float(ay), float(ax)]
    return out


def is_success(result):
    return str(result).startswith(SUCCESS_PREFIX)


def _norm_text(s):
    return re.sub(r"[\s、。・,.!！?？「」『』()（）]", "", str(s)).lower()


def _kind_match(a, b):
    a, b = _norm_text(a), _norm_text(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


# --- the notebook -----------------------------------------------------
class Notebook:
    """Objects, lessons and episodes in one directory; every change is saved at once."""

    def __init__(self, root=DEFAULT_DIR):
        self.root = Path(root)
        (self.root / "crops").mkdir(parents=True, exist_ok=True)
        self.objects = self._load_json("objects.json")
        self.lessons = self._load_json("lessons.json")
        self.episodes = []
        path = self.root / "episodes.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        self.episodes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # a hand-edited or foreign line; the rest still counts
        self.last_lessons = []  # ids of the lessons the last plan_for() leaned on

    def _load_json(self, name):
        path = self.root / name
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def save(self):
        _atomic_write(self.root / "objects.json", json.dumps(self.objects, ensure_ascii=False, indent=1))
        _atomic_write(self.root / "lessons.json", json.dumps(self.lessons, ensure_ascii=False, indent=1))

    def _save_episodes(self):
        _atomic_write(self.root / "episodes.jsonl",
                      "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in self.episodes))

    @staticmethod
    def _next_id(items, prefix):
        nums = [int(i["id"][1:]) for i in items if str(i.get("id", ""))[1:].isdigit()]
        return f"{prefix}{max(nums, default=0) + 1:03d}"

    def object(self, object_id):
        return next((o for o in self.objects if o["id"] == object_id), None)

    # appearance
    def recall(self, crop_bgr, size_cm, mask=None, threshold=MATCH_THRESHOLD, kind=None):
        """(known object, score) when the crop looks like one, else (None, best score).

        kind: what the judge (or the detector) calls it; a known object of another kind never matches,
        however alike they look (2026-10-27: a block was taken for a tissue once sizes were loosened).
        """
        f = feature(crop_bgr, size_cm, mask)
        best, score = None, 0.0
        for o in self.objects:
            if kind and not (_kind_match(kind, o.get("kind")) or _kind_match(kind, o.get("name"))):
                continue
            s = similarity(f, o.get("feature"))
            if s > score:
                best, score = o, s
        return (best, score) if best is not None and score >= threshold else (None, score)

    def remember(self, name, kind, category, verdict, crop_bgr, size_cm, mask=None):
        """Id of the object in the crop: the matching known one (seen once more) or a new entry."""
        f = feature(crop_bgr, size_cm, mask)
        match, _ = self.recall(crop_bgr, size_cm, mask, kind=kind or name)
        if match is not None:
            # Average the appearance, so one odd photo does not decide the next match.
            n = max(1, int(match.get("seen", 1)))
            old = np.asarray(match["feature"], float)
            mixed = (old * n + np.asarray(f, float)) / (n + 1)
            mixed[:-2] /= max(mixed[:-2].sum(), 1e-9)
            match["feature"] = [round(float(v), 5) for v in mixed]
            match["seen"] = n + 1
            match["kind"] = match.get("kind") or kind
            match["verdict"] = dict(verdict or {}) or match.get("verdict", {})
            match["category"] = category or match.get("category")
            match["updated"] = _now()
            self.save()
            return match["id"]
        oid = self._next_id(self.objects, "o")
        crop_rel = f"crops/{oid}.jpg"
        scale = CROP_SIDE / max(crop_bgr.shape[:2])
        thumb = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop_bgr
        ok, jpg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            _atomic_write(self.root / crop_rel, jpg.tobytes())
        self.objects.append({"id": oid, "name": name, "kind": kind, "category": category,
                             "verdict": dict(verdict or {}), "feature": f, "crop": crop_rel if ok else None,
                             "seen": 1, "attempts": 0, "successes": 0, "best_tactic": None,
                             "last_result": None, "updated": _now()})
        self.save()
        return oid

    # episodes
    def record(self, object_id, attempt, tactic, result, images=(), run="", why=None):
        """Write one grasp attempt down; returns the episode number (its line in episodes.jsonl)."""
        obj = self.object(object_id)
        t = clamp_tactic(tactic)
        ep = {"time": _now(), "run": run, "object_id": object_id, "name": obj["name"] if obj else None,
              "kind": obj["kind"] if obj else None, "attempt": attempt, "tactic": t, "result": result,
              "images": [str(p) for p in images]}
        if why:
            ep["why"] = why
        self.episodes.append(ep)
        self._save_episodes()
        if obj is not None:
            obj["attempts"] = int(obj.get("attempts", 0)) + 1
            if is_success(result):
                obj["successes"] = int(obj.get("successes", 0)) + 1
                obj["best_tactic"] = t
            elif obj.get("best_tactic") and clamp_tactic(obj["best_tactic"]) == t and result in RETHINK_RESULTS:
                # What once worked just failed the same way again (2026-10-27: a heavy block lifted once at
                # force 4 by luck, then slipped at 4 every time): stop repeating it, let the lessons decide.
                obj["best_tactic"] = None
                obj["dropped_tactic"] = t
            obj["last_result"] = result
            obj["updated"] = _now()
            self.save()
        return len(self.episodes) - 1

    def history(self, object_id, run=None):
        """This object's attempts so far (only this run's when run is given), oldest first."""
        return [e for e in self.episodes if e.get("object_id") == object_id and (run is None or e.get("run") == run)]

    # lessons
    def add_lesson(self, level, kind, text, tactic_hint=None, object_id=None, source_episode=None):
        """Id of the lesson; a near-identical one at the same level/kind is updated instead of repeated."""
        level = level if level in LEVELS else "general"
        hint = clamp_hint(tactic_hint)
        key = _norm_text(text)
        for les in self.lessons:
            if les["level"] != level or (level != "general" and _norm_text(les.get("kind")) != _norm_text(kind)):
                continue
            if level == "object" and les.get("object_id") != object_id:
                continue
            if difflib.SequenceMatcher(None, _norm_text(les["text"]), key).ratio() >= DEDUPE_RATIO:
                if hint:
                    les["tactic_hint"] = hint
                les["created"] = _now()  # still true: it moves back to the top
                self.save()
                return les["id"]
        lid = self._next_id(self.lessons, "l")
        self.lessons.append({"id": lid, "level": level, "kind": "" if level == "general" else (kind or ""),
                             "object_id": object_id if level == "object" else None, "text": str(text).strip(),
                             "tactic_hint": hint, "created": _now(), "source_episode": source_episode, "uses": 0})
        self.save()
        return lid

    def learn(self, obj, reflection, source_episode=None):
        """Write a reflection's lesson into the notebook; returns the lesson id (None if it had none)."""
        les = (reflection or {}).get("lesson") or {}
        if not str(les.get("text", "")).strip():
            return None
        level = les.get("level") if les.get("level") in LEVELS else "kind"
        return self.add_lesson(level, les.get("kind") or (obj or {}).get("kind", ""), les["text"],
                               les.get("tactic_hint"), (obj or {}).get("id"), source_episode)

    def consolidate(self, kind, model="sonnet", cwd=HERE, limit=CONSOLIDATE_OVER):
        """When a kind has more than `limit` active lessons, have Claude merge them into at most three
        consistent ones and retire the rest (2026-10-27: 28 lessons piled up, several contradicting:
        "pens: grip a bit harder" next to "pens: only the strongest grip holds"). Returns the new ids."""
        active = [les for les in self.lessons if les["level"] == "kind" and not les.get("retired")
                  and _kind_match(les.get("kind"), kind)]
        if len(active) <= limit:
            return []
        import desk_judge
        listing = "\n".join(f"- {les['text']} (hint {json.dumps(les.get('tactic_hint'), ensure_ascii=False)}, "
                             f"helped {les.get('helped', 0)}, hurt {les.get('hurt', 0)})" for les in active)
        prompt = CONSOLIDATE_PROMPT.format(kind=kind, lessons=listing)
        try:
            reply = desk_judge._claude(prompt, CONSOLIDATE_SCHEMA, model, str(cwd), timeout=REFLECT_TIMEOUT_S)
        except (RuntimeError, OSError, ValueError):
            return []
        merged = [m for m in reply.get("lessons", []) if str(m.get("text", "")).strip()][:3]
        if not merged:
            return []
        for les in active:
            les["retired"], les["retired_why"] = True, "merged"
        ids = [self.add_lesson("kind", kind, _short(m["text"], 60), clamp_hint(m.get("tactic_hint"))) for m in merged]
        self.save()
        return ids

    def lessons_for(self, kind, object_id=None, limit=8):
        """Lessons that apply: this object's, then this kind's, then general ones; most used first."""
        def ranked(items):
            return sorted(items, key=lambda les: (-int(les.get("uses", 0)), les.get("created", "")))
        own = [les for les in self.lessons if les["level"] == "object" and object_id
               and les.get("object_id") == object_id]
        kinds = [les for les in self.lessons if les["level"] == "kind" and _kind_match(les.get("kind"), kind)]
        general = [les for les in self.lessons if les["level"] == "general"]
        return (ranked(own) + ranked(kinds) + ranked(general))[:limit]

    def _merged_hints(self, lessons, base):
        """base with the lessons' hints laid on top, oldest first so the newest wins."""
        used, t = [], dict(base)
        for les in sorted(lessons, key=lambda les: les.get("created", "")):
            if les.get("tactic_hint"):
                t = clamp_tactic(les["tactic_hint"], t)
                used.append(les)
        return t, used

    def credit(self, lesson_ids, success):
        """Count whether following these lessons worked; one that failed twice more often than it helped
        is retired (no longer applied), so a wrong rule of thumb does not keep spreading."""
        for les in self.lessons:
            if les["id"] in lesson_ids:
                key = "helped" if success else "hurt"
                les[key] = int(les.get(key, 0)) + 1
                if int(les.get("hurt", 0)) >= RETIRE_AFTER and les["hurt"] > int(les.get("helped", 0)):
                    les["retired"] = True
        self.save()

    def plan_for(self, object_id, kind, verdict):
        """(tactic, Japanese reason for the screen): what the notebook says to try on this object."""
        base = default_tactic(verdict)
        obj = self.object(object_id) if object_id else None
        self.last_lessons = []
        if obj and obj.get("best_tactic"):
            return clamp_tactic(obj["best_tactic"], base), f"前回この物は {tactic_ja(obj['best_tactic'])} で成功"
        kind = kind or (obj or {}).get("kind", "")
        levels = (("object", lambda les: obj is not None and les.get("object_id") == obj["id"]),
                  ("kind", lambda les: _kind_match(les.get("kind"), kind)),
                  ("general", lambda les: True))
        for level, applies in levels:
            found = [les for les in self.lessons if les["level"] == level and applies(les) and not les.get("retired")]
            t, used = self._merged_hints(found, base)
            if not used:
                continue
            newest = used[-1]
            for les in used:
                les["uses"] = int(les.get("uses", 0)) + 1
            self.last_lessons = [les["id"] for les in used]
            self.save()
            if level == "object":
                return t, f"この物の教訓：{newest['text']}"
            if level == "kind":
                return t, f"『{newest['kind']}』のコツ：{newest['text']}"
            return t, f"全体の教訓：{newest['text']}"
        if obj is None:
            return base, "初めての物：判定どおり"
        return base, f"{obj.get('seen', 1)}回目に見る物：判定どおり"

    # summaries
    def stats(self):
        """Numbers for a learning curve: per run attempts, successes and attempts per success."""
        runs = {}
        for e in self.episodes:
            r = runs.setdefault(e.get("run", ""), {"run": e.get("run", ""), "attempts": 0, "successes": 0,
                                                    "objects": set(), "first_try": 0})
            r["attempts"] += 1
            first = e.get("object_id") not in r["objects"]
            r["objects"].add(e.get("object_id"))
            if is_success(e.get("result")):
                r["successes"] += 1
                if first:
                    r["first_try"] += 1
        per_run = []
        for r in runs.values():
            per_run.append({"run": r["run"], "attempts": r["attempts"], "successes": r["successes"],
                            "objects": len(r["objects"]), "first_try": r["first_try"],
                            "attempts_per_success": round(r["attempts"] / r["successes"], 2) if r["successes"] else None})
        attempts = len(self.episodes)
        successes = sum(1 for e in self.episodes if is_success(e.get("result")))
        return {"runs": per_run, "attempts": attempts, "successes": successes,
                "success_rate": round(successes / attempts, 3) if attempts else None,
                "lessons": len(self.lessons), "objects": len(self.objects)}

    def prompt_advice(self, limit=10):
        """Text to append to the judge's crop prompt: general lessons and the most used kind lessons."""
        def top(items):
            return sorted(items, key=lambda les: (-int(les.get("uses", 0)), les.get("created", "")))
        general = top([les for les in self.lessons if les["level"] == "general"])
        kinds = top([les for les in self.lessons if les["level"] == "kind"])
        chosen = (general + kinds)[:limit]
        if not chosen:
            return ""
        lines = [ADVICE_HEADER]
        for les in chosen:
            where = "any object" if les["level"] == "general" else f"kind \"{les['kind']}\""
            hint = f" (settings: {json.dumps(les['tactic_hint'], ensure_ascii=False)})" if les.get("tactic_hint") else ""
            lines.append(f"- {where}: {les['text']}{hint}")
        return "\n".join(lines)


# --- reflection -------------------------------------------------------
# The tactic setting names Claude sometimes writes in a Japanese lesson, in the words the person reads.
# \b does not separate "snug" from a following Japanese letter (both are word characters), so the
# boundaries are "no ASCII letter next to it".
_A = r"(?<![A-Za-z])"
_Z = r"(?![A-Za-z])"
JA_TERMS = [(r"sweep_first\s*(=|:)?\s*true", "手前に引き寄せる"), (r"sweep_first", "引き寄せ"),
            (r"turn\s*(=|:)?\s*90", "指の向きを90度回す"), (r"turn\s*(=|:)?\s*0", "指の向きを戻す"),
            (_A + "snug" + _Z, "ぴったり開く"), (_A + "wide" + _Z, "広く開く"),
            (r"soft\s*(=|:)?\s*true", "届いた所をつまむ"), (_A + "soft" + _Z, "つまむ"),
            (_A + r"force\s*(を|=|:)?\s*(\d)", r"力\2"), (_A + "force" + _Z, "力"), (_A + "opening" + _Z, "開き"),
            (r"too_flat", "平たすぎ"), (r"too_wide", "幅が広すぎ"), (_A + "true" + _Z, ""), (_A + "false" + _Z, "")]


def _ja(s):
    import re
    s = str(s or "")
    for pattern, word in JA_TERMS:
        s = re.sub(pattern, word, s, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", s).strip()


def _short(s, n):
    s = _ja(s).strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def _failures(history, result):
    return sum(1 for e in history if not is_success(e.get("result")) and e.get("result") != "swept") + \
        (0 if is_success(result) or result == "swept" else 1)


def _tried(history, key, value):
    return any(clamp_tactic(e.get("tactic")).get(key) == value for e in history)


def reflect_offline(obj, tactic, result, history):
    """Rule-based reflection with reflect()'s answer shape (simulation, or when Claude is unavailable).

    history: this object's earlier attempts (episodes) in this run, without the current one.
    """
    obj = obj or {}
    t = clamp_tactic(tactic)
    nxt = dict(t)
    kind = obj.get("kind") or obj.get("name") or "この種類"
    name = obj.get("name") or kind
    tried = list(history) + [{"tactic": t, "result": result}]
    give_up, reason = False, ""
    lesson = {"level": "object", "kind": kind, "text": "", "tactic_hint": None}
    if is_success(result):
        why = "この掴み方で成功"
        lesson.update(level="object", text=f"{name}は {tactic_ja(t)} で取れた", tactic_hint=dict(t))
    elif result == "slipped":
        why = "持ち上げたが握りが弱く途中で落ちた"
        if t["force"] < 5:
            nxt["force"] = t["force"] + 1
            lesson.update(level="kind", text=f"{kind}は滑りやすいので力を1段上げる", tactic_hint={"force": nxt["force"]})
        elif not _tried(tried, "turn", 90):
            nxt["turn"] = 90
            lesson.update(text=f"{name}は最強でも滑る。向きを変えて掴む", tactic_hint={"force": 5, "turn": 90})
        else:
            give_up, reason = True, "力を最大にしても向きを変えても滑る"
            lesson.update(text=f"{name}は滑って運べない。人に任せる")
    elif result == "missed":
        why = "指の間に物が入らず空振りした"
        if not _tried(tried, "turn", 90 - t["turn"]):
            nxt["turn"] = 90 - t["turn"]
            lesson.update(text=f"{name}は指の向きを90度変えて掴む", tactic_hint={"turn": nxt["turn"]})
        elif t["opening"] == "snug":
            nxt["opening"] = "wide"
            lesson.update(text=f"{name}は狙いがずれやすいので指を広く開く", tactic_hint={"opening": "wide"})
        else:
            nxt["turn"] = 90 - t["turn"]
            lesson.update(text=f"{name}は広く開いても空振りする", tactic_hint=None)
    elif result == "too_wide":
        why = "指を開いても幅が足りない向きだった"
        if t["turn"] == 0 and not _tried(tried, "turn", 90):
            nxt["turn"] = 90
            lesson.update(text=f"{name}は短い辺が広いので向きを変えて掴む", tactic_hint={"turn": 90})
        elif t["opening"] == "snug":
            nxt["opening"] = "wide"
            lesson.update(text=f"{name}は指をいっぱいに開いて掴む", tactic_hint={"opening": "wide", "turn": t["turn"]})
        else:
            give_up, reason = True, "どの向きでも指の幅に入らない"
            lesson.update(text=f"{name}は大きすぎて指に入らない")
    elif result == "not_taken":
        why = "柔らかい物をつまんだが取れていない"
        if t["soft"]:
            nxt["soft"] = False
            nxt["force"] = min(5, t["force"] + 1)
            lesson.update(level="kind", text=f"{kind}はつまむだけでは取れない。回り込んで少し強く握る",
                          tactic_hint={"soft": False, "force": nxt["force"]})
        elif t["opening"] == "snug":
            nxt["opening"] = "wide"
            lesson.update(level="kind", text=f"{kind}は指を広く開いて丸ごと包む", tactic_hint={"opening": "wide"})
        else:
            nxt["force"] = min(5, t["force"] + 1)
            lesson.update(level="kind", text=f"{kind}は強めに握る", tactic_hint={"force": nxt["force"]})
    elif result in ("stalled", "blocked"):
        why = "指が物や周りに当たって下りられない" if result == "stalled" else "箱や隣の物が邪魔で指を下ろせない"
        if not t["sweep_first"] and not _tried(tried, "sweep_first", True):
            nxt["sweep_first"] = True
            lesson.update(text=f"{name}は周りが混んでいるので手前に引き寄せてから掴む", tactic_hint={"sweep_first": True})
        else:
            give_up, reason = True, "引き寄せても指を下ろせない"
            lesson.update(text=f"{name}は引き寄せても掴めない場所にある")
    elif result in ("unreachable", "too_far"):
        why = "腕が届かない位置にある"
        if not t["sweep_first"] and not _tried(tried, "sweep_first", True):
            nxt["sweep_first"] = True
            lesson.update(level="general", kind="", text="遠くの物は手前に引き寄せてから掴む",
                          tactic_hint={"sweep_first": True})
        else:
            give_up, reason = True, "引き寄せても届かない"
            lesson.update(text=f"{name}は届かない所にある。人に任せる")
    elif result == "too_flat":
        why = "平たすぎて机からつまめない"
        give_up, reason = True, "薄い物は指でつまめない"
        lesson.update(level="general", kind="", text="厚さ5mm未満の硬い物は掴まず人に任せる")
    else:  # lost, failed, swept or anything new: try once more as before unless it keeps failing
        why = RESULT_JA.get(result, "うまくいかなかった")
        nxt["sweep_first"] = False if result == "swept" else t["sweep_first"]
        if result == "lost":
            give_up, reason = True, "物を見失った"
    if not is_success(result) and _failures(history, result) >= GIVE_UP_AFTER and not give_up:
        give_up, reason = True, f"{GIVE_UP_AFTER}回失敗したので人に任せる"
    if not lesson["text"]:
        lesson = {"level": "object", "kind": kind, "text": "", "tactic_hint": None}
    return {"why": _short(why, 40), "lesson": lesson, "next_tactic": clamp_tactic(nxt, t),
            "give_up": bool(give_up), "give_up_reason": reason, "source": "offline"}


def _history_text(history):
    if not history:
        return "(none - this was the first attempt)"
    lines = []
    for e in history[-8:]:
        lines.append(f"- attempt {e.get('attempt')}: {json.dumps(clamp_tactic(e.get('tactic')))} -> {e.get('result')}"
                     + (f" ({e['why']})" if e.get("why") else ""))
    return "\n".join(lines)


def reflect(notebook, obj, tactic, result, images, history, model="sonnet", cwd=HERE, timeout=REFLECT_TIMEOUT_S,
            reach_cm=None, extra=""):
    """Ask Claude (one claude -p call) why the attempt went as it did and what to try next.

    Falls back to reflect_offline when Claude does not answer. next_tactic and the lesson's hint
    are clamped to the tactic menu; give_up is forced after GIVE_UP_AFTER failures.
    """
    import desk_judge  # late: it pulls in the hand-eye calibration modules
    obj = obj or {}
    verdict = obj.get("verdict") or {}
    t = clamp_tactic(tactic)
    kind = obj.get("kind") or obj.get("name") or ""
    lessons = notebook.lessons_for(kind, obj.get("id")) if notebook is not None else []
    lessons_text = "\n".join(f"- [{les['level']}{' ' + les['kind'] if les.get('kind') else ''}] {les['text']}"
                             for les in lessons) or "(nothing yet)"
    size = verdict.get("size_cm") or ["?", "?", "?"]
    prompt = REFLECT_PROMPT.format(
        name=obj.get("name", "?"), kind=kind or "?", size=" x ".join(f"{v:.0f}" if isinstance(v, (int, float)) else
                                                                    str(v) for v in size),
        category=obj.get("category") or verdict.get("category", "?"), deformable=verdict.get("deformable", "?"),
        force=verdict.get("force", "?"), force_reason=verdict.get("force_reason", ""),
        grasp_point=verdict.get("grasp_point", ""), tactic=json.dumps(t), result=result,
        meaning=RESULT_EXPLAIN.get(result, "unknown result code"), history=_history_text(history),
        lessons=lessons_text, images="\n".join(f"- {Path(p).resolve()}" for p in images) or "(no pictures)",
        give_up_after=GIVE_UP_AFTER, reach="?" if reach_cm is None else f"{reach_cm:.0f}")
    try:
        reply = desk_judge._claude(prompt + extra, REFLECT_SCHEMA, model, str(cwd), timeout=timeout)
    except (RuntimeError, OSError, ValueError):
        return reflect_offline(obj, t, result, history)
    lesson = reply.get("lesson") if isinstance(reply.get("lesson"), dict) else {}
    level = lesson.get("level") if lesson.get("level") in LEVELS else "kind"
    give_up = _as_bool(reply.get("give_up"), False)
    reason = _short(reply.get("give_up_reason"), 60)
    if not is_success(result) and _failures(history, result) >= GIVE_UP_AFTER and not give_up:
        give_up, reason = True, f"{GIVE_UP_AFTER}回失敗したので人に任せる"
    return {"why": _short(reply.get("why"), 40),
            "lesson": {"level": level, "kind": "" if level == "general" else (lesson.get("kind") or kind),
                       "text": _short(lesson.get("text"), 60), "tactic_hint": clamp_hint(lesson.get("tactic_hint"))},
            "next_tactic": clamp_tactic(reply.get("next_tactic"), t), "give_up": give_up,
            "give_up_reason": reason if give_up else "", "source": "claude"}


# --- notes page -------------------------------------------------------
NOTES_CSS = """<style>
.notes{--bg:#fafaf7;--card:#fff;--ink:#222;--sub:#666;--line:#ddd;--accent:#2a7a4b;
 background:var(--bg);color:var(--ink);font-family:system-ui,"Noto Sans CJK JP",sans-serif;padding:12px;border-radius:8px}
@media (prefers-color-scheme:dark){.notes{--bg:#1b1c1e;--card:#25272a;--ink:#eee;--sub:#aaa;--line:#3a3c40;--accent:#6fcf97}}
.notes h2{font-size:1.15em;margin:16px 0 8px}.notes h3{font-size:1em;margin:12px 0 6px;color:var(--accent)}
.notes .counts{display:flex;gap:10px;flex-wrap:wrap}
.notes .count{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 14px}
.notes .count b{font-size:1.5em;display:block}
.notes ul{margin:0;padding-left:1.2em}.notes li{margin:3px 0}.notes .sub{color:var(--sub);font-size:.85em}
.notes .objs{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.notes .obj{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px;font-size:.9em}
.notes .obj img{width:100%;aspect-ratio:1;object-fit:contain;background:var(--line);border-radius:4px}
.notes table{border-collapse:collapse}.notes td,.notes th{border-bottom:1px solid var(--line);padding:3px 8px;text-align:right}
</style>"""


def notes_html(notebook):
    """HTML fragment (Japanese only) showing what the robot has learned."""
    e = html.escape
    s = notebook.stats()
    rate = "—" if s["success_rate"] is None else f"{s['success_rate'] * 100:.0f}%"
    out = [NOTES_CSS, '<div class="notes">', '<div class="counts">',
           f'<div class="count">知っている物<b>{s["objects"]} 個</b></div>',
           f'<div class="count">教訓<b>{s["lessons"]} 個</b></div>',
           f'<div class="count">成功率<b>{rate}</b><span class="sub">{s["successes"]} / {s["attempts"]} 回</span></div>',
           "</div>"]
    if s["runs"]:
        out.append("<h2>回ごとの記録</h2><table><tr><th>回</th><th>試した数</th><th>成功</th><th>1回目で成功</th>"
                   "<th>成功1つあたりの試行</th></tr>")
        for k, r in enumerate(s["runs"], 1):
            per = "—" if r["attempts_per_success"] is None else f"{r['attempts_per_success']:.1f}"
            out.append(f"<tr><td>{k}</td><td>{r['attempts']}</td><td>{r['successes']}</td><td>{r['first_try']}</td>"
                       f"<td>{per}</td></tr>")
        out.append("</table>")
    out.append("<h2>教訓</h2>")
    names = {o["id"]: o["name"] for o in notebook.objects}
    for level in LEVELS:
        items = sorted((les for les in notebook.lessons if les["level"] == level),
                       key=lambda les: les.get("created", ""), reverse=True)
        out.append(f"<h3>{LEVELS_JA[level]}</h3>")
        if not items:
            out.append('<p class="sub">まだありません</p>')
            continue
        out.append("<ul>")
        for les in items:
            who = names.get(les.get("object_id"), "") if level == "object" else les.get("kind", "")
            tag = f"【{e(who)}】" if who else ""
            hint = f' <span class="sub">→ {e(hint_ja(les.get("tactic_hint")))}</span>' if les.get("tactic_hint") else ""
            used = f' <span class="sub">（使った回数 {int(les.get("uses", 0))}）</span>'
            out.append(f"<li>{tag}{e(les['text'])}{hint}{used}</li>")
        out.append("</ul>")
    out.append("<h2>知っている物</h2>")
    if not notebook.objects:
        out.append('<p class="sub">まだありません</p>')
    out.append('<div class="objs">')
    for o in sorted(notebook.objects, key=lambda o: o.get("updated", ""), reverse=True):
        img = f'<img src="/notes/img/{e(Path(o["crop"]).name)}" alt="">' if o.get("crop") else ""
        best = e(tactic_ja(o["best_tactic"])) if o.get("best_tactic") else "まだ成功していない"
        cat = CATEGORY_JA.get(o.get("category"), "")
        out.append(f'<div class="obj">{img}<div><b>{e(o["name"])}</b> <span class="sub">{e(o.get("kind") or "")}'
                   f'{"・" + cat if cat else ""}</span></div>'
                   f'<div class="sub">見た回数 {o.get("seen", 0)}・成功 {o.get("successes", 0)} / {o.get("attempts", 0)}</div>'
                   f"<div>{best}</div></div>")
    out.append("</div></div>")
    return "\n".join(out)
