#!/usr/bin/env python3
"""Stand-in for arm_goto.py --execute that drives sim_world instead of the arm.

Same stdin commands, the same result lines ("到達:", "グリッパ:", "撮影:", ...)
and the same events.jsonl, so the orchestrators run unchanged with --sim.
Motion is kinematic: a move stops at the last sampled pose before the gripper
would penetrate a box, fixture, object or the desk (the real arm stalls there
too, with a lag the probe detects). Grip close takes the object between the
fingers if it fits; a squeeze below the object's min force lets it slip when
lifted, one above its max force crushes it.

    python3 sim_arm.py --world WORLD.json --output-dir DIR [arm_goto options]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import arm_goto
import sim_world
import topdown_ik

CALIB = Path(__file__).resolve().parent / "calib/video2.json"
SAG = 0.002        # m free-space settling scatter of the real arm (z)
SWAY = 0.004       # rad joint1 settling scatter (~1.5 mm sideways at 0.4 m)
FORCE_OF_SQUEEZE = sorted((v, k) for k, v in {1: 0.03, 2: 0.05, 3: 0.08, 4: 0.11, 5: 0.14}.items())


def force_level(squeeze):
    return max(k for v, k in FORCE_OF_SQUEEZE if v <= squeeze + 1e-9) if squeeze >= 0.03 - 1e-9 else 0


class SimArm:
    def __init__(self, args, emit):
        self.a, self.emit = args, emit
        self.limits = arm_goto.driver_limits(args.driver_config)
        self.q = [0.0] * 6
        self.opening = 0.0
        self.armed = False
        self.fault = None
        self.rng = np.random.default_rng(args.seed)
        self.calib = json.loads(CALIB.read_text())

    def world(self):
        return sim_world.load(self.a.world)

    def held(self, world):
        return next((o for o in world["objects"] if o.get("held")), None)

    # --- motion -------------------------------------------------------
    def move(self, target, correct):
        if self.fault:
            raise ValueError(f"faulted ({self.fault}); no further moves")
        points = arm_goto.plan(self.q, target, self.limits, self.a.max_speed)
        lowest = arm_goto.check_path(points, self.a.min_tcp_z)
        print(f"移動開始: {points[-1][0]:.1f}秒、手先の最低高さ {lowest:.3f} m。何か入力すると停止。", flush=True)
        world = self.world()
        held = self.held(world)
        ignore = (held["id"],) if held else ()
        stop, blocked = list(target), None
        samples = points[:: max(1, len(points) // 60)] + [points[-1]]
        prev = list(self.q)
        # Contact already present at the start (leaning on something) does not block leaving it.
        already = sim_world.collisions(world, self.q, self.opening, ignore)
        loose = {o["id"]: o for o in world["objects"]}
        pushed = False
        for _, q in samples[1:]:
            hit = sim_world.collisions(world, q, self.opening, ignore) - already
            if hit and hit <= set(loose):
                # Moving mostly sideways into loose objects pushes them along instead of stalling.
                a, b = topdown_ik.fk(prev).translation, topdown_ik.fk(q).translation
                if np.linalg.norm((b - a)[:2]) > abs((b - a)[2]):
                    for name in hit:
                        loose[name]["centre"] = (np.array(loose[name]["centre"]) + (b - a)[:2] * 1.05).tolist()
                    pushed = True
                    prev = list(q)
                    continue
            if hit:
                stop, blocked = prev, hit
                break
            prev = list(q)
        if pushed:
            sim_world.save(world, self.a.world)
        if blocked:
            # The real arm leans on the obstacle: a large shortfall trips the tracking watchdog.
            worst = max(abs(s - t) for s, t in zip(stop, target))
            if worst > arm_goto.MAX_TRACKING:
                self.q = stop
                self.fault = f"tracking error {worst:.3f} rad exceeds {arm_goto.MAX_TRACKING} ({sorted(blocked)})"
                self.carry(world)
                self.emit("fault", reason=self.fault, positions=stop)
                raise RuntimeError(self.fault)
        settled = list(stop)
        if not blocked:  # free-space scatter in tip height, mostly via j2, never into a solid
            sagged = list(settled)
            sagged[1] += self.rng.normal(0, SAG) / 0.45
            sagged[0] += self.rng.normal(0, SWAY)
            if not sim_world.collisions(world, sagged, self.opening, ignore):
                settled = sagged
        self.q = settled
        self.carry(world)
        self.emit("goto_settled", target=target, command=target, positions=settled, blocked=sorted(blocked or []))
        print("到達: " + " ".join(f"{q:+.3f}" for q in settled), flush=True)

    def carry(self, world):
        if getattr(self, "slip_pending", False):  # too weak a squeeze: it stays behind, fingers close
            self.slip_pending = False
            self.opening = self.squeezed[0]
        held = self.held(world)
        if held:
            T = topdown_ik.fk(self.q)
            held["centre"] = (T.translation[:2] + np.array(held["grip_offset"])).tolist()
            held["z"] = float(T.translation[2] - sim_world.TIP_REACH)
            sim_world.save(world, self.a.world)

    # --- gripper ------------------------------------------------------
    def grip_open(self, target):
        if not 0.0 < target <= self.a.grip_open_max:
            raise ValueError(f"open target must be in (0, {self.a.grip_open_max}] rad")
        world = self.world()
        if sim_world.collisions(world, self.q, target, ignore=[o["id"] for o in world["objects"] if o.get("held")]):
            target = self.opening  # fingers blocked sideways; stay
        self.opening = target
        held = self.held(world)
        if held:
            held["held"] = False
            z, owner = sim_world.surface_below(world, held["centre"], ignore=(held["id"],))
            held["z"] = z
            held.pop("grip_offset", None)
            self.emit("released", object=held["id"], onto=owner, centre=held["centre"])
            sim_world.save(world, self.a.world)
        print(f"グリッパ: {target:+.3f} rad（目標 {target:.2f}）", flush=True)

    def grip_close(self, squeeze):
        world = self.world()
        inside = sim_world.between_fingers(world, self.q, self.opening)
        result, position = "closed_empty", 0.0
        if inside:
            o, span, _ = inside[0]
            width = min(span, o["size"][1]) if not o.get("soft") else max(0.004, 0.5 * min(span, o["size"][1]))
            position = max(0.0, sim_world.rad_for_gap(width))
            if position < self.opening:
                result = "contact"
                level = force_level(squeeze)
                if level > o.get("max_force", 5):
                    o["crushed"] = True
                if level >= o.get("min_force", 1):
                    T = topdown_ik.fk(self.q)
                    o["held"] = True
                    o["grip_offset"] = (np.array(o["centre"]) - T.translation[:2]).tolist()
                else:
                    o["slipped"] = o.get("slipped", 0) + 1
                    self.slip_pending = True
                sim_world.save(world, self.a.world)
        final = max(0.0, position - squeeze) if result == "contact" else 0.0
        self.opening = position if result == "contact" else 0.0
        self.squeezed = (final, position)
        self.emit("grip_close", result=result, final_target=final, position=position)
        print(f"グリッパ: {result}、位置 {position:+.3f} rad、目標 {final:+.3f} rad", flush=True)

    def snap(self, label):
        folder = self.a.output_dir / "snaps"
        folder.mkdir(exist_ok=True)
        path = folder / f"{label}_video2.jpg"
        if path.exists():
            raise RuntimeError(f"camera capture from /dev/video2 failed ({path.name} exists)")
        sim_world.render(self.world(), self.calib, path)
        self.emit("snap", label=label, positions=self.q, files={"video2": str(path)})
        print(f"撮影: {label} video2", flush=True)

    # --- command loop -------------------------------------------------
    def run(self):
        print("commands: arm / goto / step / home / grip / snap / finish / status / disable / quit", flush=True)
        print("q=" + " ".join(f"{q:+.3f}" for q in self.q), flush=True)
        for line in sys.stdin:
            words = line.split()
            if not words:
                continue
            cmd = words[0]
            try:
                if cmd == "arm":
                    if any(abs(q) > arm_goto.FOLDED for q in self.q):
                        raise ValueError("arm requires every joint within 0.1 rad of zero")
                    self.armed = True
                    print("保持確認完了。goto で移動できます。", flush=True)
                elif cmd in ("goto", "step", "home"):
                    if not self.armed:
                        raise ValueError("arm first")
                    target = [0.0] * 6 if cmd == "home" else arm_goto.parse_target(words, {})
                    self.move(target, correct=cmd == "goto")
                elif cmd == "grip":
                    if not self.armed or self.fault:
                        raise ValueError("arm first; no fault allowed")
                    if len(words) >= 2 and words[1] == "open":
                        self.grip_open(float(words[2]) if len(words) == 3 else arm_goto.GRIP_OPEN)
                    elif len(words) == 2 and words[1] == "state":
                        print(f"グリッパ: state、位置 {self.opening:+.3f} rad", flush=True)
                    elif len(words) in (2, 3, 4) and words[1] == "close":
                        squeeze = float(words[2]) if len(words) >= 3 else arm_goto.GRIP_SQUEEZE
                        if not arm_goto.GRIP_SQUEEZE_RANGE[0] <= squeeze <= arm_goto.GRIP_SQUEEZE_RANGE[1]:
                            raise ValueError(f"squeeze must be in {arm_goto.GRIP_SQUEEZE_RANGE} rad")
                        self.grip_close(squeeze)
                    elif len(words) == 4 and words[1] == "resqueeze":
                        # The kinematic sim's objects never turn in the fingers: it stays held as it is.
                        print(f"グリッパ: held、位置 {self.opening:+.3f} rad、目標 {float(words[2]):+.3f} rad", flush=True)
                    else:
                        raise ValueError("use grip open [RAD], grip close [SQUEEZE] or grip resqueeze T S")
                elif cmd == "snap":
                    self.snap(words[1])
                elif cmd == "finish":
                    if any(abs(q) > 0.04 for q in self.q):
                        raise ValueError("use home first; finish needs every joint within 0.04 rad of zero")
                    self.armed = False
                    self.emit("normal_finish")
                    print("終了処理完了: safe_home -> disable", flush=True)
                elif cmd == "status":
                    print("q=" + " ".join(f"{q:+.3f}" for q in self.q), flush=True)
                elif cmd in ("disable", "quit"):
                    self.emit("client_exit")
                    return 0
                else:
                    raise ValueError(f"unknown command {cmd}")
            except ValueError as error:
                print("指令拒否: " + str(error), flush=True)
                self.emit("command_rejected", reason=str(error))
            except RuntimeError as error:
                print(f"停止要求: {error}。新しい移動は禁止。", flush=True)
                self.emit("fault_latched", reason=str(error))
        return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--driver-config", required=True)
    p.add_argument("--min-tcp-z", type=float, default=0.08)
    p.add_argument("--max-speed", type=float, default=arm_goto.MAX_SPEED)
    p.add_argument("--grip-open-max", type=float, default=1.6)
    p.add_argument("--seed", type=int, default=0)
    # accepted for drop-in compatibility with arm_goto.py
    for flag in ("--execute", "--gripper", "--correct", "--resume"):
        p.add_argument(flag, action="store_true")
    p.add_argument("--snap-cameras", default="/dev/video2")
    p.add_argument("--preflight")
    a = p.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    with (a.output_dir / "events.jsonl").open("x", buffering=1) as out:
        def emit(event, **fields):
            out.write(json.dumps(dict(event=event, wall_ns=time.time_ns(), **fields), ensure_ascii=False) + "\n")
        return SimArm(a, emit).run()


if __name__ == "__main__":
    raise SystemExit(main())
