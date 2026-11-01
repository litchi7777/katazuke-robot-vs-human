#!/usr/bin/env python3
"""7台の位置・電圧と、位置の繰り返し読み取りをトルクを入れずに確認する。"""

import argparse
import json
import socket
import struct
import time
from pathlib import Path
from statistics import mean, median

import probe_can_readonly as probe


class StopMeasurement(RuntimeError):
    pass


def check_link(state, reference=None):
    info = state.get("linkinfo", {}).get("info_data", {})
    if ("UP" not in state.get("flags", []) or info.get("state") != "ERROR-ACTIVE"
            or info.get("bittiming", {}).get("bitrate") != 1000000):
        raise StopMeasurement("CAN is not UP, ERROR-ACTIVE and 1Mbps")
    if reference is not None:
        if (state.get("ifindex"), state.get("parentdev")) != (
                reference.get("ifindex"), reference.get("parentdev")):
            raise StopMeasurement("CAN interface or USB connection changed")
        current = state["linkinfo"].get("info_xstats", {})
        original = reference["linkinfo"].get("info_xstats", {})
        if current != original:
            raise StopMeasurement("CAN error or restart counters changed")


def measure(bus, emit, rounds, rate_hz, get_link, before, sleep=time.sleep,
            clock=time.perf_counter):
    """1スレッド・逐次読み取り。最初の失敗以降は送信しない。"""
    def read(phase, index, motor, parameter):
        result = probe.read_once(bus, motor, parameter, 0.2)
        result.update(phase=phase, round=index, motor_id=motor,
                      parameter=f"0x{parameter:04x}",
                      unit="rad" if parameter == 0x7019 else "V")
        emit(result)
        if result["status"] != "ok":
            raise StopMeasurement(f"{phase}: motor {motor}, {parameter:#x}: {result['status']}")

    # 前回確認したjoint1より先に残り6台を確認する。
    for motor in (2, 3, 4, 5, 6, 7, 1):
        check_link(get_link(), before)
        read("initial", 0, motor, 0x7019)
        read("initial", 0, motor, 0x701C)
    check_link(get_link(), before)
    print("initial: all 7 motors responded for position and voltage", flush=True)

    started = clock()
    next_check = started
    completed = 0
    overruns = 0
    for index in range(rounds):
        now = clock()
        if now >= next_check:
            check_link(get_link(), before)
            next_check = now + 1
        cycle_started = clock()
        for motor in range(1, 8):
            read("stability", index, motor, 0x7019)
        completed += 1
        deadline = started + completed / rate_hz
        remaining = deadline - clock()
        if remaining > 0:
            sleep(remaining)
        else:
            overruns += 1
        if completed % (rate_hz * 10) == 0:
            print(f"stability: {completed}/{rounds} rounds, {completed * 7} successful reads", flush=True)
    check_link(get_link(), before)
    return {"rounds_completed": completed, "seconds_measured": clock() - started,
            "round_deadline_misses": overruns}


def timing(values):
    ordered = sorted(values)
    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return {"count": len(values), "min_ms": min(values), "mean_ms": mean(values),
            "median_ms": median(values), "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99), "max_ms": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--seconds", type=int, default=60, choices=range(1, 121), metavar="1..120")
    parser.add_argument("--rate-hz", type=int, default=10, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = {"channel": args.channel, "seconds_requested": args.seconds,
               "round_rate_hz": args.rate_hz, "rounds_requested": args.seconds * args.rate_hz,
               "read_timeout_ms": 200, "communication": "sequential SocketCAN parameter reads only",
               "parameters": {"position": "0x7019", "voltage": "0x701c"},
               "planned_requests": 14 + args.seconds * args.rate_hz * 7}
    if args.dry_run:
        print(json.dumps(dict(summary, status="dry_run"), indent=2))
        return 0
    args.output_dir.mkdir(exist_ok=False)
    records = []
    with (args.output_dir / "summary.json").open("x") as output, \
            (args.output_dir / "readings.jsonl").open("x", buffering=1) as log:
        def emit(record):
            records.append(record)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
        try:
            before = probe.link_state(args.channel)
            summary["link_before"] = before
            check_link(before)
            with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus:
                bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_ERR_FILTER,
                               struct.pack("=I", socket.CAN_ERR_MASK))
                mask = socket.CAN_EFF_FLAG | socket.CAN_RTR_FLAG | socket.CAN_EFF_MASK
                filters = b"".join(struct.pack("=II", probe.response_id(motor), mask)
                                   for motor in range(1, 8))
                bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
                bus.bind((args.channel,))
                summary.update(measure(bus, emit, args.seconds * args.rate_hz, args.rate_hz,
                                       lambda: probe.link_state(args.channel), before))
            summary["status"] = "ok"
        except KeyboardInterrupt:
            summary["status"] = "interrupted"
        except Exception as error:
            summary.update(status="stopped", reason=f"{type(error).__name__}: {error}")
        finally:
            try:
                summary["link_after"] = probe.link_state(args.channel)
            except Exception as error:
                summary["link_after_error"] = str(error)
            summary["requests_sent"] = sum(record["frames_sent"] for record in records)
            summary["successful_reads"] = sum(record["status"] == "ok" for record in records)
            summary["initial_readings"] = [record for record in records if record["phase"] == "initial"]
            summary["error_frames"] = sum(frame.get("is_error", False)
                                          for record in records for frame in record["frames"])
            for phase in ("initial", "stability"):
                readings = [record for record in records if record["phase"] == phase
                            and record["status"] == "ok"]
                if readings:
                    summary[f"{phase}_timing"] = timing([r["response_ms"] for r in readings])
                    summary[f"{phase}_by_motor"] = {
                        str(motor): timing([r["response_ms"] for r in readings if r["motor_id"] == motor])
                        for motor in range(1, 8) if any(r["motor_id"] == motor for r in readings)}
            rendered = json.dumps(summary, ensure_ascii=False, indent=2)
            output.write(rendered + "\n")
            print(rendered, flush=True)
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
