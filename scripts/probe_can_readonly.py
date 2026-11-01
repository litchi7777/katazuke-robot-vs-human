#!/usr/bin/env python3
"""RobStrideの位置または電源電圧を、SocketCANで1回だけ読み取る。

通信タイプ17の読み取り要求だけを送る。モータの有効化、モード設定、
パラメータ書き込み、MIT指令は扱わない。他のCANプログラムを止めて使う。
--dry-run は送信内容を表示するだけで、CANソケットも開かない。
アプリからの送信要求は1回であり、CANコントローラの自動再送は含まない。
"""

import argparse
import json
import math
import socket
import struct
import subprocess
import time
from pathlib import Path


FRAME = struct.Struct("=IB3x8s")
HOST_ID = 0xFD
READ_TYPE = 0x11
PARAMETERS = {"position": 0x7019, "voltage": 0x701C}


def request_frame(motor_id, parameter):
    if motor_id not in range(1, 8) or parameter not in PARAMETERS.values():
        raise ValueError("Only motor IDs 1..7 and position/voltage reads are allowed")
    can_id = socket.CAN_EFF_FLAG | (READ_TYPE << 24) | (HOST_ID << 8) | motor_id
    return FRAME.pack(can_id, 8, struct.pack("<H2x4x", parameter))


def response_id(motor_id):
    return socket.CAN_EFF_FLAG | (READ_TYPE << 24) | (motor_id << 8) | HOST_ID


def decode_reply(raw, motor_id, parameter):
    if len(raw) != FRAME.size:
        return None
    can_id, length, data = FRAME.unpack(raw)
    if can_id != response_id(motor_id) or length != 8:
        return None
    if data[:4] != struct.pack("<H2x", parameter):
        return None
    value = struct.unpack("<f", data[4:])[0]
    if not math.isfinite(value):
        raise ValueError("Response contains a non-finite value")
    return value


def read_once(bus, motor_id, parameter, timeout_s, clock=time.perf_counter):
    result = {"status": "timeout", "frames": [], "frames_sent": 0}
    started = clock()
    deadline = started + timeout_s
    bus.settimeout(timeout_s)
    try:
        sent = bus.send(request_frame(motor_id, parameter))
        if sent != FRAME.size:
            raise OSError("Incomplete SocketCAN send")
        result["frames_sent"] = 1
        while len(result["frames"]) < 256:
            remaining = deadline - clock()
            if remaining <= 0:
                break
            bus.settimeout(remaining)
            try:
                raw = bus.recv(FRAME.size)
            except socket.timeout:
                break
            item = {"elapsed_ms": (clock() - started) * 1000, "raw": raw.hex()}
            result["frames"].append(item)
            if len(raw) != FRAME.size:
                continue
            can_id, length, data = FRAME.unpack(raw)
            item.update(can_id=f"0x{can_id:08x}", length=length, data=data[:length].hex())
            item["is_error"] = bool(can_id & socket.CAN_ERR_FLAG)
            # 10/31: after the 10/28 j2 trip the adapter sends one controller-class
            # frame per new socket with no detail and both error counters (data[6:8])
            # at 0, i.e. "error-active, no errors". That is a state report, not a fault.
            if (item["is_error"] and can_id & socket.CAN_ERR_MASK == 0x04 and length == 8
                    and not any(data[:8])):
                item.update(is_error=False, state_report=True)
                continue
            if item["is_error"]:
                item["error_class"] = f"0x{can_id & socket.CAN_ERR_MASK:08x}"
                item["controller_status_byte"] = data[1]
                item["ack_error"] = bool(can_id & 0x20)
                item["bus_off"] = bool(can_id & 0x40)
                if item["bus_off"] or (can_id & 0x04 and data[1] & 0x30):
                    result["status"] = "can_error"
                    break
                continue
            value = decode_reply(raw, motor_id, parameter)
            if value is not None:
                result["value"] = value
                result["response_ms"] = item["elapsed_ms"]
                result["status"] = (
                    "response_with_can_errors"
                    if any(frame.get("is_error") for frame in result["frames"])
                    else "ok"
                )
                break
        else:
            result["status"] = "frame_limit"
    except (OSError, ValueError) as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    result["elapsed_ms"] = (clock() - started) * 1000
    return result


def link_state(channel):
    completed = subprocess.run(
        ["ip", "-json", "-details", "-statistics", "link", "show", channel],
        check=True, capture_output=True, text=True, timeout=3,
    )
    return json.loads(completed.stdout)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--motor-id", type=int, choices=range(1, 8), default=1)
    parser.add_argument("--parameter", choices=PARAMETERS, default="position")
    parser.add_argument("--timeout-ms", type=int, default=200)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.timeout_ms <= 2000:
        parser.error("--timeout-ms must be within 1..2000")
    parameter = PARAMETERS[args.parameter]
    request = request_frame(args.motor_id, parameter)
    can_id, _, data = FRAME.unpack(request)
    result = {
        "channel": args.channel, "motor_id": args.motor_id,
        "parameter": f"0x{parameter:04x}", "unit": "rad" if parameter == 0x7019 else "V",
        "request_id": f"0x{can_id:08x}", "request_data": data.hex(),
        "timeout_ms": args.timeout_ms, "frames_sent": 0,
    }
    # Reserve the output before sending, so an existing record is never overwritten.
    try:
        output = args.output.open("x") if args.output else None
    except OSError as error:
        parser.error(str(error))
    try:
        if args.dry_run:
            result["status"] = "dry_run"
        else:
            result["link_before"] = link_state(args.channel)
            before = result["link_before"]
            info = before.get("linkinfo", {}).get("info_data", {})
            if (
                "UP" not in before.get("flags", [])
                or info.get("state") != "ERROR-ACTIVE"
                or info.get("bittiming", {}).get("bitrate") != 1000000
            ):
                raise ValueError("CAN must be UP, ERROR-ACTIVE, and 1Mbps; ask the user to recover CAN first")
            with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus:
                bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_ERR_FILTER, struct.pack("=I", 0x1FFFFFFF))
                # CAN_ERR_FLAG in this mask selects the kernel's error-only list.
                # Subscribe to data replies with EFF/RTR/ID bits; errors above
                # have their own subscription (Linux SocketCAN documentation).
                reply_mask = socket.CAN_EFF_FLAG | socket.CAN_RTR_FLAG | socket.CAN_EFF_MASK
                bus.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                               struct.pack("=II", response_id(args.motor_id), reply_mask))
                bus.bind((args.channel,))
                result.update(read_once(bus, args.motor_id, parameter, args.timeout_ms / 1000))
            result["link_after"] = link_state(args.channel)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result["status"] = "interrupted"
    finally:
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if output:
            with output:
                output.write(rendered + "\n")
        print(rendered)
    return 0 if result["status"] in ("ok", "dry_run") else 2


if __name__ == "__main__":
    raise SystemExit(main())
