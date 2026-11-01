import math
import contextlib
import io
import json
import socket
import struct
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_can_readonly as probe


class FakeSocket:
    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    def settimeout(self, seconds):
        pass

    def send(self, raw):
        self.sent.append(raw)
        return len(raw)

    def recv(self, size):
        if not self.frames:
            raise socket.timeout()
        return self.frames.pop(0)


# Actual successful motor1 request and reply captured on 2026-10-17.
REQUEST = bytes.fromhex("01fd0091080000001970000000000000")
REPLY = bytes.fromhex("fd010091080000001970000072fafabc")
WARNING = bytes.fromhex("0400002008000000000c000000000000")


class ProbeTests(unittest.TestCase):
    def test_request_matches_successful_capture(self):
        self.assertEqual(probe.request_frame(1, 0x7019), REQUEST)

    def test_only_read_parameters_and_known_motors_are_allowed(self):
        for motor in (0, 8, 255):
            with self.assertRaises(ValueError):
                probe.request_frame(motor, 0x7019)
        for parameter in (0x7005, 0x7016, 0x7018, 0xFFFF):
            with self.assertRaises(ValueError):
                probe.request_frame(1, parameter)
        for motor in range(1, 8):
            for parameter in (0x7019, 0x701C):
                can_id, length, data = probe.FRAME.unpack(probe.request_frame(motor, parameter))
                self.assertEqual((can_id >> 24) & 0x1F, 0x11)
                self.assertEqual(length, 8)
                self.assertEqual(data, struct.pack("<H6x", parameter))

    def test_decodes_actual_reply(self):
        self.assertAlmostEqual(probe.decode_reply(REPLY, 1, 0x7019), -0.0306369997561)

    def test_rejects_wrong_motor_parameter_and_frame_flags(self):
        self.assertIsNone(probe.decode_reply(REPLY, 2, 0x7019))
        self.assertIsNone(probe.decode_reply(REPLY, 1, 0x701C))
        self.assertIsNone(probe.decode_reply(REPLY[:-1], 1, 0x7019))
        can_id, _, data = probe.FRAME.unpack(REPLY)
        for flag in (socket.CAN_RTR_FLAG, socket.CAN_ERR_FLAG):
            raw = probe.FRAME.pack(can_id | flag, 8, data)
            self.assertIsNone(probe.decode_reply(raw, 1, 0x7019))
        for value in (math.nan, math.inf):
            raw = probe.FRAME.pack(can_id, 8, struct.pack("<H2xf", 0x7019, value))
            with self.assertRaises(ValueError):
                probe.decode_reply(raw, 1, 0x7019)

    def test_success_sends_exactly_one_read(self):
        bus = FakeSocket([REPLY])
        result = probe.read_once(bus, 1, 0x7019, 0.2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(bus.sent, [REQUEST])

    def test_warning_is_retained_even_when_reply_arrives(self):
        bus = FakeSocket([WARNING, REPLY])
        result = probe.read_once(bus, 1, 0x7019, 0.2)
        self.assertEqual(result["status"], "response_with_can_errors")
        self.assertEqual(result["frames"][0]["controller_status_byte"], 0x0C)
        self.assertEqual(bus.sent, [REQUEST])

    def test_timeout_does_not_retry(self):
        bus = FakeSocket([])
        result = probe.read_once(bus, 1, 0x7019, 0.2)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(bus.sent, [REQUEST])

    def test_bus_off_stops_without_further_reads_or_sends(self):
        bus_off = probe.FRAME.pack(socket.CAN_ERR_FLAG | 0x40, 8, b"\0" * 8)
        bus = FakeSocket([bus_off, REPLY])
        result = probe.read_once(bus, 1, 0x7019, 0.2)
        self.assertEqual(result["status"], "can_error")
        self.assertEqual(len(bus.frames), 1)
        self.assertEqual(bus.sent, [REQUEST])

    def test_cli_refuses_unhealthy_CAN_before_opening_socket(self):
        state = {"flags": ["UP"], "linkinfo": {"info_data": {
            "state": "BUS-OFF", "bittiming": {"bitrate": 1000000},
        }}}
        output = io.StringIO()
        with patch.object(sys, "argv", ["probe"]), \
             patch.object(probe, "link_state", return_value=state), \
             patch.object(probe.socket, "socket") as open_socket, \
             contextlib.redirect_stdout(output):
            code = probe.main()
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["frames_sent"], 0)
        open_socket.assert_not_called()

    def test_cli_subscribes_to_data_reply_and_errors_separately(self):
        state = {"flags": ["UP"], "linkinfo": {"info_data": {
            "state": "ERROR-ACTIVE", "bittiming": {"bitrate": 1000000},
        }}}
        output = io.StringIO()
        with patch.object(sys, "argv", ["probe"]), \
             patch.object(probe, "link_state", return_value=state), \
             patch.object(probe.socket, "socket") as open_socket, \
             contextlib.redirect_stdout(output):
            bus = open_socket.return_value.__enter__.return_value
            bus.send.return_value = len(REQUEST)
            bus.recv.return_value = REPLY
            code = probe.main()
        self.assertEqual(code, 0)
        filters = {
            call.args[1]: call.args[2] for call in bus.setsockopt.call_args_list
        }
        can_id, mask = struct.unpack("=II", filters[socket.CAN_RAW_FILTER])
        # Linux routes masks containing CAN_ERR_FLAG to the error-only list.
        self.assertFalse(mask & socket.CAN_ERR_FLAG)
        self.assertTrue(mask & socket.CAN_EFF_FLAG)
        self.assertTrue(mask & socket.CAN_RTR_FLAG)
        self.assertEqual(mask & socket.CAN_EFF_MASK, socket.CAN_EFF_MASK)
        reply_id = probe.FRAME.unpack(REPLY)[0]
        self.assertEqual(reply_id & mask, can_id & mask)
        self.assertEqual(struct.unpack("=I", filters[socket.CAN_RAW_ERR_FILTER])[0],
                         socket.CAN_ERR_MASK)
        bus.send.assert_called_once_with(REQUEST)
        self.assertEqual(json.loads(output.getvalue())["status"], "ok")

    def test_dry_run_does_not_access_CAN(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["probe", "--dry-run"]), \
             patch.object(probe, "link_state") as get_link, \
             patch.object(probe.socket, "socket") as open_socket, \
             contextlib.redirect_stdout(output):
            code = probe.main()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["frames_sent"], 0)
        get_link.assert_not_called()
        open_socket.assert_not_called()


if __name__ == "__main__":
    unittest.main()
