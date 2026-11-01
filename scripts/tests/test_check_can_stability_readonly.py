import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_can_stability_readonly as stability


def healthy():
    return {"ifindex": 12, "parentdev": "usb-test", "flags": ["UP"],
            "linkinfo": {"info_data": {"state": "ERROR-ACTIVE", "bittiming": {"bitrate": 1000000}},
                         "info_xstats": {"error_warning": 0}}}


class StabilityTests(unittest.TestCase):
    def test_first_failed_read_stops_before_any_other_request(self):
        for status in ("timeout", "can_error", "response_with_can_errors", "error"):
            with self.subTest(status=status):
                records = []
                with patch.object(stability.probe, "read_once", return_value={"status": status}) as read:
                    with self.assertRaises(stability.StopMeasurement):
                        stability.measure(object(), records.append, 1, 10, healthy, healthy())
                read.assert_called_once()
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["motor_id"], 2)

    def test_error_counter_or_interface_change_prevents_send(self):
        states = [healthy(), healthy()]
        states[0]["linkinfo"]["info_xstats"]["error_warning"] = 1
        states[1]["ifindex"] = 13
        for state in states:
            with self.subTest(state=state), patch.object(stability.probe, "read_once") as read:
                with self.assertRaises(stability.StopMeasurement):
                    stability.measure(object(), lambda r: None, 1, 10, lambda: state, healthy())
                read.assert_not_called()

    def test_complete_run_only_requests_position_and_voltage(self):
        records = []
        result = {"status": "ok", "response_ms": 0.5}
        with patch.object(stability.probe, "read_once", side_effect=lambda *args: dict(result)) as read, \
                contextlib.redirect_stdout(io.StringIO()):
            report = stability.measure(object(), records.append, 2, 10, healthy, healthy(), sleep=lambda s: None)
        self.assertEqual(read.call_count, 28)
        self.assertEqual([r["motor_id"] for r in records[:14:2]], [2, 3, 4, 5, 6, 7, 1])
        self.assertEqual([r["motor_id"] for r in records[14:]], list(range(1, 8)) * 2)
        self.assertEqual({r["parameter"] for r in records}, {"0x7019", "0x701c"})
        self.assertEqual(report["rounds_completed"], 2)

    def test_dry_run_never_opens_CAN_or_creates_output(self):
        with patch.object(sys, "argv", ["probe", "--output-dir", "/unused", "--dry-run"]), \
                patch.object(stability.probe, "link_state") as link, \
                patch.object(stability.socket, "socket") as bus, \
                patch.object(Path, "mkdir") as mkdir, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stability.main(), 0)
        link.assert_not_called()
        bus.assert_not_called()
        mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
