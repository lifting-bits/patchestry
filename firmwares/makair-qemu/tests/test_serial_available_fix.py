#!/usr/bin/env python3
"""Regression test for the HardwareSerial::available() stash-clobber bug.

Drives a CRC-valid Control frame into the firmware and asserts the in-tree
fast-path parser actually reaches its success handler. The unpatched
firmware fails this test because available() always returns at most 1
(it clobbers the previously-stashed byte every call), so the parser's
`Serial6.available() >= 11` precondition is never satisfied — Control
bytes are silently consumed without ever being parsed.

The signal we observe is a `sendControlAck(setting, value)` ack emitted
by the fast-path success handler in src/main.cpp. The host harness sends
SET_HEARTBEAT with value 0xFFFF, which produces a ControlAck whose
6-byte (\t setting \t value_hi value_lo \n) signature is distinct from
the firmware's per-tick synthetic sendControlAck(1, 0).

Pass criteria:
  - Pre-patch  (build/makair.elf):         no parsed-frame ack -> FAIL
  - Post-patch (build/makair-patched.elf): parsed-frame ack    -> PASS

Usage:
  python3 test_serial_available_fix.py [--patched]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _makair_codec import encode_control, SET_HEARTBEAT, DISABLE_RPI_WATCHDOG


HERE = os.path.dirname(os.path.abspath(__file__))
ELF_PRISTINE = os.path.normpath(os.path.join(HERE, "..", "build", "makair.elf"))
ELF_PATCHED  = os.path.normpath(os.path.join(HERE, "..", "build", "makair-patched.elf"))

# ControlAck "tail" signature: \t setting \t value_hi value_lo \n. The frame
# layout is "\t" then 1 byte setting then "\t" then 2 bytes BE value then
# "\n", per upstream srcs/telemetry.cpp::sendControlAck.
def _ack_tail(setting: int, value: int) -> bytes:
    return bytes([0x09, setting & 0xFF, 0x09,
                  (value >> 8) & 0xFF, value & 0xFF, 0x0A])

PARSED_ACK_TAIL    = _ack_tail(SET_HEARTBEAT, DISABLE_RPI_WATCHDOG)  # parsed-frame ack
SYNTHETIC_ACK_TAIL = _ack_tail(SET_HEARTBEAT, 0)                     # per-tick synthetic ack


def _run_with_heartbeat(elf: str, wall_seconds: float) -> bytes:
    payload = encode_control(SET_HEARTBEAT, DISABLE_RPI_WATCHDOG)
    proc = subprocess.Popen(
        [
            "qemu-system-arm",
            "-M", "mps2-an386",
            "-display", "none",
            "-serial", "mon:stdio",
            "-semihosting", "-semihosting-config", "enable=on,target=native",
            "-kernel", elf,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        proc.stdin.write(payload)
        proc.stdin.flush()
        time.sleep(wall_seconds)
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
    return out


def main() -> int:
    use_patched = "--patched" in sys.argv
    elf = ELF_PATCHED if use_patched else ELF_PRISTINE
    if not os.path.isfile(elf):
        print(f"ELF not found at {elf}.", file=sys.stderr)
        if use_patched:
            print("  Run scripts/demo_makair.sh --stage=all --with-patcherex first.",
                  file=sys.stderr)
        else:
            print("  Run firmwares/makair-qemu/build-docker.sh first.", file=sys.stderr)
        return 2

    out = _run_with_heartbeat(elf, 1.5)

    saw_parsed    = PARSED_ACK_TAIL in out
    saw_synthetic = SYNTHETIC_ACK_TAIL in out

    if use_patched:
        assert saw_parsed, (
            f"FAIL: expected parsed-frame ControlAck tail {PARSED_ACK_TAIL.hex()} "
            f"in patched-ELF UART output but it was absent — did the available() "
            f"patch land?"
        )
        print(f"  patched ELF: parsed-frame ControlAck observed -> Heartbeat parsed (PASS)")
    else:
        assert saw_synthetic, (
            f"FAIL: even the per-tick synthetic ControlAck "
            f"{SYNTHETIC_ACK_TAIL.hex()} is missing — fixture broken? "
            f"(test relies on src/main.cpp's per-tick sendControlAck(1, 0))"
        )
        assert not saw_parsed, (
            f"FAIL: parsed-frame ControlAck tail {PARSED_ACK_TAIL.hex()} was "
            f"unexpectedly present in the pristine ELF — bug is no longer "
            f"reproducing, fixture stale?"
        )
        print(f"  pristine ELF: parsed-frame ack absent (only synthetic ack seen) "
              f"-> bug reproduces (PASS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
