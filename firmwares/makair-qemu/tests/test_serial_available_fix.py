#!/usr/bin/env python3
"""Regression test for the HardwareSerial::available() stash-clobber bug.

Drives a CRC-valid Heartbeat into the firmware over UART and asserts the
upstream Control parser actually reaches its Heartbeat case. The
unpatched firmware fails this test because available() always returns
at most 1 (it clobbers the previously-stashed byte every call), so the
parser's `Serial6.available() >= 11` precondition is never satisfied
and Control bytes are silently consumed without ever being parsed.

The signal we observe is the upstream parser's DBG_DO trace from
firmwares/repos/makair-firmware/srcs/serial_control.cpp:106-112, which
prints "Serial control message: setting = N, value = V" on a successful
parse. That trace is gated on -DDEBUG=1 — see README's Vulnerability
demo section for the build flag and how the patched ELF is produced.

Pass criteria:
  - Pre-patch  (build/makair.elf):         no parser trace -> FAIL
  - Post-patch (build/makair-patched.elf): parser trace -> PASS

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

PARSER_TRACE = b"Serial control message: setting = 0"


def _run_with_heartbeat(elf: str, wall_seconds: float) -> bytes:
    """Boot ELF, send a single CRC-valid Heartbeat, capture UART for the
    requested window, return raw stdout bytes."""
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
            print("  Run 'make' (with EXTRA_CXXFLAGS=-DDEBUG=1) first.", file=sys.stderr)
        return 2

    out = _run_with_heartbeat(elf, 1.5)

    saw_trace = PARSER_TRACE in out
    if use_patched:
        assert saw_trace, (
            f"FAIL: expected parser trace {PARSER_TRACE!r} in patched-ELF UART "
            f"output but it was absent — did the available() patch land?"
        )
        print(f"  patched ELF: parser trace observed -> Heartbeat parsed PASS")
    else:
        assert not saw_trace, (
            f"FAIL: parser trace {PARSER_TRACE!r} was unexpectedly present in "
            f"the pristine ELF — bug is no longer reproducing, fixture stale?"
        )
        print(f"  pristine ELF: parser trace absent -> bug reproduces PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
