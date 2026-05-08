#!/usr/bin/env python3
"""Patched-protocol regression for the makair-qemu re-host.

Runs the same v0 protocol assertions as test_protocol.py but against
build/makair-patched.elf (produced by scripts/demo_makair.sh
--stage=all --with-patcherex). The regression bar is "the patch did not
break the wire" — same Telemetry round-trips, same Heartbeat dispatch,
same drop-on-bad-CRC.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_protocol import (
    test_boot,
    test_data_snapshot_streaming,
    test_heartbeat_accepted,
    test_heartbeat_bad_crc_dropped,
)
import test_protocol as _tp

ELF = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "build", "makair-patched.elf"))


def main() -> int:
    if not os.path.isfile(ELF):
        print(f"makair-patched.elf not found at {ELF}.\n"
              f"Run scripts/demo_makair.sh --stage=all --with-patcherex first.",
              file=sys.stderr)
        return 2

    # Re-point the shared test_protocol module at the patched ELF so its
    # _run_qemu_with_stdin helper boots makair-patched.elf for every test.
    _tp.ELF = ELF

    tests = [
        test_boot,
        test_data_snapshot_streaming,
        test_heartbeat_accepted,
        test_heartbeat_bad_crc_dropped,
    ]
    failed = 0
    for t in tests:
        print(f"== {t.__name__} (patched) ==")
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {e}", file=sys.stderr)
            failed += 1
        except Exception as e:                         # noqa: BLE001
            print(f"  ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1

    if failed:
        print(f"\n--- {failed} patched test(s) failed ---", file=sys.stderr)
        return 1
    print(f"\n--- patched protocol tests passed ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
