#!/usr/bin/env python3
"""Smoke test for the makair-qemu re-host.

Boots the firmware in QEMU, captures the first ~3 s of UART output, and
asserts that:
  1. A valid Telemetry BootMessage frame arrives within the boot window.
  2. At least one DataSnapshot frame arrives within the same window.
  3. CRC32 over the framed body matches what the firmware emitted.

This is the fast CI gate — it must run in well under 5 s and exits non-zero
on any failure so it can be hooked into `make smoke` and the GitHub Actions
workflow without timing out.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _makair_codec import (
    iter_frames,
    decode_boot,
    decode_data_snapshot,
    TAG_BOOT,
    TAG_DATA_SNAPSHOT,
)

ELF = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "build", "makair.elf"))

QEMU_CMD = [
    "qemu-system-arm",
    "-M", "mps2-an386",
    "-display", "none",
    "-serial", "mon:stdio",
    "-semihosting", "-semihosting-config", "enable=on,target=native",
    "-kernel", ELF,
]


def main() -> int:
    if not os.path.isfile(ELF):
        print(f"makair.elf not found at {ELF}. Run 'make' first.", file=sys.stderr)
        return 2

    proc = subprocess.run(
        QEMU_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=3.0,
        check=False,
    ) if False else None  # no-op branch; subprocess.run is below.

    # Run with a 3 s timeout — the firmware emits BootMessage immediately
    # plus one DataSnapshot per second, so 3 s comfortably covers the bar.
    try:
        proc = subprocess.run(
            QEMU_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=3.0,
            check=False,
        )
        out = proc.stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout or b""

    # Strip the trailing QEMU "terminating on signal 15..." message — it
    # arrives on the same fd but isn't part of the firmware UART stream.
    qemu_msg = b"qemu-system-arm:"
    cut = out.find(qemu_msg)
    if cut >= 0:
        out = out[:cut]

    frames = list(iter_frames(out))
    if not frames:
        print("FAIL: no valid Telemetry frames in QEMU output", file=sys.stderr)
        sys.stderr.buffer.write(b"raw: " + out + b"\n")
        return 1

    boot = next((f for f in frames if f.tag == TAG_BOOT), None)
    snap = next((f for f in frames if f.tag == TAG_DATA_SNAPSHOT), None)

    if boot is None:
        print("FAIL: no BootMessage in first 3 s of UART output", file=sys.stderr)
        return 1
    if snap is None:
        print("FAIL: no DataSnapshot in first 3 s of UART output", file=sys.stderr)
        return 1

    bm = decode_boot(boot)
    print(f"BootMessage: proto={bm.protocol_version} fw={bm.fw_version!r} "
          f"device_id={bm.device_id!r} systick_us={bm.systick_us} "
          f"mode={bm.mode} value128={bm.value128}")
    print(f"DataSnapshot raw: {snap.payload[:16].hex()} ...")

    print(f"\n--- smoke OK ({len(frames)} frame(s) decoded) ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
