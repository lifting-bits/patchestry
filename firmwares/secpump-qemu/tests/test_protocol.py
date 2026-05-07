#!/usr/bin/env python3
"""End-to-end coverage of every wire-protocol command on a single QEMU boot.

Non-destructive — does NOT trigger the seeded MaliciousMemCpy overflow. The
exploit demo lives in scripts/smoke_test.py (it kills QEMU at the end and so
runs in its own process).

Tests share QEMU state:
  - test_pid runs in Auto mode (OPERATING_MODE == 0 at boot).
  - test_bolus and test_mode_switch flip OPERATING_MODE.
  - test_vuln_one_write advances static Attack_It from 0 to 16.
Reordering will break later assertions.
"""
import os
import sys

import pexpect

ELF = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "build", "secpump.elf"))
if not os.path.isfile(ELF):
    fallback = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "output", "secpump-qemu.elf"))
    if os.path.isfile(fallback):
        ELF = fallback

QEMU_CMD = f"qemu-system-arm -M mps2-an386 -nographic -kernel {ELF}"


def _prompt(c):
    c.expect("secpump>")


def test_boot_banner(c):
    """Boot prints upstream service-init lines, banner, and the prompt."""
    c.expect(r"\[\+\] SecPump Service Created\. Handle 0x0001")
    c.expect(r"\[\+\] MODE Charac handle: 0x0002")
    c.expect(r"\[\+\] BOLUS Charac handle: 0x0005")
    c.expect(r"\[\+\] VULNERABILTY Charac handle: 0x0008")
    c.expect(r"\[\*\] General Discoverable Mode\.")
    c.expect("SecPump QEMU re-host")
    _prompt(c)


def test_help(c):
    c.sendline("?")
    c.expect("SecPump QEMU re-host")
    _prompt(c)


def test_pid(c):
    """Auto-mode glucose feed exercises InsulinController and prints [G]/[u]."""
    for g in ("180", "200", "150", "120", "95"):
        c.sendline(f"G:{g}")
        c.expect(rf"\[G\]:{g}\.")
        c.expect(r"\[u\]:")
        _prompt(c)


def test_bolus(c):
    """Manual mode echoes BolusConfig directly as [u]."""
    c.sendline("M:1")
    c.expect(r"\[\*\] MODE request")
    c.expect(r"\[\*\] Switching to Manual MODE")
    _prompt(c)

    c.sendline("B:2.5")
    c.expect(r"\[\*\] BOLUS request")
    c.expect(r"\[\+\] BolusConfig set")
    _prompt(c)

    c.sendline("G:140")
    c.expect(r"\[G\]:140\.")
    c.expect(r"\[u\]:2\.5000")
    _prompt(c)


def test_mode_switch(c):
    """Auto/manual round-trip plus the malformed-mode error path."""
    c.sendline("M:0")
    c.expect(r"\[\*\] Switching to Auto MODE")
    _prompt(c)

    c.sendline("M:1")
    c.expect(r"\[\*\] Switching to Manual MODE")
    _prompt(c)

    c.sendline("M:7")
    c.expect(r"\[!\] ERROR MODE")
    _prompt(c)


def test_vuln_one_write(c):
    """Single V: write — exercises ProcessVulnReq dispatch without tripping
    the 7-write seeded overflow. Leaves Attack_It == 16 for the rest of run."""
    c.sendline("V:" + "00" * 16)
    c.expect(r"\[\*\] VULNERABILITY request")
    c.expect(r"Iterator: 16")
    _prompt(c)


def test_v_bad_hex(c):
    c.sendline("V:nothex")
    c.expect(r"\[!\] V: needs 32 hex chars")
    _prompt(c)


def test_gap_connect_disconnect(c):
    """C / C:<bdaddr> / D drive GAP_Connection*/Disconnection_Complete_CB.
    The address is printed in reverse byte order (PumpService.c:254-258)."""
    c.sendline("C")
    c.expect(r"Connected to device:FF-EE-DD-CC-BB-AA")
    _prompt(c)

    c.sendline("D")
    c.expect(r"Disconnected")
    _prompt(c)

    c.sendline("C:010203040506")
    c.expect(r"Connected to device:06-05-04-03-02-01")
    _prompt(c)

    c.sendline("D")
    c.expect(r"Disconnected")
    _prompt(c)


def test_read_permit(c):
    """R:M | R:B | R:V drive Read_Request_CB. The shim aci_gatt_allow_read is
    a no-op and produces no firmware output, so we assert only that the prompt
    returns. Send C first because Read_Request_CB short-circuits when
    connection_handle == 0."""
    c.sendline("C")
    c.expect(r"Connected to device:")
    _prompt(c)

    for tag in ("M", "B", "V"):
        c.sendline(f"R:{tag}")
        _prompt(c)

    c.sendline("R:Q")
    c.expect(r"\[!\] R: expected R:M, R:B, or R:V")
    _prompt(c)

    c.sendline("D")
    c.expect(r"Disconnected")
    _prompt(c)


def test_unknown_tag(c):
    c.sendline("Z:foo")
    c.expect(r"\[!\] unknown tag 'Z'")
    _prompt(c)


def test_bad_command(c):
    c.sendline("xx")
    c.expect(r"\[!\] bad command")
    _prompt(c)


def main():
    if not os.path.isfile(ELF):
        print(f"secpump.elf not found at {ELF}. Run `make` first.",
              file=sys.stderr)
        return 2

    c = pexpect.spawn(QEMU_CMD, encoding="utf-8", timeout=10)
    c.logfile_read = sys.stdout
    try:
        test_boot_banner(c)
        test_help(c)
        test_pid(c)
        test_bolus(c)
        test_mode_switch(c)
        test_vuln_one_write(c)
        test_v_bad_hex(c)
        test_gap_connect_disconnect(c)
        test_read_permit(c)
        test_unknown_tag(c)
        test_bad_command(c)
        print("\n--- all protocol tests passed ---")
        return 0
    finally:
        try:
            c.sendcontrol("a")
            c.send("x")
            c.expect(pexpect.EOF, timeout=3)
        except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF, OSError):
            pass
        if c.isalive():
            c.terminate(force=True)


if __name__ == "__main__":
    sys.exit(main())
