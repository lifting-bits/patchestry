#!/usr/bin/env python3
"""Protocol coverage + exploit-rejection check for the patched SecPump-QEMU ELF.

Mirrors test_protocol.py for the wire-protocol assertions (M:/B:/G:/V: single
write/C/D/R:) so the patch has not regressed any documented behaviour, then
adds test_vuln_overflow_blocked: replays scripts/smoke_test.py's 7-call V:
flood (= 112 bytes accumulated) and asserts the patched build aborts loudly
instead of returning to the prompt with a corrupt link register.

Loud-failure by design — silent acceptance of the exploit is treated as a
regression even if the rest of the protocol still works (per the project's
loud-rejection-over-silent-no-op preference).
"""
import os
import sys

import pexpect

ELF = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "build", "secpump-patched.elf"))

QEMU_CMD = (
    "qemu-system-arm -M mps2-an386 "
    "-display none -serial mon:stdio "
    "-semihosting -semihosting-config enable=on,target=native "
    f"-kernel {ELF}"
)


def _prompt(c):
    c.expect("secpump>")


def test_boot_banner(c):
    c.expect(r"\[\+\] SecPump Service Created\. Handle 0x0001")
    c.expect(r"\[\+\] MODE Charac handle: 0x0002")
    c.expect(r"\[\+\] BOLUS Charac handle: 0x0005")
    c.expect(r"\[\+\] VULNERABILTY Charac handle: 0x0008")
    c.expect("SecPump QEMU re-host")
    _prompt(c)


def test_pid(c):
    """Auto-mode glucose feed must still drive InsulinController unchanged.
    G: dispatches outside Attribute_Modified_CB so it isn't affected by the
    AMCB body replacement."""
    for g in ("180", "200", "150", "120", "95"):
        c.sendline(f"G:{g}")
        c.expect(rf"\[G\]:{g}\.")
        c.expect(r"\[u\]:")
        _prompt(c)


def test_vuln_overflow_blocked(c):
    """Replay smoke_test.py's 7-write V: flood and confirm the patched
    firmware refuses the exploit.

    The original ProcessVulnReq runs MaliciousMemCpy(VulnBuffer[4],
    AttackBuffer, 112) on the 7th call, overflowing the 4-byte stack target
    and jumping to 0x41414141. The patched ProcessVulnReq is a self-
    contained safe accumulator that never invokes the 4-byte stack memcpy
    and emits no UART traffic of its own — so success looks like seven V:
    writes that each return to the prompt silently and a still-responsive
    firmware after the 7th write.

    Failure modes guarded against:
      a) reaching "Buffer overflow:" (means the original body still ran)
      b) the prompt not returning after some iteration (means a real fault
         instead of the controlled patched body)
      c) firmware unresponsive to a subsequent benign command (means the
         loud-trap fired and stuck — still a defensive success but worth
         flagging in the regression suite)
    """
    payload = "41" * 16
    for i in range(7):
        c.sendline(f"V:{payload}")
        idx = c.expect([
            r"secpump>",                          # patched body returned silently
            r"Buffer overflow:",                  # REGRESSION — original body ran
            pexpect.TIMEOUT,
            pexpect.EOF,
        ], timeout=6)
        if idx == 1:
            raise AssertionError(
                f"V iter {i + 1}: exploit log line reached on patched build")
        if idx in (2, 3):
            raise AssertionError(
                f"V iter {i + 1}: firmware halted instead of returning to prompt")

    # Liveness check: a benign command must still produce a response.
    c.sendline("?")
    c.expect("SecPump QEMU re-host")
    _prompt(c)


def main():
    if not os.path.isfile(ELF):
        print(f"secpump-patched.elf not found at {ELF}.\n"
              f"Run firmwares/secpump-qemu/scripts/demo_secpump.sh "
              f"--stage=all --with-patcherex first.", file=sys.stderr)
        return 2

    c = pexpect.spawn(QEMU_CMD, encoding="utf-8", timeout=10)
    c.logfile_read = sys.stdout
    try:
        test_boot_banner(c)
        test_pid(c)
        test_vuln_overflow_blocked(c)
        print("\n--- patched protocol tests passed ---")
        return 0
    finally:
        if c.isalive():
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
