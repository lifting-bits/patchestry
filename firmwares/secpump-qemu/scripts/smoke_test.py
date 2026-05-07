#!/usr/bin/env python3
"""Drive the SecPump QEMU re-host through a deterministic session:
  - boot → banner
  - five glucose samples through the PID auto controller
  - switch to manual mode + bolus
  - 7 V: writes (= 112 bytes) to trigger the seeded MaliciousMemCpy overflow

We run QEMU with -serial stdio so its UART0 is on QEMU's stdin/stdout, and we
talk to it with pexpect.
"""
import os, sys, time, pexpect

ELF = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "output", "secpump-qemu.elf"))
QEMU_CMD = (
    "qemu-system-arm -M mps2-an386 "
    "-display none -serial mon:stdio "
    "-semihosting -semihosting-config enable=on,target=native "
    f"-kernel {ELF}"
)

def main():
    child = pexpect.spawn(QEMU_CMD, encoding="utf-8", timeout=8)
    child.logfile_read = sys.stdout
    child.expect("secpump>")

    # Auto PID: feed 5 glucose samples, watch [G] / [u] echo
    for g in ("180", "200", "150", "120", "95"):
        child.sendline(f"G:{g}")
        child.expect(r"\[u\]:")
        child.expect("secpump>")

    # Switch to manual, set bolus, send a glucose sample (echoes BolusConfig)
    child.sendline("M:1");           child.expect("secpump>")
    child.sendline("B:2.5");         child.expect("secpump>")
    child.sendline("G:140");         child.expect(r"\[u\]:2\.5"); child.expect("secpump>")

    # Trigger the seeded overflow: 7 × 16 hex bytes = 112 → MaliciousMemCpy.
    # After the 7th call the function corrupts its own saved LR with 0x41414141
    # and the CPU faults on return; we expect the prompt to NOT come back.
    payload16 = "41" * 16        # 16 bytes of 'A'
    for i in range(6):
        child.sendline(f"V:{payload16}")
        child.expect(rf"Iterator: {(i+1)*16}")
        child.expect("secpump>", timeout=4)
    child.sendline(f"V:{payload16}")
    child.expect("Iterator: 112")
    child.expect("Buffer overflow:")
    print("\n--- exploit landed: MaliciousMemCpy clobbered LR with 0x41414141 ---")
    print("--- (firmware now diverges; killing QEMU) ---")
    try:
        # Either we hang (good — exploit took effect) or we get a fault dump.
        child.expect(["secpump>", pexpect.TIMEOUT, pexpect.EOF], timeout=2)
    except pexpect.exceptions.TIMEOUT:
        pass
    child.sendcontrol('a'); child.send('x')   # qemu Ctrl-A x = quit
    try:
        child.expect(pexpect.EOF, timeout=3)
    except pexpect.exceptions.TIMEOUT:
        child.terminate(force=True)
    print("--- session complete ---")

if __name__ == "__main__":
    main()
