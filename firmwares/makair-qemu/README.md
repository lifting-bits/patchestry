# MakAir QEMU re-host

QEMU-based emulation of the [MakAir mechanical ventilator firmware][upstream]
on `qemu-system-arm -M mps2-an386`, sharing the proven skeleton from
`firmwares/secpump-qemu/`. The upstream binary normally targets an STM32
Nucleo F411RE; this re-host strips the Arduino-STM32 framework and routes
the upstream Telemetry layer through a QEMU UART so the same wire format
the device sends to its Raspberry Pi UI is observable from a host pexpect
harness.

[upstream]: https://github.com/makers-for-life/makair-firmware

The re-host is **partial by design** — see "v0 scope" below — but the
plumbing it lands (Arduino HAL shim, CRC32 shim, telemetry codec, smoke +
protocol tests, patchestry demo script) is ready for future iterations to
fill in.

## Layout

| Path | Role |
|---|---|
| `Makefile` | `arm-none-eabi-gcc/g++` cross-compile, links shim TUs + curated upstream `.cpp` files pulled live from `firmwares/repos/makair-firmware/`. |
| `mps2-an386.ld` | Linker script (copied from `firmwares/secpump-qemu/`). |
| `Dockerfile`, `build-docker.sh` | `makair-builder` image (`gcc-arm-none-eabi` + `libstdc++-arm-none-eabi-newlib`). Mounts the upstream clone read-only. |
| `run.sh` | `qemu-system-arm -M mps2-an386 -serial mon:stdio -semihosting`. Modes: `run`/`smoke`/`test`/`debug`/`patched`/`patched-test`. |
| `inc/Arduino.h` | Hollow Arduino API: `pinMode`/`digitalRead`/`analogRead` no-ops, `millis()`/`micros()` from SysTick, `HardwareSerial`, `HardwareTimer` stubs, `LL_GetUID_Word*`. |
| `inc/CRC32.h` | Arduino CRC32 library shim — IEEE 802.3 CRC, bit-by-bit. |
| `src/uart.c` | CMSDK APB UART0 driver + newlib syscalls. |
| `src/arduino_shim.cpp` | Implements the `inc/Arduino.h` API on top of `uart.c`. |
| `src/main.cpp` | QEMU re-host entry: boot, `initTelemetry()`, `sendBootMessage()`, polling loop emitting `sendDataSnapshot` and parsing Heartbeat Control frames. |
| `src/startup.c` | ARMv7-M reset + vector table; SysTick configured for a nominal 1 ms tick. |
| `tests/_makair_codec.py` | Encoder/decoder for Telemetry + Control framing — single source of truth for wire layout. |
| `tests/test_smoke.py` | Boot + first BootMessage + first DataSnapshot, under 5 s, exits 0. |
| `tests/test_protocol.py` | v0 protocol surface: boot, streaming liveness, Heartbeat round-trip, bad-CRC drop. |
| `tests/test_patched_protocol.py` | Same assertions vs `build/makair-patched.elf`. |
| `scripts/demo_makair.sh` | Patchestry pipeline (decomp → CIR → patch → lower → KLEE → Patcherex2). |

## Prerequisites

```sh
# 1. Clone upstream into firmwares/repos/makair-firmware/
firmwares/build.sh    # also builds bloodlight, ventilator, secpump-qemu

# 2. Build this re-host (Docker — works on macOS Apple Silicon, Intel, and Linux)
firmwares/makair-qemu/build-docker.sh

# 3. Run
firmwares/makair-qemu/run.sh smoke         # ~3 s
firmwares/makair-qemu/run.sh test          # v0 protocol surface
```

## Wire format

UART carries MakAir's binary frames bit-identical to upstream:

```
Telemetry (firmware -> host):  03 0C  <type-tag>  <payload>  <crc32:4 BE>  30 C0
Control   (host -> firmware):  05 0A  <setting:1>  <value:2 BE>  <crc32:4 BE>  50 A0
```

CRC is IEEE 802.3 CRC-32 (poly `0xEDB88320`). The codec module under
`tests/_makair_codec.py` encodes/decodes both directions — if upstream
changes framing in a future MakAir release, only that file needs to follow.

## v1 scope (what runs today)

**Telemetry-out (firmware → host):**

- ✅ `BootMessage` — emitted on boot
- ✅ `DataSnapshot` — periodic (cadence depends on QEMU SysTick / wall clock; see Sharp edges)
- ✅ Sender bodies for the remaining types (`MachineStateSnapshot`,
  `AlarmTrap`, `EolSnapshot`, `ControlAck`, `FatalError`, `StoppedMessage`)
  compile in via the upstream `telemetry.cpp` and are reachable from the
  upstream `serial_control.cpp` dispatch — adding them to the polling
  loop in `src/main.cpp` is now a one-liner per message.

**Control-in (host → firmware):**

- ✅ Upstream `srcs/serial_control.cpp` is in the link. All 30+ settings
  (`VentilationMode`, `PEEP`, `PIP`, `CyclesPerMinute`, trigger params,
  alarm thresholds, RPi heartbeat, …) parse + dispatch through the real
  `mainController` / `alarmController` / `activationController` / `eolTest`
  global instances. The controller bodies (upstream `main_controller.cpp`,
  `alarm_controller.cpp`, `activation.cpp`) are linked verbatim; only the
  hardware-touching layers (HardwareTimer, blower, valve PWM, screen,
  EEPROM) are stubbed via `inc/Arduino.h` + `src/arduino_shim.cpp` +
  `src/controller_stubs.cpp`.
- ✅ Heartbeat fast path remains in `src/main.cpp::serial_control_loop_v0`
  for the smoke-test scenario where the heavier upstream parser is
  unavailable.

**Negative paths:**

- ✅ Bad-CRC Heartbeat → silent drop, stream stays alive
- ✅ Out-of-range setting → upstream parser path delivers the appropriate
  `ControlAck` with the error code
- ⏸ Heartbeat timeout → `AlarmTrap` "host disconnected" — needs the
  watchdog cycle that the QEMU loop hasn't wired yet (rpi_watchdog.cpp
  is in the link; just needs a periodic `update()` call)

## Patch + verify (placeholder)

```sh
firmwares/makair-qemu/scripts/demo_makair.sh --stage=lower --skip-klee
# decomp -> CIR -> placeholder pass-through patch -> LLVM IR

firmwares/makair-qemu/scripts/demo_makair.sh --stage=all --with-patcherex
# adds: KLEE harness, KLEE run, Patcherex2 binary patch -> build/makair-patched.elf

firmwares/makair-qemu/run.sh patched-test   # protocol regression vs patched ELF
```

The default target function is `sendBootMessage` (override via
`TARGET_FUNCTION=...`) and the patch is currently a no-op pass-through —
it exercises the pipeline wiring without claiming to be a useful security
patch. Concrete patch shapes (Control-frame CRC bounds-check, alarm-trap
dispatcher hardening) land in a follow-up.

## Sharp edges

- **No real-time interrupts.** The upstream `respirator.cpp::loop()` is empty
  by design — all the work happens inside HardwareTimer interrupts driving
  the breathing cycle. We aren't running those, so the patient circuit
  state machine, blower control, valve PWM, and tidal-volume math are dark.
  The sustaining v0 emits *synthetic* DataSnapshot values from `src/main.cpp`.
- **SysTick wall-clock cadence.** `mps2-an386` under QEMU free-running mode
  doesn't honour the firmware's nominal 25 MHz / 1 kHz tick at host wall
  rate. DataSnapshot streaming is alive but its rate depends on host load.
  A future iteration could `-icount auto -rtc clock=vm` if 1 Hz wall-rate
  fidelity is needed.
- **`serial_control.cpp` not in the link.** The upstream Control-frame
  parser pulls in 30+ method calls on `mainController` / friends. v0
  works around this with a 50-line in-tree parser in `src/main.cpp` that
  validates the frame and treats Heartbeat as the only meaningful setting.
  Promoting to the upstream parser is a follow-up that requires landing
  stub class definitions for the controller globals.
- **No EEPROM / no buzzer / no LCD.** The actuator-side shims in
  `src/arduino_shim.cpp` are no-ops; nothing in the shim records actuator
  state for inspection.
- **No watchdog reset.** `IWatchdog` is not shimmed; if upstream code
  expected a watchdog tickle, the omission is silent.

## Maintenance

- **Pinned commit**: `firmwares/build.sh:MAKAIR_COMMIT` (currently
  `72d454dcf799b7a33e85f00aef7a2fea4ca5c547` = upstream tag `v4.1.0`).
  Bumping this tag may require updates to the upstream-source list in
  `Makefile:UPSTREAM_CXX_SRCS` and the codec offsets in
  `tests/_makair_codec.py`.
- **License**: AGPL-3.0, inherited from upstream MakAir.
