# SecPump QEMU re-host (mps2-an386 / Cortex-M4)

A bare-metal port of the **application-level logic** of
[`r3glisss/SecPump`](https://github.com/r3glisss/SecPump) that runs unmodified
under `qemu-system-arm`, with no STM32 HAL and no BlueNRG-MS BLE chip.

The PID insulin controller and the deliberately-vulnerable `ProcessVulnReq` /
`MaliciousMemCpy` chain are taken **verbatim** from
`SecPump-Vuln/Src/InsulinController.c` and `PumpService.c`. Only the BLE
attribute-modify dispatch is replaced — with a small UART line parser.

License: **GPLv3** (inherited from SecPump).

## Build & run (native Linux)

```sh
sudo apt install gcc-arm-none-eabi qemu-system-arm python3-pexpect
make             # builds build/secpump.elf and build/secpump.bin
make run         # qemu-system-arm -M mps2-an386 -nographic -kernel ...
make smoke       # exploit demo (destructive — kills QEMU)
make test        # protocol coverage then exploit demo
```

QEMU's UART0 is wired to stdio (multiplexed with the QEMU monitor via
`-serial mon:stdio`). To exit interactive `make run`, use any of:

- type `Q` at the `secpump>` prompt — clean exit via ARM semihosting
  (preferred);
- press `Ctrl-A` then `c` to switch to the QEMU monitor, then `quit`;
- press `Ctrl-A` then `x` (the QEMU shortcut to terminate).

`make help` lists all build/run targets.

### Docker workflow (Linux, macOS Intel, macOS Apple Silicon)

The build needs GNU binutils features (`--gc-sections`, `-Map=`) that macOS
`ld64` doesn't speak, so the ELF is built inside a dedicated minimal Docker
image. QEMU runs natively on the host.

```sh
# macOS:
brew install qemu          # provides qemu-system-arm
pip3 install pexpect       # only needed for the test scripts

# Linux:
sudo apt install qemu-system-arm python3-pexpect

./build-docker.sh          # builds build/secpump.{elf,bin} inside Docker
./run.sh                   # qemu-system-arm -nographic, interactive
./run.sh smoke             # exploit demo
./run.sh test              # full protocol coverage + exploit demo
./run.sh debug             # qemu paused, gdb stub on :1234
```

`./build-docker.sh` builds the dedicated `secpump-builder` image on first run
(~150 MB, no `armhf` cross packages, builds natively on `linux/amd64` and
`linux/arm64` — Apple Silicon needs no `--platform` flag and no Rosetta).
`./build-docker.sh clean` invokes `make clean` inside the container. Pass
extra `make` arguments through, e.g. `./build-docker.sh -j8`.

The top-level orchestrator `firmwares/build.sh` uses the same image and
copies the artifacts to `firmwares/output/secpump-qemu.elf` and
`firmwares/output/secpump-qemu.bin`.

## Wire protocol

ASCII, line-oriented. One command per line, terminated by CR or LF.

| Command | Meaning                                       | Maps to upstream                |
|---------|-----------------------------------------------|---------------------------------|
| `M:0`   | switch to Auto PID mode                       | `ProcessModeReq("0")`           |
| `M:1`   | switch to Manual bolus mode                   | `ProcessModeReq("1")`           |
| `B:1.5` | set the manual `BolusConfig` (units of insulin) | `ProcessBolusReq("1.5")`     |
| `G:200` | inject a glucose sample (mg/dL)               | `InsulinController("200")`      |
| `V:<32 hex chars>` | write 16 bytes to vuln characteristic | `ProcessVulnReq(att_data)`      |
| `C` or `C:<12 hex>` | fake LE connection                       | `GAP_ConnectionComplete_CB`     |
| `D`     | fake LE disconnect                            | `GAP_DisconnectionComplete_CB`  |
| `R:M | R:B | R:V` | fake GATT read permit on a char     | `Read_Request_CB → aci_gatt_allow_read` |
| `Q`     | exit QEMU cleanly (semihosting `SYS_EXIT`)    | n/a (rehost-only)               |
| `?`     | reprint banner                                | n/a                             |

## Demonstrating the seeded overflow

`ProcessVulnReq` accumulates `att_data` into a 256-byte global, and after **7**
writes (16 × 7 = 112 bytes) calls `MaliciousMemCpy(VulnBuffer[4], AttackBuffer, 112)`.
On a Cortex-M4 build with `-Os -mfloat-abi=soft`, the saved LR sits within the
112-byte clobber range, so the function returns to controlled bytes.

```sh
python3 scripts/smoke_test.py     # drives a full session via pexpect
```

The script exercises the PID path and then sends 7 `V:41…41` lines. The 7th
prints `Buffer overflow:`; the firmware then diverges (returns to
`0x41414141`) and QEMU is killed. That divergence is the proof.

## End-to-end patchestry demo

`scripts/demo_secpump.sh` chains every patchestry stage against this firmware
to produce a patched ELF that refuses the seeded overflow. The two
vulnerabilities targeted are:

| Function | Location | Patch |
|---|---|---|
| `MaliciousMemCpy` | `src/PumpService.c:423` | Replace with a bounded variant (`patch__replace__MaliciousMemCpy`) that aborts loudly on `n > dest_cap`. |
| `ProcessVulnReq` | `src/PumpService.c:405` | Replace with a self-contained safe accumulator (`patch__replace__ProcessVulnReq`) that never reaches the 4-byte stack target. |

Both patches and the static contracts live alongside the existing patchestry
fixtures and are reused by both the demo script and the LIT regression suite:

- `test/patchir-transform/patches/patch_malicious_memcpy.c` (patch body)
- `test/patchir-transform/patches/patch_process_vuln_req.c` (patch body)
- `test/patchir-transform/patches/secpump_security_patches.yaml` (PatchLibrary)
- `test/patchir-transform/secpump_malicious_memcpy.yaml`, `secpump_process_vuln_req.yaml` (PatchSpecs)
- `test/patchir-klee-verifier/secpump_klee_patches.yaml`, `secpump_klee_spec.yaml` (KLEE contract spec)

### Pipeline stages

```
ELF -> Ghidra headless         -> P-Code JSON
JSON -> patchir-decomp          -> CIR
CIR  -> patchir-transform       -> patched CIR + contract metadata
CIR  -> patchir-cir2llvm        -> LLVM IR
LL   -> patchir-klee-verifier   -> KLEE harness
.bc  -> run-klee.sh             -> KLEE state dirs (PASS = zero *.err)
LL + ELF + Patcherex2           -> secpump-patched.elf
```

### Quick start

```sh
# Stop after KLEE verification (default; no Patcherex2 dependency):
make demo
# Same, explicit:
./scripts/demo_secpump.sh --stage=klee

# Full chain incl. ELF instrumentation (requires ../Patcherex2/patche_binary.sh):
./scripts/demo_secpump.sh --stage=all --with-patcherex
make test-patched

# Stage flags (cumulative): build < decomp < cir < patch < lower < klee < patche < verify
./scripts/demo_secpump.sh --stage=lower         # produces patched .ll, no KLEE
./scripts/demo_secpump.sh --stage=klee --skip-klee  # generate harness, skip KLEE invocation
```

### Stage-level unit tests (LIT)

The same patch/contract YAMLs feed six LIT fixtures so the existing
`check-patchestry` target gains coverage of each pipeline stage:

| Stage | Fixture |
|---|---|
| patchir-decomp | `test/patchir-decomp/secpump_malicious_memcpy.json`, `secpump_attribute_modified_cb.json` |
| patchir-transform | `test/patchir-transform/secpump_malicious_memcpy.json`, `secpump_attribute_modified_cb.json` |
| patchir-klee-verifier | `test/patchir-klee-verifier/secpump_malicious_memcpy.ll`, `secpump_e2e.json` |

Run them via:

```sh
lit ./builds/default/test -D BUILD_TYPE=Release --filter=secpump -v
```

### Sharp edges

- **`ProcessVulnReq` lift** — `patchir-decomp -emit-cir` currently asserts on
  ProcessVulnReq's specific CFG shape (post-commit `ec09a26`); the demo script
  and LIT tests route around this by lifting `Attribute_Modified_CB` (the
  caller) instead. The patch C body is self-contained, so the pipeline still
  exercises the full replace flow.
- **KLEE harness** — KLEE inside the `patchestry-klee-ubuntu-22.04-llvm-20`
  container has been observed to segfault in `RaiseAsmPass::runOnInstruction`
  on the secpump harness shape. The demo loud-fails when KLEE returns
  non-zero; pass `--skip-klee` to bypass and inspect
  `build/demo/klee_out/klee-last/messages.txt` if you want to repro.
- **Patcherex2** — binary patching is gated on `--with-patcherex` /
  `$PATCHEREX_BIN`, defaults to `../Patcherex2/patche_binary.sh`.

## Tests

Two pexpect-driven scripts cover the firmware. They share QEMU but run in
separate processes, so the destructive exploit demo doesn't affect the
non-destructive protocol walkthrough.

| Script | Purpose |
|---|---|
| `tests/test_protocol.py` | Walks every wire-protocol command (`?`, `M:0/1/7`, `B:`, `G:` in auto + manual, single `V:`, `V:nothex`, `C`, `C:<bdaddr>`, `D`, `R:M/B/V/Q`, unknown tag, missing colon) on a single QEMU boot. Non-destructive. |
| `scripts/smoke_test.py` | Exploit demo: PID round-trip, mode/bolus, then 7×`V:` writes that trigger the seeded `MaliciousMemCpy` overflow. Destructive — kills QEMU at the end. |

Run both:

```sh
./run.sh test     # tests/test_protocol.py then scripts/smoke_test.py
make test         # equivalent
./run.sh smoke    # exploit only
```

## What was changed vs. upstream

| Upstream component                    | Re-host treatment                                         |
|---------------------------------------|-----------------------------------------------------------|
| `Src/InsulinController.c`             | **Verbatim.** Only the `#include "InsulinController.h"` is followed by `<string.h>` to satisfy the rehost; PID code is byte-identical. |
| `Inc/InsulinController.h`             | `#include "main.h"` → `<stdint.h>/<stdio.h>/<stdlib.h>`. Function prototypes unchanged. |
| `Src/PumpService.c` (entire file: `Add_Pump_Service`, `setConnectable`, `user_notify`, `Attribute_Modified_CB`, `ProcessModeReq`, `ProcessBolusReq`, `ProcessVulnReq`, `MaliciousMemCpy`, GAP CBs) | **Verbatim** in `src/PumpService.c`; only the BlueNRG include chain is replaced by `#include "PumpService.h"` (which pulls in `inc/bluenrg_shim.h`). |
| `bluenrg_gatt_aci.h` / `bluenrg_gap_aci.h` / `hci.h` / `hci_le.h` etc. | **Stubbed** in `inc/bluenrg_shim.h` + `src/bluenrg_shim.c`: types/constants kept; `aci_gatt_*` / `aci_gap_*` / `hci_*` are no-ops that hand out sequential GATT handles. |
| BlueNRG SPI radio + HCI transport     | **Removed.** No qemu board models BlueNRG. Replaced by `shim_post_attr_modified()` which synthesizes an `EVT_BLUE_GATT_ATTRIBUTE_MODIFIED` event and hands it to the upstream `user_notify()` dispatcher — same call graph as a real GATT write. |
| `MX_USART2_UART_Init` (STM32 HAL)     | **Replaced** by CMSDK APB UART0 (`src/uart.c`). |
| Cube startup + `system_stm32f4xx.c`   | **Replaced** by `src/startup.c` + `mps2-an386.ld`. |

The BlueNRG-driven dispatch chain
`user_notify → Attribute_Modified_CB → ProcessModeReq | ProcessBolusReq | ProcessVulnReq → MaliciousMemCpy`
is preserved byte-for-byte from upstream. Each `M:` / `B:` / `V:` UART line
manufactures the equivalent GATT-write event into that dispatch path. Boot
prints (`SecPump Service Created. Handle 0x0001`, `MODE Charac handle: 0x0002`,
…) come from the upstream `printf`s in `Add_Pump_Service`. Glucose samples
(`G:`) bypass GATT and go directly to `InsulinController`, matching the
upstream UART2 IRQ path.

The bug semantics, bolus-authority logic, PID, and the GATT input-parsing
surface are unchanged.

## Layout

```
.
├── Dockerfile                 # secpump-builder image (Linux + macOS, native arm64)
├── Makefile                   # bare-metal Cortex-M4 build
├── build-docker.sh            # build via secpump-builder Docker image
├── run.sh                     # local qemu wrapper (run/smoke/test/debug)
├── mps2-an386.ld
├── inc/
│   ├── InsulinController.h
│   ├── PumpService.h          # upstream prototypes + extern handles for shim
│   └── bluenrg_shim.h         # BLE/HCI/GATT type & constant surface
├── src/
│   ├── InsulinController.c    # upstream, verbatim except headers
│   ├── PumpService.c          # upstream verbatim (Add_Pump_Service, user_notify,
│   │                          #   Attribute_Modified_CB, ProcessXxxReq, etc.)
│   ├── bluenrg_shim.c         # no-op aci_*/hci_* + shim_post_attr_modified
│   ├── main.c                 # UART → shim → upstream dispatch
│   ├── uart.c                 # CMSDK UART0 + newlib syscall stubs
│   └── startup.c              # ARMv7-M vector table + Reset_Handler
├── scripts/
│   └── smoke_test.py          # exploit demo (destructive)
└── tests/
    └── test_protocol.py       # full wire-protocol coverage (non-destructive)
```

## Notes for lifting / verification

This target is small (~38 KB .text) and HAL-free, so a P-Code → CIR → LLVM IR
lift produces clean output suitable for KLEE harnessing. Suggested
`!static_contract` annotation sites:

- `InsulinController` — pre: `T_iterator < SAMPLE_TIME` after modulo;
  post: `op[T_iterator] ∈ [OP_LO, OP_HI]` (anti-windup invariant)
- `ProcessBolusReq` — `BolusConfig ∈ [0, OP_HI]` (NOT enforced upstream;
  candidate for a patch verifier)
- `ProcessVulnReq` — `Attack_It + 16 ≤ sizeof(AttackBuffer)` (violated
  before the seeded reset; trivial KLEE counterexample at iter 112)
- `MaliciousMemCpy` — `n ≤ alloc_size(dest)` (the canonical patch is to
  pass and check a `dest_cap` parameter)

## Limitations

- **`-Os` and `-mfloat-abi=soft` are load-bearing for the seeded overflow.**
  The exploit relies on the saved LR sitting inside `MaliciousMemCpy`'s
  112-byte clobber window. `-O0`, `-O2`, `-O3`, or hard-float move it and
  the demo no longer lands. Override `OPTFLAGS` / `ARCHFLAGS` only after
  re-deriving the offsets.
- **`-Wl,-u,_printf_float` is required.** The build links newlib-nano
  (`--specs=nano.specs`); without `_printf_float` the float specifiers in
  `[G]:%.4f` and `[u]:%.4f` (`InsulinController.c:116,153,164,167`) emit
  garbage and PID assertions fail. Drop both `--specs=nano.specs` and
  `-Wl,-u,_printf_float` together if you'd rather pull full newlib (binary
  grows ~20 KB).
- **`tests/test_protocol.py` runs sequentially within a single QEMU boot.**
  Tests share state: `test_pid` and `test_bolus` mutate `OPERATING_MODE` /
  `BolusConfig`, and `test_vuln_one_write` advances static `Attack_It` to
  16. There is no firmware reset between cases — reordering can break
  later assertions.
- **`R:M | R:B | R:V` produces no firmware output.** `Read_Request_CB`
  calls `aci_gatt_allow_read`, which is a no-op shim. The protocol test
  asserts only that the next `secpump>` prompt returns; it cannot verify
  the read was processed beyond "did not fault".
- **`scripts/smoke_test.py` is destructive.** After the 7th `V:` write the
  firmware diverges to `0x41414141`, so the test cannot use the in-band
  `Q` quit and falls back to `Ctrl-A x` to terminate QEMU. Do not chain it
  with non-destructive checks in the same QEMU process; `./run.sh test`
  runs the protocol test first (clean `Q` exit) in a separate process and
  then the exploit demo.
- **`Q` requires `-semihosting`.** All QEMU invocations in this directory
  pass `-semihosting -semihosting-config enable=on,target=native`. If you
  invoke `qemu-system-arm` by hand without that flag, `Q` will trigger a
  HardFault instead of exiting.
- **Two Docker builder images coexist.** secpump uses the dedicated minimal
  `secpump-builder` image (this directory's `Dockerfile`); the other
  firmwares (pulseox, bloodlight, ventilator) still use the shared
  `firmware-builder` image (`firmwares/Dockerfile`). They are not
  interchangeable.
- **No CI / LIT integration yet.** Tests run locally via `./run.sh test`
  (or `make test`) but are not wired into `.github/workflows/ci.yml`.
