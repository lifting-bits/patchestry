# QEMU Firmware Runtime Bridge Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in ARM32 QEMU runtime validation flow that proves Patchestry can generate a sub-function patch, regenerate the affected enclosing function, apply that replacement to the original firmware ELF via `patcherex2`, and validate changed serial output from the rewritten original firmware.

**Architecture:** Keep Patchestry's current sub-function patch semantics in CIR, but introduce a new bridge that lowers patched enclosing functions to ARM32 replacement artifacts and applies them as whole-function replacements inside the original ELF. Use a minimal in-repo bare-metal UART firmware for deterministic QEMU testing, and keep the runtime harness separate from the existing example-firmware and patch-matrix runners.

**Tech Stack:** QEMU `qemu-system-arm`, bare-metal ARM32 C/assembly, linker script, existing Ghidra/`patchir-*` tools, `uv`-managed Python tooling, `patcherex2`, shell scripts, markdown docs.

---

## File Map

### New files expected

- `firmwares/qemu-serial/Makefile`
- `firmwares/qemu-serial/linker.ld`
- `firmwares/qemu-serial/startup.S`
- `firmwares/qemu-serial/src/main.c`
- `firmwares/qemu-serial/src/uart.c`
- `firmwares/qemu-serial/src/uart.h`
- `firmwares/qemu-serial/src/targets.c`
- `firmwares/qemu-serial/src/targets.h`
- `scripts/test-qemu-firmware-runtime.sh`
- `scripts/patch-runtime/bridge_patched_function.py`
- `scripts/patch-runtime/common.py`
- `docs/GettingStarted/qemu_firmware_runtime.md`
- `test/patchir-transform/qemu_serial_before_patch.yaml`
- `test/patchir-transform/qemu_serial_after_patch.yaml`
- `test/patchir-transform/qemu_serial_replace_patch.yaml`
- `test/patchir-transform/qemu_serial_entry_contract.yaml`
- `test/patchir-transform/patches/qemu_serial_patch_library.yaml`
- `test/patchir-transform/patches/qemu_serial_patch.c`
- `test/patchir-transform/contracts/qemu_serial_contract.c`

### Existing files likely to modify

- `docs/system_data_flow.md`
- `docs/GettingStarted/firmware_examples.md`
- `AGENTS.md`
- `test/CMakeLists.txt`
- `cmake/options.cmake`

### Runtime output locations

- `builds/qemu-firmware-runtime/`
- `builds/qemu-firmware-runtime/<case>/`

## Chunk 1: Minimal ARM32 Firmware Fixture And Baseline QEMU Harness

### Task 1: Create the minimal firmware layout

**Files:**
- Create: `firmwares/qemu-serial/Makefile`
- Create: `firmwares/qemu-serial/linker.ld`
- Create: `firmwares/qemu-serial/startup.S`
- Create: `firmwares/qemu-serial/src/main.c`
- Create: `firmwares/qemu-serial/src/uart.c`
- Create: `firmwares/qemu-serial/src/uart.h`
- Create: `firmwares/qemu-serial/src/targets.c`
- Create: `firmwares/qemu-serial/src/targets.h`

- [ ] **Step 1: Write a failing fixture smoke script**

Create a temporary command sketch in the commit message notes or local scratch and define the expected invocation:

```sh
qemu-system-arm -M lm3s6965evb -nographic -kernel firmwares/qemu-serial/build/qemu-serial.elf
```

Expected now: fail because the firmware sources and build recipe do not exist.

- [ ] **Step 2: Add the startup path and linker script**

Implement:

- vector table and reset handler in `startup.S`
- flash/RAM memory layout in `linker.ld`
- a `_start` / `Reset_Handler` path that reaches `main`

Keep the startup code minimal and board-specific only where needed for `lm3s6965evb`.

- [ ] **Step 3: Add UART helpers**

Implement blocking UART output helpers in:

- `firmwares/qemu-serial/src/uart.c`
- `firmwares/qemu-serial/src/uart.h`

Support:

- UART init
- write char
- write string
- write line

- [ ] **Step 4: Add deterministic target functions**

Implement:

- `main.c` as a fixed-order dispatcher
- `targets.c` / `targets.h` with named functions whose output is easy to assert

Baseline output should be deterministic and finite, for example:

```text
BOOT
BASE:before
BASE:after
BASE:replace
BASE:contract
DONE
```

- [ ] **Step 5: Add the bare-metal build**

Implement the `Makefile` so it emits:

- `firmwares/qemu-serial/build/qemu-serial.elf`
- optional map file and disassembly for debugging

Use an ARM bare-metal toolchain such as `arm-none-eabi-*`, and fail fast with a clear message if it is missing.

- [ ] **Step 6: Verify the firmware builds**

Run:

```sh
make -C firmwares/qemu-serial clean all
```

Expected:

- `qemu-serial.elf` exists
- build exits 0

- [ ] **Step 7: Verify QEMU baseline output**

Run:

```sh
/opt/homebrew/bin/qemu-system-arm -M lm3s6965evb -nographic -kernel firmwares/qemu-serial/build/qemu-serial.elf
```

Expected:

- deterministic serial output
- process exits cleanly or can be terminated after the expected transcript is captured

- [ ] **Step 8: Commit**

```bash
git add firmwares/qemu-serial
git commit -m "firmware: add qemu serial test fixture."
```

## Chunk 2: Patchestry Patch Specs For Function-Scoped Runtime Cases

### Task 2: Add runtime-oriented patch and contract specs

**Files:**
- Create: `test/patchir-transform/qemu_serial_before_patch.yaml`
- Create: `test/patchir-transform/qemu_serial_after_patch.yaml`
- Create: `test/patchir-transform/qemu_serial_replace_patch.yaml`
- Create: `test/patchir-transform/qemu_serial_entry_contract.yaml`
- Create: `test/patchir-transform/patches/qemu_serial_patch_library.yaml`
- Create: `test/patchir-transform/patches/qemu_serial_patch.c`
- Create: `test/patchir-transform/contracts/qemu_serial_contract.c`

- [ ] **Step 1: Write the failing YAML validation commands**

Run:

```sh
builds/default/tools/patchir-yaml-parser/Debug/patchir-yaml-parser test/patchir-transform/qemu_serial_before_patch.yaml --validate
```

Expected now: fail because the new specs do not exist.

- [ ] **Step 2: Add a patch library for the QEMU fixture**

Define a patch library with named patch functions that support:

- before-style behavior
- after-style behavior
- replacement behavior

Keep the target output simple and serial-visible.

- [ ] **Step 3: Add a runtime contract function**

Implement `qemu_serial_contract.c` to produce a visible runtime outcome, not just static metadata.

The first case should be easy to assert from serial output, such as:

- contract emits `CONTRACT:pass`
- contract emits `CONTRACT:fail`

- [ ] **Step 4: Add four patch specs**

Create:

- `qemu_serial_before_patch.yaml`
- `qemu_serial_after_patch.yaml`
- `qemu_serial_replace_patch.yaml`
- `qemu_serial_entry_contract.yaml`

Each spec should target a function in the new firmware fixture that decompiles and recompiles cleanly as a single enclosing function replacement.

- [ ] **Step 5: Validate the specs**

Run:

```sh
builds/default/tools/patchir-yaml-parser/Debug/patchir-yaml-parser test/patchir-transform/qemu_serial_before_patch.yaml --validate
builds/default/tools/patchir-yaml-parser/Debug/patchir-yaml-parser test/patchir-transform/qemu_serial_after_patch.yaml --validate
builds/default/tools/patchir-yaml-parser/Debug/patchir-yaml-parser test/patchir-transform/qemu_serial_replace_patch.yaml --validate
builds/default/tools/patchir-yaml-parser/Debug/patchir-yaml-parser test/patchir-transform/qemu_serial_entry_contract.yaml --validate
```

Expected: all four commands exit 0.

- [ ] **Step 6: Commit**

```bash
git add test/patchir-transform
git commit -m "test: add qemu runtime patch specs."
```

## Chunk 3: Bridge Spike From Patched CIR/LLVM To Rewritten Original ELF

### Task 3: Prove a single function-replacement bridge end to end

**Files:**
- Create: `scripts/patch-runtime/bridge_patched_function.py`
- Create: `scripts/patch-runtime/common.py`
- Modify: `firmwares/qemu-serial/Makefile` as needed for bridge artifacts

- [ ] **Step 1: Inspect `patcherex2` CLI and record the exact invocation shape**

Install or inspect with `uv`:

```sh
uv tool install patcherex2
uv tool run patcherex2 --help
```

Expected:

- `patcherex2` is available through `uv`
- the CLI shape for replacing a function is understood and captured in code comments or helper docs

- [ ] **Step 2: Write a failing bridge invocation for one replacement case**

Choose the simplest runtime case, preferably the full replacement case.

Define the intended steps:

1. decompile one target function from the original ELF
2. run `patchir-decomp`
3. run `patchir-transform`
4. lower to LLVM
5. compile patched enclosing function to ARM32 replacement artifact
6. apply with `patcherex2`

Run the bridge script before implementation.

Expected: fail because the bridge helper does not yet exist.

- [ ] **Step 3: Implement helper utilities**

Add shared logic in `scripts/patch-runtime/common.py` for:

- command execution
- path validation
- artifact path creation
- clear error messages

- [ ] **Step 4: Implement the function bridge script**

Add `bridge_patched_function.py` to:

- accept original ELF, target function, spec file, and output directory
- invoke decompilation and the existing `patchir-*` flow
- identify the enclosing function to replace
- compile the patched replacement function for ARM32
- invoke `patcherex2` to rewrite the original ELF

The script must emit:

- `rewritten.elf`
- `apply.log`
- intermediate artifacts for debugging

- [ ] **Step 5: Resolve the replacement artifact shape**

During implementation, choose the smallest working bridge contract:

- relocatable object
- replacement ELF fragment
- `patcherex2` patch description

Document the choice in code comments and the runtime doc.

- [ ] **Step 6: Verify one replacement case only**

Run something equivalent to:

```sh
uv tool run python scripts/patch-runtime/bridge_patched_function.py \
  --input-elf firmwares/qemu-serial/build/qemu-serial.elf \
  --function qemu_target_replace \
  --spec test/patchir-transform/qemu_serial_replace_patch.yaml \
  --output-dir builds/qemu-firmware-runtime/replace-spike
```

Expected:

- rewritten original ELF exists
- bridge exits 0

- [ ] **Step 7: Run QEMU on the rewritten original ELF**

Run:

```sh
/opt/homebrew/bin/qemu-system-arm -M lm3s6965evb -nographic -kernel builds/qemu-firmware-runtime/replace-spike/rewritten.elf
```

Expected:

- serial output reflects the replaced behavior
- this proves the bridge before the broader matrix exists

- [ ] **Step 8: Commit**

```bash
git add scripts/patch-runtime firmwares/qemu-serial/Makefile
git commit -m "bridge: prove patched function replacement."
```

## Chunk 4: End-To-End Runtime Harness For Four Cases

### Task 4: Add the opt-in runtime matrix runner

**Files:**
- Create: `scripts/test-qemu-firmware-runtime.sh`
- Modify: `scripts/patch-runtime/bridge_patched_function.py` as needed

- [ ] **Step 1: Write the failing harness invocation**

Run:

```sh
scripts/test-qemu-firmware-runtime.sh --build-type Debug
```

Expected now: fail because the harness does not exist.

- [ ] **Step 2: Implement baseline and per-case artifact layout**

The harness should create:

```text
builds/qemu-firmware-runtime/<case>/
  original.elf
  baseline.log
  decompile.json
  patched.cir
  patched.ll
  rewritten.elf
  patched.log
  apply.log
```

- [ ] **Step 3: Implement fail-fast dependency checks**

Check for:

- `qemu-system-arm`
- bare-metal cross compiler
- `uv`
- `patcherex2`
- required `patchir-*` tools

Each failure message must tell the user what is missing and how to reproduce locally.

- [ ] **Step 4: Add one function to run QEMU and capture the transcript**

Implement a small helper in the shell script that:

- runs QEMU
- captures serial output to a file
- enforces a timeout or deterministic exit strategy

- [ ] **Step 5: Add one function to run the bridge for a case**

For each case, invoke:

- baseline run
- bridge script
- patched run
- transcript assertions

- [ ] **Step 6: Add the four v1 runtime cases**

Cases:

- `before`
- `after`
- `replace`
- `entry_contract`

Each case should assert a specific changed transcript line, not a loose substring set.

- [ ] **Step 7: Emit summary reports**

Create:

- `builds/qemu-firmware-runtime/summary.md`
- `builds/qemu-firmware-runtime/summary.tsv`

Follow the style of the existing shell runners.

- [ ] **Step 8: Verify the full runtime harness**

Run:

```sh
scripts/test-qemu-firmware-runtime.sh --build-type Debug
```

Expected:

- all four cases pass
- reports are written

- [ ] **Step 9: Commit**

```bash
git add scripts/test-qemu-firmware-runtime.sh scripts/patch-runtime
git commit -m "test: add qemu firmware runtime runner."
```

## Chunk 5: Docs, Build Flags, And Optional Test Registration

### Task 5: Document the flow and wire opt-in automation

**Files:**
- Create: `docs/GettingStarted/qemu_firmware_runtime.md`
- Modify: `docs/system_data_flow.md`
- Modify: `docs/GettingStarted/firmware_examples.md`
- Modify: `AGENTS.md`
- Modify: `cmake/options.cmake`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write the new runtime guide**

Document:

- prerequisites
- `uv` installation path for `patcherex2`
- required cross compiler
- how to build the QEMU fixture
- how to run the runtime harness
- artifact layout
- the v1 limitation: whole-function replacement of affected functions

- [ ] **Step 2: Update system data flow**

Adjust `docs/system_data_flow.md` so it still states the core repo pipeline ends at LLVM by default, but now documents the optional runtime bridge path:

```text
patched LLVM -> replacement artifact -> patcherex2 -> rewritten original ELF -> QEMU runtime validation
```

- [ ] **Step 3: Update firmware docs**

Add a short section to `docs/GettingStarted/firmware_examples.md` pointing readers to the separate QEMU runtime flow instead of overloading the current example-firmware runner.

- [ ] **Step 4: Update contributor guidance**

Update `AGENTS.md` only where needed to keep the documented owned interfaces and test commands aligned with the new runtime flow.

- [ ] **Step 5: Add opt-in test registration**

If the repo uses an opt-in CTest target for this flow, add it behind an explicit option in:

- `cmake/options.cmake`
- `test/CMakeLists.txt`

Do not make it part of the default test path yet.

- [ ] **Step 6: Verify docs commands**

Run the exact commands documented in `docs/GettingStarted/qemu_firmware_runtime.md`.

Expected:

- commands are copy-paste valid
- paths and tool names match the implementation

- [ ] **Step 7: Commit**

```bash
git add docs AGENTS.md cmake/options.cmake test/CMakeLists.txt
git commit -m "docs: add qemu runtime validation guide."
```

## Chunk 6: Final Verification

### Task 6: Prove the full feature before claiming completion

**Files:**
- Verify artifacts in `builds/qemu-firmware-runtime/`

- [ ] **Step 1: Rebuild the firmware fixture from scratch**

Run:

```sh
make -C firmwares/qemu-serial clean all
```

Expected: clean rebuild succeeds.

- [ ] **Step 2: Re-run the full runtime harness**

Run:

```sh
scripts/test-qemu-firmware-runtime.sh --build-type Debug
```

Expected:

- `before`, `after`, `replace`, and `entry_contract` pass
- rewritten original ELFs are produced

- [ ] **Step 3: Inspect the summary report**

Verify:

- each case has a baseline transcript
- each case has a patched transcript
- each case records the rewritten ELF path

- [ ] **Step 4: Inspect git diff**

Run:

```sh
git diff --stat
git diff -- docs/system_data_flow.md docs/GettingStarted/qemu_firmware_runtime.md scripts/test-qemu-firmware-runtime.sh
```

Expected:

- only the intended runtime bridge and docs changes are present

- [ ] **Step 5: Record final evidence**

Capture in the final change summary:

- firmware build command and result
- runtime harness command and result
- exact output directory

- [ ] **Step 6: Commit the final integration pass**

```bash
git add .
git commit -m "runtime: validate qemu firmware patch bridge."
```
