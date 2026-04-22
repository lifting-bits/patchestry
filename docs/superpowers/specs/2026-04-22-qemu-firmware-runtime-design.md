# QEMU Firmware Runtime Bridge Design

## Goal

Add an opt-in runtime validation path that proves Patchestry can generate a patch
from its existing sub-function CIR patching model, apply that patch to the
original ARM32 firmware ELF using `patcherex2`, and run the rewritten original
firmware under QEMU with serial-output assertions.

This design intentionally does **not** add arbitrary instruction-granularity
binary rewriting as a first step. The v1 binary application boundary is
whole-function replacement of affected functions inside the original firmware
ELF.

## Why This Exists

The repository currently validates patch generation through these stages:

1. firmware or host binary build,
2. Ghidra decompilation to JSON,
3. JSON to CIR,
4. patch and contract application in CIR,
5. CIR lowering to LLVM IR or bitcode.

That validates Patchestry's IR-level patch generation, but it stops before the
most important production outcome: creating and applying a patch to the original
binary and verifying the patched binary's behavior at runtime.

The new runtime path closes that gap while keeping scope controlled.

## Non-Goals

- No instruction-level binary insertion or relocation synthesis beyond what
  `patcherex2` already supports.
- No requirement that every current `patchir-transform` matrix case become a
  runtime case in v1.
- No integration into the existing example-firmware or patch-matrix scripts.
- No Linux user-mode ELF execution in this change.
- No full firmware image rebuilding as the test oracle. QEMU must run the
  rewritten original ELF.

## Core Model

Patchestry continues to operate with sub-function semantics:

- `apply_before`
- `apply_after`
- `replace`
- runtime contract insertion
- operation-level or call-level matching

The new binary backend narrows the application boundary:

- Determine the enclosing function for each affected patch site.
- Regenerate the patched version of that enclosing function.
- Compile the patched function to ARM32 machine code.
- Use `patcherex2` to replace the original function in the original firmware
  ELF with the regenerated patched version.

This means:

- **Patch semantics** remain sub-function in Patchestry.
- **Binary insertion semantics** are whole-function replacement for each
  affected function.

This is the key v1 compromise that keeps the bridge achievable while still
testing the real product direction.

## End-to-End v1 Flow

```text
Original firmware ELF
  -> QEMU baseline run with serial capture
  -> Ghidra JSON export for target function(s)
  -> patchir-decomp
  -> patchir-transform with existing YAML/C patch specs
  -> identify affected enclosing function(s)
  -> lower patched function(s) to LLVM IR / bitcode
  -> compile patched function(s) to ARM32 replacement code
  -> patcherex2 replaces function(s) in original ELF
  -> QEMU run of rewritten original ELF
  -> serial-output assertions
```

The baseline and patched runs must both execute the same original firmware
entrypoint under QEMU. The only difference between them is whether the original
ELF has been rewritten with Patchestry-generated replacement functions.

## Firmware Fixture

### Board

Use a minimal ARM32 bare-metal firmware fixture that runs under:

```text
qemu-system-arm -M lm3s6965evb -nographic -kernel <firmware.elf>
```

The fixture should:

- live fully in this repository,
- have a minimal startup path,
- expose deterministic UART output,
- avoid external firmware repositories,
- be small enough that function replacement and growth constraints are obvious.

### Firmware Shape

The fixture should contain a handful of small named functions whose behavior is
easy to assert from serial output. The main loop should be deterministic and
short-lived so the harness can capture output and terminate without guessing.

Suggested structure:

- UART initialization and blocking character output helpers.
- A small dispatcher that calls target functions in a fixed order.
- Target functions that produce distinct baseline lines.
- One target function suitable for `before`/`after` style patching.
- One target function suitable for full replacement.
- One target function suitable for a visible runtime-contract path.

### Runtime Oracle

The test oracle is the serial transcript written by the firmware.

Examples of valid assertions:

- baseline output contains `BASE:alpha`
- patched output contains `PATCH:before alpha`
- patched output contains `PATCH:replace beta`
- patched contract case emits `CONTRACT:fail` or `CONTRACT:pass`

The runtime oracle must stay simple and explicit. Do not add debugger-driven
inspection or memory snapshots in v1.

## Patch Coverage

v1 should cover a useful subset of the existing patch surface:

1. A `before` patch
2. An `after` patch
3. A `replace` patch
4. One runtime contract-style visible behavior change

Important detail: these remain **sub-function patches in Patchestry**, but the
binary backend applies them by replacing the whole affected function in the
original ELF.

For example:

- a `before` patch that inserts a UART line before a matched call is represented
  in CIR as insertion before the matched site,
- then lowered to a regenerated version of the enclosing function,
- then applied by replacing that entire enclosing function in the original ELF.

## Bridge Requirements

This repository does not currently implement a `patchestry -> patcherex2`
handoff. That bridge is the core new interface.

### Required Bridge Inputs

For each runtime case, the bridge needs:

- the original firmware ELF,
- the function name selected for decompilation and patching,
- the patched CIR or patched LLVM derived from Patchestry,
- the identity of the enclosing original function being replaced,
- replacement ARM32 code for that function,
- any symbol metadata `patcherex2` needs to replace the function in the
  original ELF.

### Required Bridge Output

The bridge must produce:

- a rewritten version of the original firmware ELF,
- logs or metadata sufficient to diagnose which function was replaced,
- a stable artifact path for the runtime harness to execute.

### v1 Simplification

The bridge only needs to support **function replacement** in v1.

It does not need to support:

- replacing multiple arbitrary interior basic blocks,
- binary trampolines for every insertion site,
- in-place instruction patching at operation granularity,
- generalized relocation editing beyond what `patcherex2` already handles for
  function replacement.

## Tooling Strategy

### QEMU

Use host-installed `qemu-system-arm`.

The harness must fail fast with a clear message if it is missing.

### Python Tooling

Prefer `uv` for Python-managed tooling.

The design target is one of:

1. `uv tool install patcherex2`
2. a repo-local `uv` managed environment if the installed CLI or its
   dependencies require tighter control

The repository should not assume a globally managed ad hoc Python environment.

### ARM32 Compilation

Patchestry already lowers patched CIR to LLVM IR or bitcode. The missing step is
compiling the patched function into ARM32 replacement code suitable for binary
application.

The bridge should prefer a relocatable or function-scoped replacement artifact,
not a full rebuilt firmware image.

Open technical point for implementation:

- whether the cleanest v1 artifact is a relocatable object, a replacement ELF
  fragment, or a `patcherex2`-native patch description built from compiled ARM32
  code and symbol metadata.

The implementation plan must resolve this with a spike before broad automation.

## Repository Layout

Proposed new structure:

```text
firmwares/qemu-serial/
  startup.S
  linker.ld
  Makefile or CMakeLists.txt
  src/
    main.c
    uart.c
    targets.c

scripts/test-qemu-firmware-runtime.sh
scripts/patch-runtime/
  apply_function_patch.py        # if needed
  compile_replacement.py         # if needed

docs/GettingStarted/qemu_firmware_runtime.md
docs/superpowers/specs/2026-04-22-qemu-firmware-runtime-design.md
```

The runtime harness must remain separate from:

- `scripts/test-example-firmwares.sh`
- `scripts/test-patch-matrix.sh`

## Runtime Harness Behavior

The runtime harness should:

1. Build the baseline firmware ELF.
2. Run it under QEMU and save the serial transcript.
3. Decompile the runtime target function(s) with the existing Ghidra flow.
4. Generate patched CIR with the existing Patchestry flow.
5. Build the replacement function artifact for ARM32.
6. Apply it to the original firmware ELF using `patcherex2`.
7. Run the rewritten original ELF under QEMU.
8. Assert serial-output differences for the selected case.
9. Emit a summary report plus per-case logs and artifacts.

Suggested artifact layout:

```text
builds/qemu-firmware-runtime/
  <case>/
    original.elf
    baseline.log
    decompile.json
    patched.cir
    patched.ll
    replacement.<artifact>
    rewritten.elf
    patched.log
    apply.log
  summary.md
  summary.tsv
```

## Failure Handling

The runtime harness must fail fast and clearly for:

- missing `qemu-system-arm`,
- missing bare-metal cross compiler,
- missing `patcherex2`,
- failed decompilation,
- failed patch generation,
- failed function-compilation bridge,
- failed binary application,
- missing or incorrect serial transcript.

Errors must say which stage failed, for which case, and what local command the
developer should rerun.

## Documentation Changes

Add new documentation that explains:

- prerequisites,
- how to install `patcherex2` with `uv`,
- how to build the minimal QEMU firmware fixture,
- how to run the runtime harness,
- what artifacts are produced,
- the v1 limitation that binary application is whole-function replacement of
  affected functions.

Update existing docs only where they mention the downstream handoff boundary, so
the new runtime path is documented as an optional extended validation flow
rather than the existing default tested endpoint.

## CI Strategy

Keep v1 opt-in.

Possible rollout:

1. local-only script and docs,
2. optional CTest target,
3. dedicated CI job once toolchain and runtime stability are proven.

This should not block the existing matrix until the bridge is stable.

## Risks

### 1. Function Regeneration Fidelity

Some functions may not decompile and regenerate cleanly enough for direct
function replacement. v1 should therefore use firmware fixtures designed for
clean decompile/recompile behavior.

### 2. Replacement Artifact Shape

The repository does not yet define the exact artifact contract between patched
LLVM and `patcherex2`. A short spike is required before broad implementation.

### 3. Symbol Resolution and Calling Convention Drift

Whole-function replacement only works if the regenerated replacement preserves
the target ABI, symbol expectations, and codegen assumptions for the board.

### 4. Scope Creep

Trying to support instruction-level insertion in the first version will delay
the runtime validation path significantly. The implementation must keep the v1
limit explicit.

## Acceptance Criteria

This design is complete when an implementation can demonstrate all of the
following:

1. A minimal in-repo ARM32 UART firmware runs under `qemu-system-arm`.
2. The runtime harness captures a stable baseline serial transcript.
3. Patchestry generates at least one patched function from the existing
   decompile -> CIR -> transform flow.
4. The bridge compiles that patched function into an artifact that can be
   applied to the original ELF.
5. `patcherex2` rewrites the original ELF by replacing the affected function.
6. QEMU runs the rewritten original ELF successfully.
7. The patched transcript proves changed runtime behavior for `before`,
   `after`, `replace`, and one runtime-contract case.

## Implementation Notes For Planning

The implementation plan should be staged:

1. Minimal firmware and QEMU harness
2. Bridge spike from patched LLVM to replacement artifact
3. Single-case end-to-end function replacement
4. Expand to the four runtime cases
5. Documentation and opt-in automation

The first milestone should prove the bridge with one replacement case before
attempting the broader runtime matrix.
