# QEMU Firmware Runtime Validation

This workflow validates an ARM32 firmware patch end to end:

1. Build a minimal bare-metal firmware for QEMU `lm3s6965evb`.
2. Load a checked-in decompile fixture for the target function.
3. Generate the patch with Patchestry from the normal YAML/C patch inputs.
4. Lower the patched function to LLVM IR.
5. Use `patcherex2.ModifyFunctionPatch` to compile the LLVM IR, place the
   patched function in flash, and emit a trampoline at the original
   function entry. Helper functions defined in the same module are inserted
   via `InsertFunctionPatch`. The wrapper installs `patcherex2` from the
   `trail-of-forks/Patcherex2@patche_support` branch (LLVM-IR patch support
   and ARM bare-metal trampoline fixes are not yet upstream).
6. Boot the rewritten original ELF in QEMU and assert the serial transcript.

The implementation lives in:

- `firmwares/serial-qemu/`
- `scripts/test-qemu-firmware-runtime.sh`
- `scripts/patch-runtime/qemu_firmware_runtime.py`

## Requirements

- `/opt/homebrew/bin/qemu-system-arm` or `QEMU_SYSTEM_ARM` set to a working `qemu-system-arm`
- `uv`
- a Python 3.11 interpreter
- the patched LLVM toolchain used by this repository
- an ELF-capable `ld.lld`

Docker is only required when you explicitly refresh the checked-in Ghidra JSON fixtures.

On macOS, the script bootstraps a local `libkeystone.dylib` for `patcherex2` automatically under `builds/qemu-firmware-runtime/toolchain/keystone/`.

## Copy-Paste Success Path

These commands assume:

- you are at the repository root
- your patched LLVM/ClangIR install is already available
- `qemu-system-arm`, `uv`, `cmake`, and `ninja` are installed

### Configure and Build

```sh
export LLVM_INSTALL_PREFIX=/path/to/your/llvm-install
export CC="${LLVM_INSTALL_PREFIX}/bin/clang"
export CXX="${LLVM_INSTALL_PREFIX}/bin/clang++"
export CMAKE_PREFIX_PATH="${LLVM_INSTALL_PREFIX}/lib/cmake/llvm;${LLVM_INSTALL_PREFIX}/lib/cmake/mlir;${LLVM_INSTALL_PREFIX}/lib/cmake/clang"

cmake --fresh --preset default -DLLVM_EXTERNAL_LIT="$(which lit)"
cmake --build --preset debug -j
```

### Run the QEMU Runtime Patching Test

On macOS:

```sh
export PATCHESTRY_LLVM_PREFIX="${LLVM_INSTALL_PREFIX}/bin"
export PATCHESTRY_LD_LLD="$(brew --prefix llvm@18)/bin/ld.lld"
export PATCHESTRY_PYTHON="$(command -v python3)"
export QEMU_SYSTEM_ARM="${QEMU_SYSTEM_ARM:-/opt/homebrew/bin/qemu-system-arm}"

scripts/test-qemu-firmware-runtime.sh --build-type Debug
```

On Linux:

```sh
export PATCHESTRY_LLVM_PREFIX="${LLVM_INSTALL_PREFIX}/bin"
export PATCHESTRY_LD_LLD="$(command -v ld.lld)"
export PATCHESTRY_PYTHON="$(command -v python3)"
export QEMU_SYSTEM_ARM="$(command -v qemu-system-arm)"

scripts/test-qemu-firmware-runtime.sh --build-type Debug
```

### Verify Success

```sh
cat builds/qemu-firmware-runtime/summary.tsv
```

Expected result:

```text
case	status	artifact
before	pass	.../builds/qemu-firmware-runtime/before/rewritten.elf
after	pass	.../builds/qemu-firmware-runtime/after/rewritten.elf
replace	pass	.../builds/qemu-firmware-runtime/replace/rewritten.elf
contract	pass	.../builds/qemu-firmware-runtime/contract/rewritten.elf
```

## Fixture Refresh Path

The default path uses checked-in Ghidra JSON fixtures and does not require Docker.
To regenerate those fixtures with the repository Ghidra headless flow:

```sh
scripts/ghidra/build-headless-docker.sh
PATCHESTRY_RUNTIME_REFRESH_GHIDRA=1 scripts/test-qemu-firmware-runtime.sh --build-type Debug
```

With `PATCHESTRY_RUNTIME_REFRESH_GHIDRA=1`, the script regenerates the JSON fixtures through the repository Ghidra headless flow and overwrites the fixture directory.

## Useful Overrides

```sh
PATCHESTRY_LLVM_PREFIX=/path/to/llvm/bin
PATCHESTRY_LD_LLD=/path/to/ld.lld
PATCHESTRY_PYTHON=/path/to/python3.11
QEMU_SYSTEM_ARM=/path/to/qemu-system-arm
PATCHESTRY_RUNTIME_FIXTURE_DIR=/path/to/json-fixtures
PATCHESTRY_RUNTIME_REFRESH_GHIDRA=1
```

## Outputs

The script writes results under `builds/qemu-firmware-runtime/`:

- `baseline.log`
- `<case>/rewritten.elf`
- `<case>/dependencies.json`
- `<case>/trampoline.json`
- `<case>/patch_record.json`
- `<case>/applied_bytes.json`
- `<case>/static.json`
- `<case>/binary_equivalence.json`
- `<case>/patch_record_check.json`
- `<case>/case.json`
- `<case>/validation.json`
- `<case>/qemu.log`
- `summary.tsv`
- `summary.md`

The current runtime cases are:

- `before`
- `after`
- `replace`
- `contract`
- `short`: a 4-byte target, which the entry branch fills exactly
- `tiny`: a 2-byte target, which is expected to be rejected
- `pointer`: a target reached only through a stored function address
- `handler`: the callee of an indirect call, installed in the `qemu_callback` slot
- `dispatch`: the caller, whose reinserted body reads `qemu_callback` and calls through it
- `switch`: inserted code with a `static` helper, a switch and six new string literals; the transcript shows every arm
- `divide`: inserted code that needs `__aeabi_uldivmod`, which is expected to be rejected
- `state`: inserted code with a zero-initialized writable variable of its own, which is expected to be rejected
- `initialized`: the same with an initial value of 3, which is expected to be rejected
- `length`: `qemu_target_length(buf, len)` gets a replacement body that returns an error value for a length over 16
- `forward`: the second call in `qemu_target_forward(value)` is guarded and skipped for a value over 32
- `send`: `qemu_target_send(byte)` gets an entry check that calls `qemu_reject` for a lower-case letter or above

Each case rewrites the original firmware ELF and validates the serial output from QEMU.

## Static Checks

`scripts/patch-runtime/check_static.py` reads the machine code the reinsertion added, from the rewritten image, and checks what it does to the state a function shares with its caller. It follows each inserted function from its entry along every direct branch and tracks the stack pointer, the callee-saved registers and the slots they are saved in. It does not trust the compiler that produced the code.

| Problem | Meaning |
| --- | --- |
| `callee-saved-clobbered` | at a return or tail call, one of r4-r11 or d8-d15 does not hold the value it had at entry |
| `unbalanced-stack` | at a return or tail call, the stack pointer is not where it was at entry |
| `return-address-lost` | the function returns to something other than the lr it was entered with |
| `code-size-over-budget` | the inserted functions together are larger than the code budget |
| `stack-over-budget` | their frames together are deeper than the stack budget |

A function the analysis cannot follow is `not-run` with a reason (`indirect-jump`, `untracked-stack-pointer`, `conditional-stack-change`, `inconsistent-stack`, `runs-past-end`, `undecodable`). It has not passed. A case names the reason it expects in `static_not_run`; the `switch` case does, because its helper's `switch` is compiled to a jump through a table. Any other outcome fails the case.

The report (`static.json`) also has the code size and the frame size of each inserted function, the same analysis of the function that was replaced, the growth of the frame, and `inlined_copies`: places in the original image that repeat at least half of the replaced function's body, which a reinsertion at the function's entry does not reach.

Budgets are compared only when given (`--max-code-bytes`, `--max-stack-bytes`). The patch specification has no place to declare them yet, so the runtime reports the measured sizes and the budget verdicts read `not-judged`. The stack figure is the sum of the inserted functions' own frames; frames of the image's functions that the new code calls are not included.

## Validation Corpus Targets

The last three cases patch targets whose behaviour crosses the function boundary, so that a patch can be judged by what a caller observes:

| Target | What crosses the boundary | Patch | How it rejects |
| --- | --- | --- | --- |
| `qemu_target_length` | two arguments, a read through a pointer, a write to the global `qemu_length_last`, a return value | `replace_definition` | early return of `QEMU_LENGTH_ERROR` |
| `qemu_target_forward` | two calls with arguments computed from its own | `replace` on the second call | the call is not made |
| `qemu_target_send` | two ordered writes to the UART data register between an interrupt disable and enable | `apply_at_entrypoint` | calls `qemu_reject`, which does not return |

Each patch only restricts the accepted inputs. `main` calls every target once with an input the patch accepts and once with one it rejects, and its unpatched callers print what they see (`LEN:ok`, `LEN:error`, `FWD:second:large`, ...), so each transcript shows the accepted input behaving as before and the rejected one being refused. In the `send` case the run ends at `REJECT`.

The fixtures for these three targets are written by hand, like the others; the lifted C matches the firmware source.

## Binary Comparison

After the transcript matches, `check_binary_equivalence.py` compares the patched function in `rewritten.elf` with the function in `serial.elf` by symbolic execution of the machine code of both, and writes `<case>/binary_equivalence.json`. `docs/patch_validation.md` describes the properties P1 to P4 and the verdicts. Every case that gets this far is compared. What a case tells the comparison is in its `Case` entry in `qemu_firmware_runtime.py`:

| Field | Meaning |
| --- | --- |
| `compare` | the inputs and the ways the patch rejects an input: `arg_buffers`, `reject_returns`, `reject_skips`, `reject_handlers`, `mmio_regions` and the other keys `check_binary_equivalence.py` accepts |
| `compare_differs` | the properties the patch is meant to change |

The comparison has to fail in exactly the properties in `compare_differs`, and be `proven` when there are none. Anything else fails the suite.

| Cases | Declared | Result |
| --- | --- | --- |
| `length` | a 4-byte buffer for the first argument, rejection by the return value `0xFFFFFFFF` | proven |
| `forward` | rejection by leaving out the call to `forward_second` | proven |
| `send` | rejection by a call to `qemu_reject`, the UART registers as a device range | proven, including the order of the device writes |
| `before`, `after`, `replace`, `contract`, `short`, `pointer`, `handler`, `dispatch`, `switch` | `compare_differs=("P4",)` | differs in P4 only |

The nine patches in the last row add a line to the transcript, so the calls the function makes change by design. For them the evidence gives `binary-comparison` the verdict `not-run`, with the reason that the patch is declared to change P4: a declared change is not a failure, and it is not a proof that the calls changed only in the intended way. A difference in any other property fails the case.

Without its declaration each of the three restricting patches fails the comparison (`forward` in P4, `send` in P4, `length` in P2 and P3), and the suite then exits 1.

## Globals and Indirect Calls

`<case>/bindings.manifest.json` (from `patchir-decomp -emit-bindings`) lists every global and function the lifted unit refers to, and in `unresolved_calls` every indirect call and every tail call to an address that is not a known function. Nothing binds the target of those calls; `<case>/bindings.json` carries the list through so it can be reviewed.

Patcherex2 resolves a patch's external references from function symbols only. The runtime therefore passes it the address of each original global in the manifest, and compiles the patch with `-fno-pic` so the address is used directly and not through a GOT slot that a bare-metal image does not have. Without both, reinserted code that touches a global reads from an address the patch loader made up. The `dispatch` case covers this.

When an indirect call's arguments or used result disagree with the function type recovered for its target, `patchir-decomp` reports `callind-signature-mismatch` (an error under `-strict`).

## Inserted Dependencies

The patched LLVM IR module is reinserted one function at a time, each compiled and linked on its own. `scripts/patch-runtime/patch_dependencies.py` prepares that and checks it before the image is touched, and writes `<case>/dependencies.json`.

It splits the module with `llvm-extract` and `opt` into one snippet per function. A snippet defines its function, carries the constants the function reaches (a table brings the strings it points to) as local definitions, and declares the rest. Writable globals are declared and never copied, since a copy per snippet would give each function its own variable.

It then compiles each snippet with the flags the reinsertion uses and lists the symbols the object leaves undefined. Each must be a function of the original image, a global passed by address from the bindings manifest, or a function inserted earlier. The functions are inserted in dependency order, helpers before their callers, because the patch linker only knows what is already placed. A case fails with `unresolved-dependency` when:

- a symbol has no provider, for example a compiler runtime helper (`__aeabi_uldivmod` for 64-bit division) that the firmware was never linked with;
- two inserted functions reference each other, so no insertion order exists.

Without the check the patch linker gives such a symbol an invented address and the firmware fails at run time, if that path runs at all. `divide` is rejected this way and is not run in QEMU.

A case fails with `writable-state` when the patched module defines a writable global, for example `static unsigned patch_retry_limit = 3;` in patch code. The patch link places code and read-only data only, and the runtime gives the rewriter no startup hook (`insert_points` is empty), so the variable would have no address in RAM, and an initialized one no copy of its value after reset. `dependencies.json` lists each such global under `writable_state` with its size, its section (`.data` or `.bss`) and whether it carries initial bytes. `state` and `initialized` are rejected this way. Patch code that needs state has to use a global the firmware already has.

Not checked: that a provided symbol has the type or size the snippet expects.

```sh
PATCHESTRY_LLVM_PREFIX=<llvm bin dir> python3 scripts/patch-runtime/patch_dependencies.py \
  --module builds/qemu-firmware-runtime/divide/qemu_target_replace.patched.ll \
  --elf firmwares/serial-qemu/build/serial.elf --compiler-flag=-fno-pic
```

Its unit tests (`scripts/patch-runtime/test_patch_dependencies.py`) run at the start of `scripts/test-qemu-firmware-runtime.sh`.

Patch code may use `static` helpers and `static` constants. They are imported with the patch function (`test/patchir-transform/serial_qemu_dependencies.json`).

## Reinsertion Check

After rewriting, `scripts/patch-runtime/check_trampoline.py` compares the original and rewritten ELF and writes `<case>/trampoline.json`. The rewriter either overwrites the target in place or puts a 4-byte Thumb branch at its entry and the new body in a new executable segment. The check reports which (`in-place` or `trampoline`) and fails the case on any of these:

| Problem | Meaning |
| --- | --- |
| `unchanged` | no byte of the target differs |
| `unknown-extent` | the target symbol has no size |
| `not-thumb` | the target is not a Thumb function, or the new entry is a `blx` |
| `span-outside-function` | a changed byte of the original image is not part of the target |
| `bad-destination` | the entry branch does not lead to executable Thumb code the rewriter added |
| `incoming-branch` | a direct branch in another function targets the inside of the target |
| `alternate-entry` | another function symbol starts inside the target, or its Ghidra record has several address ranges |

Indirect branches and stored pointers into the middle of a function are not checked.

A case can declare the problem it expects with `expect_reject`. It then passes only if the check reports that problem, QEMU is not run for it, and its `summary.tsv` status is `rejected:<problem>`. The `tiny` case is there because its transcript alone would pass: the branch overwrites the first instruction of the neighbouring `emit_tiny_base`, and the firmware still prints the expected lines.

The check can be run on its own:

```sh
python3 scripts/patch-runtime/check_trampoline.py \
  --elf firmwares/serial-qemu/build/serial.elf \
  --rewritten builds/qemu-firmware-runtime/replace/rewritten.elf \
  --function qemu_target_replace
```

Its unit tests (`scripts/patch-runtime/test_check_trampoline.py`) run at the start of `scripts/test-qemu-firmware-runtime.sh`.

## Patch Record

Patcherex2 writes no record of a patch and does not update the symbol table. The runtime watches each patch being applied and writes `<case>/patch_record.json` (`scripts/patch-runtime/patch_record.py`): the original and rewritten image by hash, the compiler and flags, and for every function in insertion order its address and size, the hash of the IR snippet it was compiled from, each write (file offset, address, size, hash) and the address every symbol it references had when it was linked. Writes outside every patch are listed too; the only ones expected are the ELF header and its header tables, which are rewritten on save.

The record is then checked against the rewritten image and against `trampoline.json`, which finds the new body independently by decoding the branch at the old entry. A case fails on any of these:

| Problem | Meaning |
| --- | --- |
| `image-mismatch` | the rewritten image is not the file the record was made for |
| `missing-target` | the record has no entry for the target function |
| `unmapped-write` | a write is outside the loaded segments |
| `write-mismatch` | the image does not hold the recorded bytes |
| `destination-mismatch` | the recorded placement or address of the target is not what the image's entry shows |
| `symbol-mismatch` | a referenced symbol had no address at link time, or an inserted function was linked at an address other than where it was placed |
| `unattributed-write` | bytes were written that belong to no recorded function and are not ELF structure |
| `entry-changed` | the entry point moved |
| `segment-changed` | a segment of the original image is no longer loaded at the same addresses with the same permissions |

The record says what was written and where. It does not say the bytes are the right code; that needs a comparison with a fresh compile of the snippets.

```sh
python3 scripts/patch-runtime/patch_record.py \
  --record builds/qemu-firmware-runtime/replace/patch_record.json \
  --rewritten builds/qemu-firmware-runtime/replace/rewritten.elf \
  --trampoline builds/qemu-firmware-runtime/replace/trampoline.json
```

Its unit tests (`scripts/patch-runtime/test_patch_record.py`) run at the start of `scripts/test-qemu-firmware-runtime.sh`.

## Applied Bytes

`scripts/patch-runtime/check_applied_bytes.py` checks that the rewritten image holds the intended replacement and nothing else, and writes `<case>/applied_bytes.json`. It takes the IR snippets again, compiles each with the recorded flags, links it with `ld.lld` at its recorded address against its recorded symbols, and compares the result with the image byte for byte. The rewriter gets its bytes another way (a relocatable link that a loader then relocates), so the two agree only if both applied the same relocations to the same object code. A case fails on any of these:

| Problem | Meaning |
| --- | --- |
| `snippet-mismatch` | a snippet is missing or is not the one the record was made from |
| `link-failed` | the snippet does not compile or link with the recorded symbols |
| `byte-mismatch` | the image differs from the expected bytes of a function |
| `entry-branch-mismatch` | the bytes at the old entry are not a Thumb `b.w` to the new body |
| `unrecorded-change` | a byte of the original image changed outside every recorded write |
| `unrecorded-code` | a non-zero byte was added outside every recorded write |

With the record check, every loaded byte of the rewritten image is accounted for: unchanged from the original, zero fill, or equal to code compiled from a known snippet. Flipping any single applied byte fails the check.

Both paths use the same compiler, so a compiler that mistranslates the IR is not detected here.

```sh
PATCHESTRY_LLVM_PREFIX=<llvm bin dir> python3 scripts/patch-runtime/check_applied_bytes.py \
  --record builds/qemu-firmware-runtime/replace/patch_record.json \
  --elf firmwares/serial-qemu/build/serial.elf \
  --rewritten builds/qemu-firmware-runtime/replace/rewritten.elf \
  --module builds/qemu-firmware-runtime/replace/qemu_target_replace.patched.ll
```

Its unit tests (`scripts/patch-runtime/test_check_applied_bytes.py`) run at the start of `scripts/test-qemu-firmware-runtime.sh`.

## Evidence

`scripts/patch-runtime/evidence.py CASE_DIR` writes `<case>/validation.json` from the reports in the case directory and nothing else. The suite runs it for every case, including one that stopped early.

It gives every check of the validation plan one of four verdicts:

| Verdict | Meaning |
| --- | --- |
| `proven` | the check decided its claim for everything the claim covers (every byte compared, every path analyzed), under the listed assumptions |
| `tested` | the claim held on the inputs that were run |
| `failed` | the check found a problem |
| `not-run` | the check did not run, could not decide, or does not exist yet |

A check whose report is absent is `not-run`. The verdict of a group and of the case is the weakest one in it (`failed`, then `not-run`, `tested`, `proven`). `groups` says what the checks established: `integrity` (bindings, dependencies, ABI, reinsertion, patch record, link addresses, applied bytes, rewritten bindings), `static` (calling convention, the two budgets), `runtime` (the transcript).

The file also has the bounds (the budgets and what was measured against them, the compiler and flags), every assumption with its name, source and statement, warnings (inlined copies), and each report it read with its SHA-256.

| Assumption | When it appears |
| --- | --- |
| `lift-faithful` | lifted code was used |
| `compiler-faithful` | bytes were compared against a second compilation |
| `extent-from-lifter`, `mode-declared` | the image has no symbol to confirm the function's extent or instruction set |
| `plt-stubs-from-fixture` | imports are reached through stub addresses the fixture names |
| `addresses-outside-image` | a name is an address no segment maps |
| `fixtures-hand-written` | the function JSON was written by hand |
| `fixtures-not-regenerated` | the function JSON is a checked-in file, not lifted in this run |
| `comparison-inputs`, `comparison-callees` | the rewritten function was compared with the original |
| `comparison-declared` | the case declares a rejection, or a property the patch is meant to change |
| `validation-block-absent` | the patch specification declares no intent |

The suite writes `case.json` beside the reports with what no report holds: the image and its digest, the patch, the fixture's origin, and the outcome of the run. It can call a run `tested`, `failed` or `not-run`, never `proven`. After each case the suite compares its own status with the evidence and fails on a disagreement.

`scripts/patch-runtime/collect_reports.py RUN_DIR... --output DIR` gathers the evidence and reports of whole runs into CSV tables (`cases`, `functions`, `checks`, `problems`, `assumptions`) and one JSON file.

In this suite the `abi` check does not run, so `integrity` is `not-run` for every case: the patch compiler targets `arm-linux-gnueabihf` while the firmware is a soft-float Cortex-M3 image, and nothing compares the two yet.

## Failure Inspection

If the script exits nonzero, inspect:

```sh
cat builds/qemu-firmware-runtime/summary.tsv
cat builds/qemu-firmware-runtime/baseline.log
cat builds/qemu-firmware-runtime/before/qemu.log
cat builds/qemu-firmware-runtime/after/qemu.log
cat builds/qemu-firmware-runtime/replace/qemu.log
cat builds/qemu-firmware-runtime/contract/qemu.log
```

The per-case directories also contain the intermediate patched artifacts used to rewrite the original ELF.
