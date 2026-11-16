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
- `state`: inserted code with a writable variable of its own, which is expected to be rejected

Each case rewrites the original firmware ELF and validates the serial output from QEMU.

## Globals and Indirect Calls

`<case>/bindings.manifest.json` (from `patchir-decomp -emit-bindings`) lists every global and function the lifted unit refers to, and in `unresolved_calls` every indirect call and every tail call to an address that is not a known function. Nothing binds the target of those calls; `<case>/bindings.json` carries the list through so it can be reviewed.

Patcherex2 resolves a patch's external references from function symbols only. The runtime therefore passes it the address of each original global in the manifest, and compiles the patch with `-fno-pic` so the address is used directly and not through a GOT slot that a bare-metal image does not have. Without both, reinserted code that touches a global reads from an address the patch loader made up. The `dispatch` case covers this.

When an indirect call's arguments or used result disagree with the function type recovered for its target, `patchir-decomp` reports `callind-signature-mismatch` (an error under `-strict`).

## Inserted Dependencies

The patched LLVM IR module is reinserted one function at a time, each compiled and linked on its own. `scripts/patch-runtime/patch_dependencies.py` prepares that and checks it before the image is touched, and writes `<case>/dependencies.json`.

It splits the module with `llvm-extract` and `opt` into one snippet per function. A snippet defines its function, carries the constants the function reaches (a table brings the strings it points to) as local definitions, and declares the rest. Writable globals are declared and never copied, since a copy per snippet would give each function its own variable.

It then compiles each snippet with the flags the reinsertion uses and lists the symbols the object leaves undefined. Each must be a function of the original image, a global passed by address from the bindings manifest, or a function inserted earlier. The functions are inserted in dependency order, helpers before their callers, because the patch linker only knows what is already placed. A case fails with `unresolved-dependency` when:

- a symbol has no provider, for example a compiler runtime helper (`__aeabi_uldivmod` for 64-bit division) that the firmware was never linked with, or a writable global defined by the patch, which nothing places in RAM;
- two inserted functions reference each other, so no insertion order exists.

Without the check the patch linker gives such a symbol an invented address and the firmware fails at run time, if that path runs at all. `divide` and `state` are rejected this way and are not run in QEMU.

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
