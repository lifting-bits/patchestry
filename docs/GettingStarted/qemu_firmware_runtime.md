# QEMU Firmware Runtime Validation

This workflow validates an ARM32 firmware patch end to end:

1. Build a minimal bare-metal firmware for QEMU `lm3s6965evb`.
2. Load a checked-in decompile fixture for the target function.
3. Generate the patch with Patchestry from the normal YAML/C patch inputs.
4. Lower the patched function to LLVM IR and compile it back to ARM32 code.
5. Use `patcherex2` to write the replacement blob and detour into the original ELF.
6. Boot the rewritten original ELF in QEMU and assert the serial transcript.

The implementation lives in:

- `firmwares/qemu-serial/`
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
- `<case>/qemu.log`
- `summary.tsv`
- `summary.md`

The current runtime cases are:

- `before`
- `after`
- `replace`
- `contract`

Each case rewrites the original firmware ELF and validates the serial output from QEMU.

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
