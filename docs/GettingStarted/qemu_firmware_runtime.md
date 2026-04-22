# QEMU Firmware Runtime Validation

This workflow validates an ARM32 firmware patch end to end:

1. Build a minimal bare-metal firmware for QEMU `lm3s6965evb`.
2. Decompile one target function with the repository Ghidra headless flow.
3. Generate the patch with Patchestry from the normal YAML/C patch inputs.
4. Lower the patched function to LLVM IR and compile it back to ARM32 code.
5. Use `patcherex2` to write the replacement blob and detour into the original ELF.
6. Boot the rewritten original ELF in QEMU and assert the serial transcript.

The implementation lives in:

- `firmwares/qemu-serial/`
- `scripts/test-qemu-firmware-runtime.sh`
- `scripts/patch-runtime/qemu_firmware_runtime.py`

## Requirements

- Docker with the Patchestry Ghidra image path working
- `/opt/homebrew/bin/qemu-system-arm` or `QEMU_SYSTEM_ARM` set to a working `qemu-system-arm`
- `uv`
- a Python 3.11 interpreter
- the patched LLVM toolchain used by this repository
- an ELF-capable `ld.lld`

On macOS, the script bootstraps a local `libkeystone.dylib` for `patcherex2` automatically under `builds/qemu-firmware-runtime/toolchain/keystone/`.

## Run

```sh
scripts/test-qemu-firmware-runtime.sh --build-type Debug
```

Useful environment overrides:

```sh
PATCHESTRY_LLVM_PREFIX=/path/to/llvm/bin
PATCHESTRY_LD_LLD=/path/to/ld.lld
PATCHESTRY_PYTHON=/path/to/python3.11
QEMU_SYSTEM_ARM=/path/to/qemu-system-arm
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
