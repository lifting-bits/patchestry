# Firmware Examples

These examples lift functions from firmware and related binaries, apply YAML
patch specifications, and produce patched CIR and LLVM IR. They exercise the
patching pipeline through IR generation. For binary reinsertion and execution,
see [QEMU firmware runtime](qemu_firmware_runtime.md).

Complete the [Patchestry setup](build.md) and start Docker before running the
commands below from the repository root.

## Run the examples

The [example runner](../../scripts/test-example-firmwares.sh) builds the
example binaries, prepares Ghidra, lifts the selected functions, applies their
patches, and lowers the results to LLVM IR.

```sh
scripts/test-example-firmwares.sh --build-type Debug
```

| Example | Function | Patch specification |
| --- | --- | --- |
| Pulse oximeter firmware | `measurement_update` | [measurement_update_before_patch.yaml](../../test/patchir-transform/measurement_update_before_patch.yaml) |
| Bloodlight firmware | `bl_usb__send_message` | [bl_usb__send_message_before_patch.yaml](../../test/patchir-transform/bl_usb__send_message_before_patch.yaml) |
| Bloodview application | `bl_device__process_entry` | [device_process_entry.yaml](../../test/patchir-transform/device_process_entry.yaml) |

The runner writes `summary.md` and `summary.tsv`, along with each case's JSON,
CIR, LLVM IR, and log. Use `--output-dir` to choose where to save them.

## Run the patch and contract matrix

The [matrix runner](../../scripts/test-patch-matrix.sh) exercises the broader
set of patch and contract specifications using cached firmware, Ghidra JSON,
and CIR fixtures.

```sh
scripts/test-patch-matrix.sh --build-type Debug
```

It reports each case's result and retains the generated IR and logs. Use
`--rebuild-firmware`, `--rebuild-ghidra`, or `--rebuild-fixtures` to refresh
inputs. Both runners accept `--help` for all options.

## Run the steps manually

With the Patchestry tools on `PATH` and the Ghidra image available, use the
following commands for a binary named `firmware.elf`, a function named
`measurement_update`, and a patch specification named `patch.yaml`:

```sh
scripts/ghidra/decompile-headless.sh \
  --input firmware.elf --function measurement_update --output function.json
patchir-decomp --input function.json --emit-cir --output function
patchir-yaml-parser patch.yaml --validate
patchir-transform function.cir --spec patch.yaml -o patched.cir
patchir-cir2llvm -S patched.cir -o patched.ll
```

Use `--print-tu` with `patchir-decomp` to also inspect the lifted C. See
[Patch specifications](patch_specifications.md) for authoring patches and
[Verifiable patching](../verifiable_patching.md) for contract verification
and binary validation.
