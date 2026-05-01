# How To Run Patchestry on Firmware Examples

## Automated end-to-end runner

The repository runner provides one command that:

1. builds the example firmware artifacts,
2. decompiles representative example functions to JSON,
3. converts JSON to CIR,
4. applies the in-repo example patch specs,
5. lowers the patched CIR to LLVM IR,
6. writes a report and per-case logs/artifacts.

```sh
scripts/test-example-firmwares.sh --build-type Debug
```

Artifacts and reports are written to:

```sh
builds/example-firmware-e2e/
```

The runner currently validates these repository-supported example cases:

- `pulseox_measurement_update`
- `bloodlight_usb_send_message`
- `bloodview_device_process_entry`

Generated reports:

- `builds/example-firmware-e2e/summary.md`
- `builds/example-firmware-e2e/summary.tsv`

The tested endpoint remains patched CIR and LLVM IR/bitcode, not a final
rewritten firmware binary.

## Cached patch/contract matrix runner

The matrix runner provides one command that:

1. reuses or rebuilds the example firmware artifacts,
2. reuses or rebuilds cached decompile JSON and base CIR fixtures,
3. validates the repository-supported patch and contract spec matrix,
4. lowers each patched CIR to LLVM IR,
5. writes a summary report plus per-case logs and artifacts.

```sh
scripts/test-patch-matrix.sh --build-type Debug
```

Artifacts and reports are written to:

```sh
builds/patch-matrix/
```

Fixture caches are written to:

```sh
builds/test-fixtures/
```

Firmware caches remain under:

```sh
firmwares/output/
```

By default the runner reuses any existing caches. Use `--rebuild-firmware`,
`--rebuild-ghidra`, `--rebuild-fixtures`, or `--clean` when you want to force
fresh inputs.

## Build the Ghidra docker image

First, make sure that the firwmare decompilation Ghidra docker image is set up correctly:
```sh
$ sh scripts/ghidra/build-headless-docker.sh
```

For the separate ARM32 QEMU runtime validation path, see `docs/GettingStarted/qemu_firmware_runtime.md`.

This should succeed in building `docker.io/trailofbits/patchestry-decompilation:latest`

## Build the Firmware

Use the firmware build script (which builds a Linux docker image) to build the firmware:
```sh
sh firmwares/build.sh
```

This should produce the following outputs:
```sh
ls -1 firmwares/output 
bloodlight-firmware.elf
pulseox-firmware.elf
```

## Decompile firmware to JSON blob

For each firmware blob you want to decompile, use the decompile-headless script to decompile it:

```sh
scripts/ghidra/decompile-headless.sh --input firmwares/output/bloodlight-firmware.elf --output ~/temp/patchestry/bloodlight-firmware.json 
```

This should produce the output JSON file, which can be consumed by `patchir-decomp`.

## Convert JSON to CIR

The JSON (which encompasses Ghidra high-pcode) can then be converted to CIR via
`patchir-decomp` as follows:
```sh
builds/default/tools/patchir-decomp/Debug/patchir-decomp \
  --input ~/temp/patchestry/pulseox-firmware.json \
  --emit-cir \
  --output ~/temp/patchestry/pulseox-firmware_cir \
  --print-tu
```

The `--print-tu` argument is optional; it emits C alongside the CIR. The output
looks like:
```sh
ls -1 ~/temp/patchestry/pulseox-firmware_cir*
/Users/artem/temp/patchestry/pulseox-firmware_cir.c
/Users/artem/temp/patchestry/pulseox-firmware_cir.cir
```

## Optional patching and lowering flow

Once you have CIR, the repository-supported patching flow is:

```sh
# Validate a YAML patch specification
builds/default/tools/patchir-yaml-parser/Debug/patchir-yaml-parser patch.yaml --validate

# Apply the patch spec to CIR
builds/default/tools/patchir-transform/Debug/patchir-transform \
  ~/temp/patchestry/pulseox-firmware_cir.cir \
  --spec patch.yaml \
  -o ~/temp/patchestry/pulseox-firmware_patched.cir

# Lower patched CIR to LLVM IR
builds/default/tools/patchir-cir2llvm/Debug/patchir-cir2llvm \
  -S \
  ~/temp/patchestry/pulseox-firmware_patched.cir \
  -o ~/temp/patchestry/pulseox-firmware_patched.ll
```

This repository's native tested endpoint is patched CIR and LLVM IR/bitcode.
Producing a final rewritten firmware binary is downstream of patchestry and
typically handled by external tooling.

## Opt-in automation via CTest

If you want this flow exposed through CTest, reconfigure with:

```sh
cmake --fresh --preset default \
  -DPE_ENABLE_EXAMPLE_FIRMWARE_E2E=ON \
  -DLLVM_EXTERNAL_LIT=$(which lit)
```

Then run:

```sh
ctest --preset debug -R example-firmware-e2e-tests --output-on-failure
```

This target is opt-in because it builds external example firmware repositories
and requires Docker-backed Ghidra decompilation.
