# How To Run Patchestry on Firmware Examples

## Build the Ghidra docker image

First, make sure that the firwmare decompilation Ghidra docker image is set up correctly:
```sh
$ sh scripts/ghidra/build-headless-docker.sh
```

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
