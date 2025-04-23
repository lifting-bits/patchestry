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

This should produce the output json file, which can be used with tools like `pcode-lifter`.

## Convert it to JSON to CIR

The JSON (which encompasses Ghidra high-pcode) can then be converted to ClangIR via `pcode-lifter` as follows:
```sh
builds/default/tools/pcode-lifter/Release/pcode-lifter --input ~/temp/patchestry/pulseox-firmware.json --emit-cir --output ~/temp/patchestry/pulseox-firmware_cir --print-tu
```

The `--print-tu` argument is optional, it will emit C along with the ClangIR. The output looks like:
```sh
ls -1 ~/temp/patchestry/pulseox-firmware_cir*
/Users/artem/temp/patchestry/pulseox-firmware_cir.c
/Users/artem/temp/patchestry/pulseox-firmware_cir.cir
```
