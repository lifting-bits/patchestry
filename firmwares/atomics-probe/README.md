# Atomics probes (ARM, AArch64, x86-64)

Minimal freestanding ELFs whose every public function executes a single
synchronization or LL/SC instruction. They exist so the patchir-decomp lit
suite can ground its atomic-intrinsic normalizer in real Ghidra-decompiled
output without depending on any particular vendor firmware.

One ELF per architecture:

| Source                          | Target                            | Triple used at build time   |
|---------------------------------|-----------------------------------|-----------------------------|
| `atomics_probe_cortexm4.c`      | `atomics_probe_cortexm4.elf`      | `arm-none-eabi-gcc`         |
| `atomics_probe_aarch64.c`       | `atomics_probe_aarch64.elf`       | `clang --target=aarch64-linux-gnu` |
| `atomics_probe_x86_64.c`        | `atomics_probe_x86_64.elf`        | `clang --target=x86_64-linux-gnu`  |

The ARM probe still uses the bundled `arm-none-eabi-gcc` newlib build
because that toolchain is already required for `firmwares/secpump-qemu`;
the AArch64 and x86-64 probes use clang+lld so the cross-build runs on any
host (including Apple Silicon) without an arch-specific gcc install.

## Build

```sh
# Native (requires arm-none-eabi-gcc, clang, and lld on PATH).
# Override CLANG=... if clang is not the first thing on PATH.
make all                        # cortexm4 + aarch64 + x86_64
make aarch64                    # one target at a time
make x86_64
make cortexm4

# Containerised (works on macOS, no host toolchain required for the ARM
# probe; the AArch64 and x86-64 recipes still need clang+lld on the host
# because the firmware-builder image carries gcc but not LLVM):
docker build -t secpump-builder ../secpump-qemu
docker run --rm -v "$(pwd):/work" secpump-builder -c 'cd /work && make cortexm4'
```

## Regenerate the lit fixtures

```sh
# ARM Cortex-M4
bash ../../scripts/ghidra/decompile-headless.sh \
    --input  "$(pwd)/atomics_probe_cortexm4.elf" \
    --output "$(pwd)/atomics_probe_cortexm4.json"
jq . atomics_probe_cortexm4.json \
    > ../../test/patchir-decomp/callother_atomic_arm_real_ghidra.json

# AArch64
bash ../../scripts/ghidra/decompile-headless.sh \
    --input  "$(pwd)/atomics_probe_aarch64.elf" \
    --output "$(pwd)/atomics_probe_aarch64.json"
jq . atomics_probe_aarch64.json \
    > ../../test/patchir-decomp/callother_atomic_aarch64_real_ghidra.json

# x86-64
bash ../../scripts/ghidra/decompile-headless.sh \
    --input  "$(pwd)/atomics_probe_x86_64.elf" \
    --output "$(pwd)/atomics_probe_x86_64.json"
jq . atomics_probe_x86_64.json \
    > ../../test/patchir-decomp/callother_atomic_x86_64_real_ghidra.json
```

After overwriting each JSON, re-apply the LIT `// RUN:` and `// CHECK-DAG:`
header that the fixture in `test/patchir-decomp/` carries above the JSON
body. The `jq` re-pretty-print keeps the diff stable across regenerations
when only Ghidra version drift is at play.

## Userops Ghidra 12.0.4 emits, per architecture

| Arch       | CALLOTHER intrinsic targets observed                                                  |
|------------|---------------------------------------------------------------------------------------|
| Cortex-M4  | DataMemoryBarrier, DataSynchronizationBarrier, InstructionSynchronizationBarrier, ClearExclusiveLocal, ExclusiveAccess, hasExclusiveAccess |
| AArch64    | DataMemoryBarrier, DataSynchronizationBarrier, InstructionSynchronizationBarrier, ClearExclusiveLocal (LDXR / LDAXR / LDAR lower to inline LOAD, no userop) |
| x86-64     | LOCK, UNLOCK only (MFENCE / LFENCE / SFENCE / PAUSE lower to inline P-Code, no userop) |

The normalizer in `lib/patchestry/AST/IntrinsicHandlers.cpp` canonicalizes
the Ghidra userop names in the first two rows. The x86-64 row has nothing
to normalize; `LOCK` and `UNLOCK` are meta-markers that wrap the lockable
instruction's inline P-Code body and pass through unchanged.
