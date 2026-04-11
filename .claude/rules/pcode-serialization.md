---
description: Ghidra P-Code serialization — headless decompilation, JSON export, and firmware testing
paths:
  - "scripts/ghidra/**"
  - "test/ghidra/**"
  - "test/scripts/ghidra/**"
  - "firmwares/**"
---

# P-Code Serialization Development

## Key Files

| Task | Files |
|------|-------|
| Modify P-Code serialization | `scripts/ghidra/util/PcodeSerializer.java` |
| Modify function listing | `scripts/ghidra/util/FunctionSerializer.java`, `scripts/ghidra/PatchestryListFunctions.java` |
| Modify decompilation script | `scripts/ghidra/PatchestryDecompileFunctions.java` |
| Change architecture detection | `scripts/ghidra/decompile-entrypoint.sh` |
| Change Docker image | `scripts/ghidra/decompile-headless.dockerfile` |
| Add Ghidra utility | `scripts/ghidra/util/ApiUtil.java` |
| Add domain type | `scripts/ghidra/domain/` |
| Add Ghidra LIT test | `test/ghidra/` — write a C file with `// RUN:` directives |
| Add Ghidra JUnit test | `test/scripts/ghidra/` — extend `BaseTest.java` |
| Build firmware binaries | `firmwares/build.sh` |

## Build Docker Image (prerequisite)

```sh
bash ./scripts/ghidra/build-headless-docker.sh
```

This builds `trailofbits/patchestry-decompilation:latest` with Ghidra 11.3.2 and the patchestry scripts.

## Recover a Function from Binary → JSON

```sh
# Decompile a single function
bash ./scripts/ghidra/decompile-headless.sh \
  --input path/to/binary.elf --function function_name --output /tmp/output.json

# Decompile all functions
bash ./scripts/ghidra/decompile-headless.sh \
  --input path/to/binary.elf --output /tmp/output.json

# List all functions in a binary
bash ./scripts/ghidra/decompile-headless.sh \
  --input path/to/binary.elf --list-functions
```

## Build & Test

```sh
# 1. Build Docker image (one-time, or after script changes)
bash ./scripts/ghidra/build-headless-docker.sh

# 2. Run Ghidra LIT tests (requires Docker)
# macOS
lit ./builds/default/test/ghidra -D BUILD_TYPE=Debug -v

# Linux (dev container)
lit ./builds/ci/test/ghidra -D BUILD_TYPE=Release -v
```

## LIT Test Pattern

Tests in `test/ghidra/` compile a C file to an object, decompile it via Docker, and check the JSON output:

```c
// RUN: %cc-x86_64 %s -g -c -o %t.o
// RUN: %decompile-headless --input %t.o --function test --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?test}}"
```

## JUnit Tests

JUnit tests in `test/scripts/ghidra/` run inside the Ghidra Docker container against firmware binaries (bloodlight):

- `PatchestryDecompileFunctionsTest.java` — decompiler interface, single/all function decompilation
- `PatchestryListFunctionsTest.java` — function listing
- `PcodeSerializerTest.java` — core P-Code serialization
- Base class: `BaseTest.java` (sets up Ghidra TestEnv with `ARM:LE:32:Cortex`)

## JSON Output Structure

```json
{
  "architecture": "ARM",
  "id": "ARM:LE:32:Cortex",
  "format": "Executable and Linking Format (ELF)",
  "functions": { ... },
  "globals": { ... },
  "types": { ... }
}
```

## Extraout Sanitizer

Rewrites Ghidra's `extraout_rN` / `unaff_*` / `in_*` register aliases to
the pre-call variable they shadow. On by default.

Flags: `--sanitize-extraout` (master, default on),
`--sanitize-extraout-analytical={auto|on|off}` (Tier 2 gate, default auto).

Tier 2 allowlist: `util.PcodeSerializer.TIER2_DEFAULT_ARCHITECTURES`
= `{ARM, AARCH64}`.
