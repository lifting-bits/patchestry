# How a model library is consumed

Captured from the repository at commit `da6fcf0` on 2026-11-23. Re-capture when
`tools/patchir-klee-verifier/` or `lib/patchestry/klee/models/` changes.

## patchir-klee-verifier

Output of `builds/default/tools/patchir-klee-verifier/Debug/patchir-klee-verifier --help`,
without LLVM's generic options:

```
USAGE: patchir-klee-verifier [options] <input LLVM IR file>

  -S                                    - Emit LLVM IR instead of bitcode
  --klee-init-array-expand-limit=<elements> - Above this length, arrays of aggregates fall back to flat byte symbolization
  --klee-init-max-depth=<depth>         - Runtime depth cap for recursive pointee initialization
  --model-library=<file>                - LLVM IR/bitcode file with external function models (linked before stubbing)
  -o <filename>                         - Output filename
  --strict-contracts                    - Exit non-zero when any contract predicate fails to parse
  --symbolic-ptr-size=<bytes>           - Default symbolic buffer size (bytes) for pointer arguments
  --target-function=<name>              - Function to verify (required)
  -v                                    - Enable verbose output
```

One `--model-library` file is accepted. Several model sources have to be
compiled and joined (`llvm-link`) into one file first.

## What the verifier does with it (`tools/patchir-klee-verifier/main.cpp`)

1. The input module is retargeted to x86_64 (`retargetModuleToX86_64`), whatever
   the firmware's architecture. `docs/klee-integration.md`, "Architecture
   Retargeting", explains why and what it breaks: a struct that contains a
   pointer changes layout.
2. The model library is linked with `Linker::Flags::LinkOnlyNeeded`: only the
   definitions the module references are pulled in, so a library may hold more
   models than one function needs.
3. Every function that is still only declared is stubbed by the verifier
   (`stubExternalFunctions` in `tools/patchir-klee-verifier/HarnessGen.cpp`).
   A function without a model is therefore not an error; it gets the weakest
   model:

   | Return type | The stub |
   | --- | --- |
   | `void` | returns, with no effect |
   | pointer | returns a fresh `malloc` of `--symbolic-ptr-size` bytes (default 256), made symbolic; never null |
   | sized scalar | returns a symbolic value named `<function>_ret` |
   | unsized or zero-sized | returns `undef` |

   No stub writes through a pointer argument.

   Not stubbed: `klee_*`, `llvm.*`, and `malloc`, `calloc`, `realloc`, `free`,
   `posix_memalign`, `valloc`, `memalign`, which KLEE handles itself. A model
   library must not define those seven.

   Target intrinsics such as `llvm.arm.qadd` are a known gap: the comment
   above `stubExternalFunctions` says their semantics are lost in the harness
   and that a per-architecture model library is planned. A model library
   cannot define them today, because `llvm.*` names are reserved. The
   extractor lists them under `summary.target_intrinsics` so the report can
   name them.

The library is rejected unless it matches the retargeted module:

- an empty triple and an empty datalayout are accepted and stamped with the
  module's;
- otherwise the triple must agree in architecture, vendor and OS (the
  environment field, gnu or musl, is ignored) and the datalayout string must
  be equal.

## How the existing models are built (`lib/patchestry/klee/models/CMakeLists.txt`)

```
clang -emit-llvm -c -O0 -g -ffreestanding \
    -target x86_64-unknown-linux-gnu \
    -include <models dir>/klee_stub.h \
    -o <name>.bc <name>.c
```

The built files are in `builds/default/lib/patchestry/klee/models/`. The lit test
`test/patchir-klee-verifier/bl_usb__send_message_modeled.ll` compiles
`usb_hal_models.c` the same way and passes it with `--model-library`.

## Things observed while capturing this

- `llvm-nm` does not read textual IR ("not recognized as a valid object file").
  `scripts/list_externals.py` parses the `declare` lines itself and uses
  `llvm-dis` for bitcode.
- KLEE is not installed on the host. It runs from the Docker image through
  `scripts/klee/run-klee.sh`.
