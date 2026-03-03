# Patchestry

Patchestry is an **MLIR-based binary patching framework** built by [Trail of Bits](https://www.trailofbits.com/). It enables developers to safely patch deployed firmware binaries, particularly medical device firmware, without access to original source code or deep platform expertise. The pipeline lifts Ghidra P-Code to ClangIR (CIR) and then to LLVM IR, with each layer providing a different level of abstraction for analysis and patch development.

## Dependencies

### All Platforms
| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.25 | Required |
| LLVM / Clang / MLIR | 20 | [ClangIR](https://github.com/trail-of-forks/clangir) fork, built with `CLANG_ENABLE_CIR=ON` |
| Ninja | any | Preferred build system |
| lit | any | `pip install lit` — LLVM Integrated Tester |
| Docker | any | For Ghidra headless decompilation |
| Z3 | ≥ 4.8 | SMT solver; can use vendored (`PE_USE_VENDORED_Z3=ON`) or system |

Vendored and auto-built: `gflags`, `glog`, `rellic`, optionally `z3` and [clangir](https://github.com/trail-of-forks/clangir).

### macOS
```sh
brew install cmake ninja lit colima docker docker-buildx docker-credential-helper
Test and firmware binaries are built in Docker via `firmwares/build.sh`.

### Linux
- `lld` — required as the linker (enforced by CMake toolchain)
- `Z3` — system package or use `PE_USE_VENDORED_Z3=ON`
- Alternatively, use the pre-built dev container (see Linux build below)

## Building

### Step 0: Build the LLVM/ClangIR Fork (one-time)
```sh
git clone https://github.com/trail-of-forks/clangir
cd clangir && mkdir build && cd build
cmake -G Ninja ../llvm \
  -DCMAKE_INSTALL_PREFIX=<install_dir> \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;clang-tools-extra" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCLANG_ENABLE_CIR=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_TARGETS_TO_BUILD="host;AArch64;ARM;X86"
ninja install
```

### macOS
```sh
# Configure
CC=$(which clang) CXX=$(which clang++) cmake --preset default \
  -DCMAKE_PREFIX_PATH=<llvm_install>/lib/cmake/ \
  -DLLVM_EXTERNAL_LIT=$(which lit)

# Build (choose one)
cmake --build --preset release        # Release
cmake --build --preset debug          # Debug
cmake --build --preset relwithdebinfo # RelWithDebInfo

# Install
cmake --install builds/default --prefix ./builds/run --config Release
```

### Linux (dev container — recommended)
Use the pre-built container `ghcr.io/lifting-bits/patchestry-ubuntu-22.04-llvm-20-dev:latest`,
which ships LLVM 20, MLIR, Clang, Z3, and lit pre-installed.

```sh
cmake --preset ci \
  -DPE_USE_VENDORED_Z3=OFF \
  -DLLVM_EXTERNAL_LIT=/usr/local/bin/lit \
  -DLLVM_Z3_INSTALL_DIR=/usr/local

cmake --build --preset ci --config Release -j$(nproc)
```

### Linux (without dev container)
```sh
cmake --preset default \
  -DCMAKE_PREFIX_PATH=<llvm_install>/lib/cmake/ \
  -DLLVM_EXTERNAL_LIT=$(which lit) \
  -DPE_USE_VENDORED_Z3=ON

cmake --build --preset release -j$(nproc)
```

### Key CMake Options
| Option | Default | Description |
|---|---|---|
| `PE_USE_VENDORED_Z3` | `ON` | Build Z3 from `vendor/` instead of using system Z3 |
| `PATCHESTRY_ENABLE_TESTING` | `ON` | Build test targets |
| `ENABLE_SANITIZER_ADDRESS` | `OFF` | Enable AddressSanitizer |
| `ENABLE_SANITIZER_UNDEFINED_BEHAVIOR` | `OFF` | Enable UBSan |
| `ENABLE_SANITIZER_THREAD` | `OFF` | Enable ThreadSanitizer |

## Running Tests

```sh
# Build Ghidra headless Docker image first (required for ghidra test suite)
bash ./scripts/ghidra/build-headless-docker.sh

# Run all tests
lit ./builds/default/test -D BUILD_TYPE=Release -v

# Run a specific suite
lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v
lit ./builds/default/test/patchir-decomp   -D BUILD_TYPE=Release -v

# Via CTest
cd builds/default && ctest -V --build-config Release
```

**Test suites:**
- `ghidra-output-tests` — Ghidra headless decompilation (requires Docker)
- `patchir-decomp-tests` — Decompilation pipeline
- `patchir-transform-tests` — Patch application

## Module Build & Test Quick Reference

| Module | Build Target | Test Suite |
|--------|--------------|------------|
| AST (`lib/patchestry/AST/`) | `patchestry_ast` | patchir-decomp |
| Ghidra (`lib/patchestry/Ghidra/`) | `patchestry_ghidra` | patchir-decomp |
| Codegen (`lib/patchestry/Codegen/`) | `patchestry_codegen` | both (decomp + transform) |
| YAML (`lib/patchestry/YAML/`) | `patchestry_yaml` | patchir-transform |
| Passes (`lib/patchestry/Passes/`) | `patchestry_passes` | patchir-transform |
| Contracts dialect | `MLIRContracts` | patchir-transform |

Run tests: `lit ./builds/default/test/<suite> -D BUILD_TYPE=Debug -v`

**Targeted build:** `cmake --build builds/default --config Debug --target <target>`

**Linux dev container:** swap `builds/default` → `builds/ci`, add `-j$(nproc)`

## Architecture

### End-to-End Pipeline
```
Firmware Binary
  --[Ghidra headless + patchestry Ghidra scripts]-->  P-Code JSON
  --[patchir-decomp]-->                               Clang IR (.cir)
  --[patchir-transform + YAML patch spec]-->          Patched CIR (.cir)
  --[patchir-cir2llvm]-->                             LLVM IR (.ll / .bc)
  --[Patcherex2]-->                                   Patched Binary
```

### Key Components

| Path | Role |
|---|---|
| `include/patchestry/Ghidra/` | Data model for Ghidra exports: `Program`, `Function`, `BasicBlock`, `Operation`, `Varnode` |
| `include/patchestry/AST/` | Lifts the Ghidra data model to a Clang `ASTContext` |
| `include/patchestry/Dialect/Contracts/` | MLIR dialect for static formal contracts (pre/postconditions for verification) |
| `include/patchestry/Passes/` | `InstrumentationPass`: core patch engine running on CIR modules |
| `include/patchestry/YAML/` | YAML patch specification parser |
| `lib/patchestry/Passes/` | `PatchOperationImpl`, `ContractOperationImpl`, `OperationMatcher`, `Compiler` |
| `vendor/rellic/` | Decompilation AST cleanup: raw P-Code→AST passes through LLVM IR and Rellic to improve structure |
| `include/patchestry/intrinsics/` | C header library for writing patch functions |
| `scripts/ghidra/` | Ghidra headless scripts and Docker build infrastructure |

### Patch Authoring
Patches are written in plain C (using `include/patchestry/intrinsics/patchestry_intrinsics.h`). Patchestry compiles them to CIR and merges them into the target module. A YAML spec controls which operations to match and how to apply each patch (`apply_before`, `apply_after`, or `replace`).

### Contracts
The `contracts` MLIR dialect supports two contract types. **Static contracts** attach formal pre/postconditions to CIR operations as MLIR attributes (`contract.static`); `patchir-cir2llvm` embeds these as `!static_contract` LLVM metadata for downstream formal verification tools (KLEE, SeaHorn). **Dynamic (runtime) contracts** are implemented as C/C++ functions that are inserted at match sites (before/after calls or at function entry) and execute at runtime to validate conditions; they are specified in YAML with `type: "RUNTIME"` and a `code_file`/`function_name`.

## Tools

| Tool | Description |
|---|---|
| `patchir-decomp` | Decompile Ghidra P-Code JSON → CIR / LLVM IR / ASM / object file |
| `patchir-transform` | Apply YAML-specified patches and contracts to a CIR file |
| `patchir-cir2llvm` | Lower CIR → LLVM IR text or bitcode (embeds patchestry + contract metadata) |
| `patchir-yaml-parser` | Validate and pretty-print YAML patch specification files |

### Tool Quick Reference

```sh
# Decompile a function from P-Code JSON to CIR
patchir-decomp -input func.json -emit-cir -output func

# Decompile to LLVM IR
patchir-decomp -input func.json -emit-llvm -output func

# Apply patches from a YAML spec
patchir-transform input.cir -spec patch.yaml -o patched.cir

# Lower patched CIR to LLVM IR
patchir-cir2llvm -S patched.cir -o patched.ll

# Validate a YAML patch spec
patchir-yaml-parser config.yaml --validate
```

## Repository Layout

```
patchestry/
├── cmake/                  # CMake modules, toolchain files, options
├── docs/                   # Documentation (GettingStarted/, README.md)
├── include/patchestry/     # Public C++ headers
├── lib/patchestry/         # Library implementations
├── tools/                  # Executable tool sources
│   ├── patchir-decomp/
│   ├── patchir-transform/
│   ├── patchir-cir2llvm/
│   └── patchir-yaml-parser/
├── test/                   # LIT test suites
├── vendor/                 # Vendored dependencies (gflags, glog, z3, rellic, clangir)
├── scripts/ghidra/         # Ghidra headless Docker scripts
├── firmwares/              # Example firmware targets
├── CMakeLists.txt
├── CMakePresets.json       # Build/test presets (default, ci, debug, release, ...)
└── build.sh                # Docker-based build helper
```

---

## Code Style and Conventions

### Naming
- Functions and variables: `snake_case`
- Classes and structs: `PascalCase`
- Enum values: `UPPER_CASE`
- MLIR ops / dialects / passes: `PascalCase` (e.g., `InstrumentationPass`)
- Constants: `snake_case` local, `kCamelCase` for named compile-time constants

### C++ Patterns
- Prefer `auto` for MLIR/Clang API return types; use explicit types for primitives
- MLIR casting: `mlir::dyn_cast<T>`, `mlir::isa<T>`, `llvm::dyn_cast_or_null<T>`
- Error reporting: `LOG(ERROR)` / `LOG(WARNING)` (glog) — not `llvm::errs()` or `std::cerr`
- Null-check operations before use; return early on failure with a `LOG(ERROR)` message
- Comments explain non-obvious intent or constraints only — never narrate what the code does

### What Not to Edit
- `vendor/` — pinned copies of rellic, glog, gflags, z3, clangir; override via CMake options
- `builds/` — generated build artefacts
- Any `*.inc` or `*.td` files under `include/patchestry/Dialect/` unless changing the dialect

---

## Verification

After modifying code, run the targeted build + test for the module you changed:

```sh
# Build single target
cmake --build builds/default --config Debug --target patchestry_ast

# Run relevant test suite
lit ./builds/default/test/patchir-decomp -D BUILD_TYPE=Debug -v

# Inspect decompiled output
patchir-decomp -input func.json -print-tu -output /tmp/out -verbose

# Apply and inspect patches
patchir-transform input.cir -spec patch.yaml -o patched.cir

# Run a single LIT test locally
bash scripts/run-decomp-test.sh goto_while_structurize
bash scripts/run-transform-test.sh bl_usb__send_message

# Validate a YAML spec
patchir-yaml-parser myspec.yaml --validate
```

---

## Documentation References

| Document | Purpose |
|----------|---------|
| `docs/README.md` | Project overview and technical rationale (MLIR-based binary patching) |
| `docs/statement.md` | License (Apache 2.0) and ARPA-H funding statement |
| `docs/GettingStarted/build.md` | Building and first-time setup (macOS/Linux) |
| `docs/GettingStarted/firmware_examples.md` | Running Patchestry on firmware (Ghidra image, build, decompile) |
| `docs/GettingStarted/patch_specifications.md` | Full YAML patch and contract specification reference |
| `docs/GettingStarted/intrinsic_library.md` | C intrinsics API for writing patch functions |
