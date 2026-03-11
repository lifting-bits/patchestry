# Patchestry Developer Guide (Canonical)

This is the canonical onboarding and development guide for Patchestry.

- `AGENTS.md` is the source of truth for Codex/Cursor and human contributors.
- `CLAUDE.md` exists only as a compatibility entrypoint for Claude Code and points here.

## Purpose

Patchestry is an MLIR/CIR-based binary patching framework for patching deployed firmware
without original source code.

Important terminology:

- In this document, "patch" means a firmware patch authored against a target firmware image.
- It does not mean a patch to the Patchestry repository itself.

## Architecture and Modules

### End-to-end data flow

```text
Firmware Binary
  -> Ghidra headless scripts (`scripts/ghidra/*`)
  -> P-Code JSON (`*.json`, Ghidra export schema)
  -> `patchir-decomp`
  -> CIR (`*.cir`)
  -> `patchir-transform` + YAML spec (`*.yaml`)
  -> Patched CIR (`*.cir`)
  -> `patchir-cir2llvm`
  -> LLVM IR (`*.ll`) or bitcode (`*.bc`)
  -> downstream binary rewriting / verification tools
```

### Module interface map

These interfaces are the contracts contributors should keep stable while iterating on internals.

| Module | Main paths | Stable interface files | Input format | Output format | Module test command |
|---|---|---|---|---|---|
| Ghidra model | `include/patchestry/Ghidra/`, `lib/patchestry/Ghidra/` | `include/patchestry/Ghidra/JsonDeserialize.hpp`, `include/patchestry/Ghidra/Pcode.hpp`, `include/patchestry/Ghidra/PcodeTranslation.hpp` | Ghidra export JSON | in-memory Program/Function/Block/Op model | `lit ./builds/default/test/ghidra -D BUILD_TYPE=Debug -v` |
| AST lifting | `include/patchestry/AST/`, `lib/patchestry/AST/` | `include/patchestry/AST/ASTConsumer.hpp`, `include/patchestry/AST/FunctionBuilder.hpp`, `include/patchestry/AST/OperationBuilder.hpp`, `include/patchestry/AST/TypeBuilder.hpp` | Ghidra model objects | Clang AST and CIR-ready structures | `lit ./builds/default/test/patchir-decomp -D BUILD_TYPE=Debug -v` |
| Decompiler tool | `tools/patchir-decomp/` | `tools/patchir-decomp/main.cpp` | P-Code JSON | CIR / LLVM IR / asm / object output selected by flags | `lit ./builds/default/test/patchir-decomp -D BUILD_TYPE=Debug -v` |
| YAML spec parser | `include/patchestry/YAML/`, `lib/patchestry/YAML/`, `tools/patchir-yaml-parser/` | `include/patchestry/YAML/ConfigurationFile.hpp`, `include/patchestry/YAML/PatchSpec.hpp`, `include/patchestry/YAML/ContractSpec.hpp`, `include/patchestry/YAML/YAMLParser.hpp` | YAML patch/contract spec | validated configuration objects consumed by transform passes | `lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v` |
| Patch pass engine | `include/patchestry/Passes/`, `lib/patchestry/Passes/` | `include/patchestry/Passes/InstrumentationPass.hpp`, `include/patchestry/Passes/OperationMatcher.hpp`, `lib/patchestry/Passes/PatchOperationImpl.hpp`, `lib/patchestry/Passes/ContractOperationImpl.hpp` | CIR + parsed patch/contract config | transformed CIR with inserted/replaced operations | `lit ./builds/default/test/patchir-transform/patches -D BUILD_TYPE=Debug -v` |
| Contracts dialect | `include/patchestry/Dialect/Contracts/`, `lib/patchestry/Dialect/Contracts/` | `include/patchestry/Dialect/Contracts/Contract.td`, `include/patchestry/Dialect/Contracts/ContractsDialect.hpp` | CIR ops plus contract attrs/spec | CIR attrs and LLVM metadata for verification flows | `lit ./builds/default/test/patchir-transform/contracts -D BUILD_TYPE=Debug -v` |
| CIR->LLVM lowering | `tools/patchir-cir2llvm/` | `tools/patchir-cir2llvm/main.cpp` | CIR | LLVM IR/bitcode with patch and contract metadata | `lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v` |
| Intrinsics library | `include/patchestry/intrinsics/`, `lib/patchestry/intrinsics/` | `include/patchestry/intrinsics/patchestry_intrinsics.h`, `include/patchestry/intrinsics/runtime.h`, `include/patchestry/intrinsics/safety.h` | patch C code | helper functions compiled into CIR and referenced from patch specs | `lit ./builds/default/test/patchir-transform/patches -D BUILD_TYPE=Debug -v` |

### Module-level build and test quick reference

| Area | Build target | Test command |
|---|---|---|
| Ghidra export integration | `patchir-decomp` | `lit ./builds/default/test/ghidra -D BUILD_TYPE=Debug -v` |
| AST/Ghidra/decomp | `patchir-decomp` | `lit ./builds/default/test/patchir-decomp -D BUILD_TYPE=Debug -v` |
| YAML parsing | `patchir-yaml-parser` | `lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v` |
| Patch application | `patchir-transform` | `lit ./builds/default/test/patchir-transform/patches -D BUILD_TYPE=Debug -v` |
| Contract insertion/metadata | `patchir-transform` and `patchir-cir2llvm` | `lit ./builds/default/test/patchir-transform/contracts -D BUILD_TYPE=Debug -v` |
| CIR lowering | `patchir-cir2llvm` | `lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v` |

### Data interface details by stage

1. Ghidra stage:
   `scripts/ghidra/*` and the Ghidra-side pipeline emit JSON matching the data
   model loaded by `include/patchestry/Ghidra/JsonDeserialize.hpp` and related
   P-Code headers.
2. Decomp stage:
   `patchir-decomp` consumes that JSON, builds the Ghidra model, lifts it
   through AST builders, and emits CIR as the primary editable interchange
   format for later patching.
3. Transform stage:
   `patchir-transform` consumes CIR plus YAML parsed through
   `include/patchestry/YAML/*.hpp` and applies `InstrumentationPass` and the
   patch/contract operation implementations to produce patched CIR.
4. Lowering stage:
   `patchir-cir2llvm` consumes patched CIR and emits LLVM IR text (`.ll`) or
   bitcode (`.bc`), carrying patch and contract semantics forward as LLVM-level
   metadata for downstream binary rewriting and formal verification.

## Usage and Workflows

### Who uses which tools and why

| Tool | Primary user | Why it exists | Typical use point |
|---|---|---|---|
| `patchir-decomp` | reverse engineer / decomp developer | lift Ghidra exports into editable IR | first step after obtaining P-Code JSON |
| `patchir-transform` | patch author / verification engineer | apply firmware patches and contracts onto CIR | after decompilation, before lowering |
| `patchir-cir2llvm` | verification/binary pipeline engineer | emit LLVM IR/BC for downstream toolchains | after transform stage |
| `patchir-yaml-parser` | patch author / CI | validate patch specs early and fail fast | before transform in local loops and CI |

### Patch authoring workflow

1. Write patch logic in C using `include/patchestry/intrinsics/patchestry_intrinsics.h`.
2. Declare patch placement and matching in YAML (`apply_before`, `apply_after`, `replace`).
3. Validate YAML spec with `patchir-yaml-parser`.
4. Apply with `patchir-transform` and inspect resulting CIR.
5. Lower with `patchir-cir2llvm` and continue to downstream tooling.

### Contracts workflow

Patchestry supports two contract modes:

- Static contracts:
  represented in CIR/MLIR (`contract.static`), then emitted as LLVM metadata for
  formal tools such as KLEE/SeaHorn.
- Runtime contracts:
  regular C/C++ checks injected at configured sites and executed with the binary.

### Tool quick reference

```sh
# Decompile a function from P-Code JSON to CIR
patchir-decomp -input func.json -emit-cir -output func

# Apply patches from YAML to CIR
patchir-transform input.cir -spec patch.yaml -o patched.cir

# Lower patched CIR to LLVM IR
patchir-cir2llvm -S patched.cir -o patched.ll

# Validate a YAML patch spec
patchir-yaml-parser config.yaml --validate
```

## Related Docs

- Build walkthrough: `docs/GettingStarted/build.md`
- Firmware example flow: `docs/GettingStarted/firmware_examples.md`
- Host ARM64 image build for macOS: `.devcontainer/README-HOST-BUILD.md`
- Claude-specific workflow notes: `.claude/rules/*.md`

## Dependencies

### Required host tooling

| Dependency | Version | Notes |
|---|---|---|
| CMake | >= 3.25 | Configure/build presets |
| LLVM / Clang / MLIR | 20 | ClangIR-enabled build |
| lit | any recent | LLVM integrated test runner |
| Docker engine | current | Required for Ghidra headless and firmware flows |
| Ninja | recommended | Faster local builds |
| lld (Linux) | distro package | Required by Linux toolchain setup |

### Vendored dependencies

Vendored projects live under `vendor/` and are checked out via git submodules.
The current pinned revisions in this repository are:

| Dependency | Pinned revision | Location | Purpose |
|---|---|---|---|
| clangir | `ae0e95fb` (`patche-clangir-20`) | `vendor/clangir/src` | LLVM/Clang/MLIR+CIR toolchain when using vendored clang |
| rellic | `cff5bb7b` (`llvm20`) | `vendor/rellic/src` | AST recovery and decompilation support |
| glog | `7b134a5c` | `vendor/glog/src` | Structured logging in core tools |
| gflags | `a738fdf9` | `vendor/gflags/src` | CLI flag parsing |
| z3 | `8d67feef` | `vendor/z3/src` | SMT solver used in analysis/verification flows |

### Adding a new vendored dependency

Follow this process so both humans and LLM agents can maintain consistency:

1. Add a new entry in `.gitmodules` with path `vendor/<name>/src` and upstream URL.
2. Add a `vendor/<name>/CMakeLists.txt` wrapper that initializes submodule content when missing and installs into `${PE_VENDOR_INSTALL_DIR}`.
3. Wire the dependency into top-level CMake/options (`cmake/options.cmake` and/or `CMakeLists.txt`) with an explicit `PE_USE_VENDORED_*` option when appropriate.
4. Update the vendored dependency table in this document with revision, location, and purpose.
5. Add or update tests/build checks that exercise the dependency path.

## First-Time Setup

### macOS

Install base tooling and configure Docker BuildX for Colima:

```sh
xcode-select --install
brew install colima docker docker-buildx docker-credential-helper cmake lit
brew install FiloSottile/musl-cross/musl-cross
mkdir -p ~/.docker/cli-plugins
ln -sf "$(which docker-buildx)" ~/.docker/cli-plugins/docker-buildx
colima restart
docker buildx version
```

Notes:

- Patchestry uses Colima as the Docker backend on macOS.
- `build.sh` and Ghidra headless tests assume a working Docker daemon (`docker ps` should work).
- See `docs/GettingStarted/build.md` for the maintained setup sequence.

### Linux

Choose one of these:

1. Dev container path (recommended for fastest onboarding and most reproducible environment).
2. Native host path (better for local integration with your existing toolchains and profiling tools).

The Linux bootstrap gist referenced in `docs/GettingStarted/build.md` is the fastest path to a clean from-scratch setup.

## Build Workflows

### Step 0 (one-time): LLVM/ClangIR build

If you are not using a prebuilt dev container image, build/install LLVM+ClangIR first:

```sh
git clone https://github.com/trail-of-forks/clangir
cd clangir
mkdir build && cd build
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

Expected runtime:

- Typical workstation: 30-90 minutes.
- Lower-memory systems or emulated containers: can exceed 2 hours.

If you are on Apple Silicon and this is too slow or space-constrained in Docker,
use `.devcontainer/README-HOST-BUILD.md` to build the ARM64 base image on host storage.

### Standard local configure/build

```sh
cmake --preset default \
  -DCMAKE_PREFIX_PATH=<llvm_install>/lib/cmake \
  -DLLVM_EXTERNAL_LIT=$(which lit)

cmake --build --preset debug -j
cmake --build --preset release -j
```

### `build.sh` containerized workflow

`./build.sh` is the repository helper for containerized builds. It:

1. Builds or reuses a Docker image with dependencies.
2. Runs the build inside a controlled container.
3. Places artifacts under `builds/default`.

When to use it:

- Use `build.sh` when you want reproducible containerized builds or when host setup is incomplete.
- Prefer direct preset builds when your host toolchain is already configured and you need fastest incremental iteration.

## Linux Build Mode Choice

### Dev container build (`--preset ci` / prebuilt image)

Use this when you want:

- Consistent CI-like dependencies.
- Fast onboarding with minimal host package management.
- Fewer environment-specific failures.

Tradeoffs:

- Container abstraction can make low-level host debugging/profiling less direct.

### Native Linux build (`--preset default`)

Use this when you want:

- Tight integration with local tooling (profilers, sanitizers, custom clang/lld).
- Full control over system packages and compiler layout.

Tradeoffs:

- More setup drift risk versus CI/devcontainer.

## Testing

### Core commands

```sh
# Build the Ghidra headless image first
bash ./scripts/ghidra/build-headless-docker.sh

# Run all tests
ctest --preset debug --output-on-failure

# Run via lit directly
lit ./builds/default/test -D BUILD_TYPE=Release -v
lit ./builds/default/test/patchir-decomp -D BUILD_TYPE=Debug -v
lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v
```

### Test reliability expectations

- Tests are expected to be deterministic and non-flaky.
- A PR is not considered ready if repeated local runs produce inconsistent outcomes.
- New behavior changes should include a regression test in the closest relevant suite.

## Repository Layout

| Path | Purpose |
|---|---|
| `include/patchestry/` | Public headers and component interfaces |
| `lib/patchestry/` | Core implementation |
| `tools/` | CLI tools (`patchir-*`, `pcode-translate`) |
| `test/` | LIT test suites |
| `scripts/` | Build/test helpers, including Ghidra scripts |
| `.devcontainer/` | Container build/dev environment support |
| `.claude/rules/` | Claude-targeted workflow docs |
| `vendor/` | Vendored third-party dependencies |

## Coding Conventions

- Format with `.clang-format` (LLVM style, 4-space, 96 columns).
- Keep include ordering stable.
- Prefer focused, single-purpose changes over cross-cutting refactors.

## Pull Request Expectations

Before requesting review:

1. Keep commit scope focused and component-oriented.
2. Run targeted builds/tests for touched components, plus any affected integration tests.
3. Include a clear PR description: behavior change, design notes, and test evidence.
4. If changing docs/process, ensure commands are copy/paste valid and up to date.

Commit message format:

- `component: Simple sentence with a period.`
- Keep subject <= 80 characters.
