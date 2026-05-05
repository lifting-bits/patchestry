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

### Project-owned module inventory

This inventory is intended to be exhaustive for patchestry-owned code. It does
not attempt to document LLVM/MLIR internals or vendored dependency internals.

#### Core libraries and dialects

| Module | Build target | Main paths | Primary role |
|---|---|---|---|
| Ghidra model and translation | `patchestry_ghidra` | `include/patchestry/Ghidra/`, `lib/patchestry/Ghidra/` | deserialize Ghidra JSON and register P-Code translation |
| AST lifting | `patchestry_ast` | `include/patchestry/AST/`, `lib/patchestry/AST/` | lift Ghidra model into Clang AST/CIR-ready structures |
| Codegen | `patchestry_codegen` | `include/patchestry/Codegen/`, `lib/patchestry/Codegen/` | serialize and lower internal representations during tool pipelines |
| YAML parsing | `patchestry_yaml` | `include/patchestry/YAML/`, `lib/patchestry/YAML/` | parse patch and contract YAML configuration |
| Patch passes | `patchestry_passes` | `include/patchestry/Passes/`, `lib/patchestry/Passes/` | apply patch and contract transformations to CIR |
| Contracts dialect | `MLIRContracts` | `include/patchestry/Dialect/Contracts/`, `lib/patchestry/Dialect/Contracts/` | represent contract attributes and verification metadata |
| Pcode dialect | `MLIRPcode` | `include/patchestry/Dialect/Pcode/`, `lib/patchestry/Dialect/Pcode/` | represent and deserialize P-Code as an MLIR dialect |
| Intrinsics library | `patchestry_intrinsics` | `include/patchestry/intrinsics/`, `lib/patchestry/intrinsics/` | provide patch helper/runtime functions for patch C code |
| Utility headers | no standalone target | `include/patchestry/Util/` | shared logging, diagnostics, common options, helper types |

#### Tools

| Tool | Build target | Main paths | Purpose |
|---|---|---|---|
| `patchir-decomp` | `patchir-decomp` | `tools/patchir-decomp/` | decompile Ghidra JSON into CIR/LLVM/asm/object outputs |
| `patchir-transform` | `patchir-transform` | `tools/patchir-transform/` | apply YAML-defined patches and contracts to CIR |
| `patchir-cir2llvm` | `patchir-cir2llvm` | `tools/patchir-cir2llvm/` | lower CIR to LLVM IR or bitcode |
| `patchir-yaml-parser` | `patchir-yaml-parser` | `tools/patchir-yaml-parser/` | validate and inspect YAML configuration |
| `pcode-translate` | `pcode-translate` | `tools/pcode-translate/` | standalone P-Code translation driver built on patchestry Ghidra translation |

#### Workflow scripts

| Script area | Main paths | Purpose |
|---|---|---|
| Ghidra automation | `scripts/ghidra/` | build headless container, run decompilation, serialize functions/P-Code |
| JSON rendering helper | `scripts/render_json.py` | utility script for rendering/inspecting JSON artifacts |

### Module interface map

These interfaces are the contracts contributors should keep stable while iterating on internals.

| Module | Main paths | Stable interface files | Input format | Output format | Module test command |
|---|---|---|---|---|---|
| Ghidra model | `include/patchestry/Ghidra/`, `lib/patchestry/Ghidra/` | `include/patchestry/Ghidra/JsonDeserialize.hpp`, `include/patchestry/Ghidra/Pcode.hpp`, `include/patchestry/Ghidra/PcodeTranslation.hpp` | Ghidra export JSON | in-memory Program/Function/Block/Op model | `lit ./builds/default/test/ghidra -D BUILD_TYPE=Debug -v` |
| AST lifting | `include/patchestry/AST/`, `lib/patchestry/AST/` | `include/patchestry/AST/ASTConsumer.hpp`, `include/patchestry/AST/FunctionBuilder.hpp`, `include/patchestry/AST/OperationBuilder.hpp`, `include/patchestry/AST/TypeBuilder.hpp` | Ghidra model objects | Clang AST and CIR-ready structures | `lit ./builds/default/test/patchir-decomp -D BUILD_TYPE=Debug -v` |
| Codegen | `include/patchestry/Codegen/`, `lib/patchestry/Codegen/` | `include/patchestry/Codegen/Codegen.hpp`, `include/patchestry/Codegen/PassManager.hpp`, `include/patchestry/Codegen/Serializer.hpp` | AST/CIR owned by patchestry tools | serialized/lowered outputs consumed by tool frontends | `lit ./builds/default/test/patchir-decomp -D BUILD_TYPE=Debug -v` |
| Decompiler tool | `tools/patchir-decomp/` | `tools/patchir-decomp/main.cpp` | P-Code JSON | CIR / LLVM IR / asm / object output selected by flags | `lit ./builds/default/test/patchir-decomp -D BUILD_TYPE=Debug -v` |
| YAML spec parser | `include/patchestry/YAML/`, `lib/patchestry/YAML/`, `tools/patchir-yaml-parser/` | `include/patchestry/YAML/ConfigurationFile.hpp`, `include/patchestry/YAML/PatchSpec.hpp`, `include/patchestry/YAML/ContractSpec.hpp`, `include/patchestry/YAML/YAMLParser.hpp` | YAML patch/contract spec | validated configuration objects consumed by transform passes | `lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v` |
| Patch pass engine | `include/patchestry/Passes/`, `lib/patchestry/Passes/` | `include/patchestry/Passes/InstrumentationPass.hpp`, `include/patchestry/Passes/OperationMatcher.hpp`, `lib/patchestry/Passes/PatchOperationImpl.hpp`, `lib/patchestry/Passes/ContractOperationImpl.hpp` | CIR + parsed patch/contract config | transformed CIR with inserted/replaced operations | `lit ./builds/default/test/patchir-transform/patches -D BUILD_TYPE=Debug -v` |
| Contracts dialect | `include/patchestry/Dialect/Contracts/`, `lib/patchestry/Dialect/Contracts/` | `include/patchestry/Dialect/Contracts/Contract.td`, `include/patchestry/Dialect/Contracts/ContractsDialect.hpp` | CIR ops plus contract attrs/spec | CIR attrs and LLVM metadata for verification flows | `lit ./builds/default/test/patchir-transform/contracts -D BUILD_TYPE=Debug -v` |
| Pcode dialect | `include/patchestry/Dialect/Pcode/`, `lib/patchestry/Dialect/Pcode/` | `include/patchestry/Dialect/Pcode/Deserialize.hpp`, `include/patchestry/Dialect/Pcode/PcodeDialect.hpp`, `include/patchestry/Dialect/Pcode/PcodeOps.hpp`, `include/patchestry/Dialect/Pcode/PcodeTypes.hpp` | P-Code operations and serialized dialect data | MLIR P-Code dialect objects used by translation/decomp flows | `lit ./builds/default/test/pcode-translate -D BUILD_TYPE=Debug -v` |
| CIR->LLVM lowering | `tools/patchir-cir2llvm/` | `tools/patchir-cir2llvm/main.cpp` | CIR | LLVM IR/bitcode with patch and contract metadata | `lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v` |
| Intrinsics library | `include/patchestry/intrinsics/`, `lib/patchestry/intrinsics/` | `include/patchestry/intrinsics/patchestry_intrinsics.h`, `include/patchestry/intrinsics/runtime.h`, `include/patchestry/intrinsics/safety.h` | patch C code | helper functions compiled into CIR and referenced from patch specs | `lit ./builds/default/test/patchir-transform/patches -D BUILD_TYPE=Debug -v` |
| Utility headers | `include/patchestry/Util/` | `include/patchestry/Util/Common.hpp`, `include/patchestry/Util/Diagnostic.hpp`, `include/patchestry/Util/Log.hpp`, `include/patchestry/Util/Options.hpp` | shared options, diagnostics, logging inputs | common support APIs used across patchestry components | exercised transitively by owning component tests |
| `patchir-yaml-parser` tool | `tools/patchir-yaml-parser/` | `tools/patchir-yaml-parser/main.cpp` | YAML patch/contract spec | validation results and diagnostics | `lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v` |
| `patchir-transform` tool | `tools/patchir-transform/` | `tools/patchir-transform/main.cpp` | CIR + YAML patch/contract config | patched CIR | `lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v` |
| `pcode-translate` tool | `tools/pcode-translate/` | `tools/pcode-translate/main.cpp` | MLIR translation command line + P-Code translation registration | translated P-Code output via MLIR translation driver | `lit ./builds/default/test/pcode-translate -D BUILD_TYPE=Debug -v` |

### Module-level build and test quick reference

| Area | Build target | Test command |
|---|---|---|
| Ghidra export integration | `patchir-decomp` | `lit ./builds/default/test/ghidra -D BUILD_TYPE=Debug -v` |
| AST/Ghidra/decomp | `patchir-decomp` | `lit ./builds/default/test/patchir-decomp -D BUILD_TYPE=Debug -v` |
| P-Code dialect translation | `pcode-translate` | `lit ./builds/default/test/pcode-translate -D BUILD_TYPE=Debug -v` |
| YAML parsing | `patchir-yaml-parser` | `lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v` |
| Patch application | `patchir-transform` | `lit ./builds/default/test/patchir-transform/patches -D BUILD_TYPE=Debug -v` |
| Contract insertion/metadata | `patchir-transform` and `patchir-cir2llvm` | `lit ./builds/default/test/patchir-transform/contracts -D BUILD_TYPE=Debug -v` |
| CIR lowering | `patchir-cir2llvm` | `lit ./builds/default/test/patchir-transform -D BUILD_TYPE=Debug -v` |
| Standalone intrinsics build | `patchestry_intrinsics` | validate via patch-based transform tests and standalone CMake build when editing `lib/patchestry/intrinsics/` |

### Component dependency map

Keep these dependency boundaries in mind when editing interfaces:

| Consumer | Depends on patchestry-owned components | Why it depends on them |
|---|---|---|
| `patchir-decomp` | `patchestry_ghidra`, `patchestry_ast`, `patchestry_codegen`, `patchestry_yaml` | deserialize exported firmware semantics, lift them, and emit outputs |
| `patchir-transform` | `patchestry_codegen`, `patchestry_passes`, `patchestry_yaml`, `MLIRContracts` | parse YAML, transform CIR, and preserve contract semantics |
| `patchir-yaml-parser` | `patchestry_yaml`, `patchestry_codegen`, `patchestry_passes` | validate specs against the same config and pass structures used by transform |
| `patchir-cir2llvm` | `MLIRContracts` | lower CIR while preserving contract metadata |
| `pcode-translate` | `patchestry_ghidra` | expose patchestry P-Code translation through the MLIR translation driver |
| `patchestry_ghidra` | `MLIRPcode` | build/consume patchestry's P-Code dialect layer for translation |
| `patchestry_passes` | `patchestry_yaml` | consume parsed patch/contract specs during instrumentation |

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
5. P-Code dialect stage:
   `pcode-translate` and `MLIRPcode` expose patchestry's owned representation of
   P-Code operations and types; document the dialect boundary, not MLIR's
   generic translation internals.

## Usage and Workflows

### Who uses which tools and why

| Tool | Primary user | Why it exists | Typical use point |
|---|---|---|---|
| `patchir-decomp` | reverse engineer / decomp developer | lift Ghidra exports into editable IR | first step after obtaining P-Code JSON |
| `patchir-transform` | patch author / verification engineer | apply firmware patches and contracts onto CIR | after decompilation, before lowering |
| `patchir-cir2llvm` | verification/binary pipeline engineer | emit LLVM IR/BC for downstream toolchains | after transform stage |
| `patchir-yaml-parser` | patch author / CI | validate patch specs early and fail fast | before transform in local loops and CI |
| `pcode-translate` | dialect/decomp developer | exercise and debug patchestry's P-Code translation boundary | when validating P-Code dialect behavior directly |

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

### Ghidra script workflow

The `scripts/ghidra/` tree is part of the project interface surface:

- `build-headless-docker.sh` builds the headless Ghidra environment used by tests.
- `decompile-headless.sh` and `decompile-entrypoint.sh` run repository-supported decomp flows.
- `PatchestryDecompileFunctions.java` and `PatchestryListFunctions.java` are the main script entrypoints.
- `scripts/ghidra/domain/` and `scripts/ghidra/util/` define the serialization boundary used to generate JSON consumed by patchestry tools.

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
- System data flow diagram: `docs/system_data_flow.md`
- Decompilation semantics and JSON/CFG structuring notes: `docs/system_data_flow.md`
- Claude-specific workflow notes: `.claude/rules/*.md`

## Test Suites

The repository currently registers four top-level LIT/CTest suites:

| Suite | Source path | What it validates |
|---|---|---|
| `ghidra-output-tests` | `test/ghidra/` | end-to-end headless Ghidra export and decompilation fixtures |
| `pcode-translation-tests` | `test/pcode-translate/` | standalone `pcode-translate` behavior and P-Code translation wiring |
| `patchir-decomp-tests` | `test/patchir-decomp/` | JSON-to-CIR/LLVM decompilation behavior across operations and control flow |
| `patchir-transform-tests` | `test/patchir-transform/` | YAML parsing, patch insertion/replacement, and contract workflows |

## Dependencies

### Required host tooling

| Dependency | Version | Notes |
|---|---|---|
| CMake | >= 3.25 | Configure/build presets |
| LLVM / Clang / MLIR | 22 | ClangIR-enabled build |
| lit | any recent | LLVM integrated test runner |
| Docker engine | current | Required for Ghidra headless and firmware flows |
| Ninja | recommended | Faster local builds |
| lld (Linux) | distro package | Required by Linux toolchain setup |

### Vendored dependencies

Vendored projects live under `vendor/` and are checked out via git submodules.
The current pinned revisions in this repository are:

| Dependency | Pinned revision | Location | Purpose |
|---|---|---|---|
| llvm-project | `patchir-llvmorg-22.1.4` | `vendor/llvm-project/src` | Patched LLVM/Clang/MLIR+CIR toolchain (trail-of-forks fork) when using vendored clang |
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

From a fresh checkout, initialize vendored sources before configuring or
building:

```sh
git submodule update --init --recursive
```

### macOS

Install base tooling and configure Docker BuildX for Colima:

```sh
xcode-select --install
brew install colima docker docker-buildx docker-credential-helper cmake lit
brew install FiloSottile/musl-cross/musl-cross
mkdir -p ~/.docker/cli-plugins
ln -sf "$(which docker-buildx)" ~/.docker/cli-plugins/docker-buildx
colima start --vm-type vz
docker buildx version
docker ps
```

Notes:

- Patchestry uses Colima as the Docker backend on macOS.
- Use the `vz` backend on Apple Silicon. Do not switch the documented macOS
  path to `qemu`.
- Do not treat `linux/amd64` emulation on Apple Silicon as the recommended
  macOS build path. It is materially slower and should only be used as a last
  resort when no native arm64 alternative is available.
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
git clone --branch patchir-llvmorg-22.1.4 https://github.com/trail-of-forks/llvm-project
cd llvm-project
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

This must be the patched `trail-of-forks/llvm-project` toolchain (or an
equivalent install built from the same fork). A stock Homebrew LLVM install is
not a supported replacement for host-native patchestry builds.

Expected runtime:

- Typical workstation: 30-90 minutes.
- Lower-memory systems or emulated containers: can exceed 2 hours.

If you are on Apple Silicon and this is too slow or space-constrained in Docker,
use `.devcontainer/README-HOST-BUILD.md` to build the ARM64 base image on host storage.
That workflow produces a Linux arm64 toolchain and image for container use; it
does not produce a host-native macOS LLVM/ClangIR install for the direct CMake
path below.

### Standard local configure/build

```sh
export LLVM_INSTALL_PREFIX=<llvm_install>
export CC="${LLVM_INSTALL_PREFIX}/bin/clang"
export CXX="${LLVM_INSTALL_PREFIX}/bin/clang++"
export CMAKE_PREFIX_PATH="${LLVM_INSTALL_PREFIX}/lib/cmake/llvm;${LLVM_INSTALL_PREFIX}/lib/cmake/mlir;${LLVM_INSTALL_PREFIX}/lib/cmake/clang"

cmake --fresh --preset default \
  -DLLVM_EXTERNAL_LIT=$(which lit)

cmake --build --preset debug -j
cmake --build --preset release -j
```

This is the supported host-native path on macOS and Linux when you already have
the required patched LLVM/ClangIR fork installed. On macOS, use this path only
when you have already built or installed the repository's ClangIR fork.
The verified macOS host-native path uses the fork's `clang`/`clang++` from
`<llvm_install>/bin`, not AppleClang or a stock Homebrew LLVM.

Notes:

- The main patchestry build vendors `gflags`, `glog`, and `z3` as part of
  configure.

### `build.sh` containerized workflow

`./build.sh` is the repository helper for containerized builds. It:

1. Builds or reuses a Docker image with dependencies.
2. Runs the build inside a controlled container.
3. Places artifacts under `builds/default`.

When to use it:

- Use `build.sh` when you want a containerized repository workflow.
- `build.sh` is required for Docker-backed validation paths and useful when host setup is incomplete.
- On Apple Silicon, do not recommend the default `linux/amd64` emulation path
  for routine builds. Prefer host-native builds with the patched ClangIR fork,
  or an arm64 container image when one is available.
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

# Run the example firmware end-to-end flow with artifact/report output
scripts/test-example-firmwares.sh --build-type Debug
```

### Fresh checkout to validated build

The validated Apple Silicon macOS path is the host-native flow below:

```sh
git submodule update --init --recursive

export LLVM_INSTALL_PREFIX=<llvm_install>
export CC="${LLVM_INSTALL_PREFIX}/bin/clang"
export CXX="${LLVM_INSTALL_PREFIX}/bin/clang++"
export CMAKE_PREFIX_PATH="${LLVM_INSTALL_PREFIX}/lib/cmake/llvm;${LLVM_INSTALL_PREFIX}/lib/cmake/mlir;${LLVM_INSTALL_PREFIX}/lib/cmake/clang"

cmake --fresh --preset default \
  -DLLVM_EXTERNAL_LIT=$(which lit)

cmake --build --preset debug -j

cmake -S lib/patchestry/intrinsics -B lib/patchestry/intrinsics/build_standalone \
  -DCMAKE_BUILD_TYPE=Release
cmake --build lib/patchestry/intrinsics/build_standalone -j

bash ./scripts/ghidra/build-headless-docker.sh

lit ./builds/default/test -D BUILD_TYPE=Debug -v
```

What this validates:

1. The main patchestry tools build with the same preset family used by CI.
2. The standalone intrinsics library still builds independently.
3. The headless Ghidra Docker image builds on Apple Silicon.
4. The full lit tree passes, including Ghidra, decompilation, P-Code
   translation, and transform/contract suites.

Docker-backed workflows remain relevant on macOS for `build.sh` and Ghidra
headless/container tasks, but do not present the default `linux/amd64`
emulation path as the routine Apple Silicon workflow.
The validated Ghidra image build on Apple Silicon used Colima with the `vz`
backend and built Ghidra natives for `linux_arm_64`.

### CI coherence

Local instructions should stay aligned with `.github/workflows/ci.yml`:

1. CI configures with `cmake --preset ci`.
2. CI builds with `cmake --build --preset ci --config <Debug|Release>`.
3. CI separately builds `lib/patchestry/intrinsics` as a standalone project.
4. CI builds the Ghidra headless Docker image.
5. CI validates the repository with `lit ./builds/ci/test ...`.

Local macOS instructions use `default` instead of `ci` because CI runs inside a
Linux dev image, but the sequence of configure -> build -> intrinsics build ->
Ghidra image build -> full lit test run should remain coherent.

### Test reliability expectations

- Tests are expected to be deterministic and non-flaky.
- A PR is not considered ready if repeated local runs produce inconsistent outcomes.
- New behavior changes should include a regression test in the closest relevant suite.
- If a change affects the documented example firmware flow, rerun `scripts/test-example-firmwares.sh` and update the example docs/reporting guidance in the same PR.

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
5. If changing a data-flow boundary, tool interface, or owned module interface, update `docs/system_data_flow.md` in the same PR.

Commit message format:

- `component: Simple sentence with a period.`
- Keep subject <= 80 characters.

## Gardening Tasks

These are recurring maintenance tasks that keep the repository documentation and
developer guidance aligned with the code:

1. Validate that `docs/system_data_flow.md` still matches the actual toolchain, test coverage, and repository-owned interfaces.
2. Update the module/interface inventory in this document when patchestry-owned libraries, tools, dialects, scripts, or test suites change.
3. Keep vendored dependency revisions and purposes current when submodules or integration boundaries change.
4. Ensure PRs that change affected interfaces or data-flow boundaries also update the corresponding diagram and docs in the same change.
5. Keep build and test instructions copy/paste valid from a fresh checkout, including submodule bootstrap and standalone intrinsics build.
6. Keep local build/test instructions coherent with `.github/workflows/ci.yml`; when CI stages change, update the docs in the same PR.
7. Keep `scripts/test-example-firmwares.sh` and `docs/GettingStarted/firmware_examples.md` aligned with the actual example firmware binaries, example specs, and generated reports.
