# Patchestry Developer Guide (Canonical)

This is the canonical onboarding and development guide for Patchestry.

- `AGENTS.md` is the source of truth for Codex/Cursor and human contributors.
- `CLAUDE.md` exists only as a compatibility entrypoint for Claude Code and points here.

## Purpose

Patchestry is an MLIR/CIR-based binary patching framework for patching deployed firmware
without original source code. The main workflow is:

1. Export P-Code from Ghidra.
2. Decompile to CIR.
3. Apply YAML-defined patches/contracts.
4. Lower to LLVM IR/bitcode for downstream rewriting and verification.

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
