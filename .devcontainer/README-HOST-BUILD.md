# Host-Storage ARM64 Dev Container Build

On Apple Silicon Macs, building the LLVM/ClangIR toolchain inside a Docker
container can be slow due to emulation overhead and constrained disk I/O.
This guide builds the ARM64 Linux toolchain on the host filesystem, then
packages it into a container image for use with the devcontainer workflow.

> **Note:** This produces a *Linux arm64* toolchain for container use.
> It does **not** produce a host-native macOS ClangIR install.  For a
> native macOS build, follow the instructions in
> `docs/GettingStarted/build.md`.

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4) or Linux arm64
- Docker CLI + BuildKit
- At least 30 GB of free disk space

**macOS only:** Install Docker via Homebrew and start Colima:

```sh
brew install docker docker-buildx colima
colima start --vm-type vz
```

On Linux arm64, Docker runs natively — no Colima needed.

## Step 1: Build the base image on host storage

Build the devcontainer image directly (no emulation — native arm64):

```sh
cd .devcontainer
DOCKER_BUILDKIT=1 docker build \
  --platform linux/arm64 \
  -t patchestry-dev:local \
  -f Dockerfile \
  .
```

This runs `install-clangir.sh` inside the container, which clones the
`trail-of-forks/clangir` fork (branch `patche-clangir-20`) and builds
LLVM/Clang/MLIR with CIR support.  The toolchain is installed to
`/usr/local` inside the image.

## Step 2: Verify the image

```sh
docker run --rm patchestry-dev:local /usr/local/bin/clang --version
# Should show clang version 20.x with CIR support

docker run --rm patchestry-dev:local /usr/local/bin/lit --version
# Should show lit 20.x
```

## Step 3: Use with VS Code devcontainer

Update `.devcontainer/devcontainer.json` to reference your local image
instead of building from the Dockerfile:

```jsonc
{
  "name": "C++",
  // Use the pre-built local image instead of building from Dockerfile
  "image": "patchestry-dev:local",
  // ... rest of config unchanged
}
```

Then reopen the workspace in the container (`Dev Containers: Reopen in Container`).

## Step 4: Configure and build patchestry

Inside the container:

```sh
git submodule update --init --recursive

cmake --fresh --preset default \
  -DLLVM_EXTERNAL_LIT=/usr/local/bin/lit

cmake --build --preset debug -j
cmake --build --preset release -j
```

## Rebuilding after ClangIR updates

If the `trail-of-forks/clangir` fork is updated, rebuild the image:

```sh
cd .devcontainer
DOCKER_BUILDKIT=1 docker build \
  --no-cache \
  --platform linux/arm64 \
  -t patchestry-dev:local \
  -f Dockerfile \
  .
```

The `--no-cache` flag ensures the ClangIR clone and build are re-run
from scratch.
