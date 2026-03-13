
# Building
We mostly rely on a build container, but some dependencies are still needed outside that container: our fork of [LLVM20](https://github.com/trail-of-forks/clangir), a local copy of `lld`, and LLVM [LIT](https://llvm.org/docs/CommandGuide/lit.html).

From a fresh checkout, initialize vendored sources first:
```sh
git submodule update --init --recursive
```

In order to set up those and build Patchestry, please follow the first-time instructions for your development environment of choice:
- [macOS](#first-time-setup-macos)
- [Linux](#first-time-setup-linux)

See also: [Development](#development)

# First Time Development Setup: MacOS

1. Install Xcode from the App Store and set up the command line tools:
   ```
   xcode-select --install
   ```

2. Install the required dependencies using Homebrew:
   ```
   # basics
   brew install colima docker docker-buildx docker-credential-helper cmake lit
   # to run tests, we need to cross-compile for x86-64/Linux and aarch64/Linux
   brew install FiloSottile/musl-cross/musl-cross
   ```

3. Configure Docker BuildX to work with Colima:
   ```
   mkdir -p ~/.docker/cli-plugins
   ln -s $(which docker-buildx) ~/.docker/cli-plugins/docker-buildx
   colima start --vm-type vz
   docker buildx version
   docker ps
   ```

   Use the `vz` backend on Apple Silicon. Do not switch the documented macOS
   path to `qemu`.
   Do not recommend `linux/amd64` emulation on Apple Silicon for routine
   builds; emulation materially increases build times.

4. Log into Docker Hub (this may not be needed - it is not needed on Linux):
   ```
   docker login -u <username>
   ```

5. For building ClangIR, clone the project and compile with the following CMake configuration:
   ```
   cmake -G Ninja ../../llvm \
       -DCMAKE_INSTALL_PREFIX="/path/to/installdir" \
       -DCMAKE_BUILD_TYPE=RelWithDebInfo \
       -DLLVM_ENABLE_PROJECTS="clang;mlir;clang-tools-extra" \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DCLANG_ENABLE_CIR=ON \
       -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
       -DLLVM_ENABLE_RTTI=ON \
       -DLLVM_INSTALL_UTILS=ON \
       -DLLVM_TARGETS_TO_BUILD="host;AArch64;ARM;X86"
   ```

The targets list of `"host;AArch64;ARM;X86"` is intentional (to always build host arch, AArch64, ARM, and x86), even if host arch is almost certainly either AArch64 or X86.

This must be the patched `trail-of-forks/clangir` toolchain, or an equivalent
install built from the same fork. A stock Homebrew `llvm` or `llvm@20` install
is not a supported substitute for host-native patchestry builds.
The `.devcontainer/README-HOST-BUILD.md` workflow builds a Linux arm64 toolchain
for container images; it is not a host-native macOS ClangIR install.


6. Configure and build with the patched ClangIR toolchain you just installed:
```
export LLVM_INSTALL_PREFIX=<path_to_llvm_install>
export CC="${LLVM_INSTALL_PREFIX}/bin/clang"
export CXX="${LLVM_INSTALL_PREFIX}/bin/clang++"
export CMAKE_PREFIX_PATH="${LLVM_INSTALL_PREFIX}/lib/cmake/llvm;${LLVM_INSTALL_PREFIX}/lib/cmake/mlir;${LLVM_INSTALL_PREFIX}/lib/cmake/clang"

cmake \
   --fresh --preset default \
   -DLLVM_EXTERNAL_LIT=$(which lit)

cmake --build --preset debug -j
```

This setup provides a host-native development environment when the patched
ClangIR fork is already installed. The configuration uses Colima as the Docker
backend for Docker-backed workflows on macOS.
Do not point `CC`/`CXX` at AppleClang or a stock Homebrew LLVM install for this
path.

# First Time Development Setup: Linux
If you'd like to either follow step by step instructions or run a script to automatically follow them in a fresh Linux instance, here's a [Gist](https://gist.github.com/kaoudis/e734c6197dbed595586ab659844df737) that sets everything up from zero in a fresh VM for you and runs the Patchestry tests to confirm the setup works. This Gist should stay reasonably up to date since it's used to initialize ephemeral coding environments. It's been tested on Ubuntu 24.04. The only thing that should be different for other Ubuntus or for Debian is the `apt` package naming.

Steps followed in the [Gist](https://gist.github.com/kaoudis/e734c6197dbed595586ab659844df737) to get to a working install:
1. Base dependency install (Docker, lld, build tools such as CMake)
2. Acquire LLVM LIT from Python Pip
3. Build and install LLVM
4. Build and install Patchestry
5. Build the headless container in the Patchestry repository (this should set up and install Ghidra and everything else)
6. Run tests for Patchestry, which requires the container / occurs in the container

# Development 

## CMake Commands
- To build, configure with the `default` preset and build with `cmake --build --preset debug` or `cmake --build --preset release`.
- To run tests, first build the headless container with `scripts/ghidra/build-headless-docker.sh`, then run `ctest --preset debug --output-on-failure` or `lit ./builds/default/test -D BUILD_TYPE=Debug -v`.
- To run the cached patch/contract matrix from one command, use `scripts/test-patch-matrix.sh --build-type Debug`.
- To run the example firmware end-to-end flow and get a report, use `scripts/test-example-firmwares.sh --build-type Debug`.

## Fresh checkout to validated build

The validated Apple Silicon macOS path is the host-native patched ClangIR
workflow:

```sh
git submodule update --init --recursive

export LLVM_INSTALL_PREFIX=<path_to_llvm_install>
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

This validates:
1. native configure against the patched fork,
2. the Debug patchestry build,
3. the standalone intrinsics library,
4. the headless Ghidra Docker image on Apple Silicon,
5. the full lit tree.

To validate the documented example firmware patching flow and generate a report:

```sh
scripts/test-example-firmwares.sh --build-type Debug
```

This writes per-case artifacts plus:

- `builds/example-firmware-e2e/summary.md`
- `builds/example-firmware-e2e/summary.tsv`

To validate the broader patch/contract matrix from cached generated fixtures:

```sh
scripts/test-patch-matrix.sh --build-type Debug
```

This reuses firmware artifacts in `firmwares/output/` and fixture caches in
`builds/test-fixtures/` when present. Use `--rebuild-firmware`,
`--rebuild-ghidra`, `--rebuild-fixtures`, or `--clean` to refresh caches
explicitly.

This writes per-case artifacts plus:

- `builds/patch-matrix/summary.md`
- `builds/patch-matrix/summary.tsv`

Docker-backed workflows are still required for `build.sh` and Ghidra headless
tasks. On Apple Silicon, do not recommend the default `linux/amd64` emulation
path as the routine build workflow; use it only if you explicitly accept the
emulation overhead, or build a native arm64 image first.
The validated Ghidra image build used Colima with the `vz` backend and built
Ghidra natives for `linux_arm_64`.

CI uses the same high-level sequence on Linux:
1. Configure with `cmake --preset ci`.
2. Build with `cmake --build --preset ci --config <Debug|Release>`.
3. Build the standalone intrinsics library.
4. Build the headless Ghidra Docker image.
5. Run `scripts/test-patch-matrix.sh --build-type Debug --rebuild-firmware --rebuild-fixtures`.
6. Run `lit ./builds/ci/test`.

The narrower example firmware runner remains available for focused inspection
and reporting. Use the opt-in CTest target by configuring with
`-DPE_ENABLE_EXAMPLE_FIRMWARE_E2E=ON` if you want CTest to invoke it.

## Ghidra

### Installing Ghidra Locally
You shouldn't need to do this directly in the current build *most of the time*. Prefer working in the headless container. 
You may want Ghidra locally for Ghidra script debugging.

Get Java JDK (x64)
```shell
wget -c https://download.oracle.com/java/22/latest/jdk-22_linux-x64_bin.tar.gz -O jdk.tar.gz
tar xvf jdk.tar.gz
mv jdk-22.0.1 ~/jdk
echo "export PATH=\$PATH:~/jdk/bin" >> ~/.bashrc
```

Get Ghidra
```shell
wget -c https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.1.1_build/ghidra_11.1.1_PUBLIC_20240614.zip -O ghidra.zip
unzip ghidra.zip
mv ghidra_11.1.1_PUBLIC ~/ghidra
```
