
# Building
We mostly rely on a build container, but some dependencies are still needed outside that container: our fork of [LLVM20](https://github.com/trail-of-forks/clangir), a local copy of `lld`, and LLVM [LIT](https://llvm.org/docs/CommandGuide/lit.html).

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
   colima restart
   docker buildx version
   ```

4. Log into Docker Hub:
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


6. Build with:
```
CC=$(which clang) CXX=$(which clang++) cmake \
   --preset default \
   -DCMAKE_PREFIX_PATH=<path_to_llvm_install>/lib/cmake/ \
   -DLLVM_EXTERNAL_LIT=$(which lit)
```

This setup provides a complete development environment for building and running the project on MacOS. The configuration uses Colima as a Docker backend, which provides better performance and resource management compared to Docker Desktop on MacOS.

# First Time Development Setup: Linux
If you'd like to either follow step by step instructions or run a script to automatically follow them in a fresh Linux instance, here's a [Gist](https://gist.github.com/kaoudis/e734c6197dbed595586ab659844df737) that sets everything up from zero in a fresh VM for you and runs the Patchestry tests to confirm the setup works. This Gist should stay reasonably up to date since it's used to initialize ephemeral coding environments. It's been tested on Ubuntu 24.04. The only thing that should be different for other Ubuntus or for Debian is the `apt` package naming.

Steps followed in the [Gist](https://gist.github.com/kaoudis/e734c6197dbed595586ab659844df737) to get to a working install:
1. Base dependency install (Docker, lld, build tools such as CMake)
2. Acquire LLVM LIT from Python Pip
3. Build and install LLVM
4. Build and install Patchestry
5. Build the headless container in the Patchestry repository (this should set up and install Ghidra and everything else)
6. Test Patchestry, which requires the container

# Development 

## CMake Commands
- to build, see the command referenced in step 6 [above](#first-time-development-setup-macos) or the commands used for [Linux](#first-time-development-setup-linux). You'll use the `default` preset to configure and most likely the `debug` or `release` presets for the subsequent build command after configuration.
- to run tests, ensure the headless container is available first by running `scripts/ghidra/build-headless-docker.sh`, then you may `cmake --build builds/default/ -j$((`nproc`+1)) --preset debug --target test` (using the preset of your choice but selecting the `test` target)

## Ghidra

### Installing Ghidra Scripts
Link `ghidra_scripts` directory to `$HOME`. We assume that `./patchestry` contains the cloned repository.
```shell
ln -s patchestry/ghidra_scripts ~
```

### Installing Ghidra Locally
You shouldn't need to do this directly in the current build *most of the time*. You may need Ghidra locally for Ghidra script debugging.

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

## Old Stuff

Here are old sections of documentation retained for historical reference. Please follow the above instructions instead. 

### Dev Container
You shouldn't need to do this directly with the current build.

```sh
sudo bash .devcontainer/reinstall-cmake.sh "3.29.2"
sudo bash .devcontainer/install-llvm.sh "18" all
```

With dependecies installed, we can build in `build/` in the project root.
```sh
# Assuming `/usr/lib/llvm-18/` contains an llvm-18 installation.
cmake --preset default -DCMAKE_PREFIX_PATH=/usr/lib/llvm-18/lib/cmake/
# Valid user presets are `debug`, `release`, `relwithdebinfo`.
cmake --build --preset=debug
```

### Building Patchestry

To configure project run `cmake` with following default options.
In case `clang` isn't your default compiler prefix the command with `CC=clang CXX=clang++`.
If you want to use system installed `llvm` and `mlir` (on Ubuntu) use:

The simplest way is to run

```
cmake --workflow release
```

If this method doesn't work for you, configure the project in the usual way:

```
cmake --preset default
```

To use a specific `llvm` provide `-DCMAKE_PREFIX_PATH=<llvm & mlir instalation paths>` option, where `CMAKE_PREFIX_PATH` points to directory containing `LLVMConfig.cmake` and `MLIRConfig.cmake`.

Note: Patchestry requires LLVM with RTTI enabled. Use `LLVM_ENABLE_RTTI=ON` if you build your own LLVM.


Finally, build the project:

```
cmake --build --preset release
```

Use `debug` preset for debug build.

## Test

```
ctest --preset debug
```