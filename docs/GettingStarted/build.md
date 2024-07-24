
## Dependencies

| Name | Version |
| ---- | ------- |
| [CMake](https://cmake.org/) | >= 3.25.0 |
| [LLVM](http://llvm.org/) | == 18 |

Currently, it is necessary to use `clang` (due to `gcc` bug) to build Patchestry, because of `patchestry`. On Linux it is also necessary to use `lld` at the moment.

Patchestry uses `llvm-18` which can be obtained from the [repository](https://apt.llvm.org/) provided by LLVM.

Before building (for Ubuntu) get all the necessary dependencies by running
```
apt-get install build-essential cmake ninja-build libstdc++-12-dev llvm-18 libmlir-18 libmlir-18-dev mlir-18-tools libclang-18-dev
```
or an equivalent command for your operating system of choice.

## Ubuntu 22.04

Optionally (re-)install dependecies from their respective official sources.

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

# Getting Ghidra

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

# Installing Ghidra Scripts

Link `ghidra_scripts` directory to `$HOME`. We assume that `./patchestry` contains the cloned repository.
```shell
ln -s patchestry/ghidra_scripts ~
```


## Instructions

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
