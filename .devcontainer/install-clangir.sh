#!/bin/bash

#set -e

CLANGIR_REPO="https://github.com/trail-of-forks/clangir.git"
CLANGIR_DIR="$HOME/clangir"
BUILD_DIR="$HOME/clangir-build"
INSTALL_DIR="/usr/local"

# Dependencies doesn't install llvm since it expects
# it to be installed in dev container and use the same 
# llvm version
install_dependencies() {
    echo "Installing dependencies..."
    sudo apt update
    sudo apt install -y \
        cmake \
        ninja-build \
        python3 \
        python3-pip \
        git \
        lld \
        clang
}

build_clangir() {    
    echo "Cloning Clangir repository..."
    git clone --branch patche-clangir-20 "$CLANGIR_REPO" "$CLANGIR_DIR"

    echo "Installing Clangir..."
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    CXXFLAGS="-g0" \
    CCFLAGS="-g0" \
    cmake -G Ninja \
        "$CLANGIR_DIR/llvm" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="clang;mlir" \
        -DBUILD_SHARED_LIBS:BOOL=OFF \
        -DLLVM_APPEND_VC_REV:BOOL=OFF \
        -DLLVM_BUILD_DOCS:BOOL=OFF \
        -DLLVM_INCLUDE_DOCS:BOOL=OFF \
        -DLLVM_INCLUDE_EXAMPLES:BOOL=OFF \
        -DLLVM_BUILD_EXAMPLES:BOOL=OFF \
        -DLLVM_BUILD_LLVM_DYLIB:BOOL=OFF \
        -DLLVM_BUILD_TESTS:BOOL=OFF \
        -DLLVM_ENABLE_BINDINGS:BOOL=OFF \
        -DLLVM_ENABLE_OCAMLDOC:BOOL=OFF \
        -DLLVM_ENABLE_DIA_SDK:BOOL=OFF \
        -DLLVM_ENABLE_EH:BOOL=ON \
        -DCLANG_ENABLE_CIR:BOOL=ON \
        -DLLVM_ENABLE_RTTI:BOOL=ON \
        -DLLVM_ENABLE_WARNINGS:BOOL=ON \
        -DLLVM_INCLUDE_BENCHMARKS:BOOL=OFF \
        -DLLVM_INCLUDE_EXAMPLES:BOOL=OFF \
        -DLLVM_INCLUDE_TESTS:BOOL=OFF \
        -DLLVM_INCLUDE_TOOLS:BOOL=ON \
        -DLLVM_INSTALL_UTILS:BOOL=ON \
        -DLLVM_ENABLE_ZSTD:BOOL=OFF \
        -DLLVM_TARGETS_TO_BUILD="AArch64;X86;ARM" \ \
        -DCMAKE_C_COMPILER=/usr/bin/clang \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++
    ninja install .
}

clean_up() {
    echo "Cleaning up..."
    rm -rf "$BUILD_DIR"
}

install_dependencies
build_clangir
clean_up

echo "Clangir has been successfully installed!"
