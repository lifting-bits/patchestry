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
}

build_clangir() {    
    echo "Cloning Clangir repository..."
    git clone --branch patche-clangir-20 --recursive "$CLANGIR_REPO" "$CLANGIR_DIR"

    echo "Installing Clangir..."
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    cmake -G Ninja $CLANGIR_DIR \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="clang;mlir" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCLANG_ENABLE_CIR=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DCMAKE_C_COMPILER=/usr/bin/clang \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++
    cmake --build . && sudo cmake --install .
}

clean_up() {
    echo "Cleaning up..."
    rm -rf "$BUILD_DIR"
}

install_dependencies
build_clangir
clean_up

echo "Clangir has been successfully installed!"
