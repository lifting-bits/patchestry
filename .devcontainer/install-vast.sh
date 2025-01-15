#!/bin/bash

#set -e

GAP_REPO="https://github.com/lifting-bits/gap.git"
GAP_BUILD="$HOME/gap-build"

VAST_REPO="https://github.com/trailofbits/vast.git"
BUILD_DIR="$HOME/vast-build"
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
        doctest-dev \
        libspdlog-dev
}

install_gap() {
    echo "Cloning Gap repository..."
    git clone --recursive "$GAP_REPO" "$GAP_BUILD"
    cd "$GAP_BUILD"
    mkdir -p build && cd build
    cmake -G Ninja .. -DGAP_ENABLE_MLIR=ON -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DMLIR_DIR="/usr/lib/llvm-19/lib/cmake/mlir"
    ninja && sudo ninja install
}

build_vast() {    
    echo "Cloning VAST repository..."
    git clone --branch  v0.0.66 --recursive "$VAST_REPO" "$BUILD_DIR"

    echo "Installing VAST..."
    cd "$BUILD_DIR" && mkdir -p build && cd build
    cmake -G Ninja .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_LINKER_TYPE=LLD -DMLIR_DIR="/usr/lib/llvm-19/lib/cmake/mlir"
    cmake --build . && sudo cmake --install .
}

clean_up() {
    echo "Cleaning up..."
    rm -rf "$BUILD_DIR/build"
}

install_dependencies
install_gap
build_vast
clean_up

echo "VAST has been successfully installed!"
