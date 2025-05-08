#!/bin/bash

set -e

Z3_REPO="https://github.com/Z3Prover/z3.git"
Z3_DIR="$HOME/z3"
Z3_BUILD_DIR="$HOME/z3-build"
Z3_INSTALL_DIR="/usr/local"

install_dependencies() {
    echo "Installing dependencies..."
    sudo apt update
    sudo apt install -y \
        build-essential \
        g++ \
        cmake \
        ninja-build \
        python3 \
        python3-pip \
        git
}

build_z3() {    
    echo "Cloning z3 repository..."
    git clone --branch z3-4.14.0 "$Z3_REPO" "$Z3_DIR"

    echo "Installing Z3..."
    mkdir -p "$Z3_BUILD_DIR" && cd "$Z3_BUILD_DIR"
    cmake -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$Z3_INSTALL_DIR" \
      -DZ3_BUILD_PYTHON_BINDINGS=OFF \
      -DZ3_BUILD_EXAMPLES=OFF \
      -DZ3_USE_LIB_GMP=OFF \
      $Z3_DIR
    ninja install .
}

clean_up() {
    echo "Cleaning up..."
    rm -rf "$Z3_BUILD_DIR"
}


install_dependencies
build_z3
clean_up

echo "Z3 has been successfully installed!"