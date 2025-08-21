#!/bin/bash

# Test script for building patchestry intrinsics library
# This script demonstrates different build configurations

set -e

echo "Building patchestry intrinsics library..."

# Clean previous builds
rm -rf build_*

# Build 1: Standalone build (default architecture)
mkdir -p build_standalone
pushd build_standalone
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc 2>/dev/null || echo 2)
popd

# Build 2: Standalone build for ARMv7 (with platform check)
mkdir -p build_arm
pushd build_arm

# Check if we're on macOS and warn about ARMv7 limitations
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Note: Running on macOS - ARM build will show warning about toolchain compatibility"
    exit 1
fi

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../arm-none-eabi.cmake
if make -j$(nproc 2>/dev/null || echo 2); then
    echo "ARM build completed successfully"
fi
popd
