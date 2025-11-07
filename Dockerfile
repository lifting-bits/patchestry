# Copyright (c) 2025, Trail of Bits, Inc.
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

# Build arguments for image version and LLVM version
ARG IMAGE_VERSION=22.04
ARG LLVM_VERSION=20

# Pull the base dev container image from GitHub Container Registry
FROM ghcr.io/lifting-bits/patchestry-ubuntu-${IMAGE_VERSION}-llvm-${LLVM_VERSION}-dev:latest

# Set build arguments as environment variables
ARG LLVM_VERSION
ENV LLVM_VERSION=${LLVM_VERSION}

# Set CMake configuration
ENV CMAKE_PREFIX_PATH="/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/mlir/;/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/clang/"
ENV LLVM_EXTERNAL_LIT="/usr/local/bin/lit"

# Set working directory
WORKDIR /workspace/patchestry

# Set the default command to bash
CMD ["/bin/bash"]
