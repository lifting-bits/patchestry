#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${script_dir}/output"

# Build arguments
IMAGE_VERSION="${IMAGE_VERSION:-22.04}"
LLVM_VERSION="${LLVM_VERSION:-20}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

# Get Docker memory in bytes and convert to GB
DOCKER_MEMORY_GB=16
DOCKER_CPUS=2

echo "Building patchestry Docker image..."
docker build \
    --build-arg IMAGE_VERSION="${IMAGE_VERSION}" \
    --build-arg LLVM_VERSION="${LLVM_VERSION}" \
    -t patchestry-builder \
    -f "${script_dir}/Dockerfile" \
    "${script_dir}"

echo "Building patchestry inside Docker container..."
docker run --rm --platform linux/amd64 \
    --memory="${DOCKER_MEMORY_GB}g" \
    --memory-swap="${DOCKER_MEMORY_GB}g" \
    --cpus="${DOCKER_CPUS}" \
    -v "${script_dir}:/workspace/patchestry" \
    -w /workspace/patchestry \
    patchestry-builder \
    /bin/bash -c "set -e && \
        git config --global --add safe.directory '*' && \
        JOBS=1 && \
        echo 'Cleanup previous build directory...' && \
        rm -rf builds && \
        echo 'Configuring patchestry...' && \
        export CMAKE_BUILD_PARALLEL_LEVEL=1 && \
        export FETCHCONTENT_QUIET=OFF && \
        cmake --preset default \
            -DPE_USE_VENDORED_Z3=OFF \
	    -DPATCHESTRY_INSTALL=ON \
            -DLLVM_EXTERNAL_LIT=\$(which lit) \
            -DLLVM_Z3_INSTALL_DIR=/usr/local && \
        echo 'Building patchestry...' && \
        cmake --build --preset release  -j\$JOBS && \
        echo 'Build complete!'"

echo ""
echo "Build complete! Artifacts are available in:"
echo "  - Binaries: ${script_dir}/builds/default"
echo ""
echo "Built tools:"
ls -lh "${script_dir}/builds/default/" 2>/dev/null | grep -E "patchir" || echo "  (Build directory not found yet)"
