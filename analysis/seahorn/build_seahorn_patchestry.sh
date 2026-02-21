#!/usr/bin/env bash
#
# Build SeaHorn Docker Image on Patchestry Dev Container
#
# This script builds SeaHorn on top of the patchestry-ubuntu-22.04-llvm-20-dev
# base image which already has LLVM 20 installed, avoiding the need to build
# or install LLVM separately.
#

set -euo pipefail

# Configuration
PATCHESTRY_BASE_IMAGE="ghcr.io/lifting-bits/patchestry-ubuntu-22.04-llvm-20-dev:latest"
SEAHORN_REPO="https://github.com/trail-of-forks/seahorn.git"
SEAHORN_BRANCH="dev20"
CLONE_DIR="seahorn-repo"
IMAGE_NAME="ghcr.io/lifting-bits/patchestry-seahorn"
IMAGE_TAG="ubuntu-22.04-llvm-20"
BUILD_TYPE="RelWithDebInfo"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check architecture - only build on amd64
ARCH="$(uname -m)"
if [[ "${ARCH}" != "x86_64" ]]; then
    echo "Error: This script only builds on amd64/x86_64 systems"
    echo ""
    echo "Current architecture: ${ARCH}"
    echo ""
    echo "Building for amd64 on Apple Silicon will be slower due to emulation"
    echo "and is disabled."
    echo ""
    echo "Please run this script on an amd64/x86_64 system."
    exit 1
fi

echo "=== SeaHorn on Patchestry Docker Build Script ==="
echo "Base Image: ${PATCHESTRY_BASE_IMAGE}"
echo "SeaHorn Repository: ${SEAHORN_REPO}"
echo "SeaHorn Branch: ${SEAHORN_BRANCH}"
echo "Output Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Platform: linux/amd64 (native)"
echo ""

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

# Pull the patchestry base image
echo "Pulling patchestry base image..."
docker pull "${PATCHESTRY_BASE_IMAGE}"

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to pull base image ${PATCHESTRY_BASE_IMAGE}"
    exit 1
fi
echo "✓ Base image pulled successfully"
echo ""

# Change to script directory
cd "${SCRIPT_DIR}"

# Clean up existing clone if requested
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning up existing repository clone..."
    rm -rf "${CLONE_DIR}"
    echo "✓ Cleanup completed"
fi

# Clone repository if not already present
if [[ ! -d "${CLONE_DIR}" ]]; then
    echo "Cloning SeaHorn repository..."
    git clone "${SEAHORN_REPO}" "${CLONE_DIR}"
else
    echo "Repository already cloned. Use --clean to re-clone."
fi

# Navigate to repository
cd "${CLONE_DIR}"

# Fetch latest changes and checkout dev20 branch
echo "Checking out ${SEAHORN_BRANCH} branch..."
git fetch origin
git checkout "${SEAHORN_BRANCH}"
git pull origin "${SEAHORN_BRANCH}"

# Clone llvm-seahorn if not present
if [[ ! -d "llvm-seahorn" ]]; then
    echo "Cloning llvm-seahorn repository from trail-of-forks..."
    git clone https://github.com/trail-of-forks/llvm-seahorn.git llvm-seahorn
else
    echo "llvm-seahorn already cloned."
fi

# Checkout dev20 branch for llvm-seahorn
echo "Checking out ${SEAHORN_BRANCH} branch for llvm-seahorn..."
cd llvm-seahorn
git fetch origin
git checkout "${SEAHORN_BRANCH}" || echo "Warning: ${SEAHORN_BRANCH} branch not found in llvm-seahorn, using default branch"
git pull origin "${SEAHORN_BRANCH}" 2>/dev/null || echo "Using current branch state"
cd ..

# Clone crab if not present
if [[ ! -d "crab" ]]; then
    echo "Cloning crab repository from trail-of-forks..."
    git clone https://github.com/trail-of-forks/crab.git crab
else
    echo "crab already cloned."
fi

# Checkout dev20 branch for crab
echo "Checking out ${SEAHORN_BRANCH} branch for crab..."
cd crab
git fetch origin
git checkout "${SEAHORN_BRANCH}" || echo "Warning: ${SEAHORN_BRANCH} branch not found in crab, using default branch"
git pull origin "${SEAHORN_BRANCH}" 2>/dev/null || echo "Using current branch state"
cd ..

# Clone sea-dsa if not present
if [[ ! -d "sea-dsa" ]]; then
    echo "Cloning sea-dsa repository from trail-of-forks..."
    git clone https://github.com/trail-of-forks/sea-dsa.git sea-dsa
else
    echo "sea-dsa already cloned."
fi

# Checkout dev20 branch for sea-dsa
echo "Checking out ${SEAHORN_BRANCH} branch for sea-dsa..."
cd sea-dsa
git fetch origin
git checkout "${SEAHORN_BRANCH}" || echo "Warning: ${SEAHORN_BRANCH} branch not found in sea-dsa, using default branch"
git pull origin "${SEAHORN_BRANCH}" 2>/dev/null || echo "Using current branch state"
cd ..

# Get the current commit hash for tagging
COMMIT_HASH=$(git rev-parse --short HEAD)
echo "Current commit: ${COMMIT_HASH}"
echo ""

# Create builder Dockerfile
cat > patchestry-seahorn-builder.Dockerfile <<'EOF'
#
# SeaHorn Builder Dockerfile for Patchestry
# Builds SeaHorn on top of patchestry-ubuntu-22.04-llvm-20-dev base image
#
ARG BASE_IMAGE=ghcr.io/lifting-bits/patchestry-ubuntu-22.04-llvm-20-dev:latest
FROM ${BASE_IMAGE}

# Install SeaHorn build dependencies
# LLVM 20 is already installed in the base image
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
  apt-get install -yqq \
    cmake cmake-data ninja-build \
    zlib1g-dev libzstd-dev \
    libgraphviz-dev python3-pygraphviz graphviz \
    libgmp-dev libmpfr-dev \
    libboost1.74-dev \
    python3-pip \
    less vim sudo \
    lcov gcovr rsync \
    unzip wget curl && \
  # Install gcc-multilib only on x86_64/amd64 architecture
  if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
    apt-get install -yqq gcc-multilib; \
  fi && \
  pip3 install lit OutputCheck networkx && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install z3 v4.8.9 (required version for SeaHorn)
WORKDIR /tmp
RUN wget https://github.com/Z3Prover/z3/releases/download/z3-4.8.9/z3-4.8.9-x64-ubuntu-16.04.zip && \
  unzip z3-4.8.9-x64-ubuntu-16.04.zip && \
  mv z3-4.8.9-x64-ubuntu-16.04 /opt/z3-4.8.9 && \
  rm z3-4.8.9-x64-ubuntu-16.04.zip

# Install yices 2.6.1
RUN curl -sSOL https://yices.csl.sri.com/releases/2.6.1/yices-2.6.1-x86_64-pc-linux-gnu-static-gmp.tar.gz && \
  tar xf yices-2.6.1-x86_64-pc-linux-gnu-static-gmp.tar.gz && \
  cd /tmp/yices-2.6.1/ && \
  ./install-yices /opt/yices-2.6.1 && \
  rm -rf /tmp/yices-2.6.1*

# Copy SeaHorn source
RUN mkdir -p /seahorn
COPY . /seahorn/

# Build SeaHorn
WORKDIR /seahorn
RUN rm -rf build debug release clam sea-dsa llvm-seahorn && \
  mkdir build

WORKDIR /seahorn/build

ARG BUILD_TYPE=RelWithDebInfo

# Configure and build SeaHorn
# NOTE: Using INSTALL_PREFIX=run (relative path) so binaries go to /seahorn/build/run/bin/
RUN cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DZ3_ROOT=/opt/z3-4.8.9 \
  -DYICES2_HOME=/opt/yices-2.6.1 \
  -DCMAKE_INSTALL_PREFIX=run \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DSEA_ENABLE_LLD=ON \
  -DLLVM_DIR=/usr/local/lib/cmake/llvm \
  -DCPACK_GENERATOR="TGZ" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
  cmake --build . --target extra && cmake .. && \
  cmake --build . --target crab && cmake .. && \
  cmake --build . --target install && \
  cmake --build . --target units_z3 && \
  cmake --build . --target units_yices2 && \
  cmake --build . --target test_type_checker && \
  cmake --build . --target test_hex_dump && \
  cmake --build . --target package && \
  units/units_z3 && \
  units/units_yices2 && \
  units/units_type_checker

WORKDIR /seahorn
EOF

# Create runtime Dockerfile
cat > patchestry-seahorn.Dockerfile <<'EOF'
#
# SeaHorn Runtime Dockerfile for Patchestry
# Minimal image with SeaHorn binaries and runtime dependencies
#
ARG BASE_IMAGE=ghcr.io/lifting-bits/patchestry-ubuntu-22.04-llvm-20-dev:latest
ARG BUILDER_IMAGE=ghcr.io/lifting-bits/patchestry-seahorn-builder:ubuntu-22.04-llvm-20
FROM ${BASE_IMAGE}

# Install runtime dependencies only
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
  apt-get install -yqq \
    libgraphviz-dev python3-pygraphviz graphviz \
    libgmp-dev libmpfr-dev \
    libboost1.74-dev \
    python3-pip \
    less vim sudo && \
  pip3 install lit OutputCheck networkx && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install z3 v4.8.9
WORKDIR /tmp
RUN wget https://github.com/Z3Prover/z3/releases/download/z3-4.8.9/z3-4.8.9-x64-ubuntu-16.04.zip && \
  unzip z3-4.8.9-x64-ubuntu-16.04.zip && \
  mv z3-4.8.9-x64-ubuntu-16.04 /opt/z3-4.8.9 && \
  rm z3-4.8.9-x64-ubuntu-16.04.zip

# Install yices 2.6.1
RUN curl -sSOL https://yices.csl.sri.com/releases/2.6.1/yices-2.6.1-x86_64-pc-linux-gnu-static-gmp.tar.gz && \
  tar xf yices-2.6.1-x86_64-pc-linux-gnu-static-gmp.tar.gz && \
  cd /tmp/yices-2.6.1/ && \
  ./install-yices /opt/yices-2.6.1 && \
  rm -rf /tmp/yices-2.6.1*

# Copy SeaHorn package from build context (extracted from builder image)
COPY SeaHorn-*.tar.gz /tmp/
RUN mkdir -p /opt/seahorn && \
  tar xf /tmp/SeaHorn-*.tar.gz -C /opt/seahorn --strip-components=1 && \
  rm -rf /tmp/SeaHorn-*.tar.gz

# Set up environment
ENV PATH="/opt/seahorn/bin:$PATH"
ENV Z3_ROOT="/opt/z3-4.8.9"
ENV YICES2_HOME="/opt/yices-2.6.1"

WORKDIR /work

CMD ["/bin/bash"]
EOF

echo "Building SeaHorn Docker images..."
echo "This may take 20-40 minutes..."
echo ""

# Step 1: Build the builder image
echo "Step 1/3: Building seahorn-builder image..."
BUILDER_IMAGE_NAME="${IMAGE_NAME}-builder"
docker build \
    --build-arg BASE_IMAGE="${PATCHESTRY_BASE_IMAGE}" \
    -t "${BUILDER_IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${BUILDER_IMAGE_NAME}:${IMAGE_TAG}-${COMMIT_HASH}" \
    -t "${BUILDER_IMAGE_NAME}:latest" \
    -f patchestry-seahorn-builder.Dockerfile \
    .

if [[ $? -ne 0 ]]; then
    echo "✗ Builder image build failed"
    exit 1
fi
echo "✓ Builder image created successfully"
echo ""

# Step 2: Extract SeaHorn package from builder image
echo "Step 2/3: Extracting SeaHorn package from builder image..."
CONTAINER_ID=$(docker create "${BUILDER_IMAGE_NAME}:${IMAGE_TAG}")
docker cp "${CONTAINER_ID}:/seahorn/build/SeaHorn-20.${BUILD_TYPE}.tar.gz" . 2>/dev/null || \
  docker cp "${CONTAINER_ID}:/seahorn/build/" - | tar xf - --wildcards "build/SeaHorn-*.tar.gz" --strip-components=1

if [[ $? -ne 0 ]]; then
    echo "✗ Failed to extract SeaHorn package from builder"
    docker rm "${CONTAINER_ID}"
    exit 1
fi
docker rm "${CONTAINER_ID}"
echo "✓ SeaHorn package extracted successfully"
echo ""

# Step 3: Build the runtime image
echo "Step 3/3: Building seahorn runtime image..."
docker build \
    --build-arg BASE_IMAGE="${PATCHESTRY_BASE_IMAGE}" \
    --build-arg BUILDER_IMAGE="${BUILDER_IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}-${COMMIT_HASH}" \
    -t "${IMAGE_NAME}:latest" \
    -f patchestry-seahorn.Dockerfile \
    .

# Check if build was successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=== Build completed successfully! ==="
    echo ""
    echo "Images created:"
    echo "  Builder: ${BUILDER_IMAGE_NAME}:${IMAGE_TAG}"
    echo "  Runtime: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "  Runtime: ${IMAGE_NAME}:${IMAGE_TAG}-${COMMIT_HASH}"
    echo "  Runtime: ${IMAGE_NAME}:latest"
    echo ""
    echo "To run the image:"
    echo "  docker run -it ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To verify SeaHorn is working:"
    echo "  docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} sea --version"
    echo ""
    echo "To use with existing scripts:"
    echo "  SEAHORN_IMAGE=${IMAGE_NAME}:${IMAGE_TAG} ./sea_verify.sh input.ll"
else
    echo "✗ Runtime image build failed"
    exit 1
fi

# Display image size
echo ""
echo "Image information:"
docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Cleanup
echo ""
echo "Cleaning up temporary files..."
# rm -f patchestry-seahorn-builder.Dockerfile patchestry-seahorn.Dockerfile SeaHorn-*.tar.gz
echo "✓ Cleanup completed"
