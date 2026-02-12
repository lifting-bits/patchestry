#!/usr/bin/env bash
#
# Build SeaHorn Docker Image
#
# This script clones the SeaHorn repository from Trail of Bits' fork
# and builds a Docker image from the dev20 branch.
#

set -euo pipefail

# Configuration
SEAHORN_REPO="git@github.com:trail-of-forks/seahorn.git"
SEAHORN_BRANCH="dev20"
CLONE_DIR="seahorn-repo"
IMAGE_NAME="seahorn"
IMAGE_TAG="dev20"

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
    echo "and it is disabled."
    echo ""
    echo "Please run this script on an amd64/x86_64 system."
    exit 1
fi

echo "=== SeaHorn Docker Build Script ==="
echo "Repository: ${SEAHORN_REPO}"
echo "Branch: ${SEAHORN_BRANCH}"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
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

# Change to script directory
cd "${SCRIPT_DIR}"

# Clean up existing clone if requested
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning up existing repository clone..."
    rm -rf "${CLONE_DIR}"
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

# Get the current commit hash for tagging
COMMIT_HASH=$(git rev-parse --short HEAD)
echo "Current commit: ${COMMIT_HASH}"

# Build Docker images using three-stage process
echo "Building SeaHorn Docker images (3 stages)..."
echo "This may take a while (SeaHorn has many dependencies)..."
echo ""

# Stage 1: Build dependencies image
echo "=== Stage 1/3: Building dependencies image ==="
docker build \
    -t seahorn/buildpack-deps-seahorn:jammy-llvm20 \
    -f docker/buildpack-deps-seahorn.Dockerfile \
    .

if [[ $? -ne 0 ]]; then
    echo "✗ Stage 1 failed: Could not build dependencies image"
    exit 1
fi
echo "✓ Stage 1 complete: Dependencies image built"
echo ""

# Stage 2: Build SeaHorn and create distribution package
echo "=== Stage 2/3: Building SeaHorn and creating distribution ==="
docker build \
    -t seahorn/seahorn-builder:jammy-llvm20 \
    -f docker/seahorn-builder.Dockerfile \
    .

if [[ $? -ne 0 ]]; then
    echo "✗ Stage 2 failed: Could not build SeaHorn"
    exit 1
fi

# Extract distribution tar.gz
echo "Extracting SeaHorn distribution package..."
docker run -v "$(pwd):/host" --rm seahorn/seahorn-builder:jammy-llvm20 \
    /bin/sh -c "cp build/*.tar.gz /host/" 2>/dev/null

if [[ ! -f SeaHorn-20.*.tar.gz ]]; then
    echo "✗ Stage 2 failed: Could not extract distribution package"
    exit 1
fi
echo "✓ Stage 2 complete: Distribution package created"
echo ""

# Stage 3: Build final distribution image
echo "=== Stage 3/3: Building final distribution image ==="
docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}-${COMMIT_HASH}" \
    -t "${IMAGE_NAME}:latest" \
    -f docker/seahorn.Dockerfile \
    .

# Cleanup distribution tar.gz
rm -f SeaHorn-20.*.tar.gz

# Check if final build was successful
if [[ $? -eq 0 ]]; then
    echo "✓ Stage 3 complete: Final distribution image built"
    echo ""
    echo "=== All 3 stages completed successfully! ==="
    echo ""
    echo "Images created:"
    echo "  Dependencies: seahorn/buildpack-deps-seahorn:jammy-llvm20"
    echo "  Builder: seahorn/seahorn-builder:jammy-llvm20"
    echo "  Distribution: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "                ${IMAGE_NAME}:${IMAGE_TAG}-${COMMIT_HASH}"
    echo "                ${IMAGE_NAME}:latest"
    echo ""
    echo "To run the image:"
    echo "  docker run -it ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To verify SeaHorn is working:"
    echo "  docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} sea --version"
else
    echo "✗ Stage 3 failed: Could not build final distribution image"
    exit 1
fi

# Display image size
echo ""
echo "Image information:"
docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
