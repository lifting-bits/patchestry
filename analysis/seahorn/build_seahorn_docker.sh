#!/usr/bin/env bash
#
# Build SeaHorn Docker Image with LLVM 20
#
# This script follows the official SeaHorn pattern:
# 1. Check if LLVM 20 package exists
# 2. If not, build it from llvm-seahorn source (~1 hour)
# 3. Build SeaHorn using the LLVM package (~10 minutes)
#
# The LLVM package is reusable across builds.
#

set -euo pipefail

# Configuration
SEAHORN_REPO="git@github.com:trail-of-forks/seahorn.git"
SEAHORN_BRANCH="dev20"
CLONE_DIR="seahorn-repo"
IMAGE_NAME="seahorn"
IMAGE_TAG="dev20"
LLVM_PACKAGE_PATTERN="llvm-seahorn-20.*.tar.gz"

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

echo "=== SeaHorn Docker Build Script (with LLVM 20 package) ==="
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

# Force rebuild LLVM if requested
REBUILD_LLVM=false
if [[ "${1:-}" == "--rebuild-llvm" ]] || [[ "${2:-}" == "--rebuild-llvm" ]]; then
    REBUILD_LLVM=true
    echo "Forcing LLVM rebuild..."
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
echo ""

# Check if LLVM package exists
LLVM_PACKAGE=""
shopt -s nullglob
LLVM_PACKAGES=(${LLVM_PACKAGE_PATTERN})
if [[ ${#LLVM_PACKAGES[@]} -gt 0 ]] && [[ "$REBUILD_LLVM" == false ]]; then
    LLVM_PACKAGE="${LLVM_PACKAGES[0]}"
    echo "✓ Found existing LLVM package: ${LLVM_PACKAGE}"
    echo "  Using existing package (use --rebuild-llvm to rebuild)"
    echo ""
else
    if [[ "$REBUILD_LLVM" == true ]]; then
        echo "Building LLVM 20 package (forced rebuild)..."
    else
        echo "No LLVM 20 package found. Building from source..."
    fi
    echo ""
    echo "=== Stage 0/3: Building LLVM 20 from llvm-seahorn ==="
    echo "WARNING: This will take 30-90 minutes depending on your hardware!"
    echo "The resulting package can be reused for future builds."
    echo ""

    # Build LLVM package
    docker build \
        --build-arg BUILD_TYPE=RelWithDebInfo \
        -t seahorn/llvm20-builder:latest \
        -f docker/llvm20.Dockerfile \
        .

    if [[ $? -ne 0 ]]; then
        echo "✗ Stage 0 failed: Could not build LLVM package"
        exit 1
    fi

    # Extract LLVM package
    echo "Extracting LLVM package..."
    docker run -v "$(pwd):/host" --rm seahorn/llvm20-builder:latest

    # Find the newly created package
    shopt -s nullglob
    LLVM_PACKAGES=(${LLVM_PACKAGE_PATTERN})
    if [[ ${#LLVM_PACKAGES[@]} -eq 0 ]]; then
        # Try alternative pattern (LLVM-*.tar.gz)
        LLVM_PACKAGES=(LLVM-*.tar.gz)
    fi

    if [[ ${#LLVM_PACKAGES[@]} -eq 0 ]]; then
        echo "✗ Stage 0 failed: Could not find LLVM package after build"
        exit 1
    fi

    LLVM_PACKAGE="${LLVM_PACKAGES[0]}"

    # Rename if needed to match expected pattern
    if [[ ! "$LLVM_PACKAGE" =~ ^llvm-seahorn- ]]; then
        NEW_NAME="llvm-seahorn-20.0.0-jammy-RelWithDebInfo.tar.gz"
        mv "$LLVM_PACKAGE" "$NEW_NAME"
        LLVM_PACKAGE="$NEW_NAME"
    fi

    echo "✓ Stage 0 complete: LLVM package created: ${LLVM_PACKAGE}"
    LLVM_SIZE=$(du -h "$LLVM_PACKAGE" | cut -f1)
    echo "  Package size: ${LLVM_SIZE}"
    echo "  This package can be reused for future builds!"
    echo ""
fi

# Build Docker images using three-stage process
echo "Building SeaHorn Docker images (3 stages)..."
echo ""

# Stage 1: Build dependencies image with LLVM package
echo "=== Stage 1/3: Building dependencies image with LLVM package ==="
docker build \
    -t seahorn/buildpack-deps-seahorn:jammy-llvm20-package \
    -f docker/buildpack-deps-seahorn-with-package.Dockerfile \
    .

if [[ $? -ne 0 ]]; then
    echo "✗ Stage 1 failed: Could not build dependencies image"
    exit 1
fi
echo "✓ Stage 1 complete: Dependencies image built with LLVM package"
echo ""

# Stage 2: Build SeaHorn and create distribution package
echo "=== Stage 2/3: Building SeaHorn and creating distribution ==="
docker build \
    -t seahorn/seahorn-builder:jammy-llvm20 \
    -f docker/seahorn-builder-with-package.Dockerfile \
    .

if [[ $? -ne 0 ]]; then
    echo "✗ Stage 2 failed: Could not build SeaHorn"
    exit 1
fi

# Extract distribution tar.gz
echo "Extracting SeaHorn distribution package..."
docker run -v "$(pwd):/host" --rm seahorn/seahorn-builder:jammy-llvm20 \
    /bin/sh -c "cp build/*.tar.gz /host/"

# Check if any tar.gz file matching the pattern exists
shopt -s nullglob
TAR_FILES=(SeaHorn-20.*.tar.gz)
if [[ ${#TAR_FILES[@]} -eq 0 ]]; then
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
shopt -s nullglob
rm -f SeaHorn-20.*.tar.gz

# Check if final build was successful
if [[ $? -eq 0 ]]; then
    echo "✓ Stage 3 complete: Final distribution image built"
    echo ""
    echo "=== All stages completed successfully! ==="
    echo ""
    echo "Images created:"
    echo "  LLVM Builder: seahorn/llvm20-builder:latest"
    echo "  Dependencies: seahorn/buildpack-deps-seahorn:jammy-llvm20-package"
    echo "  Builder: seahorn/seahorn-builder:jammy-llvm20"
    echo "  Distribution: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "                ${IMAGE_NAME}:${IMAGE_TAG}-${COMMIT_HASH}"
    echo "                ${IMAGE_NAME}:latest"
    echo ""
    echo "LLVM package: ${LLVM_PACKAGE} (reusable for future builds)"
    echo ""
    echo "To run the image:"
    echo "  docker run -it ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To verify SeaHorn is working:"
    echo "  docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} sea --version"
    echo "  docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} seaopt --version  # Should NOT crash"
    echo ""
    echo "To use with sea_verify:"
    echo "  SEAHORN_IMAGE=${IMAGE_NAME}:${IMAGE_TAG} ./sea_verify input.ll"
else
    echo "✗ Stage 3 failed: Could not build final distribution image"
    exit 1
fi

# Display image size
echo ""
echo "Image information:"
docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Test that it doesn't crash
echo ""
echo "Testing for ABI issues..."
if docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" seaopt --version >/dev/null 2>&1; then
    echo "✓ Success: seaopt works without crashes!"
else
    echo "✗ Warning: seaopt crashed. There may still be ABI issues."
    exit 1
fi
