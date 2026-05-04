#!/usr/bin/env bash
#
# Build the SeaHorn Docker image from scripts/seahorn/Dockerfile.
#
# The Dockerfile layers SeaHorn on top of the patchestry LLVM-20 dev image
# and clones/builds SeaHorn internally, so this script only needs to invoke
# `docker build` with the right build args and tags.
#

set -euo pipefail

# Defaults match scripts/seahorn/Dockerfile and .github/workflows/seahorn-image.yml.
LLVM_VERSION="${LLVM_VERSION:-20}"
IMAGE_VERSION="${IMAGE_VERSION:-22.04}"
SEAHORN_REPO="${SEAHORN_REPO:-https://github.com/trail-of-forks/seahorn.git}"
SEAHORN_BRANCH="${SEAHORN_BRANCH:-dev20}"
BUILD_TYPE="${BUILD_TYPE:-RelWithDebInfo}"
NO_CACHE="${NO_CACHE:-false}"

IMAGE_NAME="${IMAGE_NAME:-ghcr.io/lifting-bits/patchestry-seahorn-ubuntu-${IMAGE_VERSION}-llvm-${LLVM_VERSION}}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BASE_IMAGE="ghcr.io/lifting-bits/patchestry-ubuntu-${IMAGE_VERSION}-llvm-${LLVM_VERSION}-dev:latest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--no-cache] [--pull-base] [--help]

Environment overrides:
  LLVM_VERSION     LLVM version for the base image (default: ${LLVM_VERSION})
  IMAGE_VERSION    Ubuntu version for the base image (default: ${IMAGE_VERSION})
  SEAHORN_REPO     SeaHorn git repo URL (default: ${SEAHORN_REPO})
  SEAHORN_BRANCH   SeaHorn branch (default: ${SEAHORN_BRANCH})
  BUILD_TYPE       CMake build type (default: ${BUILD_TYPE})
  IMAGE_NAME       Output image name (default: ${IMAGE_NAME})
  IMAGE_TAG        Output image tag (default: ${IMAGE_TAG})
EOF
}

PULL_BASE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cache) NO_CACHE=true ;;
        --pull-base) PULL_BASE=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

ARCH="$(uname -m)"
if [[ "${ARCH}" != "x86_64" ]]; then
    echo "Error: SeaHorn ships x86_64-only z3/yices2 binaries; build is amd64-only." >&2
    echo "Current architecture: ${ARCH}" >&2
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed or not in PATH" >&2
    exit 1
fi
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running" >&2
    exit 1
fi

echo "=== SeaHorn Docker build ==="
echo "  Base image:    ${BASE_IMAGE}"
echo "  Output image:  ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  SeaHorn:       ${SEAHORN_REPO} @ ${SEAHORN_BRANCH}"
echo "  Build type:    ${BUILD_TYPE}"
echo "  No cache:      ${NO_CACHE}"
echo ""

if [[ "${PULL_BASE}" == "true" ]]; then
    echo "Pulling base image ${BASE_IMAGE}..."
    docker pull --platform linux/amd64 "${BASE_IMAGE}"
fi

CACHE_FLAG=()
if [[ "${NO_CACHE}" == "true" ]]; then
    CACHE_FLAG=(--no-cache)
fi

DOCKER_BUILDKIT=1 docker build \
    --platform linux/amd64 \
    "${CACHE_FLAG[@]}" \
    --build-arg IMAGE_VERSION="${IMAGE_VERSION}" \
    --build-arg LLVM_VERSION="${LLVM_VERSION}" \
    --build-arg SEAHORN_REPO="${SEAHORN_REPO}" \
    --build-arg SEAHORN_BRANCH="${SEAHORN_BRANCH}" \
    --build-arg BUILD_TYPE="${BUILD_TYPE}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}"

echo ""
echo "=== Build complete: ${IMAGE_NAME}:${IMAGE_TAG} ==="
docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

echo ""
echo "Verify SeaHorn:"
echo "  docker run --rm --platform linux/amd64 --entrypoint sea ${IMAGE_NAME}:${IMAGE_TAG} --help"
