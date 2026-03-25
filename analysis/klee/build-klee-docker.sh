#!/bin/bash
# Build the KLEE Docker image for patchestry harness execution.
#
# Usage:
#   ./analysis/klee/build-klee-docker.sh [--no-cache]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="patchestry/klee:latest"

EXTRA_ARGS=""
if [[ "${1:-}" == "--no-cache" ]]; then
    EXTRA_ARGS="--no-cache"
fi

echo "Building KLEE Docker image: ${IMAGE_NAME}"
echo "Context: ${SCRIPT_DIR}"

DOCKER_BUILDKIT=1 docker build \
    --platform linux/amd64 \
    ${EXTRA_ARGS} \
    -t "${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}"

echo ""
echo "Build complete: ${IMAGE_NAME}"
echo "Test with: docker run --rm ${IMAGE_NAME} --help"
