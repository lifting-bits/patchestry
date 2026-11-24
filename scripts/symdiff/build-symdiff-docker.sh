#!/bin/bash
# Build the PatchSymDiff Docker image.
#
# Usage:
#   ./scripts/symdiff/build-symdiff-docker.sh [--no-cache] [--test]
#
#   --no-cache   rebuild every layer
#   --test       also build the `test` stage, which runs the unit tests of
#                patch_symdiff.py inside the image; the build fails if they do
#
# The image name is $SYMDIFF_IMAGE (default patchestry/symdiff:latest).  The
# image is built for the platform of the Docker host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${SYMDIFF_IMAGE:-patchestry/symdiff:latest}"

EXTRA_ARGS=()
RUN_TESTS=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cache) EXTRA_ARGS+=("--no-cache"); shift ;;
        --test) RUN_TESTS=true; shift ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Error: unknown option: $1" >&2; exit 2 ;;
    esac
done

echo "Building PatchSymDiff Docker image: ${IMAGE_NAME}"
echo "Context: ${CONTEXT_DIR}"

DOCKER_BUILDKIT=1 docker build \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    --target runtime \
    -t "${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${CONTEXT_DIR}"

if [[ "${RUN_TESTS}" == true ]]; then
    echo ""
    echo "Running the unit tests in the image's environment"
    # pipefail inside the stage's RUN is not set, so read the log for the result.
    DOCKER_BUILDKIT=1 docker build \
        --target test \
        -t "${IMAGE_NAME%:*}-test:latest" \
        -f "${SCRIPT_DIR}/Dockerfile" \
        "${CONTEXT_DIR}"
    LOG="$(docker run --rm "${IMAGE_NAME%:*}-test:latest")"
    echo "${LOG}" | tail -n 4
    if ! echo "${LOG}" | tail -n 1 | grep -q '^OK'; then
        echo "Error: the unit tests failed in the image" >&2
        exit 1
    fi
fi

echo ""
echo "Build complete: ${IMAGE_NAME}"
echo "Test with: docker run --rm ${IMAGE_NAME} --help"
