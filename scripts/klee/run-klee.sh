#!/bin/bash
# Host wrapper to run KLEE on a harness file via Docker.
#
# Usage:
#   ./scripts/klee/run-klee.sh --input <harness.c> [--output <dir>] [OPTIONS]
#
# Examples:
#   # Run a static contract harness
#   ./scripts/klee/run-klee.sh --input tests/static_contract_harness.c
#
#   # Run with custom timeout and output directory
#   ./scripts/klee/run-klee.sh --input harness.c --output ./results --max-time 600
#
#   # Run pre-compiled bitcode
#   ./scripts/klee/run-klee.sh --run-bitcode --input harness.bc
#
#   # Interactive debugging
#   ./scripts/klee/run-klee.sh --interactive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${KLEE_IMAGE:-patchestry/klee:latest}"

# ------------------------------------------------------------------
# Ensure image exists
# ------------------------------------------------------------------
ensure_klee_image() {
    if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        echo "KLEE Docker image not found. Building..."
        bash "${SCRIPT_DIR}/build-klee-docker.sh"
    fi
}

# ------------------------------------------------------------------
# Parse arguments — separate host-level from container-level
# ------------------------------------------------------------------
INPUT_FILE=""
OUTPUT_DIR=""
MODE=""
INTERACTIVE=false
CONTAINER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input|-i)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --run-bitcode|--compile-only|--run-harness)
            MODE="$1"
            CONTAINER_ARGS+=("$1")
            shift
            ;;
        -h|--help)
            docker run --rm "${IMAGE_NAME}" --help 2>/dev/null || true
            exit 0
            ;;
        *)
            # Pass through to container
            CONTAINER_ARGS+=("$1")
            shift
            ;;
    esac
done

ensure_klee_image

# ------------------------------------------------------------------
# Interactive mode
# ------------------------------------------------------------------
if [[ "${INTERACTIVE}" == true ]]; then
    echo "Starting interactive KLEE container..."
    docker run --rm -it \
        --platform linux/amd64 \
        -v "${REPO_ROOT}:/repo:ro" \
        -v "${PWD}:/work" \
        --entrypoint /bin/bash \
        "${IMAGE_NAME}"
    exit 0
fi

# ------------------------------------------------------------------
# Validate input
# ------------------------------------------------------------------
if [[ -z "${INPUT_FILE}" ]]; then
    echo "Error: --input is required" >&2
    echo "Usage: $0 --input <harness.c> [--output <dir>] [OPTIONS]" >&2
    exit 1
fi

INPUT_FILE="$(cd "$(dirname "${INPUT_FILE}")" && pwd)/$(basename "${INPUT_FILE}")"
if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: input file not found: ${INPUT_FILE}" >&2
    exit 1
fi

INPUT_DIR="$(dirname "${INPUT_FILE}")"
INPUT_NAME="$(basename "${INPUT_FILE}")"

# Output directory defaults to <input_dir>/klee-out
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${INPUT_DIR}/klee-out"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

# ------------------------------------------------------------------
# Run container
# ------------------------------------------------------------------
echo "Running KLEE on ${INPUT_NAME}"
echo "  Output: ${OUTPUT_DIR}"

docker run --rm \
    --platform linux/amd64 \
    -v "${INPUT_DIR}:/mnt/input:ro" \
    -v "${OUTPUT_DIR}:/mnt/output:rw" \
    "${IMAGE_NAME}" \
    ${MODE:+"${MODE}"} \
    --input "/mnt/input/${INPUT_NAME}" \
    --output "/mnt/output" \
    "${CONTAINER_ARGS[@]+"${CONTAINER_ARGS[@]}"}"

echo ""
echo "Results in: ${OUTPUT_DIR}"
