#!/usr/bin/env bash
#
# SeaHorn verification wrapper using Docker
# Usage: ./sea_verify <input.ll> [additional seahorn options]
#
# This script runs SeaHorn verification inside a Docker container,
# following the same 4-step process as the standalone sea_verify.sh
#

set -euo pipefail

# Configuration
DOCKER_IMAGE="${SEAHORN_IMAGE:-ghcr.io/lifting-bits/patchestry-seahorn:latest}"
WORK_DIR="/work"

# Usage function
usage() {
    cat <<EOF
Usage: $0 <input.ll> [additional seahorn options]

Runs SeaHorn verification using Docker image: ${DOCKER_IMAGE}

The script performs 4 steps:
  1. Preprocessing (seapp with --kill-vaarg=false)
  2. Mixed semantics transformation (seapp --horn-mixed-sem)
  3. Optimization (seaopt -O3)
  4. Verification (seahorn --horn-inline-all)

Examples:
  $0 input.ll
  $0 input.ll -horn-cex=output.ll
  $0 input.ll --show-invars

Environment Variables:
  SEAHORN_IMAGE - Docker image to use (default: seahorn:dev20)

EOF
    exit 1
}

# Check arguments
if [[ $# -lt 1 ]]; then
    usage
fi

INPUT_FILE="$1"
shift  # Remove first argument, rest are passed to seahorn

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

# Check if Docker image exists
if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
    echo "Error: Docker image '$DOCKER_IMAGE' not found"
    echo ""
    echo "Please build the image first using:"
    echo "  cd analysis/seahorn && ./build_seahorn_docker.sh"
    echo ""
    echo "Or specify a different image:"
    echo "  SEAHORN_IMAGE=seahorn:latest $0 $INPUT_FILE"
    exit 1
fi

# Get absolute path to input file
INPUT_FILE_ABS=$(readlink -f "$INPUT_FILE")
INPUT_FILE_DIR=$(dirname "$INPUT_FILE_ABS")
INPUT_FILE_NAME=$(basename "$INPUT_FILE_ABS")

# Get current working directory for output files
CURRENT_DIR=$(pwd)

echo "==> SeaHorn verification with inlining enabled (Docker)"
echo "Image: $DOCKER_IMAGE"
echo "Input: $INPUT_FILE"
echo "Working directory: $CURRENT_DIR"
echo ""

# Create temporary directory for intermediate files
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Docker run command wrapper
# Mounts both input directory and temp directory, plus current dir for output
run_in_docker() {
    docker run --rm \
        -v "$INPUT_FILE_DIR:$WORK_DIR/input:ro" \
        -v "$TEMP_DIR:$WORK_DIR/temp:rw" \
        -v "$CURRENT_DIR:$WORK_DIR/output:rw" \
        -w "$WORK_DIR" \
        "$DOCKER_IMAGE" \
        "$@"
}

# Step 1: Preprocessing with --kill-vaarg=false
echo "[1/4] Preprocessing (seapp with --kill-vaarg=false)..."
run_in_docker seapp \
    -o "$WORK_DIR/temp/preprocessed.bc" \
    --simplifycfg-sink-common=false \
    --strip-extern=false \
    --promote-assumptions=false \
    --kill-vaarg=false \
    --ignore-def-verifier-fn=false \
    --horn-keep-arith-overflow=false \
    --promote-nondet-undef=true \
    --horn-replace-loops-with-nd-funcs=false \
    "$WORK_DIR/input/$INPUT_FILE_NAME"

# Step 2: Mixed semantics transformation
echo "[2/4] Mixed semantics (seapp --horn-mixed-sem)..."
run_in_docker seapp \
    --simplifycfg-sink-common=false \
    -o "$WORK_DIR/temp/mixed_sem.bc" \
    --horn-mixed-sem \
    --ms-reduce-main \
    "$WORK_DIR/temp/preprocessed.bc"

# Step 3: Optimization
echo "[3/4] Optimization (seaopt -O3)..."
run_in_docker seaopt \
    -f \
    --simplifycfg-sink-common=false \
    -o "$WORK_DIR/temp/optimized.bc" \
    -O3 \
    --unroll-threshold=150 \
    --unroll-allow-partial=false \
    --unroll-partial-threshold=0 \
    --vectorize-slp=false \
    "$WORK_DIR/temp/mixed_sem.bc"

# Step 4: Verification with --horn-inline-all
echo "[4/4] Verification (seahorn --horn-inline-all)..."

# Process additional arguments to fix file paths
# If any arguments reference files in current directory, adjust paths
SEAHORN_ARGS=()
for arg in "$@"; do
    # Check if argument looks like a file path (contains / or .)
    if [[ "$arg" == *"="* ]]; then
        # Split on = to handle -horn-cex=file.ll
        key="${arg%%=*}"
        value="${arg#*=}"
        if [[ "$value" == *.* ]] && [[ "$value" != -* ]]; then
            # Looks like a filename - prefix with output dir
            SEAHORN_ARGS+=("${key}=$WORK_DIR/output/${value}")
        else
            SEAHORN_ARGS+=("$arg")
        fi
    else
        SEAHORN_ARGS+=("$arg")
    fi
done

run_in_docker seahorn \
    --keep-shadows=true \
    --sea-dsa=ci \
    --horn-solve \
    --horn-answer \
    -horn-cex-pass \
    -horn-inter-proc \
    -horn-sem-lvl=mem \
    --horn-step=large \
    --horn-inline-all \
    "${SEAHORN_ARGS[@]}" \
    "$WORK_DIR/temp/optimized.bc"

echo ""
echo "==> Verification complete"
echo ""
echo "Output files (if any) written to: $CURRENT_DIR"
