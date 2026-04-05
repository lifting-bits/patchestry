#!/bin/bash
# Demo: KLEE catching a static contract violation on bl_usb__send_message
#
# Case 1 (VIOLATION): External usbd_ep_write_packet is fully symbolic (any i32).
#   -> KLEE finds return > 255, violating postcondition range [0, 255]
#
# Case 2 (PASS): External is modeled via usb_hal_models.c with constrained
#   return [0, 255] and buffer memory effects.
#   -> All paths satisfy the contract. 0 assertion errors.
#
# Usage:
#   ./scripts/klee/demo-contract-violation.sh [build-dir]
#
# Prerequisites:
#   - patchir-klee-verifier built (cmake --build <build-dir> --target patchir-klee-verifier)
#   - clang available in PATH
#   - KLEE available in PATH (for full end-to-end demo)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${1:-${REPO_ROOT}/builds/default}"

# Tool paths
VERIFIER="${BUILD_DIR}/tools/patchir-klee-verifier/Debug/patchir-klee-verifier"
if [ ! -f "${VERIFIER}" ]; then
    VERIFIER="${BUILD_DIR}/tools/patchir-klee-verifier/Release/patchir-klee-verifier"
fi
if [ ! -f "${VERIFIER}" ]; then
    echo "ERROR: patchir-klee-verifier not found in ${BUILD_DIR}"
    echo "Build it first: cmake --build ${BUILD_DIR} --target patchir-klee-verifier"
    exit 1
fi

TEST_LL="${REPO_ROOT}/test/patchir-klee-verifier/bl_usb__send_message.ll"
MODEL_C="${REPO_ROOT}/lib/patchestry/klee/models/usb_hal_models.c"
STUB_H="${REPO_ROOT}/lib/patchestry/klee/models/klee_stub.h"

WORK=$(mktemp -d)
trap "rm -rf ${WORK}" EXIT

echo "=== KLEE Contract Violation Demo ==="
echo ""
echo "Target: bl_usb__send_message"
echo "Contract: postcondition return in [0, 255]"
echo ""

# Step 1: Compile C model to bitcode
echo "--- Step 1: Compiling USB HAL model to bitcode ---"
clang -emit-llvm -c -O0 -include "${STUB_H}" "${MODEL_C}" -o "${WORK}/usb_hal_models.bc"
echo "  -> ${WORK}/usb_hal_models.bc"
echo ""

# Step 2: Case 1 — Violation (no model library, external fully symbolic)
echo "--- Case 1: VIOLATION (no model library) ---"
echo "  External usbd_ep_write_packet returns unconstrained i32."
echo "  KLEE can find ret > 255, violating postcondition."
"${VERIFIER}" "${TEST_LL}" \
    --target-function bl_usb__send_message \
    -v -S -o "${WORK}/violation.ll"
echo "  -> Harness: ${WORK}/violation.ll"
echo ""

# Step 3: Case 2 — Pass (with model library, external constrained [0,255])
echo "--- Case 2: PASS (with --model-library) ---"
echo "  External usbd_ep_write_packet modeled with return in [0, 255]."
echo "  All symbolic paths satisfy the contract."
"${VERIFIER}" "${TEST_LL}" \
    --target-function bl_usb__send_message \
    --model-library "${WORK}/usb_hal_models.bc" \
    -v -S -o "${WORK}/modeled.ll"
echo "  -> Harness: ${WORK}/modeled.ll"
echo ""

# Step 4: Show the difference
echo "--- Comparison ---"
echo ""
echo "VIOLATION harness (usbd_ep_write_packet auto-stubbed):"
if grep -q "define i32 @usbd_ep_write_packet" "${WORK}/violation.ll"; then
    grep -A5 "define i32 @usbd_ep_write_packet" "${WORK}/violation.ll" | head -8
fi
echo ""
echo "MODELED harness (usbd_ep_write_packet from model library):"
if grep -q "define.*@usbd_ep_write_packet" "${WORK}/modeled.ll"; then
    grep -A10 "define.*@usbd_ep_write_packet" "${WORK}/modeled.ll" | head -15
fi
echo ""

# Step 5: Check if KLEE is available for end-to-end execution
if command -v klee &>/dev/null; then
    echo "--- Running KLEE (end-to-end) ---"
    echo ""

    # Assemble to bitcode
    llvm-as "${WORK}/violation.ll" -o "${WORK}/violation.bc" 2>/dev/null || true
    llvm-as "${WORK}/modeled.ll" -o "${WORK}/modeled.bc" 2>/dev/null || true

    echo "Case 1 (VIOLATION):"
    if [ -f "${WORK}/violation.bc" ]; then
        klee --output-dir="${WORK}/klee-violation" "${WORK}/violation.bc" 2>&1 || true
        err_count=$(find "${WORK}/klee-violation" -name "*.assert.err" 2>/dev/null | wc -l | tr -d ' ')
        echo "  Assertion errors: ${err_count}"
    fi
    echo ""

    echo "Case 2 (MODELED):"
    if [ -f "${WORK}/modeled.bc" ]; then
        klee --output-dir="${WORK}/klee-modeled" "${WORK}/modeled.bc" 2>&1 || true
        err_count=$(find "${WORK}/klee-modeled" -name "*.assert.err" 2>/dev/null | wc -l | tr -d ' ')
        echo "  Assertion errors: ${err_count}"
    fi
else
    echo "KLEE not found in PATH — skipping end-to-end execution."
    echo "To run the full demo, install KLEE or use the Docker image:"
    echo "  ./scripts/klee/run-klee.sh ${WORK}/violation.ll"
fi

echo ""
echo "=== Demo complete ==="
