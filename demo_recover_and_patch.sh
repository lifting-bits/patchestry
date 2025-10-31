#!/bin/bash
#
# Demo script: Recovering bl_device__process_entry function from bloodview
# and lifting it to Clang IR
#
# This script demonstrates the patchestry workflow:
# 1. Extract function from binary using Ghidra (P-code extraction)
# 2. Lift P-code to Clang IR using patchir-decomp
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BINARY_PATH="firmwares/output/bloodlight/bloodview"
FUNCTION_NAME="bl_device__process_entry"
OUTPUT_PREFIX="process_entry"
GHIDRA_SCRIPT="scripts/ghidra/decompile-headless.sh"
PATCHIR_DECOMP="builds/default/tools/patchir-decomp/Debug/patchir-decomp"
PATCHIR_TRANSFORM="builds/default/tools/patchir-transform/Debug/patchir-transform"
PATCHIR_CIR2LLVM="builds/default/tools/patchir-cir2llvm/Debug/patchir-cir2llvm"
PATCH_SPEC="test/patchir-transform/device_process_entry.yaml"
PATCHEREX2_SCRIPT="../Patcherex2/patche_binary.sh"

# Check if required files exist
if [ ! -f "$BINARY_PATH" ]; then
    echo "Error: Binary not found at $BINARY_PATH"
    echo "Please build the firmware first by running: bash firmwares/build.sh"
    exit 1
fi

if [ ! -f "$PATCHIR_DECOMP" ]; then
    echo "Error: patchir-decomp not found at $PATCHIR_DECOMP"
    echo "Please build the project first by running: cmake --preset default && cmake --build builds/default"
    exit 1
fi

if [ ! -f "$PATCHIR_TRANSFORM" ]; then
    echo "Error: patchir-transform not found at $PATCHIR_TRANSFORM"
    echo "Please build the project first by running: cmake --preset default && cmake --build builds/default"
    exit 1
fi

if [ ! -f "$PATCHIR_CIR2LLVM" ]; then
    echo "Error: patchir-cir2llvm not found at $PATCHIR_CIR2LLVM"
    echo "Please build the project first by running: cmake --preset default && cmake --build builds/default"
    exit 1
fi

if [ ! -f "$PATCH_SPEC" ]; then
    echo "Error: Patch specification not found at $PATCH_SPEC"
    exit 1
fi

if [ ! -f "$PATCHEREX2_SCRIPT" ]; then
    echo "Error: Patcherex2 script not found at $PATCHEREX2_SCRIPT"
    echo "Please ensure Patcherex2 is cloned in the parent directory"
    exit 1
fi

echo -e "${BLUE}=== Patchestry Demo: Function Recovery, Lifting, and Patching ===${NC}"
echo ""
echo "Binary: $BINARY_PATH"
echo "Function: $FUNCTION_NAME"
echo "Patch Spec: $PATCH_SPEC"
echo "Output: ${OUTPUT_PREFIX}.{json,c,cir} and ${OUTPUT_PREFIX}_patched.{cir,ll}"
echo ""

# Step 1: Extract function from binary using Ghidra
echo -e "${GREEN}Step 1: Extracting function from binary using Ghidra...${NC}"
echo "Command: $GHIDRA_SCRIPT --input $BINARY_PATH --function $FUNCTION_NAME --output ${OUTPUT_PREFIX}.json"
echo ""

bash "$GHIDRA_SCRIPT" \
    --input "$BINARY_PATH" \
    --function "$FUNCTION_NAME" \
    --output "${OUTPUT_PREFIX}.json"

if [ ! -f "${OUTPUT_PREFIX}.json" ]; then
    echo "Error: Failed to generate ${OUTPUT_PREFIX}.json"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Successfully extracted P-code to ${OUTPUT_PREFIX}.json${NC}"
echo ""

# Step 2: Lift P-code JSON to Clang IR
echo -e "${GREEN}Step 2: Lifting P-code to Clang IR...${NC}"
echo "Command: $PATCHIR_DECOMP -input ${OUTPUT_PREFIX}.json -emit-cir -emit-mlir -emit-llvm -print-tu -output $OUTPUT_PREFIX"
echo ""

"$PATCHIR_DECOMP" \
    -input "${OUTPUT_PREFIX}.json" \
    -emit-cir \
    -print-tu \
    -output "$OUTPUT_PREFIX"

echo ""
echo -e "${GREEN}✓ Successfully lifted to Clang IR!${NC}"
echo ""

# Step 3: Patch the Clang IR using patchir-transform
echo -e "${GREEN}Step 3: Patching Clang IR using patchir-transform...${NC}"
echo "Command: $PATCHIR_TRANSFORM ${OUTPUT_PREFIX}.cir -spec $PATCH_SPEC -o ${OUTPUT_PREFIX}_patched.cir"
echo ""

"$PATCHIR_TRANSFORM" \
    "${OUTPUT_PREFIX}.cir" \
    -spec "$PATCH_SPEC" \
    -o "${OUTPUT_PREFIX}_patched.cir"

echo ""
echo -e "${GREEN}✓ Successfully patched Clang IR!${NC}"
echo ""

# Step 4: Lower patched Clang IR to LLVM IR
echo -e "${GREEN}Step 4: Lowering patched Clang IR to LLVM IR...${NC}"
echo "Command: $PATCHIR_CIR2LLVM -S ${OUTPUT_PREFIX}_patched.cir -o ${OUTPUT_PREFIX}_patched.ll"
echo ""

"$PATCHIR_CIR2LLVM" \
    -S "${OUTPUT_PREFIX}_patched.cir" \
    -o "${OUTPUT_PREFIX}_patched.ll"

echo ""
echo -e "${GREEN}✓ Successfully lowered to LLVM IR!${NC}"
echo ""

# Step 5: Patch the binary using Patcherex2
echo -e "${GREEN}Step 5: Patching binary using Patcherex2...${NC}"
echo "Command: bash $PATCHEREX2_SCRIPT $BINARY_PATH ${OUTPUT_PREFIX}_patched.ll"
echo ""

bash "$PATCHEREX2_SCRIPT" "$BINARY_PATH" "${OUTPUT_PREFIX}_patched.ll"

echo ""
echo -e "${GREEN}✓ Successfully patched binary!${NC}"
echo ""

# Display results
echo -e "${BLUE}=== Results ===${NC}"
echo ""

if [ -f "${OUTPUT_PREFIX}.c" ]; then
    echo -e "${GREEN}✓ ${OUTPUT_PREFIX}.c${NC} - Recovered C code"
fi

if [ -f "${OUTPUT_PREFIX}.cir" ]; then
    echo -e "${GREEN}✓ ${OUTPUT_PREFIX}.cir${NC} - Clang IR representation"
fi

if [ -f "${OUTPUT_PREFIX}_patched.cir" ]; then
    echo -e "${GREEN}✓ ${OUTPUT_PREFIX}_patched.cir${NC} - Patched Clang IR"
fi

if [ -f "${OUTPUT_PREFIX}_patched.ll" ]; then
    echo -e "${GREEN}✓ ${OUTPUT_PREFIX}_patched.ll${NC} - Patched LLVM IR"
fi

if [ -f "/home/akshayk/Patcherex2/outs/$(basename $BINARY_PATH)_patched" ]; then
    echo -e "${GREEN}✓ /home/akshayk/Patcherex2/outs/$(basename $BINARY_PATH)_patched${NC} - Patched binary"
fi

echo ""
echo -e "${BLUE}=== Summary ===${NC}"
echo ""
echo "The function $FUNCTION_NAME has been successfully recovered from"
echo "the binary, lifted to Clang IR, patched, lowered to LLVM IR, and"
echo "the binary has been patched with the modified function:"
echo ""
echo "  Binary → P-code JSON → Clang IR → Patched Clang IR → LLVM IR → Patched Binary"
echo ""
echo "You can now view the results:"
echo "  - View recovered C code: cat ${OUTPUT_PREFIX}.c"
echo "  - View Clang IR: cat ${OUTPUT_PREFIX}.cir"
echo "  - View patched Clang IR: cat ${OUTPUT_PREFIX}_patched.cir"
echo "  - View patched LLVM IR: cat ${OUTPUT_PREFIX}_patched.ll"
echo "  - View patch spec: cat $PATCH_SPEC"
echo "  - View patched binary: /home/akshayk/Patcherex2/outs/$(basename $BINARY_PATH)_patched"
echo ""
echo -e "${GREEN}Demo complete!${NC}"
