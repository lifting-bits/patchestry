#!/bin/bash
#
# Demo script: Recovering bl_device__process_entry function from bloodview
# and lifting it to Clang IR
#
# This script demonstrates the patchestry workflow:
# 0. Build and install patchestry tools to build/run directory
# 1. Extract function from binary using Ghidra (P-code extraction)
# 2. Lift P-code to Clang IR using patchir-decomp
# 3. Patch Clang IR using patchir-transform
# 4. Lower patched Clang IR to LLVM IR using patchir-cir2llvm
# 5. Patch the binary using Patcherex2
#

set -e  # Exit on error

# Configuration
BINARY_PATH="firmwares/output/bloodlight/bloodview"
FUNCTION_NAME="bl_device__process_entry"
OUTPUT_PREFIX="process_entry"
GHIDRA_SCRIPT="scripts/ghidra/decompile-headless.sh"
BUILD_RUN_DIR="builds/run"
PATCHIR_DECOMP="${BUILD_RUN_DIR}/bin/patchir-decomp"
PATCHIR_TRANSFORM="${BUILD_RUN_DIR}/bin/patchir-transform"
PATCHIR_CIR2LLVM="${BUILD_RUN_DIR}/bin/patchir-cir2llvm"
PATCH_SPEC="test/patchir-transform/device_process_entry.yaml"
PATCHEREX2_SCRIPT="../Patcherex2/patche_binary.sh"

# Step 0: Build and install patchestry tools
echo -e "=== Step 0: Building and Installing Patchestry Tools ==="
echo "Installing patchestry tools to: $BUILD_RUN_DIR"
echo ""

# Configure with CMake
echo "Configuring CMake..."
cmake --preset default

# Build with CMake
echo "Building patchestry (Release configuration)..."
cmake --build builds/default --config Release -j$(nproc)

# Install to build/run directory
echo "Installing tools to $BUILD_RUN_DIR..."
cmake --install builds/default --prefix ./$BUILD_RUN_DIR --config Release

echo -e "✓ Successfully built and installed patchestry tools!"

# Check if required files exist
if [ ! -f "$BINARY_PATH" ]; then
    echo "Error: Binary not found at $BINARY_PATH"
    echo "Please build the firmware first by running: bash firmwares/build.sh"
    exit 1
fi

if [ ! -f "$PATCHIR_DECOMP" ]; then
    echo "Error: patchir-decomp not found at $PATCHIR_DECOMP"
    exit 1
fi

if [ ! -f "$PATCHIR_TRANSFORM" ]; then
    echo "Error: patchir-transform not found at $PATCHIR_TRANSFORM"
    exit 1
fi

if [ ! -f "$PATCHIR_CIR2LLVM" ]; then
    echo "Error: patchir-cir2llvm not found at $PATCHIR_CIR2LLVM"
    exit 1
fi

if [ ! -f "$PATCH_SPEC" ]; then
    echo "Error: Patch specification not found at $PATCH_SPEC"
    exit 1
fi

if [ ! -f "$PATCHEREX2_SCRIPT" ]; then
    echo "Error: Patcherex2 script not found at $PATCHEREX2_SCRIPT"
    exit 1
fi

echo -e "=== Patchestry Demo: Function Recovery, Lifting, and Patching ==="
echo "Binary: $BINARY_PATH"
echo "Function: $FUNCTION_NAME"
echo "Patch Spec: $PATCH_SPEC"
echo "Output: ${OUTPUT_PREFIX}.{json,c,cir} and ${OUTPUT_PREFIX}_patched.{cir,ll}"

# Step 1: Extract function from binary using Ghidra
echo -e "Step 1: Extracting function from binary using Ghidra..."
echo "Command: $GHIDRA_SCRIPT --input $BINARY_PATH --function $FUNCTION_NAME --output ${OUTPUT_PREFIX}.json"

bash "$GHIDRA_SCRIPT" \
    --input "$BINARY_PATH" \
    --function "$FUNCTION_NAME" \
    --output "${OUTPUT_PREFIX}.json"

if [ ! -f "${OUTPUT_PREFIX}.json" ]; then
    echo "Error: Failed to generate ${OUTPUT_PREFIX}.json"
    exit 1
fi

echo -e "✓ Successfully extracted P-code to ${OUTPUT_PREFIX}.json"

# Step 2: Lift P-code JSON to Clang IR
echo -e "Step 2: Lifting P-code to Clang IR..."
echo "Command: $PATCHIR_DECOMP -input ${OUTPUT_PREFIX}.json -emit-cir -emit-mlir -emit-llvm -print-tu -output $OUTPUT_PREFIX"

"$PATCHIR_DECOMP" \
    -input "${OUTPUT_PREFIX}.json" \
    -emit-cir \
    -print-tu \
    -output "$OUTPUT_PREFIX"

echo -e "✓ Successfully lifted to Clang IR!"

# Step 3: Patch the Clang IR using patchir-transform
echo -e "Step 3: Patching Clang IR using patchir-transform..."
echo "Command: $PATCHIR_TRANSFORM ${OUTPUT_PREFIX}.cir -spec $PATCH_SPEC -o ${OUTPUT_PREFIX}_patched.cir"

"$PATCHIR_TRANSFORM" \
    "${OUTPUT_PREFIX}.cir" \
    -spec "$PATCH_SPEC" \
    -o "${OUTPUT_PREFIX}_patched.cir"

echo -e "✓ Successfully patched Clang IR!"

# Step 4: Lower patched Clang IR to LLVM IR
echo -e "Step 4: Lowering patched Clang IR to LLVM IR..."
echo "Command: $PATCHIR_CIR2LLVM -S ${OUTPUT_PREFIX}_patched.cir -o ${OUTPUT_PREFIX}_patched.ll"

"$PATCHIR_CIR2LLVM" \
    -S "${OUTPUT_PREFIX}_patched.cir" \
    -o "${OUTPUT_PREFIX}_patched.ll"

echo -e "✓ Successfully lowered to LLVM IR!"

# Step 5: Patch the binary using Patcherex2
echo -e "Step 5: Patching binary using Patcherex2..."
echo "Command: bash $PATCHEREX2_SCRIPT $BINARY_PATH ${OUTPUT_PREFIX}_patched.ll"

bash "$PATCHEREX2_SCRIPT" "$BINARY_PATH" "${OUTPUT_PREFIX}_patched.ll"

echo -e "✓ Successfully patched binary!"

# Display results
echo -e "=== Results ==="

if [ -f "${OUTPUT_PREFIX}.c" ]; then
    echo -e "✓ ${OUTPUT_PREFIX}.c - Recovered C code"
fi

if [ -f "${OUTPUT_PREFIX}.cir" ]; then
    echo -e "✓ ${OUTPUT_PREFIX}.cir - Clang IR representation"
fi

if [ -f "${OUTPUT_PREFIX}_patched.cir" ]; then
    echo -e "✓ ${OUTPUT_PREFIX}_patched.cir - Patched Clang IR"
fi

if [ -f "${OUTPUT_PREFIX}_patched.ll" ]; then
    echo -e "✓ ${OUTPUT_PREFIX}_patched.ll - Patched LLVM IR"
fi

if [ -f "/home/akshayk/Patcherex2/outs/$(basename $BINARY_PATH)_patched" ]; then
    echo -e "✓ /home/akshayk/Patcherex2/outs/$(basename $BINARY_PATH)_patched - Patched binary"
fi

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
