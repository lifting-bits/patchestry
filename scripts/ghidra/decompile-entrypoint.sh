#!/bin/bash
#
# Copyright (c) 2024, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_file> <function_name> <output_file>"
    exit 1
fi

INPUT_PATH=$1
FUNCTION_NAME=$2
OUTPUT_PATH=$3

# Create a new Ghidra project and import the file
${GHIDRA_HEADLESS} ${GHIDRA_PROJECTS} patchestry-decompilation \
    -readOnly -deleteProject \
    -import $INPUT_PATH \
    -postScript PatchestryScript.java \
    $FUNCTION_NAME \
    $OUTPUT_PATH

# Check if the decompile script was successful
if [ $? -ne 0 ]; then
    echo "Decompilation failed"
    exit 1
fi
