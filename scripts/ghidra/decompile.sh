#!/bin/bash
#
# Copyright (c) 2024, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <file_path> [<function_name>]"
    exit 1
fi

FILE_PATH=$1
FUNCTION_NAME=$2

# Create a new Ghidra project and import the file
${GHIDRA_HEADLESS} ${GHIDRA_PROJECTS} patchestry-decompilation \
    -readOnly -deleteProject \
    -import $FILE_PATH \
    -postScript PatchestryScript.java \
    $FUNCTION_NAME

# Check if the decompile script was successful
if [ $? -ne 0 ]; then
    echo "Decompilation failed"
    exit 1
fi

# TODO copy the decompiled output to the output directory
