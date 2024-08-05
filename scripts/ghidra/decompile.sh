#!/bin/bash
#
# Copyright (c) 2024, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

set -x

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_file> <function_name> <output_file>"
    exit 1
fi

INPUT_PATH=$1
FUNCTION_NAME=$2
OUTPUT_PATH=$3

# Running with non-root user may cause permission issue on ubuntu
# because binded directory will have root permission.
# This is a hacky fix to avoid the issue. It can be avoided
# by switching to using docker volume.
if [ ! -w ${OUTPUT_PATH} ]; then
  sudo chown ${USER}:${USER} ${OUTPUT_PATH}
fi

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
