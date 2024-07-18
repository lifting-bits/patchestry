#!/bin/bash
#
# Copyright (c) 2024, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <file_path> [<function_name>]"
    exit 1
fi

FILE_PATH=$1
FUNCTION_NAME=$2

# Create docker container and run the decompilation
docker build -t trailofbits/patchestry-decompilation:latest -f DecompileHeadless.Dockerfile .

if [ $? -ne 0 ]; then
    echo "Docker build failed"
    exit 1
fi

docker run --rm \
    -v $FILE_PATH:/input \
    -v DecompileHeadless.java://ghidra/Ghidra/Features/Decompiler/ghidra_scripts/DecompileHeadless.java \
    trailofbits/patchestry-decompilation:latest \
    /input $FUNCTION_NAME
