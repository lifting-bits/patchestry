#!/bin/bash
#
# Copyright (c) 2024, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
SCRIPTS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_file> <function_name> <output_file>"
    exit 1
fi

INPUT_PATH=$1
FUNCTION_NAME=$2
OUTPUT_PATH=$3

# Create docker container and run the decompilation
docker build \
    -t trailofbits/patchestry-decompilation:latest \
    -f ${SCRIPTS_DIR}/DecompileHeadless.Dockerfile \
    ${SCRIPTS_DIR}

if [ $? -ne 0 ]; then
    echo "Docker build failed"
    exit 1
fi

# Make sure $OUTPUT_PATH exists and is empty so that it can be
# mounted to the container
if [ ! -f $OUTPUT_PATH ]; then
    touch $OUTPUT_PATH
fi
truncate -s 0 $OUTPUT_PATH

docker run --rm \
    -v $INPUT_PATH:/input \
    -v $OUTPUT_PATH:/output \
    trailofbits/patchestry-decompilation:latest \
    /input $FUNCTION_NAME /output
