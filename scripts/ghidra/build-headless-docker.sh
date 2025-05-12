#!/bin/bash

# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
# This script preserves directory state for use in CI.

SCRIPTS_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

echo "Using SCRIPTS_DIR: $SCRIPTS_DIR"

DOCKER_BUILDKIT=1 docker build \
    --no-cache \
    -t trailofbits/patchestry-decompilation:latest \
    -f "${SCRIPTS_DIR}/decompile-headless.dockerfile" \
    "${SCRIPTS_DIR}"

if [ $? -eq 0 ]; then
    echo "Docker image built successfully."
else
    echo "Error: Docker build failed."
    exit 1
fi
