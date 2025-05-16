#!/bin/bash

SCRIPTS_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

echo "Using SCRIPTS_DIR: $SCRIPTS_DIR"

DOCKER_BUILDKIT=1 docker build \
    -t trailofbits/patchestry-decompilation:latest \
    -f "${SCRIPTS_DIR}/decompile-headless.dockerfile" \
    "${SCRIPTS_DIR}"

if [ $? -eq 0 ]; then
    echo "Docker image built successfully."
else
    echo "Error: Docker build failed."
    exit 1
fi
