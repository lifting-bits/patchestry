#!/bin/bash

# Get the directory of the current script
SCRIPTS_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# Print the script directory (for debugging purposes, can be removed)
echo "Using SCRIPTS_DIR: $SCRIPTS_DIR"

# Build the Docker image
docker build \
    -t trailofbits/patchestry-decompilation:latest \
    -f "${SCRIPTS_DIR}/decompile-headless.dockerfile" \
    "${SCRIPTS_DIR}"

# Check if the Docker build was successful
if [ $? -eq 0 ]; then
    echo "Docker image built successfully."
else
    echo "Error: Docker build failed."
    exit 1
fi
