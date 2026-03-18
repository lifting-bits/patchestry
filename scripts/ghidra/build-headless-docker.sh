#!/bin/bash
set -euo pipefail

# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
# This script preserves directory state for use in CI.

SCRIPTS_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

translate_to_host_path() {
  local path="$1"
  if [ -n "${HOST_WORKSPACE:-}" ]; then
    echo "${path/#\/workspace/$HOST_WORKSPACE}"
  else
    echo "$path"
  fi
}

HOST_SCRIPTS_DIR="$(translate_to_host_path "${SCRIPTS_DIR}")"

echo "Using SCRIPTS_DIR: $SCRIPTS_DIR"

DOCKER_BUILDKIT=1 docker build \
  --no-cache \
  -t trailofbits/patchestry-decompilation:latest \
  -f "${HOST_SCRIPTS_DIR}/decompile-headless.dockerfile" \
  "${HOST_SCRIPTS_DIR}"

echo "Docker image built successfully."
