#!/usr/bin/env bash
# Build secpump.elf inside the dedicated secpump-builder Docker image.
# Output: build/secpump.elf and build/secpump.bin on the host.
# Works on Linux (amd64/arm64), macOS Intel, and macOS Apple Silicon.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

docker image inspect secpump-builder >/dev/null 2>&1 \
  || docker build -t secpump-builder "${script_dir}"

docker run --rm \
  -v "${script_dir}:/work/secpump-qemu" \
  secpump-builder \
  -c "cd secpump-qemu && make ${*:-}"
