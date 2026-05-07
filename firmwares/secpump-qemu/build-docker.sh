#!/bin/bash
# Build secpump.elf inside the firmware-builder Docker image.
# Output: build/secpump.elf (also build/secpump.bin) on the host.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
firmwares_dir=$(cd "${script_dir}/.." && pwd)

docker image inspect firmware-builder >/dev/null 2>&1 \
  || docker build -t firmware-builder "${firmwares_dir}"

docker run --rm \
  -v "${script_dir}:/work/secpump-qemu" \
  firmware-builder \
  -c "cd secpump-qemu && make ${*:-}"
