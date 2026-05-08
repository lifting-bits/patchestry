#!/usr/bin/env bash
# Build makair.elf inside the makair-builder Docker image.
# Output: build/makair.elf and build/makair.bin on the host.
# Pulls upstream sources live from firmwares/repos/makair-firmware/, which
# the parent firmwares/build.sh populates by cloning the pinned MakAir
# release tag.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
firmwares_dir=$(cd "${script_dir}/.." && pwd)

upstream_dir="${firmwares_dir}/repos/makair-firmware"
if [ ! -d "${upstream_dir}" ]; then
  echo "MakAir upstream not found at ${upstream_dir}" >&2
  echo "Run firmwares/build.sh first (it clones the pinned MakAir release)." >&2
  exit 1
fi

docker image inspect makair-builder >/dev/null 2>&1 \
  || docker build -t makair-builder "${script_dir}"

# Mount both the makair-qemu working dir and the upstream clone read-write
# (the Makefile reaches into the upstream tree to compile telemetry.cpp etc.
# and writes nothing back).
docker run --rm \
  -v "${script_dir}:/work/makair-qemu" \
  -v "${upstream_dir}:/work/repos/makair-firmware:ro" \
  makair-builder \
  -c "cd /work/makair-qemu && make MAKAIR_SRC=/work/repos/makair-firmware ${*:-}"
