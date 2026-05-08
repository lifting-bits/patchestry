#!/usr/bin/env bash
# Build makair.elf inside the makair-builder Docker image.
# Output: build/makair.elf and build/makair.bin on the host.
# Self-bootstraps firmwares/repos/makair-firmware/ if missing, so this script
# can be run directly without first invoking firmwares/build.sh.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
firmwares_dir=$(cd "${script_dir}/.." && pwd)

# Must match firmwares/build.sh:MAKAIR_COMMIT (upstream tag v4.1.0).
MAKAIR_COMMIT="72d454dcf799b7a33e85f00aef7a2fea4ca5c547"
upstream_dir="${firmwares_dir}/repos/makair-firmware"

if [ ! -d "${upstream_dir}" ]; then
  echo "Cloning MakAir upstream into ${upstream_dir}..." >&2
  git clone --depth 1 https://github.com/makers-for-life/makair-firmware.git \
    "${upstream_dir}"
  (
    cd "${upstream_dir}"
    git fetch --depth=1 origin "${MAKAIR_COMMIT}"
    git checkout "${MAKAIR_COMMIT}"
    git submodule update --init --recursive
    if [ -f "${firmwares_dir}/makair-firmware-patch.diff" ]; then
      patch -s -p1 <"${firmwares_dir}/makair-firmware-patch.diff"
    fi
  )
fi

docker image inspect makair-builder >/dev/null 2>&1 \
  || docker build -t makair-builder "${script_dir}"

# Mount both the makair-qemu working dir and the upstream clone read-only
# (the Makefile reaches into the upstream tree to compile telemetry.cpp etc.
# and writes nothing back).
docker run --rm \
  -v "${script_dir}:/work/makair-qemu" \
  -v "${upstream_dir}:/work/repos/makair-firmware:ro" \
  makair-builder \
  -c "cd /work/makair-qemu && make MAKAIR_SRC=/work/repos/makair-firmware ${*:-}"
