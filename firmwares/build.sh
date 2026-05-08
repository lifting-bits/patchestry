#!/bin/bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${script_dir}/output"
mkdir -p "${script_dir}/repos"

translate_to_host_path() {
  local path="$1"
  if [ -n "${HOST_WORKSPACE:-}" ]; then
    echo "${path/#\/workspace/$HOST_WORKSPACE}"
  else
    echo "$path"
  fi
}

host_script_dir="$(translate_to_host_path "${script_dir}")"
host_output_dir="$(translate_to_host_path "${script_dir}/output")"
host_repos_dir="$(translate_to_host_path "${script_dir}/repos")"

# Repository commit hashes
PULSEOX_COMMIT="54ed8ca6bec36cc13db8f6594e3bd9941937922a"
BLOODLIGHT_COMMIT="fcc0daef9119ab09914b0c523e7d9d93aad36ea4"
VENTILATOR_COMMIT="c49fb21130de8732908d7a3d8eaf8915239a5735"
MAKAIR_COMMIT="72d454dcf799b7a33e85f00aef7a2fea4ca5c547"  # tag v4.1.0

# Clone/update repositories if needed
if [ ! -d "${script_dir}/repos/pulseox-firmware" ]; then
  git clone --depth 1 https://github.com/IRNAS/pulseox-firmware.git \
    "${script_dir}/repos/pulseox-firmware"
  cd "${script_dir}/repos/pulseox-firmware"
  git fetch --depth=1 origin "${PULSEOX_COMMIT}"
  git checkout "${PULSEOX_COMMIT}"
  git submodule update --init --recursive
  patch -s -p1 <"${script_dir}/pulseox-firmware-patch.diff"
fi

if [ ! -d "${script_dir}/repos/bloodlight-firmware" ]; then
  git clone --depth 1 https://github.com/kumarak/bloodlight-firmware.git \
    "${script_dir}/repos/bloodlight-firmware"
  cd "${script_dir}/repos/bloodlight-firmware"
  git fetch --depth=1 origin "${BLOODLIGHT_COMMIT}"
  git checkout "${BLOODLIGHT_COMMIT}"
  git submodule update --init --recursive
  patch -s -p1 <"${script_dir}/bloodlight-firmware-patch.diff"
fi

if [ ! -d "${script_dir}/repos/makair-firmware" ]; then
  git clone --depth 1 https://github.com/makers-for-life/makair-firmware.git \
    "${script_dir}/repos/makair-firmware"
  cd "${script_dir}/repos/makair-firmware"
  git fetch --depth=1 origin "${MAKAIR_COMMIT}"
  git checkout "${MAKAIR_COMMIT}"
  git submodule update --init --recursive
  if [ -f "${script_dir}/makair-firmware-patch.diff" ]; then
    patch -s -p1 <"${script_dir}/makair-firmware-patch.diff"
  fi
  cd "${script_dir}"
fi

if [ ! -d "${script_dir}/repos/ventilator" ]; then
    git clone --depth 1 https://github.com/trail-of-forks/Ventilator.git \
        "${script_dir}/repos/ventilator"
fi
cd "${script_dir}/repos/ventilator"
git fetch --depth=1 origin "${VENTILATOR_COMMIT}"
git checkout -f "${VENTILATOR_COMMIT}"
patch -s -p1 < "${script_dir}/ventilator-patch.diff"
cd "${script_dir}"

# Build using Docker
docker build -t firmware-builder "${script_dir}"

# Build pulseox firmware
docker run --rm \
  -v "${host_repos_dir}/pulseox-firmware:/work/pulseox-firmware" \
  -v "${host_output_dir}:/output" \
  firmware-builder \
  -c "git config --global --add safe.directory /work/pulseox-firmware && \
             cd pulseox-firmware && \
             cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake && \
             cmake --build build -j\$(nproc) && \
             cp build/src/firmware.elf /output/pulseox-firmware.elf"

# Build bloodlight firmware
docker run --rm \
  -v "${host_repos_dir}/bloodlight-firmware:/work/bloodlight-firmware" \
  -v "${host_output_dir}:/output" \
  firmware-builder \
  -c "git config --global --add safe.directory /work/bloodlight-firmware && \
             cd bloodlight-firmware && \
             make -C firmware/libopencm3 && \
             make -C firmware -j\$(nproc) && \
             export PKG_CONFIG_PATH=/usr/lib/arm-linux-gnueabihf/pkgconfig && \
             export PKG_CONFIG_LIBDIR=/usr/lib/arm-linux-gnueabihf/pkgconfig && \
             BL_COMMIT=\$(git rev-parse --verify HEAD) && \
             export CFLAGS=\"-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -DBL_REVISION=1 -DBL_COMMIT_SHA=\\\"\$BL_COMMIT\\\"\" && \
             make -C host CC=arm-linux-gnueabihf-gcc REVISION=1 -j\$(nproc) && \
             cp firmware/bloodlight-firmware.elf /output/bloodlight-firmware.elf && \
             mkdir -p /output/bloodlight && \
             cp -r host/build/bpm /output/bloodlight/bpm && \
             cp -r host/build/normalize /output/bloodlight/normalize && \
             cp -r host/build/bloodview /output/bloodlight/bloodview && \
             cp -r host/build/fft /output/bloodlight/fft && \
             cp -r host/build/calibrate /output/bloodlight/calibrate"

# Build secpump-qemu firmware (in-tree, no clone needed).
# Uses a dedicated minimal builder image so the build works natively on
# Linux amd64/arm64, macOS Intel, and macOS Apple Silicon (the shared
# firmware-builder image pulls armhf cross packages that fail to install
# on linux/arm64 hosts).
# Use the in-container ${script_dir} for the docker BUILD context — `docker
# build` reads the context from the CLI's filesystem and streams it to the
# daemon, so the path must be visible inside this container (where the
# host workspace is bind-mounted at /workspace). ${host_script_dir} is
# only correct for daemon-side `-v` mounts (line below) where the daemon
# itself resolves the path on the host.
docker image inspect secpump-builder >/dev/null 2>&1 \
  || docker build -t secpump-builder "${script_dir}/secpump-qemu"

docker run --rm \
  -v "${host_script_dir}/secpump-qemu:/work/secpump-qemu" \
  -v "${host_output_dir}:/output" \
  secpump-builder \
  -c "cd secpump-qemu && \
             make clean && \
             make -j\$(nproc) && \
             cp build/secpump.elf /output/secpump-qemu.elf && \
             cp build/secpump.bin /output/secpump-qemu.bin"

# Build ventilator firmware + GUI
docker build -t ventilator-builder -f "${script_dir}/Dockerfile.ventilator" "${script_dir}"

docker run --rm \
    -v "${host_script_dir}/repos/ventilator:/work/ventilator" \
    -v "${host_output_dir}:/output" \
    ventilator-builder \
    -c "set -e && \
             git config --global --add safe.directory /work/ventilator && \
             cd /work/ventilator && \
             git checkout -B build-branch && \
             cd software && \
             NANOPB_PLUGIN=\$(which protoc-gen-nanopb) && \
             EXPECTED=\${HOME}/.local/bin/protoc-gen-nanopb && \
             if [ -n \"\$NANOPB_PLUGIN\" ] && [ ! -f \"\$EXPECTED\" ]; then \
                 mkdir -p \$(dirname \"\$EXPECTED\") && \
                 ln -sf \"\$NANOPB_PLUGIN\" \"\$EXPECTED\"; \
             fi && \
             bash common/common.sh generate && \
             cd controller && \
             pio run -e stm32 && \
             cd /work/ventilator/software/common && \
             pio pkg install -e native && \
             cd /work/ventilator/software/gui && \
             conan profile detect --force 2>/dev/null || true && \
             mkdir -p build && cd build && \
             conan install .. --output-folder=. --build=missing -s build_type=Release && \
             cmake .. -DCMAKE_BUILD_TYPE=Release && \
             make -j\$(nproc) && \
             mkdir -p /output/ventilator && \
             cp /work/ventilator/software/controller/.pio/build/stm32/firmware.elf \
                /output/ventilator/controller-firmware.elf && \
             cp /work/ventilator/software/gui/build/bin/ventilator_gui_app \
                /output/ventilator/ventilator_gui_app"
