#!/bin/bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
mkdir -p "${script_dir}/output"
mkdir -p "${script_dir}/repos"

if [ -z "${BUILDX_CONFIG:-}" ]; then
  export BUILDX_CONFIG="${repo_root}/builds/docker-buildx/firmwares"
fi
mkdir -p "${BUILDX_CONFIG}"

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
VENTILATOR_COMMIT="6165c82de293d66b71f43040a2f145ab70bb49c0"

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

if [ ! -d "${script_dir}/repos/ventilator" ]; then
    git clone --depth 1 https://github.com/RespiraWorks/Ventilator.git \
        "${script_dir}/repos/ventilator"
fi
cd "${script_dir}/repos/ventilator"
git fetch --depth=1 origin "${VENTILATOR_COMMIT}"
git checkout -f "${VENTILATOR_COMMIT}"
cd "${script_dir}"

# Build using Docker
docker build -t firmware-builder "${host_script_dir}"

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

# Build ventilator firmware + GUI
docker build -t ventilator-builder -f "${script_dir}/Dockerfile.ventilator" "${script_dir}"

docker run --rm \
    -v "${script_dir}/repos/ventilator:/work/ventilator" \
    -v "${script_dir}/output:/output" \
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
             conan profile detect --force 2>/dev/null ; \
             cd /work/ventilator/software/gui && \
             mkdir -p build && cd build && \
             cmake .. -DCMAKE_BUILD_TYPE=Release && \
             make -j\$(nproc) && \
             mkdir -p /output/ventilator && \
             cp /work/ventilator/software/controller/.pio/build/stm32/firmware.elf \
                /output/ventilator/controller-firmware.elf && \
             cp /work/ventilator/software/gui/build/bin/ventilator_gui_app \
                /output/ventilator/ventilator_gui_app"
