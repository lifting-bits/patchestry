#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${script_dir}/output"
mkdir -p "${script_dir}/repos"

# Repository commit hashes
PULSEOX_COMMIT="54ed8ca6bec36cc13db8f6594e3bd9941937922a"
BLOODLIGHT_COMMIT="fcc0daef9119ab09914b0c523e7d9d93aad36ea4"

# Clone/update repositories if needed
if [ ! -d "${script_dir}/repos/pulseox-firmware" ]; then
    git clone --depth 1 https://github.com/IRNAS/pulseox-firmware.git \
        "${script_dir}/repos/pulseox-firmware"
    cd "${script_dir}/repos/pulseox-firmware"
    git fetch --depth=1 origin ${PULSEOX_COMMIT}
    git checkout ${PULSEOX_COMMIT}
    git submodule update --init --recursive
    patch -s -p1 < "${script_dir}/pulseox-firmware-patch.diff"
fi

if [ ! -d "${script_dir}/repos/bloodlight-firmware" ]; then
    git clone --depth 1 https://github.com/kumarak/bloodlight-firmware.git \
        "${script_dir}/repos/bloodlight-firmware"
    cd "${script_dir}/repos/bloodlight-firmware"
    git fetch --depth=1 origin ${BLOODLIGHT_COMMIT}
    git checkout ${BLOODLIGHT_COMMIT}
    git submodule update --init --recursive
    patch -s -p1 < "${script_dir}/bloodlight-firmware-patch.diff"
fi

# Build using Docker
docker build -t firmware-builder ${script_dir}

# Build pulseox firmware
docker run --rm \
    -v "${script_dir}/repos/pulseox-firmware:/work/pulseox-firmware" \
    -v "${script_dir}/output:/output" \
    firmware-builder \
    -c "git config --global --add safe.directory /work/pulseox-firmware && \
             cd pulseox-firmware && \
             cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake && \
             cmake --build build -j\$(nproc) && \
             cp build/src/firmware.elf /output/pulseox-firmware.elf"

# Build bloodlight firmware
docker run --rm \
    -v "${script_dir}/repos/bloodlight-firmware:/work/bloodlight-firmware" \
    -v "${script_dir}/output:/output" \
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
