#!/bin/bash

set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${script_dir}/output"
mkdir -p "${script_dir}/repos"

# Repository commit hashes
PULSEOX_COMMIT="54ed8ca6bec36cc13db8f6594e3bd9941937922a"
BLOODLIGHT_COMMIT="def737f481d6f0d16db4e94d7b26cbaae9838b41"

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
    git clone --depth 1 https://github.com/CodethinkLabs/bloodlight-firmware.git \
        "${script_dir}/repos/bloodlight-firmware"
    cd "${script_dir}/repos/bloodlight-firmware"
    git fetch --depth=1 origin ${BLOODLIGHT_COMMIT}
    git checkout ${BLOODLIGHT_COMMIT}
    git submodule update --init --recursive
fi

# Build using Docker
docker build -t firmware-builder ${script_dir}

# Build pulseox firmware
docker run --rm \
    -v "${script_dir}/repos/pulseox-firmware:/work/pulseox-firmware" \
    -v "${script_dir}/output:/output" \
    firmware-builder \
    bash -c "cd pulseox-firmware && \
             cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake && \
             cmake --build build -j\$(nproc) && \
             cp build/src/firmware.elf /output/pulseox-firmware.elf"

# Build bloodlight firmware
docker run --rm \
    -v "${script_dir}/repos/bloodlight-firmware:/work/bloodlight-firmware" \
    -v "${script_dir}/output:/output" \
    firmware-builder \
    bash -c "cd bloodlight-firmware && \
             make -C firmware/libopencm3 && \
             make -C firmware -j\$(nproc) && \
             cp firmware/bloodlight-firmware.elf /output/bloodlight-firmware.elf"
