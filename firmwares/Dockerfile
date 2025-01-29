ARG IMAGE=ubuntu:22.04
FROM ${IMAGE} AS builder

# Install dependencies
RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y \
    build-essential \
    gcc-arm-none-eabi \
    cmake \
    git \
    python3 \
    python3-pip \
    ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1

SHELL [ "/bin/bash", "-o", "pipefail", "-c" ]

WORKDIR /firmware
COPY . /firmware

RUN git clone --depth=1  --recursive https://github.com/IRNAS/pulseox-firmware.git && \
    cd pulseox-firmware && \
    patch -s -p1 < ../pulseox-firmware-patch.diff && \
    cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake && \
    cmake --build build -j$(nproc)

WORKDIR /firmware

RUN git clone --depth=1 --recursive https://github.com/CodethinkLabs/bloodlight-firmware.git && \
    cd bloodlight-firmware && \
    make -C firmware/libopencm3 && \
    make -C firmware -j$(nproc)

FROM ${IMAGE}
COPY --from=builder /firmware/pulseox-firmware/build/src/firmware.elf /output/pulseox-firmware.elf
COPY --from=builder /firmware/bloodlight-firmware/firmware/bloodlight-firmware.elf /output/bloodlight-firmware.elf
