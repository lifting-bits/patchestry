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

RUN git clone https://github.com/IRNAS/pulseox-firmware.git && \
    cd /firmware/pulseox-firmware && \
    git submodule init && \
    git submodule update && \
    patch -s -p1 < ../pulseox-firmware-patch.diff && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/arm-none-eabi.cmake .. && \
    make

WORKDIR /firmware

RUN git clone https://github.com/CodethinkLabs/bloodlight-firmware.git && \
    cd /firmware/bloodlight-firmware && \
    git submodule init && \
    git submodule update && \
    make -C firmware/libopencm3 && \
    make -C firmware

FROM ${IMAGE}
COPY --from=builder /firmware/pulseox-firmware/build/src/firmware.elf /output/pulseox-firmware.elf
COPY --from=builder /firmware/bloodlight-firmware/firmware/bloodlight-firmware.elf /output/bloodlight-firmware.elf
