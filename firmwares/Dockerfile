ARG IMAGE=ubuntu:22.04
FROM ${IMAGE}

# Install dependencies
RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc-arm-none-eabi \
    libnewlib-arm-none-eabi \
    cmake \
    git \
    python3 \
    python3-pip \
    ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1

SHELL [ "/bin/bash", "-o", "pipefail", "-c" ]
WORKDIR /work
