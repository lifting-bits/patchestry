# Minimal builder image for the makair-qemu firmware (Cortex-M4).
# Same shape as firmwares/secpump-qemu/Dockerfile but adds g++ for the
# upstream MakAir C++ sources we link in from firmwares/repos/makair-firmware.
# Works natively on linux/amd64 and linux/arm64 — no armhf cross packages.
FROM ubuntu:22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      gcc-arm-none-eabi \
      libnewlib-arm-none-eabi \
      libstdc++-arm-none-eabi-newlib \
      make \
      python3 \
      python3-pexpect \
      ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /work
ENTRYPOINT ["/bin/bash"]
