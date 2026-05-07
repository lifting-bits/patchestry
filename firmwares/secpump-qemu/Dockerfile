# Minimal builder image for the secpump-qemu firmware.
# Works natively on linux/amd64 and linux/arm64 — no armhf cross packages,
# so it builds without modification on Linux (amd64/arm64), macOS Intel,
# and macOS Apple Silicon under Docker Desktop.
FROM ubuntu:22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      gcc-arm-none-eabi \
      libnewlib-arm-none-eabi \
      make \
      python3 \
      python3-pexpect \
      ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /work
ENTRYPOINT ["/bin/bash"]
