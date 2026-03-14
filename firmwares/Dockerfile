# Disable fortified libc wrappers (__sprintf_chk, etc.) in Ubuntu 22.04
FROM ubuntu:22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Enable armhf cross-compilation; configure apt sources per host architecture
RUN dpkg --add-architecture armhf && \
    NATIVE_ARCH=$(dpkg --print-architecture) && \
    if [ "$NATIVE_ARCH" = "amd64" ]; then \
        sed -i 's/^deb /deb [arch=amd64] /' /etc/apt/sources.list && \
        echo "deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports jammy main universe multiverse restricted" >> /etc/apt/sources.list && \
        echo "deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports jammy-updates main universe multiverse restricted" >> /etc/apt/sources.list && \
        echo "deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports jammy-security main universe multiverse restricted" >> /etc/apt/sources.list && \
        echo "deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports jammy-backports main universe multiverse restricted" >> /etc/apt/sources.list ; \
    fi

# Install dependencies
RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc-arm-none-eabi \
    libnewlib-arm-none-eabi \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf \
    libc6-dev-armhf-cross \
    cmake \
    git \
    python3 \
    python3-pip \
    ninja-build \
    pkg-config \
    libudev-dev \
    libfftw3-dev \
    libsdl2-dev \
    libsdl2-ttf-dev \
    libcyaml-dev:armhf \
    libudev-dev:armhf \
    libfftw3-dev:armhf \
    libsdl2-dev:armhf \
    libsdl2-ttf-dev:armhf \
    libyaml-dev:armhf && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# --- Disable FORTIFY_SOURCE globally -----------------------------------------
ENV CFLAGS="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"
ENV CXXFLAGS="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"
ENV CPPFLAGS="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"
ENV DEB_BUILD_MAINT_OPTIONS="hardening=-fortify"
ENV DEB_CFLAGS_APPEND="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"
ENV DEB_CXXFLAGS_APPEND="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"
ENV DEB_CPPFLAGS_APPEND="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"
ENV FORCE_UNSAFE_CONFIGURE=1

WORKDIR /work
ENTRYPOINT ["/bin/bash"]
