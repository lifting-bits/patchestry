# The base image is pinned by digest and apt is pointed at one Ubuntu archive
# snapshot. amd64 packages normally come from archive.ubuntu.com and armhf
# packages from ports.ubuntu.com; when an update reaches one mirror before the
# other, the armhf copy of a Multi-Arch library is not installable and the
# build fails. A snapshot serves every architecture from the same instant.
# To update, move the digest and UBUNTU_SNAPSHOT together; the snapshot must
# not be older than the base image.
ARG UBUNTU_BASE=ubuntu:22.04@sha256:b000e69a50ff940390233025e7a337fb8469b509e2334fa4d018ac30f3e18cdb

# The snapshot service is HTTPS only and the base image has no CA
# certificates. This stage installs them from the regular mirrors; only the
# certificate bundle is copied out, so no package from a live mirror reaches
# the final image.
FROM ${UBUNTU_BASE} AS certs
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates

# Disable fortified libc wrappers (__sprintf_chk, etc.) in Ubuntu 22.04
FROM ${UBUNTU_BASE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG UBUNTU_SNAPSHOT=20261120T000000Z

COPY --from=certs /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt

# Enable armhf cross-compilation and take every architecture from the snapshot
RUN dpkg --add-architecture armhf && \
    : > /etc/apt/sources.list && \
    for suite in jammy jammy-updates jammy-security jammy-backports; do \
        echo "deb https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT} ${suite} main universe multiverse restricted" >> /etc/apt/sources.list ; \
    done

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
