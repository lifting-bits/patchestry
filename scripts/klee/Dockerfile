# KLEE Docker image for running symbolic execution harnesses
# Built from https://github.com/trail-of-forks/klee/tree/llvm20
#
# Base image: patchestry dev container with LLVM 20 pre-installed
# Platform:   linux/amd64 only (CI builds for amd64)
#
# Usage:
#   docker build --platform linux/amd64 -t patchestry/klee:latest -f scripts/klee/Dockerfile scripts/klee
#   docker run --rm --platform linux/amd64 -v /path/to/harness:/work patchestry/klee:latest \
#       --input /work/harness.c --output /work/results

ARG IMAGE_VERSION=22.04
ARG LLVM_VERSION=20
ARG TARGETPLATFORM=linux/amd64

# ============================================================
# Stage 1: Build KLEE + solver dependencies
# ============================================================
FROM --platform=$TARGETPLATFORM ghcr.io/lifting-bits/patchestry-ubuntu-${IMAGE_VERSION}-llvm-${LLVM_VERSION}-dev:latest AS builder

ARG LLVM_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV LLVM_VERSION=${LLVM_VERSION}
ENV CMAKE_PREFIX_PATH="/usr/local/;/usr/local/lib/cmake/mlir/;/usr/local/lib/cmake/clang/;/usr/local/lib/cmake/llvm/"

# Install additional build dependencies for KLEE and solvers
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsqlite3-dev libgoogle-perftools-dev \
        libboost-program-options-dev libboost-system-dev \
        libgmp-dev \
        bison flex python3 python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure clang/llvm-config symlinks exist (base image may use
# versioned names or /usr/local paths — create both)
RUN for tool in clang clang++ llvm-config llvm-link llvm-ar llvm-dis opt; do \
        if [ ! -e /usr/bin/${tool} ] && [ -e /usr/local/bin/${tool} ]; then \
            ln -sf /usr/local/bin/${tool} /usr/bin/${tool}; \
        fi; \
    done

# Detect LLVM cmake dir — base image installs to /usr/local
ENV LLVM_DIR=/usr/local/lib/cmake/llvm

# ---- Build minisat + STP solver ----
# Combined into one step because minisat's installed cmake config references
# the build-tree source path; /tmp/minisat must exist when STP compiles.
RUN git clone https://github.com/stp/minisat.git /tmp/minisat \
    && cd /tmp/minisat && git checkout 14c78206cd12d1d36b7e042fa758747c135670a4 \
    && cd / \
    && mkdir /tmp/minisat/build && cd /tmp/minisat/build \
    && cmake -G Ninja \
        -DCMAKE_INSTALL_PREFIX=/opt/minisat \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        .. \
    && ninja -j$(nproc) && ninja install \
    && git clone --depth 1 --branch 2.3.4 https://github.com/stp/stp.git /tmp/stp \
    && mkdir /tmp/stp/build && cd /tmp/stp/build \
    && cmake -G Ninja \
        -DCMAKE_INSTALL_PREFIX=/opt/stp \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DENABLE_PYTHON_INTERFACE=OFF \
        -Dminisat_DIR=/opt/minisat/lib/cmake/minisat \
        .. \
    && ninja -j$(nproc) && ninja install \
    && rm -rf /tmp/minisat /tmp/stp

# ---- Build Z3 solver ----
RUN git clone --depth 1 --branch z3-4.8.15 https://github.com/Z3Prover/z3.git /tmp/z3 \
    && mkdir /tmp/z3/build && cd /tmp/z3/build \
    && cmake -G Ninja \
        -DCMAKE_INSTALL_PREFIX=/opt/z3 \
        -DCMAKE_BUILD_TYPE=Release \
        -DZ3_BUILD_PYTHON_BINDINGS=OFF \
        -DZ3_BUILD_TEST_EXECUTABLES=OFF \
        .. \
    && ninja -j$(nproc) && ninja install \
    && rm -rf /tmp/z3

# ---- Build klee-uclibc (x86_64) ----
# Disable locale download (www.uclibc.org has intermittent SSL timeouts)
# by disabling UCLIBC_HAS_LOCALE in the config after configure.
RUN git clone --depth 1 --branch klee_uclibc_v1.4 \
        https://github.com/klee/klee-uclibc.git /tmp/klee-uclibc \
    && cd /tmp/klee-uclibc \
    && ./configure \
        --make-llvm-lib \
        --with-llvm-config=$(which llvm-config) \
    && sed -i \
        -e 's/UCLIBC_HAS_LOCALE=y/# UCLIBC_HAS_LOCALE is not set/' \
        -e 's/__UCLIBC_HAS_XLOCALE__=y/# __UCLIBC_HAS_XLOCALE__ is not set/' \
        .config \
    && yes '' | make oldconfig \
    && make -j$(nproc) \
    && mkdir -p /opt/klee-uclibc \
    && cp -a . /opt/klee-uclibc/ \
    && rm -rf /tmp/klee-uclibc

# ---- Build klee-uclibc (ARM32) ----
# Cross-compile uclibc to ARM32 LLVM bitcode so KLEE can link with
# ARM32 harnesses without data layout mismatches.
# Uses a wrapper script so configure detects ARM, patches out the
# ncurses check (only needed for menuconfig), and uses ARM sysroot.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-arm-linux-gnueabihf libc6-dev-armhf-cross \
        linux-libc-dev-armhf-cross \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 --branch klee_uclibc_v1.4 \
        https://github.com/klee/klee-uclibc.git /tmp/klee-uclibc-arm32 \
    && cd /tmp/klee-uclibc-arm32 \
    && printf '#!/bin/sh\nexec clang --target=arm-linux-gnueabihf --sysroot=/usr/arm-linux-gnueabihf "$@"\n' \
        > /tmp/arm-clang && chmod +x /tmp/arm-clang \
    && sed -i '/checkForNCurses/s/not checkForNCurses([^)]*)/False/' configure \
    && CC=/tmp/arm-clang ./configure \
        --make-llvm-lib \
        --with-llvm-config=$(which llvm-config) \
        --with-cc=/tmp/arm-clang \
    && sed -i \
        -e 's/# TARGET_arm is not set/TARGET_arm=y/' \
        -e 's/TARGET_x86_64=y/# TARGET_x86_64 is not set/' \
        -e 's|^KERNEL_HEADERS=.*|KERNEL_HEADERS="/usr/arm-linux-gnueabihf/include"|' \
        -e 's/^UCLIBC_EXTRA_CFLAGS=.*/UCLIBC_EXTRA_CFLAGS="-Wno-error=int-conversion -Wno-error=incompatible-function-pointer-types --target=arm-linux-gnueabihf --sysroot=\/usr\/arm-linux-gnueabihf"/' \
        -e 's/UCLIBC_HAS_LOCALE=y/# UCLIBC_HAS_LOCALE is not set/' \
        .config \
    && sed -i \
        -e 's|^HOSTCC.*|HOSTCC = gcc|' \
        -e 's|^BUILD_CFLAGS.*|BUILD_CFLAGS = -O2 -Wall|' \
        Makefile.klee \
    && yes '' | make oldconfig \
    && make -j$(nproc) \
    && mkdir -p /opt/klee-uclibc-arm32 \
    && cp -a . /opt/klee-uclibc-arm32/ \
    && rm -rf /tmp/klee-uclibc-arm32

# ---- Build GoogleTest (for KLEE unit tests) ----
RUN git clone --depth 1 --branch v1.16.0 \
        https://github.com/google/googletest.git /opt/googletest

# ---- Build KLEE ----
# Make minisat/STP/Z3 libs visible to the linker
ENV LIBRARY_PATH="/opt/minisat/lib:/opt/stp/lib:/opt/z3/lib"
RUN git clone --branch llvm20 \
        https://github.com/trail-of-forks/klee.git /tmp/klee \
    && cd /tmp/klee && git checkout 4006851f09af391a368578fd4644d236e36d77b7 \
    && mkdir build && cd build \
    && cmake -G Ninja \
        -DCMAKE_INSTALL_PREFIX=/opt/klee \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DLLVM_DIR=${LLVM_DIR} \
        -DLLVMCC=$(which clang) \
        -DLLVMCXX=$(which clang++) \
        -DENABLE_SOLVER_STP=ON \
        -DSTP_DIR=/opt/stp \
        -DENABLE_SOLVER_Z3=ON \
        -DZ3_INCLUDE_DIRS=/opt/z3/include \
        -DZ3_LIBRARIES=/opt/z3/lib/libz3.so \
        -DENABLE_POSIX_RUNTIME=ON \
        -DKLEE_UCLIBC_PATH=/opt/klee-uclibc \
        -DENABLE_TCMALLOC=ON \
        -DENABLE_KLEE_ASSERTS=ON \
        -DENABLE_UNIT_TESTS=ON \
        -DGTEST_SRC_DIR=/opt/googletest \
        -DENABLE_SYSTEM_TESTS=OFF \
        -DKLEE_RUNTIME_BUILD_TYPE="Release+Debug+Asserts" \
        /tmp/klee \
    && ninja -j$(nproc) && ninja install \
    && rm -rf /tmp/klee /opt/googletest

# ============================================================
# Stage 2: Runtime image (same base for LLVM tools at runtime)
# ============================================================
FROM --platform=$TARGETPLATFORM ghcr.io/lifting-bits/patchestry-ubuntu-${IMAGE_VERSION}-llvm-${LLVM_VERSION}-dev:latest AS runtime

ARG LLVM_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV LLVM_VERSION=${LLVM_VERSION}

# Install runtime-only dependencies not in the base image
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsqlite3-0 libgoogle-perftools4 python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure clang/llvm-config symlinks
RUN for tool in clang clang++ llvm-config llvm-link llvm-ar llvm-dis opt \
        llvm-nm llvm-objdump; do \
        if [ ! -e /usr/bin/${tool} ] && [ -e /usr/local/bin/${tool} ]; then \
            ln -sf /usr/local/bin/${tool} /usr/bin/${tool}; \
        fi; \
    done

# Copy KLEE installation + solver libraries
COPY --from=builder /opt/klee /opt/klee
COPY --from=builder /opt/minisat/lib /opt/minisat/lib
COPY --from=builder /opt/stp/lib /opt/stp/lib
COPY --from=builder /opt/z3/lib /opt/z3/lib
COPY --from=builder /opt/klee-uclibc /opt/klee-uclibc
COPY --from=builder /opt/klee-uclibc-arm32 /opt/klee-uclibc-arm32

ENV PATH="/opt/klee/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/klee/lib:/opt/stp/lib:/opt/z3/lib:/opt/minisat/lib"
ENV KLEE_UCLIBC_PATH="/opt/klee-uclibc"
ENV KLEE_UCLIBC_ARM32_PATH="/opt/klee-uclibc-arm32"
ENV KLEE_INCLUDE="/opt/klee/include/klee"

# Copy entrypoint
COPY klee-entrypoint.sh /usr/local/bin/klee-entrypoint.sh
RUN chmod +x /usr/local/bin/klee-entrypoint.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/klee-entrypoint.sh"]
