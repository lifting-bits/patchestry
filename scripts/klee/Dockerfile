# KLEE Docker image for running symbolic execution harnesses
# Built from trail-of-forks/klee at KLEE_BRANCH/KLEE_REF (defaults below).
#
# Base image: patchestry dev container at LLVM_VERSION (default 22).
# Platform:   linux/amd64 only (CI builds for amd64).
#
# Usage:
#   docker build --platform linux/amd64 -t patchestry/klee:latest -f scripts/klee/Dockerfile scripts/klee
#   docker run --rm --platform linux/amd64 -v /path/to/harness:/work patchestry/klee:latest \
#       --input /work/harness.c --output /work/results
#
# The KLEE branch/commit defaults below target the llvm20 lineage of
# trail-of-forks/klee. Override KLEE_BRANCH and KLEE_REF (and bump
# LLVM_VERSION) when the upstream fork lands a klee build that links
# against LLVM 22.

ARG IMAGE_VERSION=22.04
ARG LLVM_VERSION=22
ARG KLEE_REPO=https://github.com/trail-of-forks/klee.git
ARG KLEE_BRANCH=llvm20
ARG KLEE_REF=4006851f09af391a368578fd4644d236e36d77b7
ARG TARGETPLATFORM=linux/amd64

# ============================================================
# Stage 1: Build KLEE + solver dependencies
# ============================================================
FROM --platform=$TARGETPLATFORM ghcr.io/lifting-bits/patchestry-ubuntu-${IMAGE_VERSION}-llvm-${LLVM_VERSION}-dev:latest AS builder

ARG LLVM_VERSION
ARG KLEE_REPO
ARG KLEE_BRANCH
ARG KLEE_REF
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
COPY create-llvm-symlinks.sh /tmp/create-llvm-symlinks.sh
RUN chmod +x /tmp/create-llvm-symlinks.sh && /tmp/create-llvm-symlinks.sh

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

# ---- Build klee-uclibc (x86_64, compiled to LLVM bitcode) ----
# Disable locale download (www.uclibc.org has intermittent SSL timeouts)
# by disabling UCLIBC_HAS_LOCALE in the config after configure.
RUN git clone --depth 1 --branch klee_uclibc_v1.4 \
        https://github.com/klee/klee-uclibc.git /opt/klee-uclibc \
    && cd /opt/klee-uclibc \
    && ./configure \
        --make-llvm-lib \
        --with-llvm-config=$(which llvm-config) \
    && sed -i \
        -e 's/UCLIBC_HAS_LOCALE=y/# UCLIBC_HAS_LOCALE is not set/' \
        -e 's/__UCLIBC_HAS_XLOCALE__=y/# __UCLIBC_HAS_XLOCALE__ is not set/' \
        .config \
    && yes '' | make oldconfig \
    && make -j$(nproc)

# ---- Build GoogleTest (for KLEE unit tests) ----
RUN git clone --depth 1 --branch v1.16.0 \
        https://github.com/google/googletest.git /opt/googletest

# ---- Build KLEE ----
# Make minisat/STP/Z3 libs visible to the linker
ENV LIBRARY_PATH="/opt/minisat/lib:/opt/stp/lib:/opt/z3/lib"
# Copy patches applied during KLEE build
COPY patches/ /tmp/klee-patches/

RUN git clone --branch "${KLEE_BRANCH}" "${KLEE_REPO}" /tmp/klee \
    && cd /tmp/klee && git checkout "${KLEE_REF}" \
    && git apply /tmp/klee-patches/raise-asm-guard-tri.patch \
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
COPY create-llvm-symlinks.sh /tmp/create-llvm-symlinks.sh
RUN chmod +x /tmp/create-llvm-symlinks.sh && /tmp/create-llvm-symlinks.sh

# Copy KLEE installation + solver libraries
COPY --from=builder /opt/klee /opt/klee
COPY --from=builder /opt/minisat/lib /opt/minisat/lib
COPY --from=builder /opt/stp/lib /opt/stp/lib
COPY --from=builder /opt/z3/lib /opt/z3/lib
COPY --from=builder /opt/klee-uclibc /opt/klee-uclibc
ENV PATH="/opt/klee/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/klee/lib:/opt/stp/lib:/opt/z3/lib:/opt/minisat/lib"
ENV KLEE_UCLIBC_PATH="/opt/klee-uclibc"
ENV KLEE_INCLUDE="/opt/klee/include/klee"

# Copy entrypoint
COPY klee-entrypoint.sh /usr/local/bin/klee-entrypoint.sh
RUN chmod +x /usr/local/bin/klee-entrypoint.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/klee-entrypoint.sh"]
