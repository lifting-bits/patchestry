# Build SeaHorn on top of the patchestry LLVM-20 dev image.
#
#   docker build -t seahorn:dev20 -f analysis/seahorn/Dockerfile .
#
# The base image already provides LLVM 20, Clang 20, lld, CMake and Ninja.
# This Dockerfile layers SeaHorn's extra runtime/build dependencies (z3,
# yices2, boost, gmp/mpfr, graphviz) and compiles SeaHorn from
# trail-of-forks/seahorn (dev20 branch) using the in-tree LLVM toolchain.

ARG IMAGE_VERSION=22.04
ARG LLVM_VERSION=20
ARG SEAHORN_REPO=https://github.com/trail-of-forks/seahorn.git
ARG SEAHORN_BRANCH=dev20
ARG BUILD_TYPE=RelWithDebInfo

############################################################
# Stage 1: builder
############################################################
FROM ghcr.io/lifting-bits/patchestry-ubuntu-${IMAGE_VERSION}-llvm-${LLVM_VERSION}-dev:latest AS builder

ARG LLVM_VERSION
ARG SEAHORN_REPO
ARG SEAHORN_BRANCH
ARG BUILD_TYPE
ARG DEBIAN_FRONTEND=noninteractive

ENV LLVM_VERSION=${LLVM_VERSION}

# SeaHorn build/runtime deps not guaranteed to be in the base image.
RUN apt-get update && apt-get install -yqq --no-install-recommends \
        ca-certificates curl wget unzip git \
        zlib1g-dev libzstd-dev \
        libgmp-dev libmpfr-dev \
        libboost1.74-dev \
        libgraphviz-dev graphviz python3-pygraphviz \
        python3 python3-pip \
        lcov gcovr rsync sudo \
    && if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
         apt-get install -yqq --no-install-recommends gcc-multilib; \
       fi \
    && pip3 install --no-cache-dir lit OutputCheck networkx \
    && rm -rf /var/lib/apt/lists/*

# Z3 4.8.9 (matches seahorn dev20 expectations).
WORKDIR /tmp
RUN wget -q https://github.com/Z3Prover/z3/releases/download/z3-4.8.9/z3-4.8.9-x64-ubuntu-16.04.zip \
    && unzip -q z3-4.8.9-x64-ubuntu-16.04.zip \
    && mv z3-4.8.9-x64-ubuntu-16.04 /opt/z3-4.8.9 \
    && rm z3-4.8.9-x64-ubuntu-16.04.zip

# Yices2 2.6.1.
RUN curl -sSOL https://yices.csl.sri.com/releases/2.6.1/yices-2.6.1-x86_64-pc-linux-gnu-static-gmp.tar.gz \
    && tar xf yices-2.6.1-x86_64-pc-linux-gnu-static-gmp.tar.gz \
    && cd yices-2.6.1 && ./install-yices /opt/yices-2.6.1 \
    && cd /tmp && rm -rf yices-2.6.1 yices-2.6.1-x86_64-pc-linux-gnu-static-gmp.tar.gz

# Clone and build SeaHorn.
WORKDIR /seahorn
RUN git clone --depth 1 --branch "${SEAHORN_BRANCH}" "${SEAHORN_REPO}" /seahorn/src
WORKDIR /seahorn/src/build
RUN cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DZ3_ROOT=/opt/z3-4.8.9 \
        -DYICES2_HOME=/opt/yices-2.6.1 \
        -DCMAKE_INSTALL_PREFIX=/opt/seahorn \
        -DCMAKE_C_COMPILER=/usr/local/bin/clang \
        -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++ \
        -DSEA_ENABLE_LLD=ON \
        -DLLVM_DIR=/usr/local/lib/cmake/llvm \
        -DCPACK_GENERATOR=TGZ \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    && cmake --build . --target extra  && cmake .. \
    && cmake --build . --target crab   && cmake .. \
    && cmake --build . --target install

############################################################
# Stage 2: runtime
############################################################
FROM ghcr.io/lifting-bits/patchestry-ubuntu-${IMAGE_VERSION}-llvm-${LLVM_VERSION}-dev:latest

ARG LLVM_VERSION
ARG DEBIAN_FRONTEND=noninteractive

ENV LLVM_VERSION=${LLVM_VERSION} \
    PATH=/opt/seahorn/bin:${PATH}

RUN apt-get update && apt-get install -yqq --no-install-recommends \
        libgmp10 libmpfr6 libboost1.74-dev \
        libgraphviz-dev graphviz \
        python3 python3-pip \
    && pip3 install --no-cache-dir networkx \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/z3-4.8.9   /opt/z3-4.8.9
COPY --from=builder /opt/yices-2.6.1 /opt/yices-2.6.1
COPY --from=builder /opt/seahorn    /opt/seahorn

WORKDIR /workspace
CMD ["/bin/bash"]
