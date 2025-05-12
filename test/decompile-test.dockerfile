# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

# This should be kept in step with decompile-headless.dockerfile. It doesn't inherit from it only
# because the native Ghidra build takes a long time and we don't need that for test purposes.

FROM eclipse-temurin:21 AS base

# This Dockerfile runs tests for the decompilation / high pcode production part of Patchestry. 
# On Linux, you can build *from the root of the repo* with:
# $ DOCKER_BUILDKIT=1 docker build -t trailofbits/patchestry-test:latest -f test/decompile-test.dockerfile .
#
# Or, if you just want to build the test firmware for local use and *not run*
# unit tests, refer to the firmwares/Dockerfile rather than this Dockerfile.
# For the base, see scripts/ghidra/decompile-headless.dockerfile.

WORKDIR /home/user
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        gcc \
        g++ \
        build-essential \
        gcc-arm-none-eabi \
        libnewlib-arm-none-eabi \
        cmake \
        git \
        python3 \
        python3-pip \
        flex \
        bison \
        wget \
        unzip \
        ninja-build && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1

ENV PULSEOX_COMMIT="54ed8ca6bec36cc13db8f6594e3bd9941937922a"
RUN git clone https://github.com/IRNAS/pulseox-firmware.git
WORKDIR /home/user/pulseox-firmware/
RUN git fetch --depth=1 origin $PULSEOX_COMMIT && \
    git checkout $PULSEOX_COMMIT && \
    git submodule update --init --recursive
COPY firmwares/pulseox-firmware-patch.diff .
RUN patch -s -p1 < pulseox-firmware-patch.diff && \
    mkdir build && \
    cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake && \
    cmake --build build -j$(`nproc`) && \
    cp build/src/firmware.elf /home/user/pulseox-firmware.elf
# if we end up needing the fw build environment to compare to later, 
# then undo this
WORKDIR /home/user
RUN rm -rf pulseox-firmware

# we use PULSEOX_FW_PATH for tests
ENV PULSEOX_FW_PATH="/home/user/pulseox-firmware.elf"

ENV BLOODLIGHT_COMMIT="def737f481d6f0d16db4e94d7b26cbaae9838b41"
RUN git clone https://github.com/CodethinkLabs/bloodlight-firmware.git
WORKDIR /home/user/bloodlight-firmware/
RUN git fetch --depth=1 origin ${BLOODLIGHT_COMMIT} && \
    git checkout ${BLOODLIGHT_COMMIT} && \
    git submodule update --init --recursive && \
    make -C firmware/libopencm3 && \
    make -C firmware -j8 && \
    cp firmware/bloodlight-firmware.elf /home/user/bloodlight-firmware.elf
# if we end up needing the fw build environment to compare to later, 
# then undo this
WORKDIR /home/user
RUN rm -rf bloodlight-firmware

# we use BLOODLIGHT_FW_PATH for tests
ENV BLOODLIGHT_FW_PATH="/home/user/bloodlight-firmware.elf"

RUN apt-get -y autoremove --purge && \
    apt-get purge -y ninja-build cmake libnewlib-arm-none-eabi gcc-arm-none-eabi flex bison && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Ghidra source requires Gradle 8.5+ currently
ARG GRADLE_VERSION=8.5
RUN wget -q https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip -P /tmp \
    && unzip -d /home/user/gradle /tmp/gradle-${GRADLE_VERSION}-bin.zip \
    && rm /tmp/gradle-${GRADLE_VERSION}-bin.zip \
    && ln -s /home/user/gradle/gradle-${GRADLE_VERSION} /home/user/gradle/latest
ENV PATH="/home/user/gradle/latest/bin:${PATH}"

# the production release of Ghidra doesn't have test deps, so we'll build what we need here.
ARG GHIDRA_RELEASE_TAG_NAME="Ghidra_11.3.2_build"
ARG GHIDRA_REPOSITORY=https://github.com/NationalSecurityAgency/ghidra
WORKDIR /home/user
RUN git clone --depth 1 --single-branch --branch $GHIDRA_RELEASE_TAG_NAME https://github.com/NationalSecurityAgency/ghidra.git ghidra_source

ENV GHIDRA_HOME=/home/user/ghidra_source
ENV GHIDRA_INSTALL_DIR=/home/user/ghidra_source
WORKDIR /home/user/ghidra_source
# warning - this gradle dependency fetch takes a very long time
RUN gradle -I gradle/support/fetchDependencies.gradle

RUN gradle prepdev && \
    gradle :GhidraServer:yajswDevUnpack && \ 
    gradle buildGhidra --no-parallel --info --stacktrace && \
    gradle :IntegrationTest:build :IntegrationTest:testClasses :IntegrationTest:testJar

ENV GHIDRA_SCRIPTS=/home/user/ghidra_scripts
ENV GHIDRA_SCRIPT_TESTS=/home/user/ghidra_scripts/test

WORKDIR /home/user/ghidra_scripts/

COPY --chown=user:user scripts/ghidra/domain/ domain/
COPY --chown=user:user scripts/ghidra/util/ util/
COPY --chown=user:user scripts/ghidra/PatchestryDecompileFunctions.java .
COPY --chown=user:user scripts/ghidra/PatchestryListFunctions.java .
COPY --chown=user:user test/scripts/ghidra/util/ test/util/ 
COPY --chown=user:user test/scripts/ghidra/ test/
RUN mv test/build.gradle .

# should run everything including the test task
ENTRYPOINT ["gradle", "build"]