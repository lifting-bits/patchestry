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

WORKDIR /opt
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
WORKDIR /opt/pulseox-firmware/
RUN git fetch --depth=1 origin $PULSEOX_COMMIT && \
    git checkout $PULSEOX_COMMIT && \
    git submodule update --init --recursive
COPY firmwares/pulseox-firmware-patch.diff .
RUN patch -s -p1 < pulseox-firmware-patch.diff && \
    mkdir build && \
    cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake && \
    cmake --build build -j$(`nproc`) && \
    cp build/src/firmware.elf /opt/pulseox-firmware.elf
# if we end up needing the fw build environment to compare to later, 
# then undo this
WORKDIR /opt
RUN rm -rf pulseox-firmware

# we use PULSEOX_FW_PATH for tests
ENV PULSEOX_FW_PATH="/opt/pulseox-firmware.elf"

ENV BLOODLIGHT_COMMIT="def737f481d6f0d16db4e94d7b26cbaae9838b41"
RUN git clone https://github.com/CodethinkLabs/bloodlight-firmware.git
WORKDIR /opt/bloodlight-firmware/
RUN git fetch --depth=1 origin ${BLOODLIGHT_COMMIT} && \
    git checkout ${BLOODLIGHT_COMMIT} && \
    git submodule update --init --recursive && \
    make -C firmware/libopencm3 && \
    make -C firmware -j8 && \
    cp firmware/bloodlight-firmware.elf /opt/bloodlight-firmware.elf
# if we end up needing the fw build environment to compare to later, 
# then undo this
WORKDIR /opt
RUN rm -rf bloodlight-firmware

# we use BLOODLIGHT_FW_PATH for tests
ENV BLOODLIGHT_FW_PATH="/opt/bloodlight-firmware.elf"

RUN apt-get -y autoremove --purge && \
    apt-get purge -y ninja-build cmake libnewlib-arm-none-eabi gcc-arm-none-eabi flex bison && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Ghidra source requires Gradle 8.5+ currently
ARG GRADLE_VERSION=8.5
RUN wget -q https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip -P /tmp \
    && unzip -d /opt/gradle /tmp/gradle-${GRADLE_VERSION}-bin.zip \
    && rm /tmp/gradle-${GRADLE_VERSION}-bin.zip \
    && ln -s /opt/gradle/gradle-${GRADLE_VERSION} /opt/gradle/latest
ENV PATH="/opt/gradle/latest/bin:${PATH}"

# the production release of Ghidra doesn't have test deps, so we'll build what we need here.
# GHIDRA_RELEASE_TAG_NAME is defined in the parent Dockerfile so if we ever update that version,
# we also remember to update this so they match.
# for getting release build-matching source for test jars in inheritor
# should match version(s) used here (above) so that tests make sense
ARG GHIDRA_RELEASE_TAG_NAME="Ghidra_11.3.2_build"
ARG GHIDRA_REPOSITORY=https://github.com/NationalSecurityAgency/ghidra
ENV GHIDRA_HOME=/opt/ghidra
WORKDIR /opt
RUN git clone --depth 1 --single-branch --branch $GHIDRA_RELEASE_TAG_NAME https://github.com/NationalSecurityAgency/ghidra.git

WORKDIR /opt/ghidra
# warning - this gradle dependency fetch takes a very long time
RUN gradle -I gradle/support/fetchDependencies.gradle

RUN gradle prepdev && \
    gradle :GhidraServer:yajswDevUnpack && \ 
    gradle buildGhidra --no-parallel --info --stacktrace && \
    gradle :IntegrationTest:build :IntegrationTest:testClasses :IntegrationTest:testJar

# some additional deps for the Ghidra script tests
RUN wget -O dependencies/downloads/gson-2.9.0.jar \
        https://repo1.maven.org/maven2/com/google/code/gson/gson/2.9.0/gson-2.9.0.jar && \
    wget -O dependencies/downloads/junit-platform-console-standalone-1.13.0.jar \
        https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.13.0/junit-platform-console-standalone-1.13.0.jar && \
    wget -O dependencies/downloads/log4j-core-2.23.1.jar \
        https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.23.1/log4j-core-2.23.1.jar && \
    wget -O dependencies/downloads/log4j-api-2.23.1.jar \
        https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.23.1/log4j-api-2.23.1.jar && \
    wget -O dependencies/downloads/jdom-1.1.3.jar \
        https://repo1.maven.org/maven2/org/jdom/jdom/1.1.3/jdom-1.1.3.jar && \
    wget -O dependencies/downloads/commons-lang3-3.14.0.jar \
        https://repo1.maven.org/maven2/org/apache/commons/commons-lang3/3.14.0/commons-lang3-3.14.0.jar && \
    wget -O dependencies/downloads/commons-collections4-4.4.jar \ 
        https://repo1.maven.org/maven2/org/apache/commons/commons-collections4/4.4/commons-collections4-4.4.jar && \
    wget -O dependencies/downloads/commons-io-2.15.1.jar \
        https://repo1.maven.org/maven2/commons-io/commons-io/2.15.1/commons-io-2.15.1.jar && \
    wget -O dependencies/downloads/msv-rngconverter-2011.1-redhat-2.jar \
        https://maven.repository.redhat.com/earlyaccess/all/net/java/dev/msv/msv-rngconverter/2011.1-redhat-2/msv-rngconverter-2011.1-redhat-2.jar && \
    wget -O dependencies/downloads/antlr-runtime-3.5.2.jar \
        https://repo1.maven.org/maven2/org/antlr/antlr-runtime/3.5.2/antlr-runtime-3.5.2.jar && \
    wget -O dependencies/downloads/guava-32.1.2-jre.jar \
        https://repo1.maven.org/maven2/com/google/guava/guava/32.1.2-jre/guava-32.1.2-jre.jar && \
    wget -O dependencies/downloads/javahelp-2.0.05.jar \ 
        https://repo1.maven.org/maven2/javax/help/javahelp/2.0.05/javahelp-2.0.05.jar && \
    wget -O dependencies/downloads/commons-compress-1.21.jar \
        https://repo1.maven.org/maven2/org/apache/commons/commons-compress/1.21/commons-compress-1.21.jar && \
    wget -O dependencies/downloads/mockito-core-5.11.0.jar \
        https://repo1.maven.org/maven2/org/mockito/mockito-core/5.11.0/mockito-core-5.11.0.jar && \
    wget -O dependencies/downloads/byte-buddy-1.14.12.jar \
        https://repo1.maven.org/maven2/net/bytebuddy/byte-buddy/1.14.12/byte-buddy-1.14.12.jar && \
    wget -O dependencies/downloads/byte-buddy-agent-1.14.12.jar \
        https://repo1.maven.org/maven2/net/bytebuddy/byte-buddy-agent/1.14.12/byte-buddy-agent-1.14.12.jar && \
    wget -O dependencies/downloads/objenesis-3.3.jar \
        https://repo1.maven.org/maven2/org/objenesis/objenesis/3.3/objenesis-3.3.jar

ENV GHIDRA_SCRIPTS=/opt/scripts/ghidra
ENV CLASSES=/opt/classes
ENV GHIDRA_SCRIPT_TESTS=/opt/test/scripts/ghidra
ENV CLASSPATH="/opt:\
$GHIDRA_SCRIPTS:\
$GHIDRA_SCRIPT_TESTS:\
$CLASSES:\
/opt/ghidra/GPL/DMG/data/lib/*:\
/opt/ghidra/dependencies/downloads/*:\
/opt/ghidra/dependencies/flatRepo/*:\
/opt/ghidra/Ghidra/Test/IntegrationTest/build/libs/*:\
/opt/ghidra/Ghidra/Debug/Framework-AsyncComm/build/libs/*:\
/opt/ghidra/Ghidra/Debug/Debugger-rmi-trace/build/libs/*\
/opt/ghidra/Ghidra/Debug/TaintAnalysis/build/libs/*:\
/opt/ghidra/Ghidra/Debug/Debugger-isf/build/libs/*:\
/opt/ghidra/Ghidra/Debug/Debugger-jpda/build/libs/*:\
/opt/ghidra/Ghidra/Debug/Debugger-api/build/libs/*:\
/opt/ghidra/Ghidra/Debug/AnnotationValidator/build/libs/*:\
/opt/ghidra/Ghidra/Debug/Framework-TraceModeling/build/libs/*:\
/opt/ghidra/Ghidra/Debug/Debugger/build/libs/*:\
/opt/ghidra/Ghidra/Debug/ProposedUtils/build/libs/*:\
/opt/ghidra/Ghidra/Framework/Graph/build/libs/*:\
/opt/ghidra/Ghidra/Framework/SoftwareModeling/build/libs/*:\
/opt/ghidra/Ghidra/Framework/Generic/build/libs/*:\
/opt/ghidra/Ghidra/Framework/Gui/build/libs/*:\
/opt/ghidra/Ghidra/Framework/Pty/build/libs/*:\
/opt/ghidra/Ghidra/Framework/Emulation/build/libs/*:\
/opt/ghidra/Ghidra/Framework/DB/build/libs/*:\
/opt/ghidra/Ghidra/Framework/Docking/build/libs/*:\
/opt/ghidra/Ghidra/Framework/Utility/build/libs/*:\
/opt/ghidra/Ghidra/Framework/FileSystem/build/libs/*:\
/opt/ghidra/Ghidra/Framework/Help/build/libs/*:\
/opt/ghidra/Ghidra/Framework/Project/build/libs/*:\
/opt/ghidra/Ghidra/Processors/SuperH4/build/libs/*:\
/opt/ghidra/Ghidra/Processors/MIPS/build/libs/*:\
/opt/ghidra/Ghidra/Processors/Atmel/build/libs/*:\
/opt/ghidra/Ghidra/Processors/HCS12/build/libs/*:\
/opt/ghidra/Ghidra/Processors/Sparc/build/libs/*:\
/opt/ghidra/Ghidra/Processors/DATA/build/libs/*:\
/opt/ghidra/Ghidra/Processors/ARM/build/libs/*:\
/opt/ghidra/Ghidra/Processors/TI_MSP430/build/libs/*:\
/opt/ghidra/Ghidra/Processors/eBPF/build/libs/*:\
/opt/ghidra/Ghidra/Processors/JVM/build/libs/*:\
/opt/ghidra/Ghidra/Processors/Dalvik/build/libs/*:\
/opt/ghidra/Ghidra/Processors/PowerPC/build/libs/*:\
/opt/ghidra/Ghidra/Processors/Loongarch/build/libs/*:\
/opt/ghidra/Ghidra/Processors/68000/build/libs/*:\
/opt/ghidra/Ghidra/Processors/8051/build/libs/*:\
/opt/ghidra/Ghidra/Processors/PIC/build/libs/*:\
/opt/ghidra/Ghidra/Processors/Xtensa/build/libs/*:\
/opt/ghidra/Ghidra/Processors/x86/build/libs/*:\
/opt/ghidra/Ghidra/Processors/Toy/build/libs/*:\
/opt/ghidra/Ghidra/Processors/tricore/build/libs/*:\
/opt/ghidra/Ghidra/Processors/RISCV/build/libs/*:\
/opt/ghidra/Ghidra/Processors/V850/build/libs/*:\
/opt/ghidra/Ghidra/Processors/AARCH64/build/libs/*:\
/opt/ghidra/Ghidra/Features/PDB/build/libs/*:\
/opt/ghidra/Ghidra/Features/MicrosoftDemangler/build/libs/*:\
/opt/ghidra/Ghidra/Features/VersionTracking/build/libs/*:\
/opt/ghidra/Ghidra/Features/VersionTrackingBSim/build/libs/*:\
/opt/ghidra/Ghidra/Features/FileFormats/build/libs/*:\
/opt/ghidra/Ghidra/Features/MicrosoftDmang/build/libs/*:\
/opt/ghidra/Ghidra/Features/ProgramDiff/build/libs/*:\
/opt/ghidra/Ghidra/Features/GnuDemangler/build/libs/*:\
/opt/ghidra/Ghidra/Features/Base/build/libs/*:\
/opt/ghidra/Ghidra/Features/GraphServices/build/libs/*:\
/opt/ghidra/Ghidra/Features/SourceCodeLookup/build/libs/*:\
/opt/ghidra/Ghidra/Features/DebugUtils/build/libs/*:\
/opt/ghidra/Ghidra/Features/FunctionGraphDecompilerExtension/build/libs/*:\
/opt/ghidra/Ghidra/Features/WildcardAssembler/build/libs/*:\
/opt/ghidra/Ghidra/Features/GhidraServer/build/libs/*:\
/opt/ghidra/Ghidra/Features/Jython/build/libs/*:\
/opt/ghidra/Ghidra/Features/SystemEmulation/build/libs/*:\
/opt/ghidra/Ghidra/Features/CodeCompare/build/libs/*:\
/opt/ghidra/Ghidra/Features/GraphFunctionCalls/build/libs/*:\
/opt/ghidra/Ghidra/Features/Recognizers/build/libs/*:\
/opt/ghidra/Ghidra/Features/ProgramGraph/build/libs/*:\
/opt/ghidra/Ghidra/Features/SwiftDemangler/build/libs/*:\
/opt/ghidra/Ghidra/Features/MicrosoftCodeAnalyzer/build/libs/*:\
/opt/ghidra/Ghidra/Features/Sarif/build/libs/*:\
/opt/ghidra/Ghidra/Features/FunctionID/build/libs/*:\
/opt/ghidra/Ghidra/Features/PyGhidra/build/libs/*:\
/opt/ghidra/Ghidra/Features/BSim/build/libs/*:\
/opt/ghidra/Ghidra/Features/BytePatterns/build/libs/*:\
/opt/ghidra/Ghidra/Features/GhidraGo/build/libs/*:\
/opt/ghidra/Ghidra/Features/DecompilerDependent/build/libs/*:\
/opt/ghidra/Ghidra/Features/ByteViewer/build/libs/*:\
/opt/ghidra/Ghidra/Features/BSimFeatureVisualizer/build/libs/*:\
/opt/ghidra/Ghidra/Features/Decompiler/build/libs/*:\
/opt/ghidra/Ghidra/Features/FunctionGraph/build/libs/*:\
/opt/ghidra/Ghidra/Extensions/sample/build/libs/*:\
/opt/ghidra/Ghidra/Extensions/SleighDevTools/build/libs/*:\
/opt/ghidra/Ghidra/Extensions/MachineLearning/build/libs/*:\
/opt/ghidra/Ghidra/Extensions/bundle_examples/build/libs/*:\
/opt/ghidra/Ghidra/Extensions/SampleTablePlugin/build/libs/*:\
/opt/ghidra/Ghidra/Extensions/BSimElasticPlugin/build/libs/*:"


WORKDIR /opt/ghidra_scripts/

COPY test/scripts/ghidra/ $GHIDRA_SCRIPT_TESTS
COPY test/scripts/ghidra/util/ ${GHIDRA_SCRIPT_TESTS}/util/

COPY scripts/ghidra/domain/ ${GHIDRA_SCRIPTS}/domain/
COPY scripts/ghidra/util/ ${GHIDRA_SCRIPTS}/util/
COPY scripts/ghidra/*.java $GHIDRA_SCRIPTS

WORKDIR /opt
RUN mkdir $CLASSES
RUN javac \
    -Xlint:-options \
    -cp "$CLASSPATH" \
    -d "$CLASSES" \
    $GHIDRA_SCRIPTS/domain/*.java \
    $GHIDRA_SCRIPTS/util/*.java \
    $GHIDRA_SCRIPTS/*.java \
    $GHIDRA_SCRIPT_TESTS/*.java \
    $GHIDRA_SCRIPT_TESTS/util/*.java

RUN mkdir /opt/test_output && \
    java \
        -Djunit.output.dir=/opt/test_output \
        --add-opens java.desktop/sun.awt=ALL-UNNAMED \
        org.junit.platform.console.ConsoleLauncher \
            --class-path "${CLASSPATH}" \
            --disable-banner \
            --scan-classpath \
            --include-classname ".*Test$"
