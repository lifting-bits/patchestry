# You can obtain the base container by first having built 
# scripts/ghidra/decompile-headless.dockerfile in this repo.
FROM trailofbits/patchestry-decompilation:latest AS patchestry-headless

# This Dockerfile runs tests for the decompilation part of Patchestry in the same environment
# as the headless container, for uniformity. On Linux, you can build with:
# $ DOCKER_BUILDKIT=1 docker build -t trailofbits/patchestry-test:latest -f decompile-test.dockerfile .
#
# Or, if you just want to build the test firmware for local use and *not run*
# unit tests, refer to the firmwares/Dockerfile rather than this Dockerfile.
# For the base, see scripts/ghidra/decompile-headless.dockerfile.

FROM patchestry-headless AS base

WORKDIR /home/user/

ENV DEBIAN_FRONTEND=noninteractive
RUN sudo apt-get update && sudo apt-get install -y \
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
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1

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
WORKDIR /home/user/
RUN rm -rf /home/user/pulseox-firmware/

WORKDIR /home/user/    
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
WORKDIR /home/user/
RUN rm -rf /home/user/bloodlight-firmware/

# Ghidra source requires Gradle 8.5+ currently
ARG GRADLE_VERSION=8.5
RUN wget -q https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip -P /tmp \
    && sudo unzip -d /opt/gradle /tmp/gradle-${GRADLE_VERSION}-bin.zip \
    && sudo rm /tmp/gradle-${GRADLE_VERSION}-bin.zip \
    && sudo ln -s /opt/gradle/gradle-${GRADLE_VERSION} /opt/gradle/latest
ENV PATH="/opt/gradle/latest/bin:${PATH}"

# the production release of Ghidra doesn't have test deps, so we'll build what we need here.
# GHIDRA_RELEASE_TAG_NAME is defined in the parent Dockerfile so if we ever update that version,
# we also remember to update this so they match.
WORKDIR /tmp
RUN git clone --depth 1 --single-branch --branch $GHIDRA_RELEASE_TAG_NAME https://github.com/NationalSecurityAgency/ghidra.git

WORKDIR /tmp/ghidra
# warning - this gradle dependency fetch takes a very long time
RUN gradle -I gradle/support/fetchDependencies.gradle

RUN gradle prepdev && \
    gradle :GhidraServer:yajswDevUnpack && \ 
    gradle buildGhidra --no-parallel --info --stacktrace && \
    gradle :IntegrationTest:build :IntegrationTest:testClasses :IntegrationTest:testJar

# some additional deps for the Ghidra script tests
RUN wget -O dependencies/flatRepo/gson-2.9.0.jar \
        https://repo1.maven.org/maven2/com/google/code/gson/gson/2.9.0/gson-2.9.0.jar && \
    wget -O dependencies/flatRepo/junit-platform-console-standalone-1.13.0.jar \
        https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.13.0/junit-platform-console-standalone-1.13.0.jar && \
    wget -O dependencies/flatRepo/opentest4j-1.2.0.jar \
        https://repo1.maven.org/maven2/org/opentest4j/opentest4j/1.2.0/opentest4j-1.2.0.jar && \
    wget -O dependencies/flatRepo/apiguardian-api-1.1.2.jar \
        https://repo1.maven.org/maven2/org/apiguardian/apiguardian-api/1.1.2/apiguardian-api-1.1.2.jar

RUN sudo apt-get -y autoremove --purge && \
    sudo apt-get purge -y ninja-build cmake libnewlib-arm-none-eabi gcc-arm-none-eabi flex bison && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

FROM patchestry-headless AS test

ENV DEBIAN_FRONTEND=noninteractive
RUN sudo apt-get update && sudo apt-get install -y \
        python3 \
        python3-pip && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --chown=user:user --from=base /tmp/ghidra/ /home/user/ghidra_source/
COPY --chown=user:user --from=base /opt/gradle/ /opt/gradle/
ENV PATH="/opt/gradle/latest/bin:${PATH}"

WORKDIR /home/user
COPY test/scripts/java/ .

# the GHIDRA_* env vars here come from the parent Dockerfile.
ENV CLASSPATH="/home/user:\
$GHIDRA_SCRIPTS:\
/home/user/ghidra_source/GPL/DMG/data/lib/*:\
/home/user/ghidra_source/dependencies/downloads/*:\
/home/user/ghidra_source/dependencies/flatRepo/*:\
/home/user/ghidra_source/Ghidra/Test/IntegrationTest/build/libs/*:\
/home/user/ghidra_source/Ghidra/Configurations/Public_Release/build/libs/*:\
/home/user/ghidra_source/Ghidra/Debug/Framework-AsyncComm/build/libs/*:\
/home/user/ghidra_source/Ghidra/Debug/Debugger-rmi-trace/build/libs/*\
/home/user/ghidra_source/Ghidra/Debug/TaintAnalysis/build/libs/*:\
/home/user/ghidra_source/Ghidra/Debug/Debugger-isf/build/libs/*:\
/home/user/ghidra_source/Ghidra/Debug/Debugger-jpda/build/libs/*:\
/home/user/ghidra_source/Ghidra/Debug/Debugger-api/build/libs/*:\
/home/user/ghidra_source/Ghidra/Debug/AnnotationValidator/build/libs/*:\
/home/user/ghidra_source/Ghidra/Debug/Framework-TraceModeling/build/libs/*:\
/home/user/ghidra_source/Ghidra/Debug/Debugger/build/libs/*:\
/home/user/ghidra_source/Ghidra/Debug/ProposedUtils/build/libs/*:\
/home/user/ghidra_source/Ghidra/Framework/Graph/build/libs/*:\
/home/user/ghidra_source/Ghidra/Framework/SoftwareModeling/build/libs/*:\
/home/user/ghidra_source/Ghidra/Framework/Generic/build/libs/*:\
/home/user/ghidra_source/Ghidra/Framework/Gui/build/libs/*:\
/home/user/ghidra_source/Ghidra/Framework/Pty/build/libs/*:\
/home/user/ghidra_source/Ghidra/Framework/Emulation/build/libs/*:\
/home/user/ghidra_source/Ghidra/Framework/DB/build/libs/*:\
/home/user/ghidra_source/Ghidra/Framework/Docking/build/libs/*:\
/home/user/ghidra_source/Ghidra/Framework/Utility/build/libs/*:\
/home/user/ghidra_source/Ghidra/Framework/FileSystem/build/libs/*:\
/home/user/ghidra_source/Ghidra/Framework/Help/build/libs/*:\
/home/user/ghidra_source/Ghidra/Framework/Project/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/SuperH4/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/MIPS/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/Atmel/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/HCS12/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/Sparc/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/DATA/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/ARM/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/TI_MSP430/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/eBPF/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/JVM/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/Dalvik/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/PowerPC/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/Loongarch/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/68000/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/8051/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/PIC/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/Xtensa/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/x86/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/Toy/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/tricore/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/RISCV/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/V850/build/libs/*:\
/home/user/ghidra_source/Ghidra/Processors/AARCH64/build/libs/*:\
/home/user/ghidra_source/Ghidra/RuntimeScripts/Common/support/gradle/*:\
/home/user/ghidra_source/Ghidra/Features/PDB/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/MicrosoftDemangler/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/VersionTracking/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/VersionTrackingBSim/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/FileFormats/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/MicrosoftDmang/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/ProgramDiff/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/GnuDemangler/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/Base/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/GraphServices/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/SourceCodeLookup/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/DebugUtils/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/FunctionGraphDecompilerExtension/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/WildcardAssembler/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/core/commons/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/core/yajsw/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/core/permit/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/core/jna/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/core/netty/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/regex/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/commons/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/cron/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/keystore/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/vfs-dbx/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/vfs-webdav/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/abeille/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/yajsw/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/velocity/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/groovy/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/sigar/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/glazedlists/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/slf4j/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/lib/extended/jgoodies/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/data/yajsw-stable-13.12/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraServer/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/Jython/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/SystemEmulation/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/CodeCompare/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/GraphFunctionCalls/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/Recognizers/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/ProgramGraph/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/SwiftDemangler/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/MicrosoftCodeAnalyzer/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/Sarif/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/FunctionID/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/PyGhidra/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/BSim/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/BytePatterns/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/GhidraGo/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/DecompilerDependent/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/ByteViewer/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/BSimFeatureVisualizer/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/Decompiler/build/libs/*:\
/home/user/ghidra_source/Ghidra/Features/FunctionGraph/build/libs/*:\
/home/user/ghidra_source/Ghidra/Extensions/sample/build/libs/*:\
/home/user/ghidra_source/Ghidra/Extensions/SleighDevTools/build/libs/*:\
/home/user/ghidra_source/Ghidra/Extensions/MachineLearning/build/libs/*:\
/home/user/ghidra_source/Ghidra/Extensions/bundle_examples/build/libs/*:\
/home/user/ghidra_source/Ghidra/Extensions/SampleTablePlugin/build/libs/*:\
/home/user/ghidra_source/Ghidra/Extensions/BSimElasticPlugin/build/libs/*:\
/home/user/ghidra_source/GhidraBuild/LaunchSupport/build/libs/*:\
/home/user/ghidra_source/GhidraBuild/Skeleton/build/libs/*:\
/home/user/ghidra_source/GhidraBuild/BuildFiles/Doclets/build/libs/*"

# we want all the tests, even if we add more later
RUN javac -Xlint:-options -cp "$CLASSPATH" `ls $GHIDRA_SCRIPTS/*.java)` `ls ./*Test.java`
# we use PULSEOX_FW_PATH for tests
ENV PULSEOX_FW_PATH="/home/user/pulseox-firmware.elf"
COPY --chown=user:user --from=base $PULSEOX_FW_PATH $PULSEOX_FW_PATH
# we use BLOODLIGHT_FW_PATH for tests
ENV BLOODLIGHT_FW_PATH="/home/user/bloodlight-firmware.elf"
COPY --chown=user:user --from=base $BLOODLIGHT_FW_PATH $BLOODLIGHT_FW_PATH

RUN sudo mkdir -p /mnt/output && \
    java -Djunit.output.dir=/mnt/output org.junit.platform.console.ConsoleLauncher \
        --class-path "$CLASSPATH" \
        --disable-banner \
        --scan-classpath \
        --include-classname ".*Test$"
