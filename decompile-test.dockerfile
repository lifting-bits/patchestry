# You can obtain the base container by first having built 
# scripts/ghidra/decompile-headless.dockerfile in this repo.
FROM trailofbits/patchestry-decompilation:latest AS patchestry-headless

# This Dockerfile is building and running tests in the same environment
# as the headless container. On Linux, you can build with:
# $ DOCKER_BUILDKIT=1 docker build -t trailofbits/patchestry-test:latest -f decompile-test.dockerfile .
#
# Or, if you just want to build the test firmware for local use and *not run*
# unit tests, refer to the firmwares/Dockerfile rather than this Dockerfile.
# For the headless container, see scripts/ghidra/decompile-headless.dockerfile.

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
        wget \
        ninja-build && \
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1

ENV PULSEOX_COMMIT="54ed8ca6bec36cc13db8f6594e3bd9941937922a"
RUN git clone --depth 1 https://github.com/IRNAS/pulseox-firmware.git
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
# important - may not compile with the --depth 1 flag
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

RUN sudo apt-get -y autoremove --purge && \
    sudo apt-get purge -y ninja-build cmake libnewlib-arm-none-eabi gcc-arm-none-eabi && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

FROM patchestry-headless AS test

# we use PULSEOX_FW_PATH for tests
ENV PULSEOX_FW_PATH="/home/user/pulseox-firmware.elf"
COPY --chown=user:user --from=base $PULSEOX_FW_PATH $PULSEOX_FW_PATH
# we use BLOODLIGHT_FW_PATH for tests
ENV BLOODLIGHT_FW_PATH="/home/user/bloodlight-firmware.elf"
COPY --chown=user:user --from=base $BLOODLIGHT_FW_PATH $BLOODLIGHT_FW_PATH

ENV DEBIAN_FRONTEND=noninteractive
RUN sudo apt-get update && sudo apt-get install -y \
        git \
        unzip \
        wget \   
        flex \
        bison \
        build-essential \
        python3 \
        python3-pip && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ARG GRADLE_VERSION=8.5
RUN wget -q https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip -P /tmp \
    && sudo unzip -d /opt/gradle /tmp/gradle-${GRADLE_VERSION}-bin.zip \
    && sudo rm /tmp/gradle-${GRADLE_VERSION}-bin.zip \
    && sudo ln -s /opt/gradle/gradle-${GRADLE_VERSION} /opt/gradle/latest
ENV PATH="/opt/gradle/latest/bin:${PATH}"

# the production release of Ghidra doesn't have test deps, so we'll build what we need here.
WORKDIR /tmp
RUN git clone --depth 1 --branch $GHIDRA_RELEASE_TAG https://github.com/NationalSecurityAgency/ghidra.git && \
    mv ghidra /home/user/ghidra-source

WORKDIR /home/user/ghidra-source
# warning - this gradle dependency fetch takes a very long time
RUN gradle -I gradle/support/fetchDependencies.gradle
# skips building the docs
RUN gradle prepdev && \
    gradle :GhidraServer:yajswDevUnpack && \ 
    gradle buildGhidra --no-parallel --info --stacktrace

WORKDIR /home/user
COPY test/scripts/java/ .
RUN wget -O junit-platform-console-standalone.jar \
    https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.10.1/junit-platform-console-standalone-1.10.1.jar
RUN wget -O /home/user/ghidra-source/dependencies/flatRepo/gson-2.9.0.jar \
    https://repo1.maven.org/maven2/com/google/code/gson/gson/2.9.0/gson-2.9.0.jar

# the GHIDRA_* env vars here come from the parent Dockerfile.
ARG GHIDRA_SOURCE="/home/user/ghidra-source/Ghidra"
ENV CLASSPATH="/home/user:\
$GHIDRA_SCRIPTS:\
$GHIDRA_SOURCE/Features/Base/build/libs/Base.jar:\
$GHIDRA_SOURCE/Features/Base/build/libs/Base-test.jar:\
$GHIDRA_SOURCE/Framework/Generic/build/libs/Generic.jar:\
$GHIDRA_SOURCE/Framework/Generic/build/libs/Generic-test.jar:\
$GHIDRA_SOURCE/Framework/Docking/build/libs/Docking.jar:\
$GHIDRA_SOURCE/Framework/Docking/build/libs/Docking-test.jar:\
$GHIDRA_SOURCE/Framework/SoftwareModeling/build/libs/SoftwareModeling.jar:\
$GHIDRA_SOURCE/Framework/SoftwareModeling/build/libs/SoftwareModeling-test.jar:\
$GHIDRA_SOURCE/Framework/Utility/build/libs/Utility.jar:\
$GHIDRA_SOURCE/Framework/Utility/build/libs/Utility-test.jar:\
$GHIDRA_SOURCE/Framework/Project/build/libs/Project.jar:\
$GHIDRA_SOURCE/Framework/Project/build/libs/Project-test.jar:\
$GHIDRA_SOURCE/Features/Decompiler/build/libs/Decompiler.jar:\
$GHIDRA_SOURCE/Features/Decompiler/build/libs/Decompiler-test.jar:\
$GHIDRA_SOURCE/Framework/FileSystem/build/libs/FileSystem.jar:\
$GHIDRA_SOURCE/Framework/FileSystem/build/libs/FileSystem-test.jar:\
$GHIDRA_SOURCE/Framework/DB/build/libs/DB.jar:\
$GHIDRA_SOURCE/Framework/DB/build/libs/DB-test.jar:\
/home/user/ghidra-source/dependencies/flatRepo/gson-2.9.0.jar:\
junit-platform-console-standalone.jar"
RUN find $GHIDRA_SOURCE -name "*test*.jar" && javac -Xlint:-options -cp "$CLASSPATH" `ls $GHIDRA_SCRIPTS/*.java` `ls ./*Test.java`

# todo mkdir -p /mnt/output
RUN java -Djunit.output.dir=/mnt/output \
-jar junit-platform-console-standalone.jar \
    --class-path "$CLASSPATH" \
    --disable-banner \
    --scan-classpath \
    --include-classname ".*Test$"
