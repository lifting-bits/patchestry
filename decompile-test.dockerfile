# You can obtain the base container by first having built 
# scripts/ghidra/decompile-headless.dockerfile in this repo.
FROM trailofbits/patchestry-decompilation:latest

# This Dockerfile is building and running tests in the same environment
# as the headless container. On Linux, you can build with:
# $ DOCKER_BUILDKIT=1 docker build -t trailofbits/patchestry-unit-test:latest -f decompile-test.dockerfile .
#
# Or, if you just want to build the test firmware for local use and *not run*
# unit tests, refer to the firmwares/Dockerfile rather than this Dockerfile.
# For the headless container, see scripts/ghidra/decompile-headless.dockerfile.

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
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# we use PULSEOX_FW_PATH for tests
ENV PULSEOX_FW_PATH="/home/user/pulseox-firmware.elf"
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
    cp build/src/firmware.elf $PULSEOX_FW_PATH
# if we end up needing the fw build environment to compare to later, 
# then undo this
WORKDIR /home/user/
RUN rm -rf /home/user/pulseox-firmware/

WORKDIR /home/user/    
# we use BLOODLIGHT_FW_PATH for tests
ENV BLOODLIGHT_FW_PATH="/home/user/bloodlight-firmware.elf"
ENV BLOODLIGHT_COMMIT="def737f481d6f0d16db4e94d7b26cbaae9838b41"
# important - may not compile with the --depth 1 flag
RUN git clone https://github.com/CodethinkLabs/bloodlight-firmware.git
WORKDIR /home/user/bloodlight-firmware/
RUN git fetch --depth=1 origin ${BLOODLIGHT_COMMIT} && \
    git checkout ${BLOODLIGHT_COMMIT} && \
    git submodule update --init --recursive && \
    make -C firmware/libopencm3 && \
    make -C firmware -j8 && \
    cp firmware/bloodlight-firmware.elf $BLOODLIGHT_FW_PATH
# if we end up needing the fw build environment to compare to later, 
# then undo this
WORKDIR /home/user/
RUN rm -rf /home/user/bloodlight-firmware/

WORKDIR /home/user
RUN wget -O junit-platform-console-standalone.jar \
    https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.10.1/junit-platform-console-standalone-1.10.1.jar
COPY test/scripts/java/ .

# the GHIDRA_* env vars here come from the parent Dockerfile.
WORKDIR $GHIDRA_HOME

WORKDIR /home/user
ENV CLASSPATH="/home/user:$GHIDRA_SCRIPTS:$GHIDRA_HOME/support/ghidra.jar:junit-platform-console-standalone.jar"
RUN javac -cp "$CLASSPATH" `ls $GHIDRA_SCRIPTS/*.java` `ls ./*Test.java`

# todo mkdir -p /mnt/output
RUN java -Djunit.output.dir=/mnt/output \
-jar junit-platform-console-standalone.jar \
    --class-path "$CLASSPATH" \
    --disable-banner \
    --scan-classpath \
    --include-classname ".*Test$"
