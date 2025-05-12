FROM eclipse-temurin:21 AS base

FROM base AS build

ARG GHIDRA_VERSION=11.3.2
ARG GRADLE_VERSION=8.2
ARG GRADLE_HOME=/opt/gradle
ARG GHIDRA_RELEASE_TAG=20250415
ARG GHIDRA_PACKAGE=ghidra_${GHIDRA_VERSION}_PUBLIC_${GHIDRA_RELEASE_TAG}
ARG GHIDRA_SHA256=99d45035bdcc3d6627e7b1232b7b379905a9fad76c772c920602e2b5d8b2dac2
ARG GHIDRA_REPOSITORY=https://github.com/NationalSecurityAgency/ghidra

RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    unzip \
    gcc \
    g++ \
    --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives

RUN wget --progress=bar:force -O /tmp/ghidra.zip ${GHIDRA_REPOSITORY}/releases/download/Ghidra_${GHIDRA_VERSION}_build/${GHIDRA_PACKAGE}.zip && \
    echo "${GHIDRA_SHA256} /tmp/ghidra.zip" | sha256sum -c -


RUN unzip /tmp/ghidra.zip -d /tmp && \
    mv /tmp/ghidra_${GHIDRA_VERSION}_PUBLIC /ghidra && \
    chmod +x /ghidra/ghidraRun

RUN wget https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip -P /tmp \
    && unzip /tmp/gradle-${GRADLE_VERSION}-bin.zip -d /opt/ \
    && ln -s /opt/gradle-${GRADLE_VERSION} ${GRADLE_HOME} \
    && rm /tmp/gradle-${GRADLE_VERSION}-bin.zip

ENV PATH="${GRADLE_HOME}/bin:${PATH}"

WORKDIR /ghidra/support/gradle
RUN gradle buildNatives

RUN rm -rf /ghidra/Extensions/Eclipse /ghidra/licenses ghidraRun.bat docs/ &&\
    apt-get purge -y --auto-remove wget ca-certificates unzip && \
    apt-get clean

FROM base AS runtime

RUN apt-get update && apt-get install -y \
    adduser \
    sudo \
    wget \
    binutils \
    file && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives

# Add a user with no login shell and no login capabilities and add
# it to sudo group to fix the permission related issue on binding
# host directory during docker run on Ubuntu.
RUN adduser --shell /sbin/nologin --disabled-login --gecos "" user && \
    adduser user sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER user

WORKDIR /home/user/

COPY --chown=user:user --from=build /ghidra ghidra
COPY --chown=user:user --chmod=755 decompile-entrypoint.sh  .
COPY --chown=user:user --from=build /opt/gradle/ /opt/gradle/
ENV PATH="/opt/gradle/bin:${PATH}"

WORKDIR /home/user/ghidra_scripts/
COPY --chown=user:user domain/ domain/
COPY --chown=user:user util/ util/
COPY --chown=user:user PatchestryDecompileFunctions.java .
COPY --chown=user:user PatchestryListFunctions.java .
COPY --chown=user:user build.gradle .
# since we have a class structure, we need to externally trigger the Ghidra build
RUN gradle build

WORKDIR /home/user/
ENV GHIDRA_HOME=/home/user/ghidra
ENV GHIDRA_SCRIPTS=/home/user/ghidra_scripts
ENV GHIDRA_PROJECTS=/home/user/ghidra_projects
RUN mkdir $GHIDRA_PROJECTS
ENV GHIDRA_HEADLESS=${GHIDRA_HOME}/support/analyzeHeadless
ENV USER=user

ENTRYPOINT ["/home/user/decompile-entrypoint.sh"]
