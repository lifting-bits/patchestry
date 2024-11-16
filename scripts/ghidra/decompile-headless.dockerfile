FROM eclipse-temurin:17 AS base

FROM base AS build

ENV GHIDRA_VERSION=11.1.2
ENV GRADLE_VERSION=8.2
ENV GRADLE_HOME=/opt/gradle
ENV GHIDRA_RELEASE_TAG=20240709
ENV GHIDRA_PACKAGE=ghidra_${GHIDRA_VERSION}_PUBLIC_${GHIDRA_RELEASE_TAG}
ENV GHIDRA_SHA256=219ec130b901645779948feeb7cc86f131dd2da6c36284cf538c3a7f3d44b588
ENV GHIDRA_REPOSITORY=https://github.com/NationalSecurityAgency/ghidra

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

# Unzip and set up Ghidra
RUN unzip /tmp/ghidra.zip -d /tmp && \
    mv /tmp/ghidra_${GHIDRA_VERSION}_PUBLIC /ghidra && \
    chmod +x /ghidra/ghidraRun && \
    rm -rf /var/tmp/* /tmp/* /ghidra/docs /ghidra/Extensions/Eclipse /ghidra/licenses

# Download and install Gradle
RUN wget https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip -P /tmp \
    && unzip /tmp/gradle-${GRADLE_VERSION}-bin.zip -d /opt/ \
    && ln -s /opt/gradle-${GRADLE_VERSION} ${GRADLE_HOME} \
    && rm /tmp/gradle-${GRADLE_VERSION}-bin.zip

# Set the PATH for Gradle
ENV PATH="${GRADLE_HOME}/bin:${PATH}"

RUN chmod +x /ghidra/support/buildNatives && \
    /ghidra/support/buildNatives

RUN apt-get purge -y --auto-remove wget ca-certificates unzip && \
    apt-get clean

FROM base AS runtime

RUN apt-get update && apt-get install -y \
    adduser \
    sudo \
    --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives

# Add a user with no login shell and no login capabilities and add
# it to sudo group to fix the permission related issue on binding
# host directory during docker run on Ubuntu.
RUN adduser --shell /sbin/nologin --disabled-login --gecos "" user && \
    adduser user sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER user

WORKDIR /home/user/

RUN mkdir -p /home/user/ghidra_projects /home/user/ghidra_scripts

COPY --from=build /ghidra ghidra
COPY PatchestryListFunctions.java ghidra_scripts/
COPY PatchestryDecompileFunctions.java ghidra_scripts/
COPY --chown=user:user --chmod=755 decompile-entrypoint.sh  .

ENV GHIDRA_HOME=/home/user/ghidra
ENV GHIDRA_SCRIPTS=/home/user/ghidra_scripts
ENV GHIDRA_PROJECTS=/home/user/ghidra_projects
ENV GHIDRA_HEADLESS=${GHIDRA_HOME}/support/analyzeHeadless
ENV USER=user

ENTRYPOINT ["/home/user/decompile-entrypoint.sh"]
