FROM openjdk:23-jdk-slim as base

FROM base as build

# Set environment variables for Ghidra
ENV GHIDRA_VERSION 11.1.2
ENV GHIDRA_RELEASE_TAG 20240709
ENV GHIDRA_PACKAGE ghidra_${GHIDRA_VERSION}_PUBLIC_${GHIDRA_RELEASE_TAG}
ENV GHIDRA_SHA256 219ec130b901645779948feeb7cc86f131dd2da6c36284cf538c3a7f3d44b588
ENV GHIDRA_REPOSITORY https://github.com/NationalSecurityAgency/ghidra

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    unzip \
    --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives

# Download and verify Ghidra package
RUN wget --progress=bar:force -O /tmp/ghidra.zip ${GHIDRA_REPOSITORY}/releases/download/Ghidra_${GHIDRA_VERSION}_build/${GHIDRA_PACKAGE}.zip && \
    echo "${GHIDRA_SHA256} /tmp/ghidra.zip" | sha256sum -c -

# Unzip and set up Ghidra
RUN unzip /tmp/ghidra.zip -d /tmp && \
    mv /tmp/ghidra_${GHIDRA_VERSION}_PUBLIC /ghidra && \
    chmod +x /ghidra/ghidraRun && \
    rm -rf /var/tmp/* /tmp/* /ghidra/docs /ghidra/Extensions/Eclipse /ghidra/licenses

# Clean up
RUN apt-get purge -y --auto-remove wget ca-certificates unzip && \
    apt-get clean

FROM base as runtime

WORKDIR /ghidra

# Copy Ghidra from the build stage
COPY --from=build /ghidra /ghidra

# Create projects directory
RUN mkdir /ghidra/projects

# Add a user with no login shell and no login capabilities, then change ownership of /ghidra
RUN adduser --shell /sbin/nologin --disabled-login --gecos "" user && \
    chown -R user:user /ghidra

# Copy the Java and script files into the appropriate directories
# COPY DecompileHeadless.java /ghidra/Ghidra/Features/Decompiler/ghidra_scripts/DecompileHeadless.java
COPY decompile.sh /ghidra/decompile.sh

# Make the decompile script executable
RUN chmod +x /ghidra/decompile.sh

# Switch to the newly created user
USER user

# Set working directory for the user
WORKDIR /home/user/

# Copy the .dockerignore file (if necessary)
COPY .dockerignore .dockerignore

# Set environment variable for Ghidra home directory
ENV GHIDRA_HOME /ghidra
ENV GHIDRA_SCRIPTS  ${GHIDRA_HOME}/Ghidra/Features/Decompiler/ghidra_scripts
ENV GHIDRA_PROJECTS ${GHIDRA_HOME}/projects
ENV GHIDRA_HEADLESS ${GHIDRA_HOME}/support/analyzeHeadless

# Set the entrypoint
ENTRYPOINT ["/ghidra/decompile.sh"]
