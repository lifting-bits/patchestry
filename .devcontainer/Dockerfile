# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version (use Debian 11, Ubuntu 18.04/22.04 on local arm64/Apple Silicon): debian-11, debian-10, ubuntu-22.04, ubuntu-20.04, ubuntu-18.04
ARG IMAGE="ubuntu-22.04"

FROM mcr.microsoft.com/vscode/devcontainers/cpp:${IMAGE}
ARG LLVM_VERSION

# [Optional] Install CMake version different from what base image has already installed.
# CMake reinstall choices: none, 3.21.5, 3.22.2, or versions from https://cmake.org/download/
ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.29.2"

# Optionally install CMake for vcpkg
COPY ./reinstall-cmake.sh /tmp/
RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && \
        /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi && \
    rm -f /tmp/reinstall-cmake.sh

COPY ./dependencies.txt /tmp/
RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    xargs -a /tmp/dependencies.txt apt-get install -y --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Docker
RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

# Install Python packages
RUN pip3 install lit codespell jinja2 Pygments

COPY ./install-clangir.sh /tmp/
RUN chmod +x /tmp/install-clangir.sh && \
    /tmp/install-clangir.sh

# Remove old versions of LLVM and Clang
RUN apt-get -y purge llvm-14 clang-14

# Add docker group and add root to it
RUN groupadd docker || true && \
    usermod -aG docker root
