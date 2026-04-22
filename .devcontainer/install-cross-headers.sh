#!/bin/bash
#
# Install glibc headers for the non-native target architecture so LIT tests
# that cross-compile C sources via `clang -target <arch>-linux-gnu` can find
# system headers like <stdio.h>.
#
# The LIT suite under test/ghidra/ compiles `%cc-x86_64` sources to produce
# x86_64 object files for Ghidra to decompile; `extraout_sanitize.c` does the
# same for aarch64. On arm64 hosts the x86_64 headers are missing; on amd64
# hosts the base image already ships with both sides' multiarch headers.

set -eu

ARCH=$(dpkg --print-architecture)
if [ "$ARCH" != "arm64" ]; then
    echo "Host arch is '$ARCH'; no cross-arch headers needed."
    exit 0
fi

# arm64 Ubuntu's sources.list points at ports.ubuntu.com, which does not serve
# amd64 packages. Scope the existing entries to arm64 and add archive.ubuntu.com
# for amd64.
sed -i 's|^deb http|deb [arch=arm64] http|' /etc/apt/sources.list
cat > /etc/apt/sources.list.d/amd64-cross.list <<'EOF'
deb [arch=amd64] http://archive.ubuntu.com/ubuntu jammy main universe
deb [arch=amd64] http://archive.ubuntu.com/ubuntu jammy-updates main universe
deb [arch=amd64] http://archive.ubuntu.com/ubuntu jammy-security main universe
EOF

dpkg --add-architecture amd64
apt-get update
apt-get install -y --no-install-recommends libc6-dev-amd64-cross
apt-get clean
rm -rf /var/lib/apt/lists/*

# clang searches `/usr/include/<triple>/` by default, but the cross package
# installs to `/usr/x86_64-linux-gnu/include/`. Symlink so clang finds the
# bits/ subdirectory under its default multiarch path.
ln -sf /usr/x86_64-linux-gnu/include /usr/include/x86_64-linux-gnu
