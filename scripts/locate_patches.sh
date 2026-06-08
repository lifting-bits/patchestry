#!/usr/bin/env bash
# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
# Thin wrapper around scripts/patch_locator.py. Relays applied patches back to
# their binary addresses using the patch-location map emitted by
# `patchir-transform --emit-patch-map`.
#
# Usage:
#   scripts/locate_patches.sh --binary <elf> --map <patchmap.json> [--out <prefix>]
#   scripts/locate_patches.sh --binary <elf> --ll <patched.ll> --json <pcode.json>
#
# All arguments are forwarded to patch_locator.py; see its --help for details.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/patch_locator.py" "$@"
