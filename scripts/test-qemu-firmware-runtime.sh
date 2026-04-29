#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

build_type="Debug"
output_dir="$repo_root/builds/qemu-firmware-runtime"
fixture_dir="${PATCHESTRY_RUNTIME_FIXTURE_DIR:-$repo_root/test/qemu-firmware-runtime/fixtures}"
refresh_ghidra="${PATCHESTRY_RUNTIME_REFRESH_GHIDRA:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-type)
      build_type="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

llvm_prefix="${PATCHESTRY_LLVM_PREFIX:-/Users/artem/projects/patchestry/bin}"
ld_lld="${PATCHESTRY_LD_LLD:-/opt/homebrew/Cellar/llvm@18/18.1.8/bin/ld.lld}"
python_bin="${PATCHESTRY_PYTHON:-}"
qemu_system_arm="${QEMU_SYSTEM_ARM:-}"

if [[ -z "$python_bin" ]]; then
  if [[ -x /opt/homebrew/bin/python3.11 ]]; then
    python_bin=/opt/homebrew/bin/python3.11
  else
    python_bin="$(command -v python3)"
  fi
fi

if [[ -z "$qemu_system_arm" ]]; then
  if command -v qemu-system-arm >/dev/null 2>&1; then
    qemu_system_arm="$(command -v qemu-system-arm)"
  elif [[ -x /opt/homebrew/bin/qemu-system-arm ]]; then
    qemu_system_arm=/opt/homebrew/bin/qemu-system-arm
  else
    echo "Missing qemu-system-arm. Set QEMU_SYSTEM_ARM to your binary." >&2
    exit 1
  fi
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-/tmp/uv-tools}"

check_tool() {
  local path="$1"
  local hint="$2"
  if [[ ! -x "$path" ]]; then
    echo "Missing required tool at $path. $hint" >&2
    exit 1
  fi
}

ensure_docker() {
  if ! docker ps >/dev/null 2>&1; then
    echo "Docker is not available. Start Colima or your Docker daemon before running this script." >&2
    exit 1
  fi
}

ensure_macos_keystone() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    return
  fi

  local dylib_dir="$output_dir/toolchain/keystone/lib"
  local dylib_path="$dylib_dir/libkeystone.dylib"
  mkdir -p "$dylib_dir"

  if [[ -f "$dylib_path" ]]; then
    export DYLD_LIBRARY_PATH="$dylib_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    return
  fi

  uv run --python "$python_bin" --with 'patcherex2 @ git+https://github.com/trail-of-forks/Patcherex2.git@patche_support' python -c 'print("patcherex2 cache primed")' >/dev/null

  local cmake_file
  cmake_file="$(
    find "$UV_CACHE_DIR"/sdists-*/pypi/keystone-engine -name CMakeLists.txt 2>/dev/null \
      | rg '/src/src/CMakeLists.txt$' \
      | head -n 1
  )"
  if [[ -z "$cmake_file" ]]; then
    echo "Could not find keystone-engine source in $UV_CACHE_DIR after priming patcherex2." >&2
    exit 1
  fi

  local keystone_src
  keystone_src="$(dirname "$cmake_file")"
  local patched_src build_dir
  patched_src="$(mktemp -d /tmp/keystone-src.XXXXXX)"
  build_dir="$(mktemp -d /tmp/keystone-build.XXXXXX)"

  cp -R "$keystone_src/." "$patched_src/"
  perl -0pi -e 's/if \(POLICY CMP0051\)\n.*?endif\(\)\n/if (POLICY CMP0051)\n  cmake_policy(SET CMP0051 NEW)\nendif()\n/s' \
    "$patched_src/CMakeLists.txt" \
    "$patched_src/llvm/CMakeLists.txt"

  cmake -G Ninja "$patched_src" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_LIBS_ONLY=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -B "$build_dir" >/dev/null
  ninja -C "$build_dir" >/dev/null

  cp "$build_dir/llvm/lib/libkeystone.dylib" "$dylib_path"
  export DYLD_LIBRARY_PATH="$dylib_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
}

check_tool "$llvm_prefix/clang" "Set PATCHESTRY_LLVM_PREFIX to your patched LLVM bin directory."
check_tool "$ld_lld" "Set PATCHESTRY_LD_LLD to an ELF-capable lld binary."
check_tool "$python_bin" "Set PATCHESTRY_PYTHON to a Python 3.11 binary."
check_tool "$qemu_system_arm" "Set QEMU_SYSTEM_ARM to your qemu-system-arm binary."
check_tool "$(command -v uv)" "Install uv."
check_tool "$(command -v cmake)" "Install CMake."
check_tool "$(command -v ninja)" "Install Ninja."

if [[ "$refresh_ghidra" == "1" ]]; then
  check_tool "$(command -v docker)" "Install Docker."
  ensure_docker
fi

ensure_macos_keystone

# Patcherex2 (trail-of-forks/Patcherex2 @ patche_support) hardcodes
# clang_version=19 for bare-metal ARM, so it shells out to clang-19 and
# ld.lld-19 unconditionally. Expose the host's existing clang/ld.lld
# under those names via a shim dir on PATH instead of forcing every
# environment to install LLVM 19 alongside.
patcherex_shim_dir="${PATCHESTRY_PATCHEREX_SHIM_DIR:-${TMPDIR:-/tmp}/patchestry-patcherex-shim}"
rm -rf "$patcherex_shim_dir"
mkdir -p "$patcherex_shim_dir"
ln -sf "$llvm_prefix/clang" "$patcherex_shim_dir/clang-19"
ln -sf "$ld_lld" "$patcherex_shim_dir/ld.lld-19"
export PATH="$patcherex_shim_dir:$PATH"

make -C "$repo_root/firmwares/qemu-serial" clean all \
  LLVM_PREFIX="$llvm_prefix" \
  LD_LLD="$ld_lld"

runtime_args=(
  "$repo_root/scripts/patch-runtime/qemu_firmware_runtime.py"
  --repo-root "$repo_root"
  --build-type "$build_type"
  --qemu-system-arm "$qemu_system_arm"
  --fixture-dir "$fixture_dir"
  --output-dir "$output_dir"
)

if [[ "$refresh_ghidra" == "1" ]]; then
  runtime_args+=(--refresh-ghidra-fixtures)
fi

uv run --python "$python_bin" --with 'patcherex2 @ git+https://github.com/trail-of-forks/Patcherex2.git@patche_support' python "${runtime_args[@]}"
