#!/usr/bin/env bash
#
# Lift functions from prebuilt ARM Linux binaries, reinsert each one unchanged
# and check the rewritten binary (scripts/patch-runtime/identity_reinsertion.py).
#
#   scripts/test-identity-reinsertion.sh --binary-root DIR [--cases FILE]
#       [--build-type Debug] [--output-dir DIR] [--only CASE]...
#
# The binaries are not in this repository.  --binary-root is the directory the
# `binary` paths of the cases file are relative to; each case pins the sha256
# of the binary its fixture was lifted from.
#
# Environment, as for scripts/test-qemu-firmware-runtime.sh:
#   PATCHESTRY_LLVM_PREFIX  bin directory of the patched LLVM (clang, ld.lld, ...)
#   QEMU_SYSTEM_ARM         qemu-system-arm, for the equivalence comparison (default: on PATH)
#   PATCHESTRY_LD_LLD       ld.lld to use (default: the one in that directory)
#   PATCHESTRY_PYTHON       Python 3.11 for the Patcherex2 environment
#   PATCHESTRY_KEYSTONE_LIB_DIR  macOS only: directory with libkeystone.dylib
#       (default: the one scripts/test-qemu-firmware-runtime.sh builds)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

build_type="Debug"
output_dir="$repo_root/builds/identity-reinsertion"
cases="$repo_root/test/identity-reinsertion/aarno_cases.json"
binary_root=""
only=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-type) build_type="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --cases) cases="$2"; shift 2 ;;
    --binary-root) binary_root="$2"; shift 2 ;;
    --only) only+=(--only "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$binary_root" ]]; then
  echo "Missing --binary-root: the directory the cases' binaries are under." >&2
  exit 1
fi

llvm_prefix="${PATCHESTRY_LLVM_PREFIX:-}"
if [[ -z "$llvm_prefix" || ! -x "$llvm_prefix/clang" ]]; then
  echo "Set PATCHESTRY_LLVM_PREFIX to your patched LLVM bin directory." >&2
  exit 1
fi
ld_lld="${PATCHESTRY_LD_LLD:-$llvm_prefix/ld.lld}"

python_bin="${PATCHESTRY_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  if [[ -x /opt/homebrew/bin/python3.11 ]]; then
    python_bin=/opt/homebrew/bin/python3.11
  else
    python_bin="$(command -v python3)"
  fi
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-/tmp/uv-tools}"

# Same pins as scripts/test-qemu-firmware-runtime.sh; see the note there.
patcherex_env=(
  --with 'patcherex2 @ git+https://github.com/trail-of-forks/Patcherex2.git@patche_support'
  --with 'bitstring<5'
  --with 'pycparser==3.0'
)

if [[ "$(uname -s)" == "Darwin" ]]; then
  keystone_dir="${PATCHESTRY_KEYSTONE_LIB_DIR:-$repo_root/builds/qemu-firmware-runtime/toolchain/keystone/lib}"
  if [[ ! -f "$keystone_dir/libkeystone.dylib" ]]; then
    echo "Missing $keystone_dir/libkeystone.dylib." >&2
    echo "Run scripts/test-qemu-firmware-runtime.sh once (it builds it), or set PATCHESTRY_KEYSTONE_LIB_DIR." >&2
    exit 1
  fi
  export DYLD_LIBRARY_PATH="$keystone_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

# Patcherex2 runs clang-19 and ld.lld-19 by name.
shim_dir="${PATCHESTRY_PATCHEREX_SHIM_DIR:-${TMPDIR:-/tmp}/patchestry-patcherex-shim}"
rm -rf "$shim_dir"
mkdir -p "$shim_dir"
ln -sf "$llvm_prefix/clang" "$shim_dir/clang-19"
ln -sf "$ld_lld" "$shim_dir/ld.lld-19"
export PATH="$shim_dir:$PATH"
export PATCHESTRY_LLVM_PREFIX="$llvm_prefix" PATCHESTRY_LD_LLD="$ld_lld"
# The equivalence comparison builds a harness with patchir-klee-verifier and
# runs it under qemu-system-arm (QEMU_SYSTEM_ARM, or the one on PATH).
export PATCHESTRY_KLEE_VERIFIER="${PATCHESTRY_KLEE_VERIFIER:-$repo_root/builds/default/tools/patchir-klee-verifier/$build_type/patchir-klee-verifier}"

for test in test_linux_arm_images.py test_check_abi.py test_check_static.py test_evidence.py test_check_equivalence.py; do
  uv run --python "$python_bin" "${patcherex_env[@]}" \
    python "$repo_root/scripts/patch-runtime/$test"
done

uv run --python "$python_bin" "${patcherex_env[@]}" \
  python "$repo_root/scripts/patch-runtime/identity_reinsertion.py" \
  --cases "$cases" \
  --binary-root "$binary_root" \
  --build-type "$build_type" \
  --output-dir "$output_dir" \
  ${only[@]+"${only[@]}"}

echo "Summary: $output_dir/summary.tsv"
