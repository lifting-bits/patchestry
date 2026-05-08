#!/bin/bash
# Emulate secpump.elf locally with qemu-system-arm (works on macOS).
# Usage: ./run.sh [run|smoke|test|debug|patched|patched-test]
#   run          - interactive qemu (-nographic). Exit with Ctrl-A x.
#   smoke        - run scripts/smoke_test.py (exploit demo; destructive).
#   test         - run tests/test_protocol.py then scripts/smoke_test.py.
#   debug        - qemu paused, gdb stub on tcp::1234 (-s -S).
#   patched      - boot build/secpump-patched.elf interactively.
#   patched-test - protocol regression + exploit-blocked check against the
#                  patched ELF (tests/test_patched_protocol.py).
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mode="${1:-run}"

case "$mode" in
  patched|patched-test)
    ELF="${script_dir}/build/secpump-patched.elf"
    if [ ! -f "$ELF" ]; then
      echo "secpump-patched.elf not found. Run scripts/demo_secpump.sh --stage=all --with-patcherex first." >&2
      exit 1
    fi
    ;;
  *)
    ELF="${script_dir}/build/secpump.elf"
    if [ ! -f "$ELF" ]; then
      ELF="${script_dir}/../output/secpump-qemu.elf"
    fi
    if [ ! -f "$ELF" ]; then
      echo "secpump.elf not found. Run ./build-docker.sh first." >&2
      exit 1
    fi
    ;;
esac

command -v qemu-system-arm >/dev/null \
  || { echo "qemu-system-arm not found. macOS: brew install qemu" >&2; exit 1; }

QEMU_BASE_ARGS=(
  -M mps2-an386
  -display none
  -serial mon:stdio
  -semihosting -semihosting-config enable=on,target=native
  -kernel "$ELF"
)

case "$mode" in
  run|patched)
    exec qemu-system-arm "${QEMU_BASE_ARGS[@]}"
    ;;
  smoke)
    exec python3 "${script_dir}/scripts/smoke_test.py"
    ;;
  test)
    python3 "${script_dir}/tests/test_protocol.py"
    exec python3 "${script_dir}/scripts/smoke_test.py"
    ;;
  patched-test)
    exec python3 "${script_dir}/tests/test_patched_protocol.py"
    ;;
  debug)
    exec qemu-system-arm "${QEMU_BASE_ARGS[@]}" -s -S
    ;;
  *)
    echo "usage: $0 [run|smoke|test|debug|patched|patched-test]" >&2
    exit 2
    ;;
esac
