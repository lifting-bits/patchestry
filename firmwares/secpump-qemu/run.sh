#!/bin/bash
# Emulate secpump.elf locally with qemu-system-arm (works on macOS).
# Usage: ./run.sh [run|smoke|debug]
#   run   - interactive qemu (-nographic). Exit with Ctrl-A x.
#   smoke - run scripts/smoke_test.py end-to-end (requires python3 pexpect).
#   debug - qemu paused, gdb stub on tcp::1234 (-s -S).
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

ELF="${script_dir}/build/secpump.elf"
if [ ! -f "$ELF" ]; then
  ELF="${script_dir}/../output/secpump-qemu.elf"
fi
if [ ! -f "$ELF" ]; then
  echo "secpump.elf not found. Run ./build-docker.sh first." >&2
  exit 1
fi

command -v qemu-system-arm >/dev/null \
  || { echo "qemu-system-arm not found. macOS: brew install qemu" >&2; exit 1; }

mode="${1:-run}"
case "$mode" in
  run)
    exec qemu-system-arm -M mps2-an386 -nographic -kernel "$ELF"
    ;;
  smoke)
    exec python3 "${script_dir}/scripts/smoke_test.py"
    ;;
  debug)
    exec qemu-system-arm -M mps2-an386 -nographic -kernel "$ELF" -s -S
    ;;
  *)
    echo "usage: $0 [run|smoke|debug]" >&2
    exit 2
    ;;
esac
