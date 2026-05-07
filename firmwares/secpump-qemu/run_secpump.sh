#!/bin/bash
# Emulate secpump.elf locally with qemu-system-arm (works on macOS).
# Usage: ./run_secpump.sh [run|smoke|test|debug]
#   run   - interactive qemu (-nographic). Exit with Ctrl-A x.
#   smoke - run scripts/smoke_test.py (exploit demo; destructive).
#   test  - run tests/test_protocol.py then scripts/smoke_test.py.
#   debug - qemu paused, gdb stub on tcp::1234 (-s -S).
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

ELF="${script_dir}/../output/secpump-qemu.elf"
if [ ! -f "$ELF" ]; then
  echo "${ELF} not found." >&2
  echo "Build it with one of:" >&2
  echo "  cd $(dirname "${script_dir}") && ./build.sh   # Docker, all firmwares" >&2
  echo "  cd ${script_dir} && make                       # native arm-none-eabi-gcc" >&2
  exit 1
fi

command -v qemu-system-arm >/dev/null \
  || { echo "qemu-system-arm not found. macOS: brew install qemu" >&2; exit 1; }

QEMU_BASE_ARGS=(
  -M mps2-an386
  -display none
  -serial mon:stdio
  -semihosting -semihosting-config enable=on,target=native
  -kernel "$ELF"
)

mode="${1:-run}"
case "$mode" in
  run)
    exec qemu-system-arm "${QEMU_BASE_ARGS[@]}"
    ;;
  smoke)
    exec python3 "${script_dir}/scripts/smoke_test.py"
    ;;
  test)
    python3 "${script_dir}/tests/test_protocol.py"
    exec python3 "${script_dir}/scripts/smoke_test.py"
    ;;
  debug)
    exec qemu-system-arm "${QEMU_BASE_ARGS[@]}" -s -S
    ;;
  *)
    echo "usage: $0 [run|smoke|test|debug]" >&2
    exit 2
    ;;
esac
