#!/bin/bash
# Emulate the MakAir firmware locally via qemu-system-arm (mps2-an386).
# Usage: ./run.sh [run|smoke|test|debug|patched|patched-test]
#   run          - interactive qemu with stdio UART. Exit with Ctrl-A x.
#   smoke        - run tests/test_smoke.py (boot + first BootMessage).
#   test         - run smoke then tests/test_protocol.py (Telemetry + Control surface).
#   debug        - qemu paused, gdb stub on tcp::1234 (-s -S).
#   patched      - boot build/makair-patched.elf interactively.
#   patched-test - run tests/test_patched_protocol.py against the patched ELF.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mode="${1:-run}"

case "$mode" in
  patched|patched-test)
    ELF="${script_dir}/build/makair-patched.elf"
    if [ ! -f "$ELF" ]; then
      echo "makair-patched.elf not found. Run scripts/demo_makair.sh --stage=all --with-patcherex first." >&2
      exit 1
    fi
    ;;
  *)
    ELF="${script_dir}/build/makair.elf"
    if [ ! -f "$ELF" ]; then
      echo "makair.elf not found. Run 'make' first." >&2
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
    exec python3 "${script_dir}/tests/test_smoke.py"
    ;;
  test)
    python3 "${script_dir}/tests/test_smoke.py"
    exec python3 "${script_dir}/tests/test_protocol.py"
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
