#!/bin/bash
# Host wrapper: compare a function in an original and a patched binary by
# symbolic execution, in the PatchSymDiff Docker image.
#
# Usage:
#   ./scripts/symdiff/run-symdiff.sh --original <elf> --patched <elf> \
#       --arch arm|thumb (--function <name> | --address <addr>) \
#       [--output <dir>] [--image <name>] [-- <patch_symdiff.py options>]
#
#   --original   the binary before the patch
#   --patched    the binary after the patch
#   --arch       architecture of both binaries.  32-bit ARM only for now:
#                `arm` or `thumb`.  With --function the instruction set comes
#                from the symbol; with --address it comes from --arch.
#   --function   symbol of the function to compare, looked up in the original
#   --address    its address, for a binary without symbols
#   --output     directory for the results (default ./symdiff-out):
#                  report.json  verdict, differences with a witness input, assumptions
#                  cozy.txt     cozy's own report
#                  stdout.txt   the summary printed by the script
#   --image      Docker image (default $SYMDIFF_IMAGE or patchestry/symdiff:latest)
#   --interactive  open a shell in the image, with the current directory at /work
#
# Everything after `--` goes to patch_symdiff.py unchanged, for example
#   -- --stub memcpy --stub-args memcpy:3 --reject-return 0xffffffff --max-seconds 600
# `run-symdiff.sh --help-script` prints its options and what each difference means.
#
# Exit code: 0 proven, 1 failed (a difference was found), 3 incomplete,
# 2 usage error.
#
# The container has no network and sees the two binaries read-only.

set -euo pipefail

IMAGE_NAME="${SYMDIFF_IMAGE:-patchestry/symdiff:latest}"

usage() {
    sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() {
    echo "Error: $1" >&2
    echo "Run '$0 --help' for usage." >&2
    exit 2
}

ensure_image() {
    if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        echo "Error: Docker image '${IMAGE_NAME}' not found." >&2
        echo "       Build it with scripts/symdiff/build-symdiff-docker.sh," >&2
        echo "       or name another with --image or SYMDIFF_IMAGE." >&2
        exit 2
    fi
}

ORIGINAL=""
PATCHED=""
ARCH=""
FUNCTION=""
ADDRESS=""
OUTPUT_DIR="./symdiff-out"
INTERACTIVE=false
HELP_SCRIPT=false
SCRIPT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --original) ORIGINAL="${2:?--original needs a value}"; shift 2 ;;
        --patched) PATCHED="${2:?--patched needs a value}"; shift 2 ;;
        --arch) ARCH="${2:?--arch needs a value}"; shift 2 ;;
        --function) FUNCTION="${2:?--function needs a value}"; shift 2 ;;
        --address) ADDRESS="${2:?--address needs a value}"; shift 2 ;;
        --output|-o) OUTPUT_DIR="${2:?--output needs a value}"; shift 2 ;;
        --image) IMAGE_NAME="${2:?--image needs a value}"; shift 2 ;;
        --interactive) INTERACTIVE=true; shift ;;
        --help-script) HELP_SCRIPT=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; SCRIPT_ARGS=("$@"); break ;;
        *) die "unknown option: $1 (options of patch_symdiff.py go after --)" ;;
    esac
done

if [[ "${HELP_SCRIPT}" == true ]]; then
    ensure_image
    docker run --rm --network none --entrypoint python "${IMAGE_NAME}" -c \
        "import sys; sys.path.insert(0, '/opt/symdiff'); import patch_symdiff; print(patch_symdiff.__doc__)"
    exit 0
fi

if [[ "${INTERACTIVE}" == true ]]; then
    ensure_image
    docker run --rm -it --network none -v "${PWD}:/work" --entrypoint /bin/bash "${IMAGE_NAME}"
    exit 0
fi

[[ -n "${ORIGINAL}" ]] || die "--original is required"
[[ -n "${PATCHED}" ]] || die "--patched is required"
[[ -n "${ARCH}" ]] || die "--arch is required"
[[ -f "${ORIGINAL}" ]] || die "original binary not found: ${ORIGINAL}"
[[ -f "${PATCHED}" ]] || die "patched binary not found: ${PATCHED}"
if [[ -n "${FUNCTION}" && -n "${ADDRESS}" ]]; then
    die "give --function or --address, not both"
fi
[[ -n "${FUNCTION}" || -n "${ADDRESS}" ]] || die "--function or --address is required"

case "${ARCH}" in
    arm|thumb) ;;
    *) die "--arch ${ARCH}: only arm and thumb are handled (patch_symdiff.py is 32-bit ARM)" ;;
esac

TARGET_ARGS=()
if [[ -n "${FUNCTION}" ]]; then
    # The symbol says whether the function is ARM or Thumb.
    TARGET_ARGS=(--function "${FUNCTION}")
else
    TARGET_ARGS=(--address "${ADDRESS}" --mode "${ARCH}")
fi

absolute() {
    echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}
ORIGINAL="$(absolute "${ORIGINAL}")"
PATCHED="$(absolute "${PATCHED}")"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

ensure_image

echo "Comparing ${FUNCTION:-${ADDRESS}} (${ARCH})"
echo "  original: ${ORIGINAL}"
echo "  patched:  ${PATCHED}"
echo "  output:   ${OUTPUT_DIR}"

# Each binary is mounted as a file under its own directory, so two files with
# the same name do not collide.
set +e
docker run --rm \
    --network none \
    -v "${ORIGINAL}:/mnt/original/$(basename "${ORIGINAL}"):ro" \
    -v "${PATCHED}:/mnt/patched/$(basename "${PATCHED}"):ro" \
    -v "${OUTPUT_DIR}:/mnt/output:rw" \
    "${IMAGE_NAME}" \
    --original "/mnt/original/$(basename "${ORIGINAL}")" \
    --rewritten "/mnt/patched/$(basename "${PATCHED}")" \
    "${TARGET_ARGS[@]}" \
    --output /mnt/output/report.json \
    --text /mnt/output/cozy.txt \
    "${SCRIPT_ARGS[@]+"${SCRIPT_ARGS[@]}"}" \
    2>&1 | tee "${OUTPUT_DIR}/stdout.txt"
STATUS="${PIPESTATUS[0]}"
set -e

case "${STATUS}" in
    0) echo "Verdict: proven" ;;
    1) echo "Verdict: failed (see ${OUTPUT_DIR}/report.json)" ;;
    3) echo "Verdict: incomplete (see ${OUTPUT_DIR}/report.json)" ;;
    2) echo "Usage error" >&2 ;;
    *) echo "Error: the container exited with ${STATUS}" >&2 ;;
esac
exit "${STATUS}"
