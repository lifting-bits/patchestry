#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_TYPE="Debug"
OUTPUT_DIR="${REPO_ROOT}/builds/example-firmware-e2e"
RUN_BUILD_FIRMWARES=true
RUN_BUILD_GHIDRA_IMAGE=auto
REPORT_PREFIX="summary"

show_help() {
  cat <<USAGE
Usage: $0 [OPTIONS]

Run the repository-supported example firmware flow end to end:
  firmware build -> Ghidra JSON -> CIR -> patched CIR -> LLVM IR

Options:
  --build-type <Debug|Release|RelWithDebInfo>
      Tool build configuration to use. Default: Debug
  --output-dir <dir>
      Directory to store artifacts and reports.
      Default: ${REPO_ROOT}/builds/example-firmware-e2e
  --skip-firmware-build
      Reuse existing files under firmwares/output instead of rebuilding them.
  --skip-ghidra-image-build
      Reuse an existing trailofbits/patchestry-decompilation:latest image.
  --build-ghidra-image
      Force rebuilding the Ghidra headless image before running cases.
  --report-prefix <name>
      Prefix for generated report files. Default: summary
  -h, --help
      Show this help message and exit.

Generated artifacts:
  <output-dir>/<case>/decompile.json
  <output-dir>/<case>/<case>.cir
  <output-dir>/<case>/<case>.patched.cir
  <output-dir>/<case>/<case>.patched.ll
  <output-dir>/<case>/run.log
  <output-dir>/<report-prefix>.md
  <output-dir>/<report-prefix>.tsv
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --skip-firmware-build)
      RUN_BUILD_FIRMWARES=false
      shift
      ;;
    --skip-ghidra-image-build)
      RUN_BUILD_GHIDRA_IMAGE=false
      shift
      ;;
    --build-ghidra-image)
      RUN_BUILD_GHIDRA_IMAGE=true
      shift
      ;;
    --report-prefix)
      REPORT_PREFIX="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
SUMMARY_MD="${OUTPUT_DIR}/${REPORT_PREFIX}.md"
SUMMARY_TSV="${OUTPUT_DIR}/${REPORT_PREFIX}.tsv"

PATCHIR_DECOMP="${REPO_ROOT}/builds/default/tools/patchir-decomp/${BUILD_TYPE}/patchir-decomp"
PATCHIR_TRANSFORM="${REPO_ROOT}/builds/default/tools/patchir-transform/${BUILD_TYPE}/patchir-transform"
PATCHIR_CIR2LLVM="${REPO_ROOT}/builds/default/tools/patchir-cir2llvm/${BUILD_TYPE}/patchir-cir2llvm"
PATCHIR_YAML_PARSER="${REPO_ROOT}/builds/default/tools/patchir-yaml-parser/${BUILD_TYPE}/patchir-yaml-parser"
DECOMPILER_HEADLESS="${REPO_ROOT}/scripts/ghidra/decompile-headless.sh"
GHIDRA_IMAGE="trailofbits/patchestry-decompilation:latest"

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

require_executable() {
  local path="$1"
  if [[ ! -x "${path}" ]]; then
    echo "Missing required executable: ${path}" >&2
    exit 1
  fi
}

require_executable "${PATCHIR_DECOMP}"
require_executable "${PATCHIR_TRANSFORM}"
require_executable "${PATCHIR_CIR2LLVM}"
require_executable "${PATCHIR_YAML_PARSER}"
require_executable "${DECOMPILER_HEADLESS}"
require_file "${REPO_ROOT}/firmwares/build.sh"

if ${RUN_BUILD_FIRMWARES}; then
  echo "Building example firmwares..."
  (cd "${REPO_ROOT}" && bash ./firmwares/build.sh)
fi

if [[ "${RUN_BUILD_GHIDRA_IMAGE}" == "auto" ]]; then
  if docker image inspect "${GHIDRA_IMAGE}" >/dev/null 2>&1; then
    RUN_BUILD_GHIDRA_IMAGE=false
  else
    RUN_BUILD_GHIDRA_IMAGE=true
  fi
fi

if [[ "${RUN_BUILD_GHIDRA_IMAGE}" == "true" ]]; then
  echo "Building Ghidra headless image..."
  (cd "${REPO_ROOT}" && bash ./scripts/ghidra/build-headless-docker.sh)
fi

cat > "${SUMMARY_MD}" <<EOF_MD
# Example Firmware E2E Report

| Case | Status | Binary | Function | Spec |
|---|---|---|---|---|
EOF_MD
printf 'case\tstatus\tbinary\tfunction\tspec\tjson\tcir\tpatched_cir\tllvm_ir\tlog\n' > "${SUMMARY_TSV}"

PASS_COUNT=0
FAIL_COUNT=0

append_summary() {
  local case_name="$1"
  local status="$2"
  local binary="$3"
  local function_name="$4"
  local spec="$5"
  local json="$6"
  local cir="$7"
  local patched_cir="$8"
  local llvm_ir="$9"
  local log_file="${10}"

  printf '| `%s` | `%s` | `%s` | `%s` | `%s` |\n' \
    "${case_name}" "${status}" "${binary}" "${function_name}" "${spec}" >> "${SUMMARY_MD}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${case_name}" "${status}" "${binary}" "${function_name}" "${spec}" \
    "${json}" "${cir}" "${patched_cir}" "${llvm_ir}" "${log_file}" >> "${SUMMARY_TSV}"
}

run_case() {
  local case_name="$1"
  local binary="$2"
  local function_name="$3"
  local spec="$4"
  shift 4
  local patterns=("$@")

  local case_dir="${OUTPUT_DIR}/${case_name}"
  local json="${case_dir}/decompile.json"
  local cir_prefix="${case_dir}/${case_name}"
  local cir="${cir_prefix}.cir"
  local patched_cir="${case_dir}/${case_name}.patched.cir"
  local llvm_ir="${case_dir}/${case_name}.patched.ll"
  local log_file="${case_dir}/run.log"

  mkdir -p "${case_dir}"
  : > "${log_file}"

  {
    echo "case=${case_name}"
    echo "binary=${binary}"
    echo "function=${function_name}"
    echo "spec=${spec}"
  } >> "${log_file}"

  if [[ ! -e "${binary}" ]]; then
    echo "Missing required path: ${binary}" >> "${log_file}"
    append_summary "${case_name}" "FAIL" "${binary}" "${function_name}" "${spec}" "${json}" "${cir}" "${patched_cir}" "${llvm_ir}" "${log_file}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    return 1
  fi

  local status="PASS"
  local failed_step=""
  run_step() {
    local step_name="$1"; shift
    echo "+ ${step_name}" >> "${log_file}"
    if ! "$@" >> "${log_file}" 2>&1; then
      failed_step="${step_name}"
      return 1
    fi
  }

  if ! run_step "decompile-headless" \
       "${DECOMPILER_HEADLESS}" --input "${binary}" --function "${function_name}" --output "${json}" \
    || ! run_step "patchir-decomp" \
       "${PATCHIR_DECOMP}" -input "${json}" -emit-cir -output "${cir_prefix}" \
    || ! run_step "patchir-yaml-parser" \
       "${PATCHIR_YAML_PARSER}" "${spec}" --validate \
    || ! run_step "patchir-transform" \
       "${PATCHIR_TRANSFORM}" "${cir}" --spec "${spec}" -o "${patched_cir}" \
    || ! run_step "patchir-cir2llvm" \
       "${PATCHIR_CIR2LLVM}" -S "${patched_cir}" -o "${llvm_ir}"; then
    echo "Pipeline failed at step: ${failed_step}" >> "${log_file}"
    status="FAIL"
  fi

  if [[ "${status}" == "PASS" ]]; then
    for check_file in "${json}" "${cir}" "${patched_cir}" "${llvm_ir}"; do
      if [[ ! -s "${check_file}" ]]; then
        echo "Expected non-empty file missing or empty: ${check_file}" >> "${log_file}"
        status="FAIL"
      fi
    done
  fi

  if [[ "${status}" == "PASS" ]]; then
    for pattern in "${patterns[@]}"; do
      if ! grep -Fq "${pattern}" "${patched_cir}"; then
        echo "Missing expected pattern in patched CIR: ${pattern}" >> "${log_file}"
        status="FAIL"
        break
      fi
    done
  fi

  append_summary "${case_name}" "${status}" "${binary}" "${function_name}" "${spec}" "${json}" "${cir}" "${patched_cir}" "${llvm_ir}" "${log_file}"

  if [[ "${status}" == "PASS" ]]; then
    PASS_COUNT=$((PASS_COUNT + 1))
    return 0
  fi

  FAIL_COUNT=$((FAIL_COUNT + 1))
  return 1
}

CASE_FAILURE=0
run_case \
  "pulseox_measurement_update" \
  "${REPO_ROOT}/firmwares/output/pulseox-firmware.elf" \
  "measurement_update" \
  "${REPO_ROOT}/test/patchir-transform/measurement_update_before_patch.yaml" \
  "patch__before__spo2_lookup" || CASE_FAILURE=1

run_case \
  "bloodlight_usb_send_message" \
  "${REPO_ROOT}/firmwares/output/bloodlight-firmware.elf" \
  "bl_usb__send_message" \
  "${REPO_ROOT}/test/patchir-transform/bl_usb__send_message_before_patch.yaml" \
  "patch__before__usbd_ep_write_packet" \
  "contract__before__test_contract" || CASE_FAILURE=1

run_case \
  "bloodview_device_process_entry" \
  "${REPO_ROOT}/firmwares/output/bloodlight/bloodview" \
  "bl_device__process_entry" \
  "${REPO_ROOT}/test/patchir-transform/device_process_entry.yaml" \
  "patch__replace__sprintf" \
  "contract__sprintf" || CASE_FAILURE=1

cat >> "${SUMMARY_MD}" <<EOF_MD

## Totals

- Passed: ${PASS_COUNT}
- Failed: ${FAIL_COUNT}
- Output directory: ${OUTPUT_DIR}
EOF_MD

if [[ ${CASE_FAILURE} -ne 0 ]]; then
  echo "Example firmware end-to-end validation failed. See ${SUMMARY_MD}" >&2
  exit 1
fi

echo "Example firmware end-to-end validation passed. See ${SUMMARY_MD}"
