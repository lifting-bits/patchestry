#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_TYPE="Debug"
BUILD_ROOT="default"
FIXTURE_DIR="${REPO_ROOT}/builds/test-fixtures"
OUTPUT_DIR="${REPO_ROOT}/builds/patch-matrix"
REPORT_PREFIX="summary"
REBUILD_FIRMWARE=false
REBUILD_GHIDRA=false
REBUILD_FIXTURES=false
CLEAN=false

PATCHIR_DECOMP="${REPO_ROOT}/builds/${BUILD_ROOT}/tools/patchir-decomp/${BUILD_TYPE}/patchir-decomp"
PATCHIR_TRANSFORM="${REPO_ROOT}/builds/${BUILD_ROOT}/tools/patchir-transform/${BUILD_TYPE}/patchir-transform"
PATCHIR_CIR2LLVM="${REPO_ROOT}/builds/${BUILD_ROOT}/tools/patchir-cir2llvm/${BUILD_TYPE}/patchir-cir2llvm"
PATCHIR_YAML_PARSER="${REPO_ROOT}/builds/${BUILD_ROOT}/tools/patchir-yaml-parser/${BUILD_TYPE}/patchir-yaml-parser"
DECOMPILER_HEADLESS="${REPO_ROOT}/scripts/ghidra/decompile-headless.sh"
GHIDRA_IMAGE="trailofbits/patchestry-decompilation:latest"

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Run the repository patch/contract validation matrix from cached generated fixtures:
  firmware build -> Ghidra JSON -> base CIR -> patch/contract matrix -> LLVM IR

Options:
  --build-type <Debug|Release|RelWithDebInfo>
      Tool build configuration to use. Default: Debug
  --build-root <default|ci>
      Build tree root under builds/. Default: default
  --fixture-dir <dir>
      Directory for cached decompile/base-CIR fixtures.
      Default: ${REPO_ROOT}/builds/test-fixtures
  --output-dir <dir>
      Directory for per-matrix outputs and reports.
      Default: ${REPO_ROOT}/builds/patch-matrix
  --report-prefix <name>
      Prefix for generated report files. Default: summary
  --rebuild-firmware
      Rebuild firmware artifacts in firmwares/output before running.
      Implies --rebuild-fixtures.
  --rebuild-ghidra
      Rebuild the Ghidra headless Docker image before running.
  --rebuild-fixtures
      Regenerate cached JSON/CIR fixtures before running the matrix.
  --clean
      Remove cached fixtures and matrix outputs before running.
  -h, --help
      Show this help message and exit.
EOF
}

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

prepare_tool_env() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    export SDKROOT
    SDKROOT="$(xcrun --show-sdk-path)"
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
    --build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    --build-root)
      BUILD_ROOT="$2"
      shift 2
      ;;
    --fixture-dir)
      FIXTURE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --report-prefix)
      REPORT_PREFIX="$2"
      shift 2
      ;;
    --rebuild-firmware)
      REBUILD_FIRMWARE=true
      REBUILD_FIXTURES=true
      shift
      ;;
    --rebuild-ghidra)
      REBUILD_GHIDRA=true
      shift
      ;;
    --rebuild-fixtures)
      REBUILD_FIXTURES=true
      shift
      ;;
    --clean)
      CLEAN=true
      shift
      ;;
    -h | --help)
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

  PATCHIR_DECOMP="${REPO_ROOT}/builds/${BUILD_ROOT}/tools/patchir-decomp/${BUILD_TYPE}/patchir-decomp"
  PATCHIR_TRANSFORM="${REPO_ROOT}/builds/${BUILD_ROOT}/tools/patchir-transform/${BUILD_TYPE}/patchir-transform"
  PATCHIR_CIR2LLVM="${REPO_ROOT}/builds/${BUILD_ROOT}/tools/patchir-cir2llvm/${BUILD_TYPE}/patchir-cir2llvm"
  PATCHIR_YAML_PARSER="${REPO_ROOT}/builds/${BUILD_ROOT}/tools/patchir-yaml-parser/${BUILD_TYPE}/patchir-yaml-parser"
}

setup_paths() {
  if ${CLEAN}; then
    rm -rf "${FIXTURE_DIR}" "${OUTPUT_DIR}"
  fi

  mkdir -p "${FIXTURE_DIR}" "${OUTPUT_DIR}"
  FIXTURE_DIR="$(cd "${FIXTURE_DIR}" && pwd)"
  OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
  SUMMARY_MD="${OUTPUT_DIR}/${REPORT_PREFIX}.md"
  SUMMARY_TSV="${OUTPUT_DIR}/${REPORT_PREFIX}.tsv"
}

ensure_requirements() {
  require_executable "${PATCHIR_DECOMP}"
  require_executable "${PATCHIR_TRANSFORM}"
  require_executable "${PATCHIR_CIR2LLVM}"
  require_executable "${PATCHIR_YAML_PARSER}"
  require_executable "${DECOMPILER_HEADLESS}"
  require_file "${REPO_ROOT}/firmwares/build.sh"
}

ensure_firmware_artifacts() {
  local pulseox="${REPO_ROOT}/firmwares/output/pulseox-firmware.elf"
  local bloodlight="${REPO_ROOT}/firmwares/output/bloodlight-firmware.elf"
  local bloodview="${REPO_ROOT}/firmwares/output/bloodlight/bloodview"

  if ${REBUILD_FIRMWARE} || [[ ! -e "${pulseox}" ]] || [[ ! -e "${bloodlight}" ]] || [[ ! -e "${bloodview}" ]]; then
    echo "Building example firmwares..."
    (cd "${REPO_ROOT}" && bash ./firmwares/build.sh)
  fi
}

ensure_ghidra_image() {
  if ${REBUILD_GHIDRA}; then
    echo "Building Ghidra headless image..."
    (cd "${REPO_ROOT}" && bash ./scripts/ghidra/build-headless-docker.sh)
    return
  fi

  if ! docker image inspect "${GHIDRA_IMAGE}" >/dev/null 2>&1; then
    echo "Building Ghidra headless image..."
    (cd "${REPO_ROOT}" && bash ./scripts/ghidra/build-headless-docker.sh)
  fi
}

fixture_binary_path() {
  case "$1" in
  pulseox_measurement_update)
    echo "${REPO_ROOT}/firmwares/output/pulseox-firmware.elf"
    ;;
  bloodlight_usb_send_message)
    echo "${REPO_ROOT}/firmwares/output/bloodlight-firmware.elf"
    ;;
  bloodview_device_process_entry)
    echo "${REPO_ROOT}/firmwares/output/bloodlight/bloodview"
    ;;
  bloodlight_led_loop)
    echo "${REPO_ROOT}/firmwares/output/bloodlight-firmware.elf"
    ;;
  *)
    echo "Unknown fixture name: $1" >&2
    exit 1
    ;;
  esac
}

fixture_function_name() {
  case "$1" in
  pulseox_measurement_update)
    echo "measurement_update"
    ;;
  bloodlight_usb_send_message)
    echo "bl_usb__send_message"
    ;;
  bloodview_device_process_entry)
    echo "bl_device__process_entry"
    ;;
  bloodlight_led_loop)
    echo "bl_led_loop"
    ;;
  *)
    echo "Unknown fixture name: $1" >&2
    exit 1
    ;;
  esac
}

prepare_fixture() {
  local fixture_name="$1"
  local case_dir="${FIXTURE_DIR}/${fixture_name}"
  local binary
  local function_name

  binary="$(fixture_binary_path "${fixture_name}")"
  function_name="$(fixture_function_name "${fixture_name}")"

  require_file "${binary}"
  mkdir -p "${case_dir}"

  local json="${case_dir}/${fixture_name}.json"
  local cir_prefix="${case_dir}/${fixture_name}"
  local cir="${cir_prefix}.cir"
  local fixture_binary
  fixture_binary="${case_dir}/$(basename "${binary}")"

  if ! ${REBUILD_FIXTURES} && [[ -s "${json}" && -s "${cir}" ]]; then
    return
  fi

  echo "Preparing fixture ${fixture_name}..."
  if [[ -n "${HOST_WORKSPACE:-}" ]]; then
    cp "${binary}" "${fixture_binary}"
    "${DECOMPILER_HEADLESS}" \
      --input "${fixture_binary}" \
      --function "${function_name}" \
      --output "${json}" \
      --ci "${case_dir}"
  else
    "${DECOMPILER_HEADLESS}" --input "${binary}" --function "${function_name}" --output "${json}"
  fi
  "${PATCHIR_DECOMP}" -input "${json}" -emit-cir -output "${cir_prefix}"
}

init_summary() {
  cat >"${SUMMARY_MD}" <<EOF
# Patch Matrix Report

| Case | Status | Type | Fixture | Spec |
|---|---|---|---|---|
EOF
  printf 'case\tstatus\ttype\tfixture\tspec\tpatched_cir\tllvm_ir\tlog\n' >"${SUMMARY_TSV}"
}

append_summary() {
  local case_name="$1"
  local status="$2"
  local case_type="$3"
  local fixture_name="$4"
  local spec="$5"
  local patched_cir="$6"
  local llvm_ir="$7"
  local log_file="$8"

  printf "| \`%s\` | \`%s\` | \`%s\` | \`%s\` | \`%s\` |\n" \
    "${case_name}" "${status}" "${case_type}" "${fixture_name}" "${spec}" >>"${SUMMARY_MD}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${case_name}" "${status}" "${case_type}" "${fixture_name}" "${spec}" \
    "${patched_cir}" "${llvm_ir}" "${log_file}" >>"${SUMMARY_TSV}"
}

check_patterns() {
  local file="$1"
  shift
  local pattern
  for pattern in "$@"; do
    if ! grep -Fq "${pattern}" "${file}"; then
      echo "Missing expected pattern in ${file}: ${pattern}" >&2
      return 1
    fi
  done
}

run_matrix_case() {
  local case_name="$1"
  local fixture_name="$2"
  local spec="$3"
  local cir_patterns="$4"
  local llvm_patterns="$5"

  local fixture_cir="${FIXTURE_DIR}/${fixture_name}/${fixture_name}.cir"
  local case_dir="${OUTPUT_DIR}/${case_name}"
  local patched_cir="${case_dir}/${case_name}.patched.cir"
  local llvm_ir="${case_dir}/${case_name}.patched.ll"
  local log_file="${case_dir}/run.log"
  local status="PASS"

  mkdir -p "${case_dir}"
  : >"${log_file}"

  {
    echo "case=${case_name}"
    echo "fixture=${fixture_name}"
    echo "spec=${spec}"
    echo "+ patchir-yaml-parser"
    "${PATCHIR_YAML_PARSER}" "${spec}" --validate
    echo "+ patchir-transform"
    "${PATCHIR_TRANSFORM}" "${fixture_cir}" --spec "${spec}" -o "${patched_cir}"
    echo "+ patchir-cir2llvm"
    "${PATCHIR_CIR2LLVM}" -S "${patched_cir}" -o "${llvm_ir}"
  } >>"${log_file}" 2>&1 || status="FAIL"

  if [[ "${status}" == "PASS" ]]; then
    for check_file in "${patched_cir}" "${llvm_ir}"; do
      if [[ ! -s "${check_file}" ]]; then
        echo "Expected non-empty file missing or empty: ${check_file}" >> "${log_file}"
        status="FAIL"
      fi
    done
  fi

  if [[ "${status}" == "PASS" && -n "${cir_patterns}" ]]; then
    local cir_pattern_array=()
    IFS='|' read -r -a cir_pattern_array <<<"${cir_patterns}"
    check_patterns "${patched_cir}" "${cir_pattern_array[@]}" >>"${log_file}" 2>&1 || status="FAIL"
  fi

  if [[ "${status}" == "PASS" && -n "${llvm_patterns}" ]]; then
    local llvm_pattern_array=()
    IFS='|' read -r -a llvm_pattern_array <<<"${llvm_patterns}"
    check_patterns "${llvm_ir}" "${llvm_pattern_array[@]}" >>"${log_file}" 2>&1 || status="FAIL"
  fi

  append_summary \
    "${case_name}" \
    "${status}" \
    "positive" \
    "${fixture_name}" \
    "${spec}" \
    "${patched_cir}" \
    "${llvm_ir}" \
    "${log_file}"

  if [[ "${status}" != "PASS" ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    return 1
  fi

  PASS_COUNT=$((PASS_COUNT + 1))
}

run_negative_case() {
  local case_name="$1"
  local fixture_name="$2"
  local spec="$3"
  local command="$4"
  local error_patterns="$5"

  local case_dir="${OUTPUT_DIR}/${case_name}"
  local log_file="${case_dir}/run.log"
  local status="PASS"

  mkdir -p "${case_dir}"
  : >"${log_file}"

  {
    echo "case=${case_name}"
    echo "fixture=${fixture_name}"
    echo "spec=${spec}"
    echo "+ expecting failure"
  } >>"${log_file}"

  if bash -c "set -euo pipefail; ${command}" >>"${log_file}" 2>&1; then
    echo "Command unexpectedly succeeded." >>"${log_file}"
    status="FAIL"
  fi

  if [[ "${status}" == "PASS" ]]; then
    local error_pattern_array=()
    IFS='|' read -r -a error_pattern_array <<<"${error_patterns}"
    check_patterns "${log_file}" "${error_pattern_array[@]}" >>"${log_file}" 2>&1 || status="FAIL"
  fi

  append_summary \
    "${case_name}" \
    "${status}" \
    "negative" \
    "${fixture_name}" \
    "${spec}" \
    "-" \
    "-" \
    "${log_file}"

  if [[ "${status}" != "PASS" ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    return 1
  fi

  PASS_COUNT=$((PASS_COUNT + 1))
}

main() {
  parse_args "$@"
  prepare_tool_env
  setup_paths
  ensure_requirements
  ensure_firmware_artifacts
  ensure_ghidra_image

  prepare_fixture pulseox_measurement_update
  prepare_fixture bloodlight_usb_send_message
  prepare_fixture bloodview_device_process_entry
  prepare_fixture bloodlight_led_loop

  init_summary
  PASS_COUNT=0
  FAIL_COUNT=0
  CASE_FAILURE=0

  run_matrix_case \
    "measurement_before_patch" \
    "pulseox_measurement_update" \
    "${REPO_ROOT}/test/patchir-transform/measurement_update_before_patch.yaml" \
    'patch__before__spo2_lookup' \
    'patchestry_operation' || CASE_FAILURE=1

  run_matrix_case \
    "measurement_after_patch" \
    "pulseox_measurement_update" \
    "${REPO_ROOT}/test/patchir-transform/measurement_update_after_patch.yaml" \
    'patch__after__spo2_lookup' \
    'patchestry_operation' || CASE_FAILURE=1

  run_matrix_case \
    "measurement_replace_patch" \
    "pulseox_measurement_update" \
    "${REPO_ROOT}/test/patchir-transform/measurement_update_replace_patch.yaml" \
    'patch__replace__spo2_lookup' \
    'patch__replace__spo2_lookup|patchestry_operation' || CASE_FAILURE=1

  run_matrix_case \
    "measurement_before_operation" \
    "pulseox_measurement_update" \
    "${REPO_ROOT}/test/patchir-transform/measurement_update_before_operation.yaml" \
    'patch__before__spo2_lookup' \
    'patchestry_operation' || CASE_FAILURE=1

  run_matrix_case \
    "measurement_operation_after" \
    "pulseox_measurement_update" \
    "${REPO_ROOT}/test/patchir-transform/operation_apply_after.yaml" \
    'patch__after__spo2_lookup' \
    'patchestry_operation' || CASE_FAILURE=1

  run_matrix_case \
    "measurement_operation_replace" \
    "pulseox_measurement_update" \
    "${REPO_ROOT}/test/patchir-transform/operation_replace.yaml" \
    'patch__replace__spo2_lookup' \
    'patch__replace__spo2_lookup|patchestry_operation' || CASE_FAILURE=1

  # inline-patches multisite: patch__before__spo2_lookup inlined at three
  # __aeabi_fdiv sites. The value of this case is the end-to-end pipeline
  # signal — the multi-site UAF that Phase 1 fixed would crash
  # patchir-transform here. Post-inline content patterns are fragile
  # across platforms (e.g. `isfinite_float` gets inlined by ClangEmitter
  # on some targets but kept as a call on others), so no CIR/LLVM pattern
  # assertions: passing = the pipeline ran to completion.
  run_matrix_case \
    "measurement_inline_multisite" \
    "pulseox_measurement_update" \
    "${REPO_ROOT}/test/patchir-transform/inline_patches_multisite.yaml" \
    '' \
    '' || CASE_FAILURE=1

  # The USB pre-instrumentation spec used to dispatch a runtime contract
  # `contract__before__test_contract`; post-PR#199 that's migrated to a
  # second patch_action emitting `patch__before__usb_state_check`.
  run_matrix_case \
    "usb_before_patch" \
    "bloodlight_usb_send_message" \
    "${REPO_ROOT}/test/patchir-transform/bl_usb__send_message_before_patch.yaml" \
    'patch__before__usbd_ep_write_packet|patch__before__usb_state_check' \
    'patch__before__usbd_ep_write_packet|patch__before__usb_state_check|patchestry_operation' || CASE_FAILURE=1

  run_matrix_case \
    "usb_after_patch" \
    "bloodlight_usb_send_message" \
    "${REPO_ROOT}/test/patchir-transform/bl_usb__send_message_after_patch.yaml" \
    'patch__after__usbd_ep_write_packet' \
    'patch__after__usbd_ep_write_packet|patchestry_operation' || CASE_FAILURE=1

  run_matrix_case \
    "usb_before_update_patch" \
    "bloodlight_usb_send_message" \
    "${REPO_ROOT}/test/patchir-transform/bl_usb__send_message_before_update_patch.yaml" \
    'patch__before__usbd_cp_write_packet__update_state' \
    'patch__before__usbd_cp_write_packet__update_state|patchestry_operation' || CASE_FAILURE=1

  # APPLY_AT_ENTRYPOINT is patch-only now (contracts are static-only, no
  # runtime dispatch). Switched from entrypoint_contract.yaml (which
  # expected a runtime-contract call that never gets emitted) to the
  # flat-surface entrypoint_patch.yaml replacement.
  run_matrix_case \
    "usb_entrypoint_patch" \
    "bloodlight_usb_send_message" \
    "${REPO_ROOT}/test/patchir-transform/entrypoint_patch.yaml" \
    'patch__entrypoint__message_entry_check|patchestry_operation' \
    'patch__entrypoint__message_entry_check|patchestry_operation' || CASE_FAILURE=1

  # Contracts are static-only: `contract.static` attribute is attached
  # directly to the patch__replace__sprintf call; no separate
  # `contract__sprintf` runtime call is emitted.
  run_matrix_case \
    "bloodview_device_process_entry" \
    "bloodview_device_process_entry" \
    "${REPO_ROOT}/test/patchir-transform/device_process_entry.yaml" \
    'patch__replace__sprintf|contract.static' \
    'patch__replace__sprintf|static_contract' || CASE_FAILURE=1

  run_matrix_case \
    "bloodview_all_predicates" \
    "bloodview_device_process_entry" \
    "${REPO_ROOT}/test/patchir-transform/all_predicates.yaml" \
    'patch__replace__sprintf|contract.static|return_value_nonnegative' \
    'patch__replace__sprintf|static_contract|dest_nonnull|return_value_nonnegative' || CASE_FAILURE=1

  run_matrix_case \
    "led_loop_before_cmp_operation" \
    "bloodlight_led_loop" \
    "${REPO_ROOT}/test/patchir-transform/bl_led_loop_before_cmp_operation.yaml" \
    'patch__before__cmp__bl_spi_mode' \
    'patch__before__cmp__bl_spi_mode|patchestry_operation' || CASE_FAILURE=1

  run_negative_case \
    "negative_missing_spec_file" \
    "pulseox_measurement_update" \
    "${FIXTURE_DIR}/does-not-exist.yaml" \
    "\"${PATCHIR_TRANSFORM}\" \"${FIXTURE_DIR}/pulseox_measurement_update/pulseox_measurement_update.cir\" --spec \"${FIXTURE_DIR}/does-not-exist.yaml\" -o \"${OUTPUT_DIR}/negative_missing_spec_file/unused.cir\"" \
    'File does not exist|Failed to load file|failed to load config' || CASE_FAILURE=1

  # llvm::yaml stops processing after the first mapping error, so only
  # the offending entry's diagnostics reach the log. The negative
  # fixture is shaped to always trigger a "missing required key" error
  # on the first (intentionally broken) `patches:` entry.
  run_negative_case \
    "negative_malformed_schema" \
    "-" \
    "${REPO_ROOT}/test/patchir-transform/test_patch.yaml" \
    "\"${PATCHIR_YAML_PARSER}\" \"${REPO_ROOT}/test/patchir-transform/test_patch.yaml\" --validate" \
    'error|missing required key' || CASE_FAILURE=1

  run_negative_case \
    "negative_bad_patch_reference" \
    "pulseox_measurement_update" \
    "${REPO_ROOT}/test/patchir-transform/bad_patch_ref.yaml" \
    "\"${PATCHIR_TRANSFORM}\" \"${FIXTURE_DIR}/pulseox_measurement_update/pulseox_measurement_update.cir\" --spec \"${REPO_ROOT}/test/patchir-transform/bad_patch_ref.yaml\" -o \"${OUTPUT_DIR}/negative_bad_patch_reference/unused.cir\"" \
    'Patch specification for ID|not found|Failed to run instrumentation passes' || CASE_FAILURE=1

  run_negative_case \
    "negative_missing_library_file" \
    "pulseox_measurement_update" \
    "${REPO_ROOT}/test/patchir-transform/missing_library.yaml" \
    "\"${PATCHIR_TRANSFORM}\" \"${FIXTURE_DIR}/pulseox_measurement_update/pulseox_measurement_update.cir\" --spec \"${REPO_ROOT}/test/patchir-transform/missing_library.yaml\" -o \"${OUTPUT_DIR}/negative_missing_library_file/unused.cir\"" \
    'Failed to load library|File does not exist|Failed to run instrumentation passes' || CASE_FAILURE=1

  run_negative_case \
    "negative_unsupported_argument_source" \
    "-" \
    "${REPO_ROOT}/test/patchir-transform/unsupported_argument_source.yaml" \
    "\"${PATCHIR_YAML_PARSER}\" \"${REPO_ROOT}/test/patchir-transform/unsupported_argument_source.yaml\" --validate" \
    'Unknown argument source type|unsupported' || CASE_FAILURE=1

  cat >>"${SUMMARY_MD}" <<EOF

## Totals

- Passed: ${PASS_COUNT}
- Failed: ${FAIL_COUNT}
- Fixture cache: ${FIXTURE_DIR}
- Output directory: ${OUTPUT_DIR}
EOF

  if [[ ${CASE_FAILURE} -ne 0 ]]; then
    echo "Patch matrix validation failed. See ${SUMMARY_MD}" >&2
    # Dump the summary + any FAIL case's run.log so CI captures the
    # full failure context in its step log (the summary file itself
    # isn't uploaded as an artifact).
    if [[ -f "${SUMMARY_MD}" ]]; then
      echo "----- ${SUMMARY_MD} -----" >&2
      cat "${SUMMARY_MD}" >&2
      echo "-----" >&2
    fi
    if [[ -f "${SUMMARY_TSV}" ]]; then
      while IFS=$'\t' read -r case status case_type fixture spec patched_cir llvm_ir log; do
        if [[ "${status}" == "FAIL" ]]; then
          echo "----- FAIL case '${case}' log: ${log} -----" >&2
          [[ -f "${log}" ]] && cat "${log}" >&2
          echo "-----" >&2
        fi
      done < <(tail -n +2 "${SUMMARY_TSV}")
    fi
    exit 1
  fi

  echo "Patch matrix validation passed. See ${SUMMARY_MD}"
}

main "$@"
