#!/bin/bash -x
#
# Copyright (c) 2024, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

INPUT_FILE=""
COMMAND=""
FUNCTION_NAME=""
OUTPUT_FILE=""
HIGH_PCODE=""

function help {
  cat << EOF
Usage: ./decompile-entrypoint.sh [OPTIONS]

Options:
  --help, -h
      Display this help message.

  --input <INPUT_FILE>
      Specify the input binary file to load.

  --command <COMMAND>
      Specify the command to execute. Available commands are:
        - list-functions: List all functions in the binary.
        - decompile: Decompile a single function.
        - decompile-all: Decompile all functions in the binary.

  --function <FUNCTION_NAME>
      Decompile a specific function to extract pcode. This option is required when using the 'decompile' command.

  --output <OUTPUT_FILE>
      Specify the output file to write the results.

Examples:
  ./decompile-entrypoint.sh --input /path/to/file --command list-functions --output /path/to/output.json
  ./decompile-entrypoint.sh --input /path/to/file --command decompile --function main --output /path/to/output.json
  ./decompile-entrypoint.sh --input /path/to/file --command decompile-all --output /path/to/output.json
EOF
}

function die {
  echo "Error: $1" >&2
  exit 1
}

# Parse arguments
function parse_args {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help|-h)
        help
        exit 0
        ;;
      --input)
        if [[ -n "$2" ]]; then
          INPUT_FILE="$2"
          shift
        else
          die "--input requires an argument."
        fi
        ;;
      --command)
        if [[ -n "$2" ]]; then
          COMMAND="$2"
          shift
        else
          die "--command requires an argument."
        fi
        ;;
      --function)
        if [[ -n "$2" ]]; then
          FUNCTION_NAME="$2"
          shift
        else
          die "--function requires an argument."
        fi
        ;;
      --output)
        if [[ -n "$2" ]]; then
          OUTPUT_FILE="$2"
          shift
        else
          die "--output requires an argument."
        fi
        ;;
      *)
        die "Invalid option '$1'."
        ;;
    esac
    shift
  done
}

function validate_args {
  # Validate required arguments
  if [[ -z "$INPUT_FILE" ]]; then
    die "--input is required."
  fi

  if [[ -z "$COMMAND" ]]; then
    die "--command is required."
  fi

  if [[ "$COMMAND" == "decompile" && -z "$FUNCTION_NAME" ]]; then
    die "--function is required when --command decompile is specified."
  fi

  if [[ -z "$OUTPUT_FILE" ]]; then
    die "--output is required."
  fi
}

# Function to run Ghidra headless script for listing functions
function run_list_functions {
  echo "Running Ghidra headless script to list functions..."
  ${GHIDRA_HEADLESS} ${GHIDRA_PROJECTS} patchestry-decompilation \
    -readOnly -deleteProject \
    -import $INPUT_FILE \
    -postScript "PatchestryListFunctions.java" \
    $OUTPUT_FILE

  if [ $? -ne 0 ]; then
    die "List function failed"
  fi
}

# Function to run Ghidra headless script for decompiling
function run_decompile_single {
  echo "Running Ghidra headless script to decompile function..."
  local ghidra_script="PatchestryDecompileFunctions.java"
  
  ${GHIDRA_HEADLESS} ${GHIDRA_PROJECTS} patchestry-decompilation \
    -readOnly -deleteProject \
    -import $INPUT_FILE \
    -postScript ${ghidra_script} \
    single \
    $FUNCTION_NAME \
    $OUTPUT_FILE

  if [ $? -ne 0 ]; then
    die "Decompilation failed"
  fi
}

function run_decompile_all {
  echo "Running Ghidra headless script to decompile function..."
  local ghidra_script="PatchestryDecompileFunctions.java"

  ${GHIDRA_HEADLESS} ${GHIDRA_PROJECTS} patchestry-decompilation \
    -readOnly -deleteProject \
    -import $INPUT_FILE \
    -postScript ${ghidra_script} \
    all \
    $OUTPUT_FILE

  if [ $? -ne 0 ]; then
    die "Decompilation failed"
  fi
}


function main {
  parse_args $@
  validate_args

  if [ ! -w "$OUTPUT_FILE" ]; then
    sudo chmod 777 "$OUTPUT_FILE" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Error: Failed to change permissions on output file '$OUTPUT_FILE'."
        exit 1
    fi
  fi

  case "$COMMAND" in
    list-functions)
      run_list_functions
      ;;
    decompile)
      run_decompile_single
      ;;
    decompile-all)
      run_decompile_all
      ;;
    *)
      die "Unknown command '$COMMAND'."
      ;;
  esac
}

main "$@"
exit $?
