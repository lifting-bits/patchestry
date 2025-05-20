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

ARCHITECTURE=""

# Create the Ghidra processor string (ARCHITECTURE) from attributes visible to 
# readelf in the input binary that we intend to pass to Ghidra eventually.
# The processor string tells Ghidra how to decompile the input binary, based on
# the pspec and cspec files available to it. The string has four colon-sep 
# parts (processor_name:endianness:mode:variant), the first
# three of which relate to and help define which pspec Ghidra will use:
#
# The first part is the processor name (e.g., x86, ARM, MIPS).
# The second part is the endianness (e.g., LE, BE).
# The third part is the address size or mode (e.g., 32, 64, 16).
# The fourth part is the compiler specification or a processor variant (e.g., 
# gcc, windows, clang, or a custom variant for special processor features).
# This defines which cspec Ghidra will use to decompile. This doesn't quite mean
# "what compiler was used to compile the input binary", i.e., if a binary was
# originally compiled with clang for linux, you'd still put gcc or default for
# the variant string.
function detect_processor {
    local file_output = $(file "$INPUT_FILE")
    local variant="default" 
    local mode="32" 

    local endianness="LE"
    if [["$file_output" == *"big-endian"*]]; then
        endianness="BE"
    fi

    if [[ "$file_output" == *"ELF"* ]]; then
      variant="gcc"
    elif [[ "$file_output" == *"PE32"* || "$file_output" == *"PE32+"* ]]; then
      variant="windows"
    elif [[ "$file_output" == *"Mach-O"* ]]; then
      variant="clang"
    fi 

    local processor_name=$("$file_output" | grep -o -E 'x86-64|Intel 80386|ARMv[0-9]+|armv[0-9]+|ARM aarch64|AArch64|ARM, big-endian|ARM, little-endian|ARM')
    if [[ "$processor_name" =~ "x86-64" ]]; then
        ARCHITECTURE="x86:${endianness}:64:${variant}"
        return
    elif [[ "$processor_name" =~ "Intel 80386" ]]; then
        ARCHITECTURE="x86:${endianness}:{$mode}:${variant}"
        return
    elif [[ "$processor_name" =~ "ARM aarch64|AArch64" ]]; then
        ARCHITECTURE="AARCH64:${endianness}:64:v8A"
        return
    fi

    if [[ "$processor_name" =~ ARM || "$processor_name" =~ armv ]]; then
        variant="v7"
        if readelf -S "$INPUT_FILE" | grep -q -E "\.text.*08000000|\.data.*20000000"; then
            # Cortex-M memory layout detected
            ARCHITECTURE="ARM:${endianness}:${mode}:${variant}"
        else
            # Check ARM attributes for version
            local arm_attrs=$(readelf -A "$INPUT_FILE" 2>/dev/null)
            if echo "$arm_attrs" | grep -q "Tag_CPU_arch: v8"; then
                ARCHITECTURE="ARM:${endianness}:${mode}:v8"
            elif echo "$arm_attrs" | grep -q "Tag_CPU_arch: v7"; then
                if [["$arm_attrs" =~ "Cortex-M"]]; then
                  variant="Cortex"
                fi

                ARCHITECTURE="ARM:${endianness}:${mode}:${variant}"
            elif echo "$arm_attrs" | grep -q "Tag_CPU_arch: v6"; then
                ARCHITECTURE="ARM:${endianness}:${mode}:v6"
            elif echo "$arm_attrs" | grep -q "Tag_CPU_arch: v5"; then
                ARCHITECTURE="ARM:${endianness}:${mode}:v5"
            elif echo "$arm_attrs" | grep -q "Tag_CPU_arch: v4"; then
                ARCHITECTURE="ARM:${endianness}:${mode}:v4"
            else
                echo "Not sure what we have here... defaulting to ARMv7 for now... Attributes: '${arm_attrs}'"
                ARCHITECTURE="ARM:${endianness}:${mode}:${variant}"
            fi
        fi
    else
        echo "Could not determine full input-binary architecture"
        observed '$processor_name':'$endianness':'$mode':'$variant'
        exit 1
    fi

    echo "observed '$processor_name':'$endianness':'$mode':'$variant'"
}

# Function to run Ghidra headless script for listing functions
function run_list_functions {
  echo "Running Ghidra headless script to list functions..."
  ${GHIDRA_HEADLESS} ${GHIDRA_PROJECTS} patchestry-decompilation \
    -readOnly -deleteProject \
    -import $INPUT_FILE \
    -processor $ARCHITECTURE \
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
    -processor $ARCHITECTURE \
    -postScript ${ghidra_script} \
    single \
    $FUNCTION_NAME \
    $OUTPUT_FILE

  if [ $? -ne 0 ]; then
    die "Decompilation failed"
  fi
}

function run_decompile_all {
  echo "Running Ghidra headless script to decompile all functions..."
  local ghidra_script="PatchestryDecompileFunctions.java"

  ${GHIDRA_HEADLESS} ${GHIDRA_PROJECTS} patchestry-decompilation \
    -readOnly -deleteProject \
    -import $INPUT_FILE \
    -processor $ARCHITECTURE \
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

  detect_processor

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
