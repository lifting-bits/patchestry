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
Note: In general, this script is intended to be run (in Docker) via `decompile-headless.sh`, not directly.

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

  --test-decomp
      Runs in-situ decompilation tests against the Bloodlight and PulseOX firmware samples. Requires compiling
      these binaries outside of the devcontainer environment so they can be used in test cases.

Examples:
  ./decompile-entrypoint.sh --input /path/to/file --command list-functions --output /path/to/output.json
  ./decompile-entrypoint.sh --input /path/to/file --command decompile --function main --output /path/to/output.json
  ./decompile-entrypoint.sh --input /path/to/file --command decompile-all --output /path/to/output.json
  ./decompile-entrypoint.sh --test-decomp
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
      --test-decomp)
        # running in-situ tests requires only this argument
        test_decomp
        exit 0
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

<<<<<<< Updated upstream
=======
# Runs in-situ unit tests of the decompilation to pcode. todo(kaoudis) this 
# or some test suite in the folder needs to also check to see if we have at 
# least something in the output pcode for every single function in the 
# functions list. Maybe this means converting some of the existing end to end
# tests?
function test_decomp {
  local path="ghidra_scripts"
  local junit="$path/test/junit-platform-console-standalone.jar" 
  local classpath="$junit:$GHIDRA_PATH/Ghidra/Framework/Generic/lib/*:$GHIDRA_PATH/Ghidra/Framework/SoftwareModeling/lib/*:$path:$path/test"
  local script_sources=$(find "$path" -maxdepth 1 -name "*.java" | tr '\n' ' ')
  local test_sources=$(find "$path/test" -name "*Test.java" | tr '\n' ' ')

  javac -cp "$classpath" $script_sources $test_sources

  java -jar $junit \
    --class-path "$classpath" \
    --disable-banner \
    --scan-classpath \
    --include-classname ".*Test$"
}


>>>>>>> Stashed changes
ARCHITECTURE=""

# Create the Ghidra processor string (ARCHITECTURE) from attributes visible to 
# readelf in the input binary that we intend to pass to Ghidra eventually.
function detect_processor {
    local arch=$(file "$INPUT_FILE" | grep -o -E 'x86-64|Intel 80386|ARMv[0-9]+|armv[0-9]+|ARM aarch64|AArch64|ARM, big-endian|ARM, little-endian|ARM')
    local endian="LE"
    
    # First check basic architecture from file command
    if [[ "$arch" =~ x86-64 ]]; then
        ARCHITECTURE="x86:LE:64:default"
        return
    elif [[ "$arch" =~ "Intel 80386" ]]; then
        ARCHITECTURE="x86:LE:32:default"
        return
    elif [[ "$arch" =~ "ARM aarch64|AArch64" ]]; then
        ARCHITECTURE="AARCH64:LE:64:v8A"
        return
    fi

    # For ARM, check endianness first
    if file "$INPUT_FILE" | grep -q "big-endian"; then
        endian="BE"
    fi

    if [[ "$arch" =~ ARM || "$arch" =~ armv ]]; then
        if readelf -S "$INPUT_FILE" | grep -q -E "\.text.*08000000|\.data.*20000000"; then
            # Cortex-M memory layout detected
            ARCHITECTURE="ARM:${endian}:32:v7"
        else
            # Check ARM attributes for version
            local arm_attrs=$(readelf -A "$INPUT_FILE" 2>/dev/null)
            if echo "$arm_attrs" | grep -q "Tag_CPU_arch: v8"; then
                ARCHITECTURE="ARM:${endian}:32:v8"
            elif echo "$arm_attrs" | grep -q "Tag_CPU_arch: v7"; then
                ARCHITECTURE="ARM:${endian}:32:v7"
            elif echo "$arm_attrs" | grep -q "Tag_CPU_arch: v6"; then
                ARCHITECTURE="ARM:${endian}:32:v6"
            elif echo "$arm_attrs" | grep -q "Tag_CPU_arch: v5"; then
                ARCHITECTURE="ARM:${endian}:32:v5"
            elif echo "$arm_attrs" | grep -q "Tag_CPU_arch: v4"; then
                ARCHITECTURE="ARM:${endian}:32:v4"
            else
                # If all else fails, default to v7
                ARCHITECTURE="ARM:${endian}:32:v7"
            fi
        fi
    else
        echo "Unknown input-binary architecture '$arch'?"
        exit 1
    fi

    echo "Binary Architecture was: '$arch', '$arm_attrs' ($ARCHITECTURE)"
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
