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
# default, windows, clang, or a custom variant for special processor features).
# This defines which cspec Ghidra will use to decompile. This doesn't quite mean
# "what compiler was used to compile the input binary", i.e., if a binary was
# originally compiled with clang for linux, you'd still put default for
# the variant string.
#
# Currently, this function is specific to ELF architecture detection, and covers
# the bare basics of PE architecture detection.
# todo(kaoudis) Future work should include more support for binary blob, and PE.
function detect_processor {
    local file_output=$(file "$INPUT_FILE")

    local endianness="LE"
    if [[ "$file_output" == *"big-endian"* ]]; then
        endianness="BE"
    fi

    local variant="default"
    if [[ "$file_output" == *"PE32"* ]]; then
        variant="windows"
    elif [[ "$file_output" == *"Mach-O"* ]]; then
        variant="clang"
    fi

    local mode="32"
    # this grep will take only the first match
    local processor_name=$(echo "$file_output" | grep -o -E \
        'x86-64|Intel 80386|ARMv[0-9]+|armv[0-9]+|ARM aarch64|ARM|AArch64|MIPS|PowerPC|AVR|MSP430|8051|68k|SPARC|RISC-V|Xtensa|CR16C|Z80|6502|PIC')
            case "$processor_name" in
                "x86-64")
                    processor_name="x86"
                    mode="64"
                    ;;
                "Intel 80386")
                    processor_name="x86"
                    ;;
                "ARM aarch64"|"AArch64")
                    processor_name="AARCH64"
                    mode="64"
                    variant="v8A"
                    ;;
                ARM*|armv*)
                    variant="v7" # also includes/will be used for Cortex-A
                    processor_name="ARM"

                    local arm_attrs=$(readelf -A "$INPUT_FILE" 2>/dev/null)
                    if [[ "$arm_attrs" == *"Tag_CPU_arch: v8"* ]]; then
                        variant="v8"
					# If we're using variant="Cortex", the binary uses Thumb (T32) 
					# instructions and we want Ghidra to correspondingly/correctly 
					# disassemble them. While we *hope* if we get here that the ELF 
					# attributes show that the binary is compiled for a Cortex 
					# microcontroller, we fall back to section-addressing as attribs
					# are not always present (eg, for a raw .bin or an Intel HEX file,
					# or if ELF attributes are otherwise stripped). 
					# Section addressing varies for other archs, 
					# but 0x20000000 is by standard the start of SRAM for Cortex-M.
					# https://developer.arm.com/documentation/dui0646/c/The-Cortex-M7-Processor/Memory-model
					# https://developer.arm.com/documentation/dui0662/latest/The-Cortex-M0--Processor/Memory-model 
                    elif [[ "$arm_attrs" == *"Cortex-M"* ]] || [[ "$arm_attrs" == *"Tag_CPU_arch_profile: Microcontroller"* ]] || readelf -S "$INPUT_FILE" | grep -q -E "\.data.*20000000"; then
                        variant="Cortex"
                    elif [[ "$file_output" == *"EABI2"* || "$file_output" == *"EABI1"* ]]; then
                        variant="v5"
                    fi
                    ;;
                "MIPS")
                    if [[ "$file_output" == *"MIPS64"* ]]; then
                        mode="64"
                    fi

                    if [[ "$file_output" == *"microMIPS"* ]]; then
                        variant="micro"
                    fi
                    ;;
                "PowerPC")
                    if [[ "$file_output" == *"VLE"* ]]; then
                        variant="VLE"
                    fi

                    if [[ "$file_output" == *"64-bit"* ]]; then
                        mode="64"
                    fi
                    ;;
                "68k")
                    processor_name="68000"
                    ;;
                "SPARC")
                    processor_name="Sparc"
                    if [[ "$file_output" == *"64-bit"* ]]; then
                        mode="64"
                        variant="V9"
                    fi
                    ;;
                "RISC-V")
                    processor_name="RISCV"
                    if [[ "$file_output" == *"64-bit"* ]]; then
                        mode="64"
                    fi
                    ;;
                "AVR"|"MSP430"|"8051"|"PIC"|"Z80"|"6502")
                    # Ghidra's language definitions for these architectures should not need 
                    # an explicit address size or endianness in the processor string
                    ARCHITECTURE="${processor_name}:${variant}"
                    echo "Detected architecture: ${ARCHITECTURE}"
                    return
                    ;;
                *)
                    echo "Unsupported architecture: '${processor_name}'"
                    exit 1
                    ;;
            esac

            ARCHITECTURE="${processor_name}:${endianness}:${mode}:${variant}"
            echo "Detected architecture: ${ARCHITECTURE}"
        }

# Function to run Ghidra headless script for listing functions
function run_list_functions {
    echo "Running Ghidra headless script to list functions..."

    local guess_architecture=""
    if [[ -n "$ARCHITECTURE" ]]; then
        guess_architecture="-processor ${ARCHITECTURE}"
    fi

    ${GHIDRA_HEADLESS} ${GHIDRA_PROJECTS} patchestry-decompilation \
        -readOnly \
        -deleteProject \
        -import ${INPUT_FILE} \
        ${guess_architecture}\
        -postScript "PatchestryListFunctions" \
        ${OUTPUT_FILE}

    if [ $? -ne 0 ]; then
        die "List function failed"
    fi
}

# Function to run Ghidra headless script for decompiling
function run_decompile_single {
    echo "Running Ghidra headless script to decompile the single function '$FUNCTION_NAME'..."

    local guess_architecture=""
    if [[ -n "$ARCHITECTURE" ]]; then
        guess_architecture="-processor ${ARCHITECTURE}"
    fi

    ${GHIDRA_HEADLESS} ${GHIDRA_PROJECTS} patchestry-decompilation \
        -readOnly \
        -deleteProject \
        -import ${INPUT_FILE} \
        ${guess_architecture}\
        -postScript "PatchestryDecompileFunctions" \
        single \
        ${FUNCTION_NAME} \
        ${OUTPUT_FILE}

    if [ $? -ne 0 ]; then
        die "Decompilation failed"
    fi
}

function run_decompile_all {
    echo "Running Ghidra headless script to decompile all functions..."

    local guess_architecture=""
    if [[ -n "$ARCHITECTURE" ]]; then
        guess_architecture="-processor ${ARCHITECTURE}"
    fi

    ${GHIDRA_HEADLESS} ${GHIDRA_PROJECTS} patchestry-decompilation \
        -readOnly \
        -deleteProject \
        -import ${INPUT_FILE} \
        ${guessArchitecture}\
        -postScript "PatchestryDecompileFunctions" \
        all \
        ${OUTPUT_FILE}

    if [ $? -ne 0 ]; then
        die "Decompilation failed"
    fi
}


function check_output_writable {
    if [ -e "$OUTPUT_FILE" ]; then
        # File exists - check if it's writable
        if [ ! -w "$OUTPUT_FILE" ]; then
            die "Output file '$OUTPUT_FILE' exists but is not writable."
        fi
    else
        # File doesn't exist - check if parent directory is writable
        local parent_dir=$(dirname "$OUTPUT_FILE")
        if [ ! -d "$parent_dir" ]; then
            die "Parent directory '$parent_dir' does not exist."
        fi
        if [ ! -w "$parent_dir" ]; then
            die "Parent directory '$parent_dir' is not writable."
        fi
    fi
}

function main {
    parse_args $@
    validate_args

    check_output_writable
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
