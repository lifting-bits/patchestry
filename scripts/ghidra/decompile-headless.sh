#!/bin/bash
#
# Copyright (c) 2024, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help        Show this help message and exit"
    echo "  -i, --input       Path to the input file"
    echo "  -f, --function    Name of the function to decompile"
    echo "  -o, --output      Path to the output file where results will be saved"
    echo "  -v, --verbose     Enable verbose output"
    echo "  -t, --interactive Start Docker container in interactive mode"
    echo
}

INPUT_PATH=""
FUNCTION_NAME=""
OUTPUT_PATH=""
VERBOSE=false
INTERACTIVE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--input)
            INPUT_PATH="$2"
            shift 2
            ;;
        -f|--function)
            FUNCTION_NAME="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--interactive)
            INTERACTIVE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [ -z "$INPUT_PATH" ]; then
    echo "Error: Missing required option: -i, --input <input_file>"
    exit 1
fi

if [ -z "$FUNCTION_NAME" ]; then
    echo "Error: Missing required option: -f, --function <function_name>"
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    echo "Error: Missing required option: -o, --output <output_file>"
    exit 1
fi

if [ "$VERBOSE" = true ]; then
    echo "Input file: $INPUT_PATH"
    echo "Function name: $FUNCTION_NAME"
    echo "Output file: $OUTPUT_PATH"
fi

if [ ! -e "$OUTPUT_PATH" ]; then
    if [ "$VERBOSE" = true ]; then
        echo "Creating output file: $OUTPUT_PATH"
    fi
    touch "$OUTPUT_PATH"
fi

if [ "$VERBOSE" = true ]; then
    echo "Running Docker container..."
fi

absolute_path() {
    if [[ "$1" = /* ]]; then
        echo "$1"
    else
        echo "$(pwd)/$1"
    fi
}

INPUT_ABS_PATH=$(absolute_path "$INPUT_PATH")
OUTPUT_ABS_PATH=$(absolute_path "$OUTPUT_PATH")

RUN="docker run --rm \
    -v \"$INPUT_ABS_PATH:/input.o\" \
    -v \"$OUTPUT_ABS_PATH:/output.json\" \
    trailofbits/patchestry-decompilation:latest"

if file "$INPUT_PATH" | grep -q "Mach-O"; then
    FUNCTION_NAME="_$FUNCTION_NAME"
fi

if [ "$INTERACTIVE" = true ]; then
    RUN=$(echo "$RUN" | sed 's/docker run --rm/docker run -it --rm --entrypoint \/bin\/bash/')
else
    RUN="$RUN /input.o \"$FUNCTION_NAME\" /output.json"
fi

if [ "$VERBOSE" = true ]; then
    echo "Running Docker container with the following command:"
    echo "$RUN"
fi

if [ "$VERBOSE" = true ]; then
    echo "Starting Docker container..."
fi

eval "$RUN"
