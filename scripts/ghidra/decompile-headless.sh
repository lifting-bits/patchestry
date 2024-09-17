#!/bin/bash
#
# Copyright (c) 2024, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

show_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  -h, --help            Show this help message and exit
  -i, --input           Path to the input file
  -f, --function        Name of the function to decompile
  -l, --list-functions  List all functions from input file
  -o, --output          Path to the output file where results will be saved
  -v, --verbose         Enable verbose output
  -t, --interactive     Start Docker container in interactive mode
  -c, --ci              Run in CI mode

Examples:
  ./decompile-headless.sh --input /path/to/file --output /path/to/output.json  // Decompile all functions
  ./decompile-headless.sh --input /path/to/file --function main --output /path/to/output.json // Decompile single function
  ./decompile-headless.sh --input /path/to/file --list-functions --output /path/to/output.json // List all functions from binary
EOF
}

parse_args() {
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
            -l|--list-functions)
                LIST_FUNCTIONS="true"
                shift
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
            -c|--ci)
                CI_OUTPUT_FOLDER="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

validate_args() {
    if [ -z "$INPUT_PATH" ]; then
        echo "Error: Missing required option: -i, --input <input_file>"
        exit 1
    fi

    if [ -z "$OUTPUT_PATH" ]; then
        echo "Error: Missing required option: -o, --output <output_file>"
        exit 1
    fi
}

prepare_paths() {
    INPUT_PATH=$(realpath "$INPUT_PATH")

    if [ ! -e "$INPUT_PATH" ]; then
        echo "Error: Input file does not exist: $INPUT_PATH"
        exit 1
    fi

    if [ ! -e "$OUTPUT_PATH" ]; then
        if [ "$VERBOSE" = true ]; then
            echo "Creating output file: $OUTPUT_PATH"
        fi
        touch "$OUTPUT_PATH"
    fi
    # realpath may fail of OUTPUT_PATH does not exist
    OUTPUT_PATH=$(realpath "$OUTPUT_PATH")
}

is_not_absolute_path() {
    case "$1" in
        /*)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

validate_paths() {
    # CI_OUTPUT_FOLDER should be absolute to avoid any issue with mounting locations
    if [ -n "$CI_OUTPUT_FOLDER" ] && is_not_absolute_path "$CI_OUTPUT_FOLDER"; then
        echo "$CI_OUTPUT_FOLDER path is not absolute. Exiting!"
        exit 1
    fi

    # Expect both input and output file to exist
    if [! -f "$INPUT_PATH" ]; then
        echo "Input file $INPUT_PATH doesn't exist. Exiting!"
        exit 1
    fi

    if [! -f "$OUTPUT_PATH" ]; then
        echo "Output file $OUTPUT_PATH doesn't exist. Exiting!"
        exit 1
    fi
}

build_docker_command() {
    if [ -n "$CI_OUTPUT_FOLDER" ]; then
        INPUT_PATH=$(basename "$INPUT_PATH")
        OUTPUT_PATH=$(basename "$OUTPUT_PATH")
        RUN="docker run --rm \
            -v $CI_OUTPUT_FOLDER:/mnt/output:rw \
            trailofbits/patchestry-decompilation:latest \
            --input /mnt/output/$INPUT_PATH \
            --command decompile --function \"$FUNCTION_NAME\" \
            --output /mnt/output/$OUTPUT_PATH"

    elif [ -n "$LIST_FUNCTIONS" ]; then
        RUN="docker run --rm \
            -v \"$INPUT_PATH:/input.o\" \
            -v \"$OUTPUT_PATH:/output.json\" \
            trailofbits/patchestry-decompilation:latest \
            --input /input.o --command list-functions \
            --output /output.json"

    else
        RUN="docker run --rm \
            -v \"$INPUT_PATH:/input.o\" \
            -v \"$OUTPUT_PATH:/output.json\" \
            trailofbits/patchestry-decompilation:latest"

        if file "$INPUT_PATH" | grep -q "Mach-O"; then
            FUNCTION_NAME="_$FUNCTION_NAME"
        fi

        if [ "$INTERACTIVE" = true ]; then
            RUN="${RUN} --entrypoint /bin/bash"
        elif [ -n "$FUNCTION_NAME" ]; then
            RUN="${RUN} \
                --input /input.o \
                --command decompile \
                --function \"$FUNCTION_NAME\" \
                --output /output.json"
        else
            RUN="${RUN} \
                --input /input.o \
                --command decompile-all \
                --output /output.json"
        fi
    fi
}

main() {
    parse_args "$@"
    validate_args
    prepare_paths
    validate_paths
    build_docker_command

    if [ "$VERBOSE" = true ]; then
        echo "Running Docker container with the following command:"
        echo "$RUN"
    fi

    eval "$RUN"
}

main "$@"