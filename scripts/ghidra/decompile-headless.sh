#!/bin/bash
#
# Copyright (c) 2024, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

show_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  -h, --help                           Show this help message and exit
  -i, --input                          Path to the input file
  -f, --function                       Name of the function to decompile
                                       (with --c-source, may also be an
                                       address e.g. 0x0000f69c, including
                                       any address inside the function body)
  -l, --list-functions                 List all functions from input file
      --c-source                       Emit Ghidra's plain-C decompilation
                                       for the function selected by --function
                                       (debugging aid; output is a .c file,
                                       not JSON). Requires --function.
  -o, --output                         Path to the output file where results will be saved
  -v, --verbose                        Enable verbose output
  -t, --interactive                    Start Docker container in interactive mode
  -c, --ci                             Run in CI mode
      --sanitize-extraout[=on|off]     Control the extraout/unaff/in_
                                       register-alias sanitizer in PcodeSerializer.
                                       Default: on. Use --no-sanitize-extraout
                                       to disable.
      --sanitize-extraout-analytical <auto|on|off>
                                       Control Tier 2 (analytical callee-walk)
                                       preservation analysis. Default: auto
                                       (architecture allowlist).
      --repair-function-boundaries     Run the TailCallAnalysis pre-pass
                                       (default on). Pass
                                       --no-repair-function-boundaries to
                                       skip the pass.

Environment:
  HOST_WORKSPACE        When running in Docker-in-Docker, set this to the host
                        path that maps to /workspace in the container.

Examples:
  ./decompile-headless.sh --input /path/to/file --output /path/to/output.json  // Decompile all functions
  ./decompile-headless.sh --input /path/to/file --function main --output /path/to/output.json // Decompile single function
  ./decompile-headless.sh --input /path/to/file --list-functions --output /path/to/output.json // List all functions from binary
  ./decompile-headless.sh --input /path/to/file --function bl_usb__send_message \\
      --output /tmp/out.json --sanitize-extraout
  ./decompile-headless.sh --input /path/to/file --c-source --function FUN_0000f69c \\
      --output /tmp/FUN_0000f69c.c // Ghidra C decompilation of one function
EOF
}

# Forwarded to the inner entrypoint / Java script.
SANITIZER_ARGS=()

# Translate container path to host path for Docker-in-Docker scenarios.
# When HOST_WORKSPACE is set, replaces /workspace prefix with the host path.
translate_to_host_path() {
    local path="$1"
    if [ -n "$HOST_WORKSPACE" ]; then
        echo "${path/#\/workspace/$HOST_WORKSPACE}"
    else
        echo "$path"
    fi
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
            --c-source)
                C_SOURCE="true"
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
                if [ -z "$CI_OUTPUT_FOLDER" ]; then
                    echo "Error: CI output folder path cannot be empty"
                    exit 1
                fi
                shift 2
                ;;
            --sanitize-extraout)
                SANITIZER_ARGS+=("--sanitize-extraout")
                shift
                ;;
            --sanitize-extraout=*)
                SANITIZER_ARGS+=("$1")
                shift
                ;;
            --no-sanitize-extraout)
                SANITIZER_ARGS+=("--no-sanitize-extraout")
                shift
                ;;
            --sanitize-extraout-analytical)
                if [ -z "$2" ]; then
                    echo "Error: --sanitize-extraout-analytical requires a value (auto|on|off)"
                    exit 1
                fi
                SANITIZER_ARGS+=("--sanitize-extraout-analytical" "$2")
                shift 2
                ;;
            --sanitize-extraout-analytical=*)
                SANITIZER_ARGS+=("$1")
                shift
                ;;
            --repair-function-boundaries)
                SANITIZER_ARGS+=("--repair-function-boundaries")
                shift
                ;;
            --no-repair-function-boundaries)
                SANITIZER_ARGS+=("--no-repair-function-boundaries")
                shift
                ;;
            --repair-function-boundaries=*)
                SANITIZER_ARGS+=("$1")
                shift
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

prepare_paths() {
    INPUT_PATH=$(realpath "$INPUT_PATH")

    if [ ! -e "$INPUT_PATH" ]; then
        echo "Error: Input file does not exist: $INPUT_PATH"
        exit 1
    fi

    local output_dir=$(dirname "$OUTPUT_PATH")
    local output_name=$(basename "$OUTPUT_PATH")
    OUTPUT_PATH="$(realpath "$output_dir")/$output_name"

    # In CI mode, don't pre-create file - let container create it to avoid permission issues
    if [ -z "$CI_OUTPUT_FOLDER" ]; then
        if [ ! -e "$OUTPUT_PATH" ]; then
            if [ "$VERBOSE" = true ]; then
                echo "Creating output file: $OUTPUT_PATH"
            fi
            touch "$OUTPUT_PATH"
        fi
        chmod 666 "$OUTPUT_PATH" 2>/dev/null || true
    fi
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
    # CI_OUTPUT_FOLDER should be absolute and exist as a directory
    if [ -n "$CI_OUTPUT_FOLDER" ]; then
        if is_not_absolute_path "$CI_OUTPUT_FOLDER"; then
            echo "$CI_OUTPUT_FOLDER path is not absolute. Exiting!"
            exit 1
        fi
        if [ ! -d "$CI_OUTPUT_FOLDER" ]; then
            echo "$CI_OUTPUT_FOLDER does not exist or is not a directory. Exiting!"
            exit 1
        fi
    fi

    if [ ! -f "$INPUT_PATH" ]; then
        echo "Input file $INPUT_PATH doesn't exist. Exiting!"
        exit 1
    fi

    # In non-CI mode, output file should exist (we created it in prepare_paths)
    if [ -z "$CI_OUTPUT_FOLDER" ] && [ ! -f "$OUTPUT_PATH" ]; then
        echo "Output file $OUTPUT_PATH doesn't exist. Exiting!"
        exit 1
    fi
}

# Assemble the `docker run` invocation in the DOCKER_CMD array rather than a
# string. The array is executed directly as "${DOCKER_CMD[@]}", so no element
# is ever re-parsed by a shell — a FUNCTION_NAME or path containing shell
# metacharacters ($(...), backticks, ';') is passed through as inert data
# instead of being evaluated. (Building a string and running it with `eval`
# would execute such metacharacters on the caller's machine.)
build_docker_command() {
    DOCKER_CMD=(docker run --rm)

    if [ -n "$CI_OUTPUT_FOLDER" ]; then
        # Translate to host path for Docker-in-Docker scenarios
        local HOST_CI_OUTPUT_FOLDER
        HOST_CI_OUTPUT_FOLDER=$(translate_to_host_path "$CI_OUTPUT_FOLDER")
        # Make directory writable by container user (for DinD scenarios)
        chmod 1777 "$CI_OUTPUT_FOLDER" 2>/dev/null || true
        DOCKER_CMD+=(-v "$HOST_CI_OUTPUT_FOLDER:/mnt/output:rw")
    fi

    # Build the entrypoint script arguments (command mode, function, flags).
    local SCRIPT_ARGS=()
    if [ -n "$C_SOURCE" ]; then
        if [ -z "$FUNCTION_NAME" ]; then
            echo "Error: --c-source requires --function"
            exit 1
        fi
        # --c-source also accepts an address (0x...) or a Ghidra auto-name
        # (FUN_...) in addition to a real C symbol. Only a C symbol gets the
        # Mach-O leading-underscore treatment; addresses and FUN_ names must be
        # passed through verbatim so PatchestryDecompileCFunction can resolve
        # them (prefixing them with '_' yields an unresolvable target).
        if file "$INPUT_PATH" | grep -q "Mach-O"; then
            case "$FUNCTION_NAME" in
                0x*|0X*|FUN_*) ;;
                *) FUNCTION_NAME="_$FUNCTION_NAME" ;;
            esac
        fi
        SCRIPT_ARGS=(--command decompile-c --function "$FUNCTION_NAME")
    elif [ -n "$LIST_FUNCTIONS" ]; then
        SCRIPT_ARGS=(--command list-functions)
    elif [ -n "$FUNCTION_NAME" ]; then
        if file "$INPUT_PATH" | grep -q "Mach-O"; then
            FUNCTION_NAME="_$FUNCTION_NAME"
        fi
        SCRIPT_ARGS=(--command decompile --function "$FUNCTION_NAME")
    else
        SCRIPT_ARGS=(--command decompile-all)
    fi

    # Sanitizer flags are forwarded verbatim.
    SCRIPT_ARGS+=("${SANITIZER_ARGS[@]}")

    if [ -n "$CI_OUTPUT_FOLDER" ]; then
        local input_base output_base
        input_base=$(basename "$INPUT_PATH")
        output_base=$(basename "$OUTPUT_PATH")
        DOCKER_CMD+=(
            trailofbits/patchestry-decompilation:latest
            --input "/mnt/output/$input_base"
            "${SCRIPT_ARGS[@]}"
            --output "/mnt/output/$output_base"
        )
        echo "CMD: ${DOCKER_CMD[*]}"
    else
        DOCKER_CMD+=(
            -v "$INPUT_PATH:/input.o"
            -v "$OUTPUT_PATH:/output.json"
            trailofbits/patchestry-decompilation:latest
        )

        if [ "$INTERACTIVE" = true ]; then
            DOCKER_CMD+=(--entrypoint /bin/bash)
        else
            DOCKER_CMD+=(
                --input /input.o
                "${SCRIPT_ARGS[@]}"
                --output /output.json
            )
        fi
    fi
}

main() {
    parse_args "$@"

    if [ -z "$INPUT_PATH" ]; then
        echo "Error: Missing required option: -i, --input <input_file>"
        exit 1
    fi

    if [ -z "$OUTPUT_PATH" ]; then
        echo "Error: Missing required option: -o, --output <output_file>"
        exit 1
    fi

    prepare_paths
    validate_paths

    build_docker_command

    if [ "$VERBOSE" = true ]; then
        echo "Running Docker container with the following command:"
        echo "${DOCKER_CMD[*]}"
    fi

    "${DOCKER_CMD[@]}"
}

main "$@"
