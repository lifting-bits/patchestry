# Decompilation Framework

The directory contains a script that runs Ghidra in headless mode to
decompile binary files, identifying and listing all functions while
extracting their corresponding pcode.

We support two decompilation modes:

1. **GUI-based:** A user-friendly interface where functions can be decompiled interactively.
2. **Headless:** Decompilation through a Docker script for automated, non-interactive use.

## Prerequisites

Before running the scripts, make sure you have the following installed:

- **Docker**: The scripts use a Docker container to run Ghidra in headless mode.

To perform headless decompilation, you need to build a Docker container (`decompile-headless.dockerfile`) configured to run Ghidra in headless mode. You can do this by running the `build-headless-docker.sh` script.

## Running Headless Decompilation Script

The `decompile-headless.sh` script decompiles a binary file using Ghidra
in headless mode, extracting pcode for either a specific function or all
functions by default, and saving the output as json to a specified file `<output-file>`.

To extract P-code for a particular function, use the `--function` flag; otherwise,
it decompiles all functions if no function name is specified.

```sh ./decompile-headless.sh --input <binary> --function <function-name> --output <output-file> ```

The script also list all functions in the binary using the `--list-functions` flag.

```sh ./decompile-headless.sh --input <binary> --list-functions --output <output-file> ```

## Running Patchestry via Ghidra GUI

1. Ensure Patchestry is available via PATH:
    ```shell
    patchestry
    ```

2. Start Ghidra GUI:
    ```shell
    ~/ghidra/ghidraRun
    ```

3. Create a project and import a binary file.

4. Run `PatchestryDecompileFunctions.java` in `single` or `all` mode to decompile single or all functions from a binary file.

5. Run `PatchestryListFunctions.java` script to list all the functions in a binary file.

**Note:** Ghidra scripts must be installed. See the [build](build.md) section for details.
