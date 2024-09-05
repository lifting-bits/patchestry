# Decompilation Framework

The directory includes a decompilation script that runs Ghidra in headless mode to extract pcode for specific functions from binary files.

We support two decompilation modes:

1. **GUI-based:** A user-friendly interface where functions can be decompiled interactively.
2. **Headless:** Decompilation through a Docker script for automated, non-interactive use.

## Prerequisites

Before running the scripts, make sure you have the following installed:

- **Docker**: The scripts use a Docker container to run Ghidra in headless mode.

To perform headless decompilation, you need to build a Docker container (`decompile-headless.dockerfile`) configured to run Ghidra in headless mode. You can do this by running the `build-headless-docker.sh` script.

## Running Headless Decompilation Script

To decompile and extract pcode for a specific function from a binary file, use
the `decompile-headless.sh` script. This script extracts the pcode for the
specified function and writes the json output to a file named `<output-file>`.

```sh ./decompile-headless.sh --input <binary> --function <function-name> --output <output-file> ```

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

4. Run `PatchestryScript.java`.

**Note:** Ghidra scripts must be installed. See the [build](build.md) section for details.
