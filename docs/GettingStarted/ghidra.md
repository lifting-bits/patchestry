# Ghidra integration

We support two decompilation modes:

1. **GUI-based:** A user-friendly interface where functions can be decompiled interactively.
2. **Headless:** Decompilation through a Docker script for automated, non-interactive use.

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

## Headless Decompilation

*TBD*