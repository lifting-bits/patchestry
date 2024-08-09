# Decompilation and Testing Framework

The directory includes a decompilation script that runs Ghidra in headless mode to extract pcode for specific functions from binary files. It also features a testing framework that compiles the source files, generates the corresponding pcode for the specified function, and validates the output using the FileCheck verifier.

## Prerequisites

Before running the scripts, make sure you have the following installed:

- **Docker**: The scripts use a Docker container to run Ghidra in headless mode.
- **FileCheck**: The `FileCheck` tool is installed and available in your PATH. It is typically part of LLVM suite.


## Usage

### Run Decompilation Script

To decompile and extract pcode for a specific function from a binary file, use the `decompile-headless.sh` script. This script extracts the pcode for the specified function and writes the output to a file named `patchestry.out.json` in the output directory.

```bash
./decompile-headless.sh <binary> <function-name> <output-file>
```

### Testing the Output

#### Using Test Script
You can run all the tests using `decompile-headless-test.sh` script:
```
./decompile-headless-test.sh
```

#### Testing with CMake
Alternatively, you can use CMake to configure, build, and run the tests.
```
cmake -B /path/to/build -S /path/to/test
cmake --build /path/to/build
ctest  --test-dir /path/to/build
```
