/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "llvm/Support/CommandLine.h"

using namespace llvm;

int main(int argc, char **argv) {
    // Command line options
    cl::opt< std::string > inputFilename(
        cl::Positional, cl::desc("<input mlir file>"), cl::Required
    );
    cl::opt< std::string > outputFilename(
        "o", cl::desc("Output filename"), cl::value_desc("filename")
    );
    cl::opt< std::string > configFilename(
        "config", cl::desc("YAML configuration file"), cl::Required
    );

    cl::ParseCommandLineOptions(argc, argv);

    return 0;
}
