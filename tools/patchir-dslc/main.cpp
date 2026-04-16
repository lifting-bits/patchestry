/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>

namespace {
    const llvm::cl::opt< std::string > input_filename(
        llvm::cl::Positional, llvm::cl::desc("<input .patch file>"), llvm::cl::init("-")
    );

    const llvm::cl::opt< std::string > output_filename(
        "o", llvm::cl::desc("Output .patchmod filename"), llvm::cl::value_desc("filename"),
        llvm::cl::init("")
    );

    const llvm::cl::opt< bool > check_only(
        "check", llvm::cl::desc("Parse and type-check only; do not write output"),
        llvm::cl::init(false)
    );

    const llvm::cl::opt< bool > dump_config(
        "dump-config", llvm::cl::desc("Print the lowered Configuration in YAML form"),
        llvm::cl::init(false)
    );

    const llvm::cl::list< std::string > import_paths(
        "I", llvm::cl::desc("Additional search root for imports"),
        llvm::cl::value_desc("dir")
    );
} // namespace

int main(int argc, char **argv) {
    llvm::InitLLVM init(argc, argv);
    llvm::cl::ParseCommandLineOptions(argc, argv, "patchir-dslc: compile PatchDSL to .patchmod\n");

    llvm::errs() << "patchir-dslc v1 stub: input=" << input_filename;
    if (!output_filename.empty()) {
        llvm::errs() << " output=" << output_filename;
    }
    if (check_only) {
        llvm::errs() << " (--check)";
    }
    if (dump_config) {
        llvm::errs() << " (--dump-config)";
    }
    for (auto const &dir : import_paths) {
        llvm::errs() << " -I " << dir;
    }
    llvm::errs() << "\n";
    return 0;
}
