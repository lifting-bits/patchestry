/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/PatchDSL/Compiler.hpp>

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

    if (input_filename == "-") {
        llvm::errs() << "patchir-dslc: error: input .patch file required (use: patchir-dslc <file.patch>)\n";
        return 2;
    }

    patchestry::patchdsl::CompilerOptions opts;
    opts.import_paths.assign(import_paths.begin(), import_paths.end());

    // `--check` only needs parsing, no lowering.
    if (check_only) {
        auto ast_or = patchestry::patchdsl::ParseFile(input_filename, opts);
        if (!ast_or) {
            llvm::errs() << "patchir-dslc: " << llvm::toString(ast_or.takeError())
                         << "\n";
            return 1;
        }
        llvm::outs() << "parsed OK: " << input_filename << "\n";
        return 0;
    }

    // Everything else (--dump-config, -o, summary) requires the lowered
    // Configuration.
    auto cfg_or = patchestry::patchdsl::CompileFile(input_filename, opts);
    if (!cfg_or) {
        llvm::errs() << "patchir-dslc: " << llvm::toString(cfg_or.takeError()) << "\n";
        return 1;
    }
    auto const &cfg = *cfg_or;

    if (dump_config) {
        llvm::outs() << patchestry::patchdsl::ConfigurationToYAML(cfg);
        return 0;
    }

    if (!output_filename.empty()) {
        llvm::errs() << "patchir-dslc: -o not implemented until Phase 4 "
                        "(use --check or --dump-config for now)\n";
        return 1;
    }

    // No explicit flag — print a short summary of the lowered Configuration.
    llvm::outs() << "compiled `" << input_filename << "`: "
                 << cfg.libraries.patches.size() << " library patch(es), "
                 << cfg.meta_patches.size() << " meta-patch(es), "
                 << cfg.meta_contracts.size() << " meta-contract(s)\n";
    return 0;
}
