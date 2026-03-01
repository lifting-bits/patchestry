/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

// patchir-c-normalize: parse a goto-heavy C file, run the AST normalization
// pipeline, and emit structured C output.  Designed for lightweight LIT test
// fixtures that exercise specific normalization pass patterns without the
// overhead of a full Ghidra P-Code JSON fixture.

#include <memory>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/LangStandard.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Parse/ParseAST.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <patchestry/AST/ASTNormalizationPipeline.hpp>
#include <patchestry/Util/Diagnostic.hpp>
#include <patchestry/Util/Log.hpp>
#include <patchestry/Util/Options.hpp>

namespace {

    const llvm::cl::opt< std::string > input_filename( // NOLINT(cert-err58-cpp)
        "input", llvm::cl::desc("Input C file"), llvm::cl::Required
    );

    const llvm::cl::opt< std::string > output_filename( // NOLINT(cert-err58-cpp)
        "output", llvm::cl::desc("Output file prefix (appends .c)"),
        llvm::cl::value_desc("prefix"), llvm::cl::init("")
    );

    const llvm::cl::opt< bool > print_tu( // NOLINT(cert-err58-cpp)
        "print-tu", llvm::cl::desc("Write normalized C to <prefix>.c"), llvm::cl::init(false)
    );

    const llvm::cl::opt< bool > verbose( // NOLINT(cert-err58-cpp)
        "verbose", llvm::cl::desc("Enable debug logs"), llvm::cl::init(false)
    );

    const llvm::cl::opt< bool > enable_goto_elimination( // NOLINT(cert-err58-cpp)
        "enable-goto-elimination",
        llvm::cl::desc("Enable goto elimination AST pipeline"), llvm::cl::init(true)
    );

    const llvm::cl::opt< bool > goto_elimination_strict( // NOLINT(cert-err58-cpp)
        "goto-elimination-strict",
        llvm::cl::desc("Fail when goto elimination verification detects remaining gotos"),
        llvm::cl::init(false)
    );

    patchestry::Options parseCommandLineOptions(int argc, char **argv) {
        llvm::cl::ParseCommandLineOptions(
            argc, argv, "patchir-c-normalize: normalize goto-heavy C to structured C\n"
        );

        return {
            .verbose                 = verbose.getValue(),
            .enable_goto_elimination = enable_goto_elimination.getValue(),
            .goto_elimination_strict = goto_elimination_strict.getValue(),
            .output_file             = output_filename.getValue(),
            .input_file              = input_filename.getValue(),
            .print_tu                = print_tu.getValue(),
        };
    }

    // =========================================================================
    // ASTConsumer that runs the normalization pipeline and optionally prints
    // the resulting translation unit as C source.
    // =========================================================================

    class CNormalizeConsumer : public clang::ASTConsumer
    {
      public:
        explicit CNormalizeConsumer(patchestry::Options opts) : opts_(std::move(opts)) {}

        void HandleTranslationUnit(clang::ASTContext &ctx) override {
            if (opts_.enable_goto_elimination) {
                if (!patchestry::ast::runASTNormalizationPipeline(ctx, opts_)) {
                    LOG(ERROR) << "Normalization pipeline failed\n";
                    if (opts_.goto_elimination_strict) {
                        return;
                    }
                }
            }

            if (opts_.print_tu) {
                std::error_code ec;
                auto out = std::make_unique< llvm::raw_fd_ostream >(
                    opts_.output_file + ".c", ec, llvm::sys::fs::OF_Text
                );
                if (ec) {
                    LOG(ERROR) << "Failed to open output file '" << opts_.output_file
                               << ".c': " << ec.message() << "\n";
                    return;
                }
                ctx.getTranslationUnitDecl()->print(
                    *llvm::dyn_cast< llvm::raw_ostream >(out), ctx.getPrintingPolicy(), 0
                );
            }
        }

      private:
        patchestry::Options opts_;
    };

} // namespace

int main(int argc, char **argv) {
    auto options = parseCommandLineOptions(argc, argv);

    clang::CompilerInstance ci;
    clang::CompilerInvocation &invocation = ci.getInvocation();
    invocation.getTargetOpts().Triple     = llvm::sys::getDefaultTargetTriple();

    ci.createDiagnostics(*llvm::vfs::getRealFileSystem());
    ci.getDiagnostics().setClient(new patchestry::DiagnosticClient());
    if (!ci.hasDiagnostics()) {
        LOG(ERROR) << "Failed to initialize diagnostics.\n";
        return EXIT_FAILURE;
    }

    // Suppress warnings expected from decompiled 32-bit firmware code parsed
    // on a 64-bit host (pointer/int size mismatch, char signedness).
    auto &diags = ci.getDiagnostics();
    diags.setSeverityForGroup(clang::diag::Flavor::WarningOrError,
        "int-to-pointer-cast", clang::diag::Severity::Ignored);
    diags.setSeverityForGroup(clang::diag::Flavor::WarningOrError,
        "pointer-to-int-cast", clang::diag::Severity::Ignored);
    diags.setSeverityForGroup(clang::diag::Flavor::WarningOrError,
        "pointer-sign", clang::diag::Severity::Ignored);

    // Accept C11 with GNU extensions; GNUKeywords allows __attribute__ etc.
    ci.getLangOpts().C11         = true;
    ci.getLangOpts().GNUMode     = true;
    ci.getLangOpts().GNUKeywords = true;

    ci.getFrontendOpts().ProgramAction = clang::frontend::ParseSyntaxOnly;

    clang::FrontendInputFile input_fif(
        options.input_file, clang::InputKind(clang::Language::C)
    );
    ci.getFrontendOpts().Inputs.emplace_back(input_fif);

    ci.createFileManager(llvm::vfs::getRealFileSystem());
    ci.createSourceManager(ci.getFileManager());

    std::shared_ptr< clang::TargetOptions > target_opts =
        std::make_shared< clang::TargetOptions >();
    target_opts->Triple = llvm::sys::getDefaultTargetTriple();
    ci.setTarget(clang::TargetInfo::CreateTargetInfo(ci.getDiagnostics(), target_opts));

    ci.createPreprocessor(clang::TU_Complete);
    ci.createASTContext();
    ci.setASTConsumer(std::make_unique< CNormalizeConsumer >(options));
    ci.createSema(clang::TU_Complete, nullptr);

    if (!ci.InitializeSourceManager(input_fif)) {
        LOG(ERROR) << "Failed to open input file '" << options.input_file << "'\n";
        return EXIT_FAILURE;
    }

    clang::ParseAST(ci.getSema());

    return EXIT_SUCCESS;
}
