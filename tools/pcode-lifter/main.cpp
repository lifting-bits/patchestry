/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <cstdlib>
#include <memory>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendOptions.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Util/Log.hpp>

const llvm::cl::opt< std::string > input_filename(
    llvm::cl::Positional, llvm::cl::desc("<input JSON file>"), llvm::cl::Required
);

const llvm::cl::opt< bool >
    verbose("v", llvm::cl::desc("Enable debug logs"), llvm::cl::init(false));

const llvm::cl::opt< bool > pprint(
    "pretty-print", llvm::cl::desc("Pretty print translation unit"), llvm::cl::init(false)
);

const llvm::cl::opt< std::string > output_filename(
    "output", llvm::cl::desc("Specify output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("/tmp/output.c")
);

bool debug_mode = false;

int main(int argc, char **argv) {
    llvm::cl::ParseCommandLineOptions(
        argc, argv, "pcode-lifter to lift high pcode into clang ast\n"
    );

    if (verbose) {
        debug_mode = true;
    }

    llvm::ErrorOr< std::unique_ptr< llvm::MemoryBuffer > > file_or_err =
        llvm::MemoryBuffer::getFile(input_filename);

    if (std::error_code error_code = file_or_err.getError()) {
        LOG(ERROR) << "Error reading json file : " << error_code.message() << "\n";
        return EXIT_FAILURE;
    }

    std::unique_ptr< llvm::MemoryBuffer > buffer = std::move(file_or_err.get());
    auto json                                    = llvm::json::parse(buffer->getBuffer());
    if (!json) {
        LOG(ERROR) << "Failed to parse pcode JSON: " << json.takeError();
        return EXIT_FAILURE;
    }

    auto program = patchestry::ghidra::JsonParser().deserialize_program(*json->getAsObject());
    if (!program.has_value()) {
        LOG(ERROR) << "Failed to process json object" << json.takeError();
        return EXIT_FAILURE;
    }

    clang::CompilerInstance ci;
    ci.createDiagnostics();
    if (!ci.hasDiagnostics()) {
        LOG(ERROR) << "Failed to initialize diagnostics.\n";
        return EXIT_FAILURE;
    }

    clang::CompilerInvocation &invocation = ci.getInvocation();
    clang::TargetOptions &inv_target_opts = invocation.getTargetOpts();
    inv_target_opts.Triple                = llvm::sys::getDefaultTargetTriple();

    std::shared_ptr< clang::TargetOptions > target_options =
        std::make_shared< clang::TargetOptions >();
    target_options->Triple = llvm::sys::getDefaultTargetTriple();
    ci.setTarget(clang::TargetInfo::CreateTargetInfo(ci.getDiagnostics(), target_options));

    ci.getFrontendOpts().ProgramAction = clang::frontend::ParseSyntaxOnly;
    ci.getLangOpts().C99               = true;
    // Setup file manager and source manager
    ci.createFileManager();
    ci.createSourceManager(ci.getFileManager());

    auto &sm              = ci.getSourceManager();
    std::string file_data = "/patchestry";
    llvm::ErrorOr< clang::FileEntryRef > file_entry_ref_or_err =
        ci.getFileManager().getVirtualFileRef("/tmp/patchestry", file_data.size(), 0);
    clang::FileID file_id = sm.createFileID(
        *file_entry_ref_or_err, clang::SourceLocation(), clang::SrcMgr::C_User, 0
    );

    sm.setMainFileID(file_id);

    // Create the preprocessor and AST context
    ci.createPreprocessor(clang::TU_Complete);
    ci.createASTContext();

    auto &ast_context = ci.getASTContext();

    std::string outfile = output_filename.getValue();
    std::unique_ptr< patchestry::ast::PcodeASTConsumer > consumer =
        std::make_unique< patchestry::ast::PcodeASTConsumer >(ci, program.value(), outfile);
    ci.setASTConsumer(std::move(consumer));
    ci.createSema(clang::TU_Complete, nullptr);

    auto &ast_consumer = ci.getASTConsumer();
    ast_consumer.HandleTranslationUnit(ast_context);

    return EXIT_SUCCESS;
}
