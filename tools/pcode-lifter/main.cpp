/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <cstdlib>
#include <memory>

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <patchestry/AST/ASTBuilder.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>

const llvm::cl::opt< std::string > input_filename(
    llvm::cl::Positional, llvm::cl::desc("<input JSON file>"), llvm::cl::Required
);

const llvm::cl::opt< bool >
    verbose("v", llvm::cl::desc("Enable debug logs"), llvm::cl::init(false));

clang::ASTContext &create_ast_context(void) {
    clang::CompilerInstance compiler;

    compiler.createDiagnostics();
    if (!compiler.hasDiagnostics()) {
        llvm::errs() << "Failed to initialize diagnostics.\n";
    }

    std::shared_ptr< clang::TargetOptions > target_options =
        std::make_shared< clang::TargetOptions >();
    target_options->Triple = llvm::sys::getDefaultTargetTriple();
    compiler.setTarget(
        clang::TargetInfo::CreateTargetInfo(compiler.getDiagnostics(), target_options)
    );

    // Set up file manager and source manager
    compiler.createFileManager();
    compiler.createSourceManager(compiler.getFileManager());

    // Create the preprocessor and AST context
    compiler.createPreprocessor(clang::TU_Complete);
    compiler.createASTContext();

    return compiler.getASTContext();
}

int main(int argc, char **argv) {
    llvm::cl::ParseCommandLineOptions(
        argc, argv, "pcode-lifter to lift high pcode into clang ast\n"
    );

    if (verbose) {
        llvm::outs() << "Enable debug logs";
    }

    llvm::ErrorOr< std::unique_ptr< llvm::MemoryBuffer > > file_or_err =
        llvm::MemoryBuffer::getFile(input_filename);

    if (std::error_code error_code = file_or_err.getError()) {
        llvm::errs() << "Error reading json file : " << error_code.message() << "\n";
        return EXIT_FAILURE;
    }

    std::unique_ptr< llvm::MemoryBuffer > buffer = std::move(file_or_err.get());
    auto json                                    = llvm::json::parse(buffer->getBuffer());
    if (!json) {
        llvm::errs() << "Failed to parse pcode JSON: " << json.takeError();
        return EXIT_FAILURE;
    }

    auto program = patchestry::ghidra::json_parser().parse_program(*json->getAsObject());
    if (!program.has_value()) {
        llvm::errs() << "Failed to process json object" << json.takeError();
        return EXIT_FAILURE;
    }

    clang::ASTContext &context = create_ast_context();
    patchestry::ast::ast_builder ast_builder(context);
    ast_builder.build_ast(program.value());

    return EXIT_SUCCESS;
}
