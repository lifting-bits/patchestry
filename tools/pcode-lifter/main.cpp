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
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <patchestry/AST/ASTConsumer.hpp>
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

    auto program = patchestry::ghidra::JsonParser().deserialize_program(*json->getAsObject());
    if (!program.has_value()) {
        llvm::errs() << "Failed to process json object" << json.takeError();
        return EXIT_FAILURE;
    }

    clang::CompilerInstance ci;
    ci.createDiagnostics();
    if (!ci.hasDiagnostics()) {
        llvm::errs() << "Failed to initialize diagnostics.\n";
        return EXIT_FAILURE;
    }

    clang::CompilerInvocation &invocation        = ci.getInvocation();
    clang::TargetOptions &invocation_target_opts = invocation.getTargetOpts();

    invocation_target_opts.Triple = llvm::sys::getDefaultTargetTriple();
    llvm::outs() << "Target triple: " << invocation_target_opts.Triple << "\n";

    std::shared_ptr< clang::TargetOptions > target_options =
        std::make_shared< clang::TargetOptions >();
    target_options->Triple = llvm::sys::getDefaultTargetTriple();
    ci.setTarget(clang::TargetInfo::CreateTargetInfo(ci.getDiagnostics(), target_options));

    ci.getPreprocessorOpts().addRemappedFile(
        "dummy.cpp", llvm::MemoryBuffer::getMemBuffer("").release()
    );
    ci.getFrontendOpts().Inputs.push_back(
        clang::FrontendInputFile("dummy.cpp", clang::Language::C)
    );

    ci.getFrontendOpts().ProgramAction = clang::frontend::ParseSyntaxOnly;
    // Setup file manager and source manager
    ci.createFileManager();
    ci.createSourceManager(ci.getFileManager());

    // Create the preprocessor and AST context
    ci.createPreprocessor(clang::TU_Complete);
    ci.createASTContext();

    auto &ast_context = ci.getASTContext();

    std::unique_ptr< patchestry::ast::PcodeASTConsumer > consumer =
        std::make_unique< patchestry::ast::PcodeASTConsumer >(ast_context, program.value());
    ci.setASTConsumer(std::move(consumer));

    auto &ast_consumer = ci.getASTConsumer();
    ast_consumer.HandleTranslationUnit(ast_context);

    return EXIT_SUCCESS;
}
