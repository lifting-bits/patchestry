/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Util/Options.hpp"
#include <cstdlib>
#include <fstream>
#include <memory>
#include <ranges>
#include <string_view>
#include <vector>

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
#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Util/Log.hpp>

/*************************/
// Command line options
/**************************/

namespace {

    const llvm::cl::opt< patchestry::EmitMLIRType > emit_mlir_type_option(
        "emit-mlir", llvm::cl::desc("MLIR Emission Type"),
        llvm::cl::values(
            clEnumVal(patchestry::EmitMLIRType::hl, "High-Level VAST MLIR Representation"),
            clEnumVal(patchestry::EmitMLIRType::cir, "ClangIR representation")
        ),
        llvm::cl::init(patchestry::EmitMLIRType::hl)
    );

    const llvm::cl::opt< bool > emit_tower(
        "emit-tower", llvm::cl::desc("Emit MLIR tower representation"), llvm::cl::init(false)
    );

    const llvm::cl::opt< bool > emit_llvm(
        "emit-llvm", llvm::cl::desc("Emit LLVM IR Representation"), llvm::cl::init(false)
    );

    const llvm::cl::opt< bool >
        emit_asm("emit-asm", llvm::cl::desc("Emit ASM Representation"), llvm::cl::init(false));

    const llvm::cl::opt< bool >
        emit_obj("emit-obj", llvm::cl::desc("Emit Object file"), llvm::cl::init(false));

    const llvm::cl::opt< std::string >
        input_filename("input", llvm::cl::desc("Input JSON file"), llvm::cl::Required);

    llvm::cl::opt< std::string > output_filename(
        "output", llvm::cl::desc("Specify output filename"), llvm::cl::value_desc("filename"),
        llvm::cl::init("") // Initialize with empty string
    );

    const llvm::cl::opt< bool >
        verbose("verbose", llvm::cl::desc("Enable debug logs"), llvm::cl::init(false));

    const llvm::cl::opt< bool > print_tu(
        "print-tu", llvm::cl::desc("Pretty print translation unit"), llvm::cl::init(false)
    );

    const llvm::cl::opt< std::string > pipelines(
        "pipelines", llvm::cl::desc("Specify pipelines for lowering steps"),
        llvm::cl::value_desc("string"), llvm::cl::init("")
    );

    patchestry::Options parse_command_line_options(int argc, char **argv) {
        llvm::cl::ParseCommandLineOptions(
            argc, argv, "patche-front to represent high pcode into mlir representations\n"
        );

        auto split_pipelines = [&](std::string_view pipelines,
                                   char delim = ',') -> std::vector< std::string > {
            std::vector< std::string > vec;
            for (auto part : pipelines | std::views::split(delim)) {
                vec.emplace_back(part.begin(), part.end());
            }
            return vec;
        };

        patchestry::Options opts;
        opts.emit_mlir   = true;
        opts.emit_tower  = emit_tower.getValue();
        opts.emit_llvm   = emit_llvm.getValue();
        opts.emit_asm    = emit_asm.getValue();
        opts.emit_obj    = emit_obj.getValue();
        opts.mlir_type   = emit_mlir_type_option.getValue();
        opts.output_file = output_filename.getValue();
        opts.input_file  = input_filename.getValue();
        opts.print_tu    = print_tu.getValue();
        opts.pipelines   = split_pipelines(pipelines.getValue());

        return opts;
    }

} // namespace

void create_source_manager(clang::CompilerInstance &ci) {
    // Create file manager and setup source manager
    ci.createFileManager();
    ci.createSourceManager(ci.getFileManager());

    // get source manager and setup main_file_id for the source manager
    auto &sm = ci.getSourceManager();

    // Create fake file to support real file system needed for vast
    // location translation
    std::string data      = "/patchestry";
    std::string file_name = "/tmp/patchestry";
    std::ofstream(file_name) << data;
    llvm::ErrorOr< clang::FileEntryRef > file_entry_ref_or_err =
        ci.getFileManager().getVirtualFileRef(file_name, data.size(), 0);
    clang::FileID file_id = sm.createFileID(
        *file_entry_ref_or_err, clang::SourceLocation(), clang::SrcMgr::C_User, 0
    );
    sm.setMainFileID(file_id);
}

void create_ast_context(clang::CompilerInstance &ci) {}

void set_codegen_options(clang::CompilerInstance &ci) {
    clang::CodeGenOptions &cg_opts = ci.getCodeGenOpts();

    cg_opts.OptimizationLevel = 0;
    cg_opts.StrictReturn      = false;
    cg_opts.StrictEnums       = false;
}

int main(int argc, char **argv) {
    auto options = parse_command_line_options(argc, argv);

    llvm::ErrorOr< std::unique_ptr< llvm::MemoryBuffer > > file_or_err =
        llvm::MemoryBuffer::getFile(options.input_file);

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

    create_source_manager(ci);
    set_codegen_options(ci);

    // Create the preprocessor and AST context
    ci.createPreprocessor(clang::TU_Complete);
    ci.createASTContext();

    auto &ast_context = ci.getASTContext();
    std::unique_ptr< patchestry::ast::PcodeASTConsumer > consumer =
        std::make_unique< patchestry::ast::PcodeASTConsumer >(ci, program.value(), options);
    ci.setASTConsumer(std::move(consumer));
    ci.createSema(clang::TU_Complete, nullptr);

    auto &ast_consumer = ci.getASTConsumer();
    ast_consumer.HandleTranslationUnit(ast_context);

    auto *pcode_consumer = dynamic_cast< patchestry::ast::PcodeASTConsumer * >(&ast_consumer);
    if (pcode_consumer != nullptr) {
        auto codegen          = std::make_unique< patchestry::codegen::CodeGenerator >(ci);
        const auto &locations = pcode_consumer->locations();
        if (options.emit_tower) {
            codegen->emit_tower(ast_context, locations, options);
        } else {
            codegen->emit_source_ir(ast_context, locations, options);
        }
    }

    return EXIT_SUCCESS;
}
