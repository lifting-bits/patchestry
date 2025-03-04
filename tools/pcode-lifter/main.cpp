/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Util/Diagnostic.hpp"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <llvm/TargetParser/Triple.h>
#include <memory>
#include <string_view>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Stmt.h>
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
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Util/Log.hpp>
#include <patchestry/Util/Options.hpp>

namespace {

    const llvm::cl::opt< bool > emit_mlir( // NOLINT(cert-err58-cpp)
        "emit-mlir", llvm::cl::desc("Emit High-level MLIR representation"),
        llvm::cl::init(false)
    );

    const llvm::cl::opt< bool > emit_cir( // NOLINT(cert-err58-cpp)
        "emit-cir", llvm::cl::desc("Emit CIR MLIR representation"), llvm::cl::init(false)
    );

    const llvm::cl::opt< bool > emit_llvm( // NOLINT(cert-err58-cpp)
        "emit-llvm", llvm::cl::desc("Emit LLVM IR Representation"), llvm::cl::init(false)
    );

    const llvm::cl::opt< bool > emit_asm( // NOLINT(cert-err58-cpp)
        "emit-asm", llvm::cl::desc("Emit ASM Representation"), llvm::cl::init(false)
    );

    const llvm::cl::opt< bool > emit_obj( // NOLINT(cert-err58-cpp)
        "emit-obj", llvm::cl::desc("Emit Object file"), llvm::cl::init(false)
    );

    const llvm::cl::opt< std::string > input_filename( // NOLINT(cert-err58-cpp)
        "input", llvm::cl::desc("Input JSON file"), llvm::cl::Required
    );

    const llvm::cl::opt< std::string > output_filename( // NOLINT(cert-err58-cpp)
        "output", llvm::cl::desc("Specify output filename"), llvm::cl::value_desc("filename"),
        llvm::cl::init("") // Initialize with empty string
    );

    const llvm::cl::opt< bool > verbose( // NOLINT(cert-err58-cpp)
        "verbose", llvm::cl::desc("Enable debug logs"), llvm::cl::init(false)
    ); // NOLINT(cert-err58-cpp)

    const llvm::cl::opt< bool > print_tu( // NOLINT(cert-err58-cpp)
        "print-tu", llvm::cl::desc("Pretty print translation unit"), llvm::cl::init(false)
    );

    patchestry::Options parseCommandLineOptions(int argc, char **argv) {
        llvm::cl::ParseCommandLineOptions(
            argc, argv, "patche-lifter to represent high pcode into mlir representations\n"
        );

        return {
            .emit_cir    = emit_cir.getValue(),
            .emit_mlir   = emit_mlir.getValue(), // It is set to true by default
            .emit_llvm   = emit_llvm.getValue(),
            .emit_asm    = emit_asm.getValue(),
            .emit_obj    = emit_obj.getValue(),
            .verbose     = verbose.getValue(),
            .output_file = output_filename.getValue(),
            .input_file  = input_filename.getValue(),
            .print_tu    = print_tu.getValue(),
        };
    }

    void createSourceManager(clang::CompilerInstance &ci) {
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
            ci.getFileManager().getVirtualFileRef(
                file_name, static_cast< off_t >(data.size()), 0
            );
        clang::FileID file_id = sm.createFileID(
            *file_entry_ref_or_err, clang::SourceLocation(), clang::SrcMgr::C_User, 0
        );
        sm.setMainFileID(file_id);
    }

    void setCodegenOptions(clang::CompilerInstance &ci) {
        clang::CodeGenOptions &cg_opts = ci.getCodeGenOpts();
        cg_opts.OptimizationLevel      = 0;
        cg_opts.StrictReturn           = false;
        cg_opts.StrictEnums            = false;
    }

    std::string createTargetTriple(std::string &arch, const std::string &lang) {
        llvm::Triple target_triple;

        // Utility function to split the language identifier (lang) string
        auto split_language = [](const std::string &lang_id,
                                 char delim = ':') -> std::vector< std::string > {
            std::vector< std::string > tokens;
            std::stringstream ss(lang_id);
            std::string token;

            while (std::getline(ss, token, delim)) {
                tokens.push_back(token);
            }
            return tokens;
        };

        // Ghidra export lang id in the format - arch:endianess:size:variant
        auto lang_vec = split_language(lang);
        if (lang_vec.size() < 3) {
            LOG(ERROR
            ) << "Error: Invalid language format. Expected 'arch:endianess:size:variant'.\n";
            return "";
        }

        int bit_size = std::stoi(lang_vec[2]);
        auto is_le   = (lang_vec[1] == "LE");

        auto is_equal = [&](std::string astr, std::string bstr) -> bool {
            // transform both the string to lower-case and compare
            std::transform(astr.begin(), astr.end(), astr.begin(), [](unsigned char c) {
                return std::toupper(c);
            });
            std::transform(bstr.begin(), bstr.end(), bstr.begin(), [](unsigned char c) {
                return std::toupper(c);
            });
            return astr == bstr;
        };

        if (is_equal(arch, "x86") || is_equal(arch, "x86-64")) {
            target_triple.setArch(bit_size == 32U ? llvm::Triple::x86 : llvm::Triple::x86_64);
        } else if (is_equal(arch, "ARM") || is_equal(arch, "AARCH64")) {
            target_triple.setArch(
                bit_size == 32U ? (is_le ? llvm::Triple::arm : llvm::Triple::armeb)
                                : (is_le ? llvm::Triple::aarch64 : llvm::Triple::aarch64_be)
            );
        }

        else if (is_equal(arch, "MIPS"))
        {
            target_triple.setArch(
                bit_size == 32U ? (is_le ? llvm::Triple::mipsel : llvm::Triple::mips)
                                : (is_le ? llvm::Triple::mips64el : llvm::Triple::mips64)
            );
        } else if (is_equal(arch, "POWERPC")) {
            target_triple.setArch(
                bit_size == 32U ? (is_le ? llvm::Triple::ppcle : llvm::Triple::ppc)
                                : (is_le ? llvm::Triple::ppc64le : llvm::Triple::ppc64)
            );
        } else {
            target_triple.setArch(llvm::Triple::UnknownArch);
        }

        target_triple.setVendor(llvm::Triple::UnknownVendor);
        target_triple.setOS(llvm::Triple::Linux);

        // Set environment (for specific cases)
        if (is_equal(arch, "ARM") && bit_size == 32) {
            target_triple.setEnvironment(llvm::Triple::GNUEABIHF); // Hard float ABI
        }

        return target_triple.str();
    }

} // namespace

int main(int argc, char **argv) {
    auto options = parseCommandLineOptions(argc, argv);

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
    clang::CompilerInvocation &invocation = ci.getInvocation();
    clang::TargetOptions &inv_target_opts = invocation.getTargetOpts();
    inv_target_opts.Triple                = llvm::sys::getDefaultTargetTriple();

    ci.createDiagnostics(*llvm::vfs::getRealFileSystem());
    ci.getDiagnostics().setClient(new patchestry::DiagnosticClient());
    if (!ci.hasDiagnostics()) {
        LOG(ERROR) << "Failed to initialize diagnostics.\n";
        return EXIT_FAILURE;
    }
    createSourceManager(ci);

    std::shared_ptr< clang::TargetOptions > target_options =
        std::make_shared< clang::TargetOptions >();
    target_options->Triple = createTargetTriple(*program->arch, *program->lang);
    ci.setTarget(clang::TargetInfo::CreateTargetInfo(ci.getDiagnostics(), target_options));

    ci.getFrontendOpts().ProgramAction = clang::frontend::ParseSyntaxOnly;
    ci.getLangOpts().C99               = true;

    setCodegenOptions(ci);

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
        auto codegen = std::make_unique< patchestry::codegen::CodeGenerator >(ci);
        codegen->emit_cir(ast_context, options);
    }

    return EXIT_SUCCESS;
}
