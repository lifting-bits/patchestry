/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <algorithm>
#include <memory>
#include <string>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Parse/ParseAST.h>

#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Util/Diagnostic.hpp>

namespace patchestry::passes {

    namespace {

        constexpr int kArch32BitSize = 32U; // NOLINT

        llvm::Triple::SubArchType getSubArch(const std::string &variant) {
            static const std::unordered_map< std::string, llvm::Triple::SubArchType >
                variantMap = {
                    // ARM 32-bit variants
                    {          "v4t",            llvm::Triple::ARMSubArch_v4t },
                    {           "v5",             llvm::Triple::ARMSubArch_v5 },
                    {         "v5te",           llvm::Triple::ARMSubArch_v5te },
                    {           "v6",             llvm::Triple::ARMSubArch_v6 },
                    {         "v6t2",           llvm::Triple::ARMSubArch_v6t2 },
                    {          "v6k",            llvm::Triple::ARMSubArch_v6k },
                    {          "v6m",            llvm::Triple::ARMSubArch_v6m },
                    {           "v7",             llvm::Triple::ARMSubArch_v7 },
                    {          "v7a",             llvm::Triple::ARMSubArch_v7 },
                    {          "v7r",             llvm::Triple::ARMSubArch_v7 },
                    {          "v7m",            llvm::Triple::ARMSubArch_v7m },
                    {         "v7em",           llvm::Triple::ARMSubArch_v7em },
                    {           "v8",             llvm::Triple::ARMSubArch_v8 },
                    {          "v8a",             llvm::Triple::ARMSubArch_v8 },
                    {          "v8r",            llvm::Triple::ARMSubArch_v8r },
                    {          "v8m",   llvm::Triple::ARMSubArch_v8m_baseline },
                    {        "v8.1m", llvm::Triple::ARMSubArch_v8_1m_mainline },

                    // Thumb variants
                    {    "v4t_thumb",            llvm::Triple::ARMSubArch_v4t },
                    {   "v5te_thumb",           llvm::Triple::ARMSubArch_v5te },
                    {     "v6_thumb",             llvm::Triple::ARMSubArch_v6 },
                    {   "v6t2_thumb",           llvm::Triple::ARMSubArch_v6t2 },
                    {     "v7_thumb",             llvm::Triple::ARMSubArch_v7 },
                    {    "v7m_thumb",            llvm::Triple::ARMSubArch_v7m },

                    // Cortex-M specific variants
                    {       "Cortex",            llvm::Triple::ARMSubArch_v7m },

                    // AArch64 variants (ARM 64-bit)
                    {          "v8A",     llvm::Triple::AArch64SubArch_arm64e }, // Generic v8A
                    { "AppleSilicon",     llvm::Triple::AArch64SubArch_arm64e },
            };
            auto it = variantMap.find(variant);
            if (it != variantMap.end()) {
                return it->second;
            }
            return llvm::Triple::NoSubArch;
        }

        std::string createTargetTriple(const std::string &lang) {
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
                LOG(ERROR) << "Error: Invalid language format. Expected "
                              "'arch:endianess:size:variant'.\n";
                return "";
            }

            const std::string &arch(lang_vec[0]);
            int bit_size = std::stoi(lang_vec[2]);
            auto is_le   = (lang_vec[1] == "LE");
            auto variant = lang_vec.size() > 3 ? lang_vec[3] : "";

            auto is_equal = [&](std::string astr, std::string bstr) -> bool {
                // transform both the string to lower-case and compare
                std::ranges::transform(astr, astr.begin(), [](unsigned char c) {
                    return std::toupper(c);
                });
                std::ranges::transform(bstr, bstr.begin(), [](unsigned char c) {
                    return std::toupper(c);
                });
                return astr == bstr;
            };

            if (is_equal(arch, "x86") || is_equal(arch, "x86-64")) {
                target_triple.setArch(
                    bit_size == kArch32BitSize ? llvm::Triple::x86 : llvm::Triple::x86_64
                );
            } else if (is_equal(arch, "ARM") || is_equal(arch, "AARCH64")) {
                target_triple.setArch(
                    bit_size == kArch32BitSize
                        ? (is_le ? llvm::Triple::arm : llvm::Triple::armeb) // NOLINT
                        : (is_le ? llvm::Triple::aarch64 : llvm::Triple::aarch64_be),
                    getSubArch(variant)
                );
            }

            else if (is_equal(arch, "MIPS"))
            {
                target_triple.setArch(
                    bit_size == kArch32BitSize
                        ? (is_le ? llvm::Triple::mipsel : llvm::Triple::mips) // NOLINT
                        : (is_le ? llvm::Triple::mips64el : llvm::Triple::mips64)
                );
            } else if (is_equal(arch, "POWERPC")) {
                target_triple.setArch(
                    bit_size == kArch32BitSize
                        ? (is_le ? llvm::Triple::ppcle : llvm::Triple::ppc) // NOLINT
                        : (is_le ? llvm::Triple::ppc64le : llvm::Triple::ppc64)
                );
            } else {
                target_triple.setArch(llvm::Triple::UnknownArch);
            }

            target_triple.setVendor(llvm::Triple::UnknownVendor);
            target_triple.setOS(llvm::Triple::Linux);

            // Set environment (for specific cases)
            if (is_equal(arch, "ARM") && bit_size == kArch32BitSize) {
                target_triple.setEnvironment(llvm::Triple::GNUEABIHF); // Hard float ABI
            }

            return target_triple.str();
        }

        std::unique_ptr< clang::CompilerInstance >
        createCompilerInstance(const std::string &filename, const std::string &lang) { // NOLINT
            auto ci                = std::make_unique< clang::CompilerInstance >();
            auto &invocation       = ci->getInvocation();
            auto &inv_target_opts  = invocation.getTargetOpts();
            inv_target_opts.Triple = llvm::sys::getDefaultTargetTriple();

            ci->createDiagnostics(*llvm::vfs::getRealFileSystem());
            ci->getDiagnostics().setClient(new patchestry::DiagnosticClient());
            if (!ci->hasDiagnostics()) {
                llvm::errs() << "Failed to initialize diagnostics.\n";
                return nullptr;
            }

            std::shared_ptr< clang::TargetOptions > target_options =
                std::make_shared< clang::TargetOptions >();
            target_options->Triple = createTargetTriple(lang);
            ci->setTarget(
                clang::TargetInfo::CreateTargetInfo(ci->getDiagnostics(), target_options)
            );

            ci->createFileManager();
            ci->createSourceManager(ci->getFileManager());
            auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(filename);
            if (!buffer) {
                llvm::errs() << "Failed to open file: " << filename << "\n";
                return nullptr;
            }
            llvm::ErrorOr< clang::FileEntryRef > file_entry_ref_or_err =
                ci->getFileManager().getVirtualFileRef(
                    filename, static_cast< off_t >(buffer->get()->getBufferSize()), 0
                );

            clang::FileID file_id = ci->getSourceManager().createFileID(
                *file_entry_ref_or_err, clang::SourceLocation(), clang::SrcMgr::C_User
            );
            ci->getSourceManager().setMainFileID(file_id);

            ci->getFrontendOpts().ProgramAction = clang::frontend::ParseSyntaxOnly;
            ci->getLangOpts().C99               = true;

            ci->createPreprocessor(clang::TU_Complete);
            ci->createASTContext();
            ci->setASTConsumer(std::make_unique< clang::ASTConsumer >());
            ci->createSema(clang::TU_Complete, nullptr);
            return ci;
        }

    } // namespace

    std::optional< std::string >
    emitModuleAsString(const std::string &filename, const std::string &lang) { // NOLINT
        auto ci = createCompilerInstance(filename, lang);
        if (!ci) {
            return {};
        }
        clang::ParseAST(ci->getSema());
        auto codegen = std::make_unique< patchestry::codegen::CodeGenerator >(*ci);
        auto module  = codegen->lower_ast_to_mlir(ci->getASTContext());
        if (!module.has_value()) {
            return {};
        }
        std::string module_string;
        llvm::raw_string_ostream os(module_string);
        auto flags = mlir::OpPrintingFlags();
        flags.enableDebugInfo(true, false);
        module->print(os, flags);
        return module_string;
    }

} // namespace patchestry::passes
