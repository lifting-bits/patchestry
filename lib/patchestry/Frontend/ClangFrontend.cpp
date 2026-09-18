/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/Frontend/ClangFrontend.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <clang/Basic/CodeGenOptions.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Lex/Preprocessor.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/TargetParser/Triple.h>

#include <patchestry/Util/Diagnostic.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::frontend {

    namespace {

        // Create the target from the invocation-owned TargetOptions.  TargetInfo
        // keeps a pointer to the options it was created from, so they must
        // outlive the CompilerInstance; a function-local object would dangle.
        bool setTarget(clang::CompilerInstance &ci, const std::string &triple) {
            ci.getTargetOpts().Triple = triple;
            ci.setTarget(
                clang::TargetInfo::CreateTargetInfo(ci.getDiagnostics(), ci.getTargetOpts())
            );
            if (!ci.hasTarget()) {
                LOG(ERROR) << "Failed to create target for triple '" << triple << "'\n";
                return false;
            }
            return true;
        }

        // The decompiler's codegen options: no optimization and no strict
        // return/enum assumptions, so lowering does not drop lifted code.
        void setDecompCodegenOptions(clang::CompilerInstance &ci) {
            clang::CodeGenOptions &cg_opts = ci.getCodeGenOpts();
            cg_opts.OptimizationLevel      = 0;
            cg_opts.StrictReturn           = false;
            cg_opts.StrictEnums            = false;
        }

        // Source manager whose main file is a placeholder: the lifter builds the
        // translation unit programmatically and never parses it.
        void createPlaceholderSourceManager(clang::CompilerInstance &ci) {
            // Create file manager and setup source manager
            ci.createFileManager();
            ci.createSourceManager();

            // get source manager and setup main_file_id for the source manager
            auto &sm = ci.getSourceManager();

            // Create fake file to support real file system needed for vast
            // location translation
            std::string data      = "// temporary patchestry data";
            std::string file_name = "/tmp/patchestry.c";
            std::ofstream(file_name) << data;
            llvm::ErrorOr< clang::FileEntryRef > file_entry_ref_or_err =
                ci.getFileManager().getVirtualFileRef(
                    file_name, static_cast< off_t >(data.size()), 0
                );
            clang::FileID file_id = sm.createFileID(
                *file_entry_ref_or_err, clang::SourceLocation(), clang::SrcMgr::C_User, 0
            );
            sm.setMainFileID(file_id);

            ci.getFrontendOpts().ProgramAction = clang::frontend::ParseSyntaxOnly;
            ci.getFrontendOpts().Inputs.emplace_back(
                clang::FrontendInputFile(file_name, clang::InputKind(clang::Language::C))
            );
            ci.getLangOpts().C99 = true;
        }

    } // namespace

    std::string ghidraLangToTriple(const std::string &arch, const std::string &lang) {
        llvm::Triple target_triple;

        // Utility function to split the language identifier (lang) string
        auto split_language = [](const std::string &lang_id,
                                 char delim = ':') -> std::vector< std::string > {
            std::vector< std::string > tokens;
            std::stringstream ss(lang_id);
            std::string token;

            while (std::getline(ss, token, delim)) { tokens.push_back(token); }
            return tokens;
        };

        // Ghidra export lang id in the format - arch:endianess:size:variant
        auto lang_vec = split_language(lang);
        if (lang_vec.size() < 3) {
            LOG(
                ERROR
            ) << "Error: Invalid language format. Expected 'arch:endianess:size:variant'.\n";
            return "";
        }

        int bit_size = 0;
        if (llvm::StringRef(lang_vec[2]).getAsInteger(10, bit_size)) {
            LOG(ERROR) << "Invalid bit size in language id: " << lang_vec[2] << "\n";
            return "";
        }
        auto is_le = (lang_vec[1] == "LE");

        auto is_equal = [&](std::string astr, std::string bstr) -> bool {
            // transform both the string to lower-case and compare
            std::ranges::transform(astr, astr.begin(), [](unsigned char c) {
                return static_cast< char >(std::toupper(c));
            });
            std::ranges::transform(bstr, bstr.begin(), [](unsigned char c) {
                return static_cast< char >(std::toupper(c));
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

    std::optional< std::string > getClangResourceDir() {
        if (const char *clang_resource = std::getenv("CLANG_RESOURCE_DIR")) {
            if (!std::string(clang_resource).empty()) { return std::string(clang_resource); }
        }

        std::array< char, 1024 > buffer{};
        auto close_pipe = [](FILE *pipe) {
            if (pipe != nullptr) { pclose(pipe); }
        };
        std::unique_ptr< FILE, decltype(close_pipe) > pipe(
            popen("clang --print-resource-dir 2>/dev/null", "r"), close_pipe
        );

        if (!pipe) { return std::nullopt; }

        if (std::fgets(buffer.data(), static_cast< int >(buffer.size()), pipe.get()) == nullptr)
        {
            return std::nullopt;
        }

        std::string resource_dir = buffer.data();
        if (!resource_dir.empty() && resource_dir.back() == '\n') { resource_dir.pop_back(); }

        if (resource_dir.empty()) { return std::nullopt; }

        return resource_dir;
    }

    std::vector< std::string > getPatchestryIncludePaths() {
        std::vector< std::string > paths;

        if (auto resource_dir = getClangResourceDir()) {
            llvm::SmallString< 256 > resource_include(resource_dir.value());
            llvm::sys::path::append(resource_include, "include");
            if (llvm::sys::fs::exists(resource_include)) {
                paths.push_back(std::string(resource_include));
            }
        }

        // 2. Use CMake configured path
#ifdef PATCHESTRY_INCLUDE_DIR
        paths.push_back(PATCHESTRY_INCLUDE_DIR);
#endif

        // 3. Check environment variable
        if (const char *patchestry_root = std::getenv("PATCHESTRY_ROOT")) {
            paths.push_back(std::string(patchestry_root) + "/include");
        }

        // 4. Check CMAKE_INSTALL_PREFIX if available at runtime
        if (const char *install_prefix = std::getenv("CMAKE_INSTALL_PREFIX")) {
            paths.push_back(std::string(install_prefix) + "/include");
        }

        // 5. Note: We skip macOS SDK and system paths (/usr/include, etc.)
        // because they are platform-specific and cause conflicts when
        // cross-compiling for other platforms like ARM Linux.
        // The Clang resource directory (added above) provides the necessary
        // compiler built-in headers like stdarg.h.

        // 6. Try relative include paths for project-specific headers
        paths.push_back("../include");
        paths.push_back("../../include");
        paths.push_back("../../../include");
        paths.push_back("include");

        std::vector< std::string > valid_paths;
        for (const auto &path : paths) {
            llvm::SmallString< 256 > intrinsics_path(path);
            if (llvm::sys::fs::exists(intrinsics_path)) { valid_paths.push_back(path); }
        }

        return valid_paths;
    }

    std::unique_ptr< clang::CompilerInstance > createCompilerInstance(
        const std::string &filename, const FrontendConfig &config
    ) { // NOLINT
        auto ci = std::make_unique< clang::CompilerInstance >();

        // install custom diagnostic client. Under LLVM 22, DiagnosticsEngine
        // stores DiagnosticOptions by reference, so the options must outlive
        // the engine. Use the DiagnosticOptions already owned by the
        // Invocation (held there via shared_ptr) instead of leaking a raw new.
        auto diag_ids    = new clang::DiagnosticIDs();
        auto diagnostics = new clang::DiagnosticsEngine(
            llvm::IntrusiveRefCntPtr< clang::DiagnosticIDs >(diag_ids), ci->getDiagnosticOpts(),
            new patchestry::DiagnosticClient(), true
        );
        ci->setDiagnostics(diagnostics);
        if (!ci->hasDiagnostics()) {
            LOG(ERROR) << "Failed to initialize diagnostics.\n";
            return nullptr;
        }

        if (!setTarget(*ci, config.triple)) { return nullptr; }

        ci->createVirtualFileSystem();
        ci->createFileManager();
        ci->createSourceManager();
        auto buffer_or_error = llvm::MemoryBuffer::getFileOrSTDIN(filename);
        if (!buffer_or_error) {
            LOG(ERROR) << "Failed to open file: " << filename << "\n";
            return nullptr;
        }
        auto buffer = std::move(*buffer_or_error);

        llvm::ErrorOr< clang::FileEntryRef > file_entry_ref_or_err =
            ci->getFileManager().getVirtualFileRef(
                filename, static_cast< off_t >(buffer->getBufferSize()), 0
            );

        if (!file_entry_ref_or_err) {
            LOG(ERROR) << "Failed to create file entry ref: "
                       << file_entry_ref_or_err.getError().message() << "\n";
            return nullptr;
        }

        ci->getSourceManager().overrideFileContents(*file_entry_ref_or_err, std::move(buffer));

        clang::FileID file_id = ci->getSourceManager().createFileID(
            *file_entry_ref_or_err, clang::SourceLocation(), clang::SrcMgr::C_User
        );

        ci->getSourceManager().setMainFileID(file_id);

        ci->getFrontendOpts().ProgramAction = clang::frontend::ParseSyntaxOnly;
        ci->getLangOpts().C99               = true;
        ci->getLangOpts().GNUMode           = true; // Enable GNU extensions including builtins
        ci->getLangOpts().NoBuiltin = false; // Enable builtin functions (NoBuiltin=false)

        auto &header_search_opts = ci->getHeaderSearchOpts();

        // Set Clang's resource directory for built-in headers
        // Try environment variable first, then clang command
        auto resource_dir_opt    = getClangResourceDir();
        std::string resource_dir = resource_dir_opt.value_or("");

        if (!resource_dir.empty() && llvm::sys::fs::exists(resource_dir)) {
            header_search_opts.ResourceDir = resource_dir;
        }

        auto include_paths = getPatchestryIncludePaths();
        for (const auto &path : include_paths) {
            if (llvm::sys::fs::exists(path)) {
                header_search_opts.AddPath(path, clang::frontend::System, false, false);
            }
        }

        ci->createPreprocessor(clang::TU_Complete);
        auto &pp      = ci->getPreprocessor();
        auto &headers = pp.getHeaderSearchInfo();

        // Re-add the patchestry include path to the preprocessor's header search
        for (const auto &header_path : include_paths) {
            if (llvm::sys::fs::exists(header_path)) {
                auto dir_ref = pp.getFileManager().getOptionalDirectoryRef(header_path);
                if (dir_ref) {
                    headers.AddSearchPath(
                        clang::DirectoryLookup(*dir_ref, clang::SrcMgr::C_System, false),
                        true // isAngled
                    );
                }
            }
        }

        ci->createASTContext();
        ci->setASTConsumer(std::make_unique< clang::ASTConsumer >());
        ci->createSema(clang::TU_Complete, nullptr);
        return ci;
    }

    std::unique_ptr< clang::CompilerInstance > createSyntheticCompilerInstance(
        const FrontendConfig &config, ConsumerFactory make_consumer
    ) {
        auto ci = std::make_unique< clang::CompilerInstance >();

        ci->createVirtualFileSystem();
        ci->createDiagnostics(new patchestry::DiagnosticClient(), /*ShouldOwnClient=*/true);
        if (!ci->hasDiagnostics()) {
            LOG(ERROR) << "Failed to initialize diagnostics.\n";
            return nullptr;
        }

        createPlaceholderSourceManager(*ci);

        if (!setTarget(*ci, config.triple)) { return nullptr; }

        setDecompCodegenOptions(*ci);

        // Create the preprocessor and AST context, then the consumer and Sema:
        // Sema's constructor takes the consumer, so it must be installed first.
        ci->createPreprocessor(clang::TU_Complete);
        ci->createASTContext();
        ci->setASTConsumer(make_consumer(*ci));
        ci->createSema(clang::TU_Complete, nullptr);
        return ci;
    }

} // namespace patchestry::frontend
