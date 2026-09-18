/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/CSourceUnit.hpp>

#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/Basic/Builtins.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Parse/ParseAST.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/MemoryBuffer.h>

#include <patchestry/AST/PcodeValidator.hpp>
#include <patchestry/Frontend/ClangFrontend.hpp>
#include <patchestry/Ghidra/Target.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    namespace {

        struct TUHeader
        {
            std::string target;
            std::string arch;
        };

        // Parse the `key=value` fields of a marker line.
        void parseMarkerFields(
            llvm::StringRef rest,
            llvm::function_ref< void(llvm::StringRef, llvm::StringRef) > on_field
        ) {
            llvm::SmallVector< llvm::StringRef, 4 > tokens;
            rest.split(tokens, ' ', -1, /*KeepEmpty=*/false);
            for (auto token : tokens) {
                auto [key, value] = token.split('=');
                on_field(key, value);
            }
        }

        // Scan the file for the `patchestry:tu` header and the
        // `patchestry:function-begin` markers.
        void scanMarkers(
            llvm::StringRef text, TUHeader &header, std::vector< MarkerEntry > &markers
        ) {
            llvm::SmallVector< llvm::StringRef, 64 > lines;
            text.split(lines, '\n');
            for (auto line : lines) {
                line = line.trim();
                if (line.consume_front("// patchestry:tu ")) {
                    parseMarkerFields(line, [&](llvm::StringRef key, llvm::StringRef value) {
                        if (key == "target") {
                            header.target = value.str();
                        } else if (key == "arch") {
                            header.arch = value.str();
                        }
                    });
                } else if (line.consume_front("// patchestry:function-begin ")) {
                    MarkerEntry entry;
                    auto [key, rest] = line.split(' ');
                    entry.key        = key.str();
                    parseMarkerFields(rest, [&](llvm::StringRef field, llvm::StringRef value) {
                        if (field == "name") {
                            entry.name = value.str();
                        } else if (field == "symbol") {
                            entry.symbol = value.str();
                        }
                    });
                    if (entry.symbol.empty()) { entry.symbol = entry.name; }
                    if (!entry.name.empty()) { markers.push_back(std::move(entry)); }
                }
            }
        }

        // The lifter declares target intrinsics (`__wfi`, `__dmb`) as plain
        // functions.  Re-parsed with builtins live they resolve to clang
        // builtins that CIRGen cannot lower; drop that identity again.
        // Library builtins (ceilf, memcpy) keep it, as in an -input run.
        void dropIntrinsicBuiltinIdentity(clang::ASTContext &ctx) {
            for (auto *decl : ctx.getTranslationUnitDecl()->decls()) {
                auto *fn = llvm::dyn_cast< clang::FunctionDecl >(decl);
                if (fn == nullptr || fn->isImplicit() || fn->getIdentifier() == nullptr) {
                    continue;
                }
                // Sema records the identity as a BuiltinAttr on every declaration.
                const unsigned id = fn->getBuiltinID();
                if (id == 0 || ctx.BuiltinInfo.isPredefinedLibFunction(id)
                    || fn->getName().starts_with("__builtin_"))
                {
                    continue;
                }
                for (auto *redecl : fn->redecls()) { redecl->dropAttr< clang::BuiltinAttr >(); }
                fn->getIdentifier()->clearBuiltinID();
            }
        }

    } // namespace

    std::unique_ptr< TranslationUnit >
    ParseCTranslationUnit(const CSourceOptions &options, const ghidra::Program *program) {
        auto file_or_err = llvm::MemoryBuffer::getFile(options.path);
        if (std::error_code error_code = file_or_err.getError()) {
            LOG(ERROR) << "from-c: cannot read " << options.path << ": " << error_code.message()
                       << "\n";
            return nullptr;
        }
        TUHeader header;
        std::vector< MarkerEntry > markers;
        scanMarkers(file_or_err.get()->getBuffer(), header, markers);

        // The target comes from the override, else the `patchestry:tu`
        // header, else the program the C was lifted from.
        std::string lang = options.target_lang;
        if (lang.empty()) { lang = header.target; }
        if (lang.empty() && program != nullptr && program->lang) { lang = *program->lang; }
        if (lang.empty()) {
            LOG(ERROR) << "from-c: no target language: pass -target-lang, keep the "
                          "patchestry:tu header, or pass -input\n";
            return nullptr;
        }
        std::string arch = header.arch;
        if (arch.empty() && program != nullptr && program->arch) { arch = *program->arch; }
        if (arch.empty()) { arch = lang.substr(0, lang.find(':')); }
        auto triple = ghidra::targetTriple(lang, ghidra::VariantPolicy::Ignore, arch);
        if (triple.empty()) { return nullptr; }

        auto ci = frontend::createCompilerInstance(
            options.path,
            frontend::FrontendConfig{ triple, frontend::CompilationPolicy::ReenteredCode }
        );
        if (!ci) { return nullptr; }

        clang::ParseAST(ci->getSema());
        auto &diags = ci->getDiagnostics();
        if (diags.hasErrorOccurred()) {
            LOG(ERROR) << "from-c: " << diags.getNumErrors() << " error(s) parsing "
                       << options.path << "\n";
            return nullptr;
        }
        auto &ctx = ci->getASTContext();
        dropIntrinsicBuiltinIdentity(ctx);

        auto unit     = std::make_unique< TranslationUnit >();
        unit->markers = std::move(markers);
        unit->ci      = std::move(ci);
        return unit;
    }

} // namespace patchestry::ast
