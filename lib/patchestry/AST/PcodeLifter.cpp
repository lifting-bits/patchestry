/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/PcodeLifter.hpp>

#include <memory>
#include <string>

#include <clang/AST/ASTConsumer.h>
#include <clang/Frontend/CompilerInstance.h>

#include <patchestry/AST/ASTConsumer.hpp>
#include <patchestry/Frontend/ClangFrontend.hpp>
#include <patchestry/Ghidra/Target.hpp>

namespace patchestry::ast {

    std::unique_ptr< TranslationUnit >
    LiftProgram(ghidra::Program &program, const LiftOptions &options) {
        const std::string arch = program.arch.value_or("");
        const std::string lang = program.lang.value_or("");
        auto ci                = frontend::createSyntheticCompilerInstance(
            frontend::FrontendConfig{
                ghidra::targetTriple(lang, ghidra::VariantPolicy::Ignore, arch),
                frontend::CompilationPolicy::LiftedCode },
            [&](clang::CompilerInstance &instance) -> std::unique_ptr< clang::ASTConsumer > {
                return std::make_unique< PcodeASTConsumer >(instance, program, options);
            }
        );
        if (!ci) { return nullptr; }

        ci->getASTConsumer().HandleTranslationUnit(ci->getASTContext());
        return std::make_unique< TranslationUnit >(TranslationUnit{ std::move(ci) });
    }

} // namespace patchestry::ast
