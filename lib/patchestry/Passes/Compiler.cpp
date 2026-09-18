/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <memory>
#include <string>

#include <clang/AST/ASTContext.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Parse/ParseAST.h>

#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Frontend/ClangFrontend.hpp>
#include <patchestry/Ghidra/Target.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::passes {

    std::optional< std::string >
    emitModuleAsString(const std::string &filename, const std::string &lang) { // NOLINT
        auto ci = patchestry::frontend::createCompilerInstance(
            filename,
            patchestry::frontend::FrontendConfig{
                ghidra::targetTriple(lang, ghidra::VariantPolicy::Preserve),
                frontend::CompilationPolicy::PatchCode }
        );
        if (!ci) { return {}; }
        clang::ParseAST(ci->getSema());
        // Don't gate on hasErrorOccurred — clang recovery may still emit a
        // usable patch symbol. Hard failures surface downstream via
        // ensurePatchFunctionAvailable -> signal_failure. #244
        patchestry::codegen::CodeGenerator codegen(ci->getASTContext(), ci->getCodeGenOpts());
        auto module = codegen.lower_ast_to_mlir();
        if (!module.has_value()) { return {}; }
        std::string module_string;
        llvm::raw_string_ostream os(module_string);
        auto flags = mlir::OpPrintingFlags();
        flags.enableDebugInfo(true, false);
        module->print(os, flags);
        return module_string;
    }

} // namespace patchestry::passes
