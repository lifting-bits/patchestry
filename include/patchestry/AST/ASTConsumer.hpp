/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Sema/Sema.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/AST/LiftOptions.hpp>
#include <patchestry/AST/TUPrinter.hpp>
#include <patchestry/AST/TypeBuilder.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>

namespace patchestry::ast {
    using namespace patchestry::ghidra;

    using ASTTypeMap = std::unordered_map< std::string, clang::QualType >;
    using ASTDeclMap = std::unordered_map< std::string, clang::Decl * >;

    /// Fills the CompilerInstance's translation unit from a P-Code program:
    /// types, globals, then one function at a time through CGraph, SNode and
    /// Clang AST emission.  Builds the AST only; printing and lowering are the
    /// caller's stages.
    class PcodeASTConsumer : public clang::ASTConsumer
    {
      public:
        explicit PcodeASTConsumer(clang::CompilerInstance &ci, Program &prog, LiftOptions opts)
            : program(prog), ci(ci), options(opts), type_builder(nullptr) {}

        void HandleTranslationUnit(clang::ASTContext &ctx) override;

      private:
        void create_globals(clang::ASTContext &ctx, VariableMap &serialized_variables);

        Program &get_program(void) const { return program; }

      public:
        /// Emitted definition -> lifted function, for `-print-tu` markers.
        const DefinitionMap &get_definitions(void) const { return definitions; }

      private:
        Program &program;
        clang::CompilerInstance &ci;

        LiftOptions options;
        std::unique_ptr< TypeBuilder > type_builder;

        std::unordered_map< std::string, clang::FunctionDecl * > function_declarations;
        std::unordered_map< std::string, clang::VarDecl * > global_variable_declarations;
        std::unordered_map< std::string, clang::FunctionDecl * > intrinsic_declarations;

        // Emitted definition -> lifted function, for `-print-tu` markers
        // and comments.
        DefinitionMap definitions;
    };

} // namespace patchestry::ast
