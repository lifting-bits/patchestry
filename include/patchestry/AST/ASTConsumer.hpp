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
#include <clang/Frontend/FrontendAction.h>
#include <clang/Sema/Sema.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/AST/TypeBuilder.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Options.hpp>

namespace patchestry::ast {
    using namespace patchestry::ghidra;

    using ASTTypeMap = std::unordered_map< std::string, clang::QualType >;
    using ASTDeclMap = std::unordered_map< std::string, clang::Decl * >;

    class PcodeASTConsumer : public clang::ASTConsumer
    {
      public:
        explicit PcodeASTConsumer(
            clang::CompilerInstance &ci, Program &prog, patchestry::Options &opts
        )
            : program(prog), ci(ci), options(opts), type_builder(nullptr) {}

        void HandleTranslationUnit(clang::ASTContext &ctx) override;

      private:
        void set_sema_context(clang::DeclContext *dc);

        void write_to_file(void);

        void create_globals(clang::ASTContext &ctx, VariableMap &serialized_variables);

        void create_functions(
            clang::ASTContext &ctx, FunctionMap &serialized_functions, TypeMap &serialized_types
        );

        Program &get_program(void) const { return program.get(); }

        clang::Sema &sema(void) const { return ci.get().getSema(); }

        std::reference_wrapper< Program > program;
        std::reference_wrapper< clang::CompilerInstance > ci;

        const patchestry::Options &options; // NOLINT
        std::unique_ptr< TypeBuilder > type_builder;

        std::unordered_map< std::string, clang::FunctionDecl * > function_declarations;
        std::unordered_map< std::string, clang::VarDecl * > global_variable_declarations;
    };

} // namespace patchestry::ast
