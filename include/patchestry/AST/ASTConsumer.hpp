/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include "patchestry/Ghidra/PcodeOperations.hpp"
#include <functional>
#include <unordered_map>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>

#include <patchestry/Ghidra/JsonDeserialize.hpp>

namespace patchestry::ast {
    using namespace patchestry::ghidra;

    using ASTTypeMap = std::unordered_map< std::string, clang::QualType >;
    using ASTDeclMap = std::unordered_map< std::string, clang::Decl * >;

    class PcodeASTConsumer : public clang::ASTConsumer
    {
      public:
        explicit PcodeASTConsumer(clang::ASTContext &ctx, Program &prog)
            : program(prog), context(ctx), serialized_types_clang({}) {}

        void HandleTranslationUnit(clang::ASTContext &ctx) override;

      private:
        void create_types(clang::ASTContext &ctx, TypeMap &type_map);

        void create_functions(
            clang::ASTContext &ctx, FunctionMap &serialized_functions, TypeMap &serialized_types
        );

        clang::QualType
        create_function_prototype(clang::ASTContext &ctx, FunctionPrototype &proto);

        void create_function_parameters(
            clang::ASTContext &ctx, clang::FunctionDecl *func_decl, const BasicBlock &entry
        );

        clang::VarDecl *create_variable_decl(
            clang::ASTContext &ctx, clang::DeclContext &dc, const std::string &var_name,
            clang::QualType var_type
        );

        clang::BinaryOperator *
        create_assignment_stmt(clang::ASTContext &ctx, clang::Expr *lhs, clang::Expr *rhs);

        clang::DeclStmt *create_decl_stmt(clang::ASTContext &ctx, clang::Decl *decl);

        clang::QualType
        create_type(clang::ASTContext &ctx, const std::shared_ptr< VarnodeType > &vnode_type);

        clang::QualType
        create_typedef_type(clang::ASTContext &ctx, const TypedefType &typedef_type);

        clang::QualType
        create_pointer_type(clang::ASTContext &ctx, const PointerType &pointer_type);

        clang::QualType create_array_type(clang::ASTContext &ctx, const ArrayType &array_type);

        clang::QualType
        create_composite_type(clang::ASTContext &ctx, const VarnodeType &composite_type);

        void create_record_definition(
            clang::ASTContext &ctx, const CompositeType &varnode, clang::Decl *prev_decl,
            const ASTTypeMap &clang_types
        );

        clang::QualType create_enum_type(clang::ASTContext &ctx, const EnumType &enum_type);

        Program &get_program(void) const { return program.get(); }

        clang::ASTContext &get_context(void) const { return context.get(); }

        std::reference_wrapper< Program > program;
        std::reference_wrapper< clang::ASTContext > context;
        ASTTypeMap serialized_types_clang;
        std::unordered_map< std::string, clang::Decl * > incomplete_definition;
    };

} // namespace patchestry::ast
