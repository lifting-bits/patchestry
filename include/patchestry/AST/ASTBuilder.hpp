/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>

#include <patchestry/Ghidra/JsonDeserialize.hpp>

using namespace patchestry::ghidra;

namespace patchestry::ast {

    class ast_builder
    {
      public:
        explicit ast_builder(clang::ASTContext &context);

        // Build AST from pCode instructions
        clang::FunctionDecl *build_ast(const program &pcode_program);

      private:
        clang::ASTContext &m_context;
        clang::FunctionDecl *create_function();
        clang::VarDecl *create_variable(const std::string &name);
    };

} // namespace patchestry::ast
