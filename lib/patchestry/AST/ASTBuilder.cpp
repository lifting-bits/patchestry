/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Ghidra/JsonDeserialize.hpp"
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/Basic/SourceLocation.h>

#include <patchestry/AST/ASTBuilder.hpp>

namespace patchestry::ast {

    ast_builder::ast_builder(clang::ASTContext &ctx) : m_context(ctx) {}

    clang::FunctionDecl *ast_builder::build_ast(const program & /*prog*/) {
        clang::FunctionDecl *funcDecl = nullptr;
        return funcDecl;
    }

} // namespace patchestry::ast
