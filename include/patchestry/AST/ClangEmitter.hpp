/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/SNode.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>

namespace patchestry::ast {

    namespace detail {
        // Check if a stmt ends with a control flow terminator
        // (goto/break/continue/return).  Recurses into CompoundStmt,
        // LabelStmt, and IfStmt (both arms must terminate).
        bool EndsWithTerminator(clang::Stmt *s);
    } // namespace detail

    // Convert an SNode tree back to a Clang CompoundStmt and set it as the
    // function body.
    void EmitClangAST(SNode *root, clang::FunctionDecl *fn,
                      clang::ASTContext &ctx);

    // Post-emission cleanup for prettier C output.
    // Flattens nested CompoundStmts and pushes LabelStmts inside
    // CompoundStmt bodies. Only call for patchir-decomp path.
    void CleanupPrettyPrint(clang::FunctionDecl *fn, clang::ASTContext &ctx);

} // namespace patchestry::ast
