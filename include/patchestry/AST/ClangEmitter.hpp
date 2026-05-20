/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/SNode.hpp>

#include <string_view>

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

    // Convert a function-body SNode sequence to a Clang CompoundStmt and
    // set it as the function body.
    void EmitClangAST(const std::vector< SNode * > &root_children,
                      clang::FunctionDecl *fn, clang::ASTContext &ctx);

    // Post-emission Clang-AST cleanup driver.  Runs the F1-F8 pass
    // pipeline (terminal-label inlining, goto-forwarder folds,
    // goto-to-next-label elimination, dead-control removal, if-goto
    // scopeification, cross-scope label hoist, loop recovery, switch-
    // local case folds) plus cosmetic normalizers (while->for promotion,
    // condition normalization, label-into-compound pushes).  Only call
    // for patchir-decomp path.  When `report_cleanup` is set, emits a
    // CLANG_CLEANUP_SUMMARY diagnostic tagged with `function_name`.
    void CleanupPrettyPrint(
        clang::FunctionDecl *fn, clang::ASTContext &ctx,
        bool report_cleanup = false, std::string_view function_name = {});

} // namespace patchestry::ast
