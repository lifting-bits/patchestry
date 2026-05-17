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

#include <string>
#include <vector>

namespace patchestry::ast {

    namespace detail {
        // Check if a stmt ends with a control flow terminator
        // (goto/break/continue/return).  Recurses into CompoundStmt,
        // LabelStmt, and IfStmt (both arms must terminate).
        bool EndsWithTerminator(clang::Stmt *s);
    } // namespace detail

    // Convert an SNode tree back to a Clang CompoundStmt and set it as the
    // function body.
    void EmitClangAST(SNode *root, clang::FunctionDecl *fn, clang::ASTContext &ctx);

    // Layer C Stage 4 overload: take the function-body slot directly as a
    // std::vector<SNode*> instead of routing through an intermediate SSeq.
    // Materializes a single CompoundStmt from the vector and sets it as
    // the function body.
    void EmitClangAST(
        const std::vector< SNode * > &root_children, clang::FunctionDecl *fn,
        clang::ASTContext &ctx
    );

    // Post-emission cleanup for prettier C output.
    // Flattens nested CompoundStmts and pushes LabelStmts inside
    // CompoundStmt bodies. Only call for patchir-decomp path.
    void CleanupPrettyPrint(clang::FunctionDecl *fn, clang::ASTContext &ctx);

    struct ClangEmissionValidationReport
    {
        size_t emitted_labels         = 0;
        size_t emitted_gotos          = 0;
        size_t dangling_gotos         = 0;
        size_t duplicate_labels       = 0;
        size_t empty_labels           = 0;
        size_t unreachable_statements = 0;
        size_t switch_stmts           = 0;
        size_t case_stmts             = 0;
        size_t default_stmts          = 0;
        size_t break_stmts            = 0;
        size_t continue_stmts         = 0;
        std::vector< std::string > diagnostics;

        bool ok() const { return diagnostics.empty(); }
    };

    /// Verify the final Clang AST after SNode emission and pretty-print
    /// cleanup.  This catches cleanup-time label/goto damage before CIR
    /// lowering observes the function body.
    ClangEmissionValidationReport ValidateEmittedClangAST(clang::FunctionDecl *fn);

    struct StructuringImprovementReport
    {
        size_t residual_gotos            = 0;
        size_t emitted_labels            = 0;
        size_t dangling_gotos            = 0;
        size_t layout_fallthrough        = 0;
        size_t pass_through_collapse     = 0;
        size_t terminal_clone_inline     = 0;
        size_t single_ref_inline_reorder = 0;
        size_t small_target_clone        = 0;
        size_t terminal_epilogue         = 0;
        size_t cross_scope_gotos         = 0;
        size_t residual_hard             = 0;
    };

    /// Classify residual gotos in the final Clang AST after cleanup.  This is
    /// a report-only estimator for future structuring improvements; it must
    /// not mutate the AST.
    StructuringImprovementReport AnalyzeStructuringImprovements(clang::FunctionDecl *fn);

} // namespace patchestry::ast
