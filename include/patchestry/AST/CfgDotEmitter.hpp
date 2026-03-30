/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/CGraph.hpp>
#include <patchestry/AST/SNode.hpp>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include <string>
#include <unordered_set>

namespace patchestry::ast {

    namespace detail {
        using patchestry::ast::CNode;
        using patchestry::ast::CGraph;
    } // namespace detail

    /// Emit a CGraph snapshot as a Graphviz DOT digraph.
    void EmitCGraphDot(const detail::CGraph &g, llvm::raw_ostream &os);

    /// Step-tracking emitter for CGraph fold tracing.
    struct CGraphDotTracer {
        std::string fn_name;
        unsigned step   = 0;
        bool enabled    = false;
        bool audit      = false;
        size_t original_stmt_count = 0;

        /// Baseline stmt pointers for pointer-level diff tracking.
        std::unordered_set<const clang::Stmt *> baseline_stmts;

        /// Dump current CGraph state to a numbered DOT file.
        void Dump(const detail::CGraph &g, llvm::StringRef rule_name,
                  bool is_phase_boundary = false);

        /// Audit statement counts after a fold step.
        void AuditAfterFold(const detail::CGraph &g, llvm::StringRef rule_name);
    };

    /// Count all clang::Stmt* reachable in an SNode tree (output).
    size_t CountSNodeStmts(const SNode *root);

    /// Count all stmts across active CGraph nodes (raw stmts + structured).
    size_t CountCGraphStmts(const detail::CGraph &g);

    /// Collect all clang::Stmt* pointers reachable from an SNode tree.
    void CollectSNodeStmtPtrs(const SNode *root,
                               std::unordered_set<const clang::Stmt *> &out);

    /// Collect all clang::Stmt* from active CGraph nodes (raw + structured).
    void CollectCGraphStmtPtrs(const detail::CGraph &g,
                                std::unordered_set<const clang::Stmt *> &out);

    /// Report which stmts from baseline are missing in current.
    size_t ReportMissingStmts(const std::unordered_set<const clang::Stmt *> &baseline,
                               const std::unordered_set<const clang::Stmt *> &current,
                               llvm::StringRef context_label);

    /// Emit an SNode tree as DOT.
    void EmitDot(const SNode *root, llvm::raw_ostream &os);

} // namespace patchestry::ast
