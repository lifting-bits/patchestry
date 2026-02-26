/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/CfgBuilder.hpp>
#include <patchestry/AST/CfgFoldStructure.hpp>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include <string>
#include <unordered_set>

namespace patchestry::ast {

    /// Emit a Cfg (CfgBlock-based) as a Graphviz DOT digraph.
    void EmitCfgDot(const Cfg &cfg, llvm::raw_ostream &os);

    /// Emit a CGraph snapshot as a Graphviz DOT digraph.
    void EmitCGraphDot(const detail::CGraph &g, llvm::raw_ostream &os);

    /// Step-tracking emitter for CGraph fold tracing.
    /// Creates numbered DOT files: <fn_name>.step_<NNN>_<rule>.dot
    struct CGraphDotTracer {
        std::string fn_name;
        unsigned step   = 0;
        bool enabled    = false;
        bool audit      = false;
        size_t original_stmt_count = 0;

        /// Baseline stmt pointers collected from the input CFG.
        /// Populated when audit=true; used for pointer-level diff tracking.
        std::unordered_set<const clang::Stmt *> baseline_stmts;

        /// Dump current CGraph state to a numbered DOT file.
        /// When verbose=false, only dumps if is_phase_boundary is true.
        void Dump(const detail::CGraph &g, llvm::StringRef rule_name,
                  bool is_phase_boundary = false);

        /// Audit statement counts after a fold step.
        void AuditAfterFold(const detail::CGraph &g, llvm::StringRef rule_name);
    };

    // -----------------------------------------------------------------------
    // Statement counting utilities for structuring verification
    // -----------------------------------------------------------------------

    /// Count all clang::Stmt* in a Cfg (input baseline).
    size_t CountCfgStmts(const Cfg &cfg);

    /// Count all clang::Stmt* reachable in an SNode tree (output).
    size_t CountSNodeStmts(const SNode *root);

    /// Count all stmts across active CGraph nodes (raw stmts + structured).
    size_t CountCGraphStmts(const detail::CGraph &g);

    // -----------------------------------------------------------------------
    // Stmt-pointer tracking for precise loss detection
    // -----------------------------------------------------------------------

    /// Collect all clang::Stmt* pointers from a Cfg into a set.
    void CollectCfgStmtPtrs(const Cfg &cfg,
                             std::unordered_set<const clang::Stmt *> &out);

    /// Collect all clang::Stmt* pointers reachable from an SNode tree.
    void CollectSNodeStmtPtrs(const SNode *root,
                               std::unordered_set<const clang::Stmt *> &out);

    /// Collect all clang::Stmt* from active CGraph nodes (raw + structured).
    void CollectCGraphStmtPtrs(const detail::CGraph &g,
                                std::unordered_set<const clang::Stmt *> &out);

    /// Report which specific stmts from `baseline` are missing in `current`.
    /// Logs the stmt class name and a one-line pretty-print for each missing stmt.
    /// Returns the number of missing stmts.
    size_t ReportMissingStmts(const std::unordered_set<const clang::Stmt *> &baseline,
                               const std::unordered_set<const clang::Stmt *> &current,
                               llvm::StringRef context_label);

} // namespace patchestry::ast
