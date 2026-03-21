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

        /// Dump current CGraph state to a numbered DOT file.
        void Dump(const detail::CGraph &g, llvm::StringRef rule_name);

        /// Audit statement counts after a fold step.
        /// Counts stmts in all uncollapsed CNodes (raw + structured).
        /// Logs a warning if the count drops below original_stmt_count.
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

} // namespace patchestry::ast
