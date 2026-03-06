/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>

namespace patchestry::ghidra {
    struct Function;
} // namespace patchestry::ghidra

namespace patchestry::ast {

    /// A case entry for switch blocks -- carries Ghidra case metadata.
    struct SwitchCaseEntry {
        int64_t value;          // case constant value
        size_t succ_index;      // index into CfgBlock::succs[] for this case target
        bool has_exit = false;  // whether Ghidra marked this case as having a break/exit
    };

    // A basic block in the CFG
    struct CfgBlock {
        std::string label;                  // empty if unlabeled entry block
        std::vector< clang::Stmt * > stmts; // statements in the block
        std::vector< size_t > succs;        // successor block indices
        bool is_conditional = false;        // true if block ends with if(cond) goto
        clang::Expr *branch_cond = nullptr; // condition expr if conditional
        size_t taken_succ = 0;              // index of "then" successor
        size_t fallthrough_succ = 0;        // index of "else" / fallthrough successor
        std::vector< SwitchCaseEntry > switch_cases;  // non-empty iff this is a switch block
    };

    // Per-function CFG
    struct Cfg {
        const clang::FunctionDecl *function = nullptr;
        std::vector< CfgBlock > blocks;
        size_t entry = 0;

        size_t blockCount() const { return blocks.size(); }
    };

    // Build CFGs for all functions in the translation unit.
    // Splits at label boundaries, extracts goto edges, reorders in RPO.
    std::vector< Cfg > buildCfgs(clang::ASTContext &ctx);

    // Build CFG for a single function.
    Cfg buildCfg(const clang::FunctionDecl *fn);

    // Reorder blocks in reverse post-order.
    void reorderBlocksRPO(Cfg &cfg);

    // Populate CfgBlock::switch_cases from Ghidra switch metadata.
    // Must be called AFTER buildCfg + reorderBlocksRPO so that CfgBlock labels
    // and succ indices are final.  Matches CfgBlock labels to ghidra block keys
    // (via labelNameFromKey) and maps each ghidra::SwitchCase target to the
    // corresponding succs[] index.
    void populateSwitchMetadata(Cfg &cfg, const ghidra::Function &func);

} // namespace patchestry::ast
