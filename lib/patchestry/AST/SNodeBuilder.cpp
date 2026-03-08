/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/SNodeBuilder.hpp>

namespace patchestry::ast {

    SNode *buildSNodeTree(const Cfg &cfg, SNodeFactory &factory) {
        auto *root = factory.make< SSeq >();

        for (size_t i = 0; i < cfg.blocks.size(); ++i) {
            const auto &blk = cfg.blocks[i];

            // Create an SBlock with the block's statements
            auto *sblock = factory.make< SBlock >();
            for (auto *stmt : blk.stmts) {
                sblock->addStmt(stmt);
            }

            // Wrap in SLabel if the block has a label
            SNode *block_node = sblock;
            if (!blk.label.empty()) {
                auto label_sv = factory.intern(blk.label);
                block_node = factory.make< SLabel >(label_sv, sblock);
            }

            root->addChild(block_node);

            // Add terminal control flow
            if (blk.is_conditional) {
                // Conditional branch: if(cond) goto taken; else goto fallthrough;
                auto taken_label = cfg.blocks[blk.taken_succ].label;
                auto fall_label = cfg.blocks[blk.fallthrough_succ].label;

                SNode *then_node = nullptr;
                SNode *else_node = nullptr;

                if (!taken_label.empty()) {
                    then_node = factory.make< SGoto >(factory.intern(taken_label));
                }
                if (!fall_label.empty() && blk.fallthrough_succ != i + 1) {
                    // Only emit else-goto if it's not a trivial fallthrough
                    else_node = factory.make< SGoto >(factory.intern(fall_label));
                }

                if (then_node) {
                    auto *ite = factory.make< SIfThenElse >(
                        blk.branch_cond, then_node, else_node
                    );
                    root->addChild(ite);
                }
            } else if (!blk.succs.empty()) {
                // Unconditional goto (only if not fallthrough to next block)
                size_t succ = blk.succs[0];
                if (succ != i + 1 && !cfg.blocks[succ].label.empty()) {
                    auto target_label = factory.intern(cfg.blocks[succ].label);
                    root->addChild(factory.make< SGoto >(target_label));
                }
            }
            // No successors = return/exit — already in stmts
        }

        return root;
    }

} // namespace patchestry::ast
