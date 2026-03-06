/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/CfgBuilder.hpp>
#include <patchestry/Util/Log.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/RecursiveASTVisitor.h>

#include <algorithm>
#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace patchestry::ast {

    namespace {

        // Split a CompoundStmt into basic blocks at label boundaries.
        // A new block starts at each LabelStmt.
        struct BlockSplitter {
            std::vector< CfgBlock > blocks;
            std::unordered_map< std::string, size_t > label_to_block;

            void split(clang::CompoundStmt *body) {
                // Start with entry block
                blocks.push_back(CfgBlock{});

                for (auto *stmt : body->body()) {
                    // LabelStmt starts a new block
                    if (auto *label_stmt = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                        // If current block is non-empty, start a new one
                        if (!blocks.back().stmts.empty() || !blocks.back().label.empty()) {
                            blocks.push_back(CfgBlock{});
                        }

                        std::string name = label_stmt->getDecl()->getNameAsString();
                        blocks.back().label = name;
                        label_to_block[name] = blocks.size() - 1;

                        // Add the sub-statement (what the label wraps)
                        clang::Stmt *sub = label_stmt->getSubStmt();
                        if (sub && !llvm::isa< clang::NullStmt >(sub)) {
                            // If sub is another label, recurse; otherwise add it
                            if (auto *nested_label = llvm::dyn_cast< clang::LabelStmt >(sub)) {
                                // Finish this block, start new one for nested label
                                blocks.push_back(CfgBlock{});
                                std::string nested_name = nested_label->getDecl()->getNameAsString();
                                blocks.back().label = nested_name;
                                label_to_block[nested_name] = blocks.size() - 1;
                                sub = nested_label->getSubStmt();
                                if (sub && !llvm::isa< clang::NullStmt >(sub)) {
                                    blocks.back().stmts.push_back(sub);
                                }
                            } else {
                                blocks.back().stmts.push_back(sub);
                            }
                        }
                    } else {
                        blocks.back().stmts.push_back(stmt);
                    }
                }

                // Remove completely empty trailing block
                if (blocks.size() > 1 && blocks.back().stmts.empty()
                    && blocks.back().label.empty()) {
                    blocks.pop_back();
                }
            }
        };

        // Check if a stmt is an if-goto or bare goto (i.e., a block terminator)
        bool isBlockTerminator(clang::Stmt *stmt) {
            if (llvm::isa< clang::GotoStmt >(stmt)) return true;
            if (llvm::isa< clang::ReturnStmt >(stmt)) return true;
            if (auto *ifs = llvm::dyn_cast< clang::IfStmt >(stmt)) {
                auto *then_s = ifs->getThen();
                if (then_s && llvm::isa< clang::GotoStmt >(then_s)) return true;
                if (auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(then_s)) {
                    if (cs->size() == 1 && llvm::isa< clang::GotoStmt >(cs->body_front()))
                        return true;
                }
            }
            return false;
        }

        // Split blocks at internal control flow (if-goto, bare goto) that aren't
        // the last statement.  After the label-based split, a block may contain:
        //   stmt0; if(cond) goto L; stmt1; stmt2;
        // This should become two blocks: [stmt0, if(cond) goto L] and [stmt1, stmt2].
        void splitAtInternalControlFlow(std::vector< CfgBlock > &blocks,
                                         std::unordered_map< std::string, size_t > &label_to_block,
                                         unsigned &synth_counter) {
            for (size_t i = 0; i < blocks.size(); ++i) {
                auto &blk = blocks[i];
                for (size_t s = 0; s + 1 < blk.stmts.size(); ++s) {
                    if (!isBlockTerminator(blk.stmts[s])) continue;

                    // Found a terminator that isn't the last stmt — split here
                    CfgBlock new_block;
                    std::string synth_label =
                        "__synth_" + std::to_string(synth_counter++);
                    new_block.label = synth_label;
                    new_block.stmts.assign(
                        blk.stmts.begin() + static_cast< ptrdiff_t >(s + 1),
                        blk.stmts.end()
                    );
                    blk.stmts.resize(s + 1);

                    // Insert new block right after current one
                    auto pos = blocks.begin()
                        + static_cast< ptrdiff_t >(i + 1);
                    blocks.insert(pos, std::move(new_block));

                    // Update label_to_block for the new block and all
                    // blocks after it (indices shifted)
                    label_to_block[synth_label] = i + 1;
                    for (size_t j = i + 2; j < blocks.size(); ++j) {
                        if (!blocks[j].label.empty()) {
                            label_to_block[blocks[j].label] = j;
                        }
                    }

                    break; // re-process current block (now shorter)
                }
            }
        }

        // Extract goto targets from a block's terminal statement
        std::string getGotoTarget(clang::Stmt *stmt) {
            if (auto *go = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                return go->getLabel()->getNameAsString();
            }
            return "";
        }

        // Check if a statement is an if(cond) with goto arms
        bool isConditionalGoto(clang::Stmt *stmt, clang::Expr *&cond,
                               std::string &then_target, std::string &else_target) {
            auto *if_stmt = llvm::dyn_cast< clang::IfStmt >(stmt);
            if (!if_stmt) return false;

            cond = if_stmt->getCond();
            clang::Stmt *then_s = if_stmt->getThen();
            clang::Stmt *else_s = if_stmt->getElse();

            // then must be a goto (possibly wrapped in compound)
            if (auto *go = llvm::dyn_cast_or_null< clang::GotoStmt >(then_s)) {
                then_target = go->getLabel()->getNameAsString();
            } else if (auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(then_s)) {
                if (cs->size() == 1) {
                    if (auto *go = llvm::dyn_cast< clang::GotoStmt >(cs->body_front())) {
                        then_target = go->getLabel()->getNameAsString();
                    }
                }
            }

            if (then_target.empty()) return false;

            // else can be a goto or absent (fallthrough)
            if (else_s) {
                if (auto *go = llvm::dyn_cast< clang::GotoStmt >(else_s)) {
                    else_target = go->getLabel()->getNameAsString();
                } else if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(else_s)) {
                    if (cs->size() == 1) {
                        if (auto *go = llvm::dyn_cast< clang::GotoStmt >(cs->body_front())) {
                            else_target = go->getLabel()->getNameAsString();
                        }
                    }
                }
            }

            return true;
        }

        void resolveEdges(BlockSplitter &bs) {
            for (size_t i = 0; i < bs.blocks.size(); ++i) {
                auto &blk = bs.blocks[i];
                if (blk.stmts.empty()) {
                    // Empty block falls through to next
                    if (i + 1 < bs.blocks.size()) {
                        blk.succs.push_back(i + 1);
                        blk.fallthrough_succ = i + 1;
                    }
                    continue;
                }

                clang::Stmt *last = blk.stmts.back();

                // Check for conditional goto (if(cond) goto L1; else goto L2;)
                clang::Expr *cond = nullptr;
                std::string then_target, else_target;
                if (isConditionalGoto(last, cond, then_target, else_target)) {
                    blk.is_conditional = true;
                    blk.branch_cond = cond;
                    blk.stmts.pop_back(); // remove the if-stmt

                    // Ghidra convention: succs[0] = false/fallthrough,
                    //                    succs[1] = true/taken
                    if (!else_target.empty()) {
                        auto it_else = bs.label_to_block.find(else_target);
                        if (it_else != bs.label_to_block.end()) {
                            blk.fallthrough_succ = it_else->second;
                            blk.succs.push_back(it_else->second);
                        }
                    } else if (i + 1 < bs.blocks.size()) {
                        // No else → fallthrough to next block
                        blk.fallthrough_succ = i + 1;
                        blk.succs.push_back(i + 1);
                    }

                    auto it_then = bs.label_to_block.find(then_target);
                    if (it_then != bs.label_to_block.end()) {
                        blk.taken_succ = it_then->second;
                        blk.succs.push_back(it_then->second);
                    }
                    continue;
                }

                // Check for unconditional goto
                std::string target = getGotoTarget(last);
                if (!target.empty()) {
                    blk.stmts.pop_back(); // remove the goto stmt
                    auto it = bs.label_to_block.find(target);
                    if (it != bs.label_to_block.end()) {
                        blk.succs.push_back(it->second);
                        blk.fallthrough_succ = it->second;
                    }
                    continue;
                }

                // Check for return statement — no successors
                if (llvm::isa< clang::ReturnStmt >(last)) {
                    continue;
                }

                // Default: fallthrough to next block
                if (i + 1 < bs.blocks.size()) {
                    blk.succs.push_back(i + 1);
                    blk.fallthrough_succ = i + 1;
                }
            }
        }

    } // namespace

    Cfg buildCfg(const clang::FunctionDecl *fn) {
        Cfg cfg;
        cfg.function = fn;

        if (!fn->hasBody()) return cfg;

        auto *body = llvm::dyn_cast< clang::CompoundStmt >(fn->getBody());
        if (!body) return cfg;

        BlockSplitter bs;
        bs.split(body);

        // Split blocks at internal if-goto / bare goto patterns
        unsigned synth_counter = 0;
        splitAtInternalControlFlow(bs.blocks, bs.label_to_block, synth_counter);

        resolveEdges(bs);

        cfg.blocks = std::move(bs.blocks);
        cfg.entry = 0;

        return cfg;
    }

    void reorderBlocksRPO(Cfg &cfg) {
        if (cfg.blocks.empty()) return;

        size_t n = cfg.blocks.size();

        // Iterative post-order DFS
        std::vector< size_t > post_order;
        std::vector< bool > visited(n, false);
        std::stack< std::pair< size_t, size_t > > stack; // (block, succ_index)

        stack.push({cfg.entry, 0});
        visited[cfg.entry] = true;

        while (!stack.empty()) {
            auto &[block, succ_idx] = stack.top();
            auto &succs = cfg.blocks[block].succs;

            if (succ_idx < succs.size()) {
                size_t next = succs[succ_idx];
                ++succ_idx;
                if (!visited[next]) {
                    visited[next] = true;
                    stack.push({next, 0});
                }
            } else {
                post_order.push_back(block);
                stack.pop();
            }
        }

        // RPO = reverse of post-order
        std::reverse(post_order.begin(), post_order.end());

        // Add any unreachable blocks at the end
        for (size_t i = 0; i < n; ++i) {
            if (!visited[i]) {
                post_order.push_back(i);
            }
        }

        // Build remapping: old_index → new_index
        std::vector< size_t > remap(n);
        for (size_t new_idx = 0; new_idx < post_order.size(); ++new_idx) {
            remap[post_order[new_idx]] = new_idx;
        }

        // Reorder blocks
        std::vector< CfgBlock > new_blocks(n);
        for (size_t new_idx = 0; new_idx < n; ++new_idx) {
            new_blocks[new_idx] = std::move(cfg.blocks[post_order[new_idx]]);
            // Remap successor indices
            for (auto &s : new_blocks[new_idx].succs) {
                s = remap[s];
            }
            if (new_blocks[new_idx].is_conditional) {
                new_blocks[new_idx].taken_succ = remap[new_blocks[new_idx].taken_succ];
                new_blocks[new_idx].fallthrough_succ = remap[new_blocks[new_idx].fallthrough_succ];
            } else if (!new_blocks[new_idx].succs.empty()) {
                new_blocks[new_idx].fallthrough_succ = remap[new_blocks[new_idx].fallthrough_succ];
            }
        }

        cfg.blocks = std::move(new_blocks);
        cfg.entry = 0; // entry is always first in RPO
    }

    std::vector< Cfg > buildCfgs(clang::ASTContext &ctx) {
        std::vector< Cfg > cfgs;

        auto *tu = ctx.getTranslationUnitDecl();
        for (auto *decl : tu->decls()) {
            auto *fn = llvm::dyn_cast< clang::FunctionDecl >(decl);
            if (!fn || !fn->doesThisDeclarationHaveABody()) continue;

            Cfg cfg = buildCfg(fn);
            if (!cfg.blocks.empty()) {
                reorderBlocksRPO(cfg);
                cfgs.push_back(std::move(cfg));
            }
        }

        return cfgs;
    }

} // namespace patchestry::ast
