/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/CfgBuilder.hpp>

#include <algorithm>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/RecursiveASTVisitor.h>

#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace patchestry::ast {

    namespace {

        // Retrieve the label name from a LabelDecl without going through
        // DeclarationName::getAsString(), which can crash in some LLVM
        // builds.  Falls back to getName().str() via the IdentifierInfo.
        std::string GetLabelName(const clang::LabelDecl *decl) {
            if (const auto *ii = decl->getIdentifier()) {
                return ii->getName().str();
            }
            return {};
        }

        // Split a CompoundStmt into basic blocks at label boundaries.
        // A new block starts at each LabelStmt.
        struct BlockSplitter {
            std::vector< CfgBlock > blocks;
            std::unordered_map< std::string, size_t > label_to_block;

            void Split(clang::CompoundStmt *body) {
                // Start with entry block
                blocks.push_back(CfgBlock{});

                for (auto *stmt : body->body()) {
                    // LabelStmt starts a new block
                    if (auto *label_stmt = llvm::dyn_cast< clang::LabelStmt >(stmt)) {
                        // If current block is non-empty, start a new one
                        if (!blocks.back().stmts.empty() || !blocks.back().label.empty()) {
                            blocks.push_back(CfgBlock{});
                        }

                        std::string name = GetLabelName(label_stmt->getDecl());
                        blocks.back().label = name;
                        label_to_block[name] = blocks.size() - 1;

                        // Add the sub-statement (what the label wraps)
                        clang::Stmt *sub = label_stmt->getSubStmt();
                        if (sub && !llvm::isa< clang::NullStmt >(sub)) {
                            // If sub is another label, recurse; otherwise add it
                            if (auto *nested_label = llvm::dyn_cast< clang::LabelStmt >(sub)) {
                                // Finish this block, start new one for nested label
                                blocks.push_back(CfgBlock{});
                                std::string nested_name =
                                    GetLabelName(nested_label->getDecl());
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
        bool IsBlockTerminator(clang::Stmt *stmt) {
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
        void SplitAtInternalControlFlow(std::vector< CfgBlock > &blocks,
                                         std::unordered_map< std::string, size_t > &label_to_block,
                                         unsigned &synth_counter) {
            for (size_t i = 0; i < blocks.size(); ++i) {
                auto &blk = blocks[i];
                for (size_t s = 0; s + 1 < blk.stmts.size(); ++s) {
                    if (!IsBlockTerminator(blk.stmts[s])) continue;

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
        std::string GetGotoTarget(clang::Stmt *stmt) {
            if (auto *go = llvm::dyn_cast< clang::GotoStmt >(stmt)) {
                return GetLabelName(go->getLabel());
            }
            return "";
        }

        // Check if a statement is an if(cond) with goto arms
        bool IsConditionalGoto(clang::Stmt *stmt, clang::Expr *&cond,
                               std::string &then_target, std::string &else_target) {
            auto *if_stmt = llvm::dyn_cast< clang::IfStmt >(stmt);
            if (!if_stmt) return false;

            cond = if_stmt->getCond();
            clang::Stmt *then_s = if_stmt->getThen();
            clang::Stmt *else_s = if_stmt->getElse();

            // then must be a goto (possibly wrapped in compound)
            if (auto *go = llvm::dyn_cast_or_null< clang::GotoStmt >(then_s)) {
                then_target = GetLabelName(go->getLabel());
            } else if (auto *cs = llvm::dyn_cast_or_null< clang::CompoundStmt >(then_s)) {
                if (cs->size() == 1) {
                    if (auto *go = llvm::dyn_cast< clang::GotoStmt >(cs->body_front())) {
                        then_target = GetLabelName(go->getLabel());
                    }
                }
            }

            if (then_target.empty()) return false;

            // else can be a goto or absent (fallthrough)
            if (else_s) {
                if (auto *go = llvm::dyn_cast< clang::GotoStmt >(else_s)) {
                    else_target = GetLabelName(go->getLabel());
                } else if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(else_s)) {
                    if (cs->size() == 1) {
                        if (auto *go = llvm::dyn_cast< clang::GotoStmt >(cs->body_front())) {
                            else_target = GetLabelName(go->getLabel());
                        }
                    }
                }
            }

            return true;
        }

        void ResolveEdges(BlockSplitter &bs) {
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
                if (IsConditionalGoto(last, cond, then_target, else_target)) {
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
                std::string target = GetGotoTarget(last);
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

                // Check for switch statement (emitted by create_branchind as
                // CompoundStmt { SwitchStmt, goto fallback }).  Extract case
                // goto targets as successors so PopulateSwitchMetadata can
                // match them.
                {
                    clang::SwitchStmt *sw = nullptr;
                    clang::Stmt *post_switch = nullptr;
                    if (auto *cs = llvm::dyn_cast< clang::CompoundStmt >(last)) {
                        for (auto *s : cs->body()) {
                            if (auto *s_sw = llvm::dyn_cast< clang::SwitchStmt >(s)) {
                                sw = s_sw;
                            } else if (sw && !post_switch) {
                                post_switch = s;
                            }
                        }
                    } else {
                        sw = llvm::dyn_cast< clang::SwitchStmt >(last);
                    }

                    if (sw) {
                        // Set branch_cond from the switch discriminant so that
                        // CfgFoldStructure::foldSwitch can use it.
                        blk.branch_cond = sw->getCond();

                        // Extract goto targets from case statements.
                        if (auto *body =
                                llvm::dyn_cast_or_null< clang::CompoundStmt >(sw->getBody()))
                        {
                            for (auto *s : body->body()) {
                                auto *case_stmt = llvm::dyn_cast< clang::CaseStmt >(s);
                                if (!case_stmt) continue;
                                clang::Stmt *sub = case_stmt->getSubStmt();
                                if (auto *go =
                                        llvm::dyn_cast_or_null< clang::GotoStmt >(sub))
                                {
                                    std::string target =
                                        GetLabelName(go->getLabel());
                                    auto it = bs.label_to_block.find(target);
                                    if (it != bs.label_to_block.end()) {
                                        size_t tgt = it->second;
                                        if (std::find(blk.succs.begin(),
                                                      blk.succs.end(), tgt)
                                            == blk.succs.end())
                                        {
                                            blk.succs.push_back(tgt);
                                        }
                                    }
                                }
                                // Inlined cases with a terminal goto (back-edge
                                // to loop header) contribute a successor edge.
                                else if (auto *cs = llvm::dyn_cast_or_null<
                                             clang::CompoundStmt>(sub))
                                {
                                    if (!cs->body_empty()) {
                                        if (auto *go = llvm::dyn_cast<
                                                clang::GotoStmt>(cs->body_back()))
                                        {
                                            std::string target =
                                                GetLabelName(go->getLabel());
                                            auto it = bs.label_to_block.find(target);
                                            if (it != bs.label_to_block.end()) {
                                                // Avoid duplicate edges (multiple cases
                                                // may goto the same loop header).
                                                size_t tgt = it->second;
                                                if (std::find(blk.succs.begin(),
                                                              blk.succs.end(), tgt)
                                                    == blk.succs.end())
                                                {
                                                    blk.succs.push_back(tgt);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Fallback goto after the switch.
                        if (post_switch) {
                            std::string target = GetGotoTarget(post_switch);
                            if (!target.empty()) {
                                auto it = bs.label_to_block.find(target);
                                if (it != bs.label_to_block.end()) {
                                    blk.succs.push_back(it->second);
                                    blk.fallthrough_succ = it->second;
                                }
                            }
                        }

                        // Keep the switch in stmts — FunctionBuilder already
                        // emitted it with inlined case bodies.  The SNode
                        // pipeline wraps it in an SBlock and passes it through.
                        continue;
                    }
                }

                // Default: fallthrough to next block
                if (i + 1 < bs.blocks.size()) {
                    blk.succs.push_back(i + 1);
                    blk.fallthrough_succ = i + 1;
                }
            }
        }

    } // namespace

    Cfg BuildCfg(const clang::FunctionDecl *fn) {
        Cfg cfg;
        cfg.function = fn;

        if (!fn->hasBody()) return cfg;

        auto *body = llvm::dyn_cast< clang::CompoundStmt >(fn->getBody());
        if (!body) return cfg;

        BlockSplitter bs;
        bs.Split(body);

        // Split blocks at internal if-goto / bare goto patterns
        unsigned synth_counter = 0;
        SplitAtInternalControlFlow(bs.blocks, bs.label_to_block, synth_counter);

        ResolveEdges(bs);

        cfg.blocks = std::move(bs.blocks);
        cfg.entry = 0;

        return cfg;
    }

    void ReorderBlocksRPO(Cfg &cfg) {
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

    std::vector< Cfg > BuildCfgs(clang::ASTContext &ctx) {
        std::vector< Cfg > cfgs;

        auto *tu = ctx.getTranslationUnitDecl();
        for (auto *decl : tu->decls()) {
            auto *fn = llvm::dyn_cast< clang::FunctionDecl >(decl);
            if (!fn || !fn->doesThisDeclarationHaveABody()) continue;

            Cfg cfg = BuildCfg(fn);
            if (!cfg.blocks.empty()) {
                ReorderBlocksRPO(cfg);
                cfgs.push_back(std::move(cfg));
            }
        }

        return cfgs;
    }

    void PopulateSwitchMetadata(Cfg &cfg, const ghidra::Function &func) {
        using namespace patchestry::ghidra;

        // Build a label→CfgBlock-index map for quick lookup.
        // CfgBlock::label stores the sanitized name (LabelNameFromKey(key)).
        std::unordered_map< std::string, size_t > label_to_idx;
        for (size_t i = 0; i < cfg.blocks.size(); ++i) {
            if (!cfg.blocks[i].label.empty()) {
                label_to_idx[cfg.blocks[i].label] = i;
            }
        }

        // For each ghidra block, scan operations for BRANCHIND with switch_cases.
        for (const auto &[bb_key, bb] : func.basic_blocks) {
            for (const auto &op_key : bb.ordered_operations) {
                if (!bb.operations.contains(op_key)) continue;
                const auto &op = bb.operations.at(op_key);

                if (op.mnemonic != Mnemonic::OP_BRANCHIND) continue;
                if (op.switch_cases.empty()) continue;

                // Find the CfgBlock for this ghidra block.
                std::string sanitized_label = LabelNameFromKey(bb_key);
                auto blk_it = label_to_idx.find(sanitized_label);
                if (blk_it == label_to_idx.end()) {
                    // Entry block has no label — check if it matches the entry block key.
                    if (bb_key == func.entry_block && !cfg.blocks.empty()
                        && cfg.blocks[cfg.entry].label.empty())
                    {
                        blk_it = label_to_idx.end(); // use entry directly
                    } else {
                        LOG(WARNING) << "PopulateSwitchMetadata: no CfgBlock for ghidra block '"
                                     << bb_key << "'\n";
                        continue;
                    }
                }

                size_t cfg_idx = (blk_it != label_to_idx.end())
                    ? blk_it->second
                    : cfg.entry;

                auto &blk = cfg.blocks[cfg_idx];

                // Build a target-label → succs-index map for this block.
                // Each successor's label needs to be matched against the sanitized
                // target_block from the ghidra SwitchCase.
                std::unordered_map< std::string, size_t > target_to_succ_idx;
                for (size_t si = 0; si < blk.succs.size(); ++si) {
                    size_t succ_blk = blk.succs[si];
                    if (succ_blk < cfg.blocks.size()) {
                        const auto &succ_label = cfg.blocks[succ_blk].label;
                        if (!succ_label.empty()) {
                            target_to_succ_idx[succ_label] = si;
                        }
                    }
                }

                // Populate switch_cases on the CfgBlock.
                for (const auto &sc : op.switch_cases) {
                    std::string target_label = LabelNameFromKey(sc.target_block);
                    auto it = target_to_succ_idx.find(target_label);
                    if (it != target_to_succ_idx.end()) {
                        blk.switch_cases.push_back(SwitchCaseEntry{
                            sc.value,
                            it->second,
                            sc.has_exit
                        });
                    } else {
                        LOG(WARNING) << "PopulateSwitchMetadata: switch case target '"
                                     << sc.target_block
                                     << "' not found in succs of block '"
                                     << bb_key << "'\n";
                    }
                }
            }
        }
    }

} // namespace patchestry::ast
