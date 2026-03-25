/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/CGraph.hpp>
#include <patchestry/AST/FunctionBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>

namespace patchestry::ast {

    namespace {

        /// Compute RPO from P-Code CFG (same algorithm as FunctionBuilder's compute_rpo).
        std::vector<std::string> compute_rpo_from_function(const ghidra::Function &function) {
            // Build per-block successor lists from branch operations.
            std::unordered_map<std::string, std::vector<std::string>> succs;
            for (const auto &[key, blk] : function.basic_blocks) {
                for (const auto &op_key : blk.ordered_operations) {
                    if (!blk.operations.contains(op_key)) continue;
                    const auto &op = blk.operations.at(op_key);
                    const auto &blocks = function.basic_blocks;
                    if (op.taken_block && blocks.contains(*op.taken_block)) {
                        succs[key].push_back(*op.taken_block);
                    }
                    if (op.not_taken_block && blocks.contains(*op.not_taken_block)) {
                        succs[key].push_back(*op.not_taken_block);
                    }
                    if (op.target_block && blocks.contains(*op.target_block)) {
                        succs[key].push_back(*op.target_block);
                    }
                    for (const auto &s : op.successor_blocks) {
                        if (blocks.contains(s)) succs[key].push_back(s);
                    }
                    for (const auto &sc : op.switch_cases) {
                        if (blocks.contains(sc.target_block))
                            succs[key].push_back(sc.target_block);
                    }
                    if (op.fallback_block && blocks.contains(*op.fallback_block)) {
                        succs[key].push_back(*op.fallback_block);
                    }
                }
            }

            // Iterative post-order DFS
            std::unordered_set<std::string> visited;
            std::vector<std::string> post_order;
            struct Frame {
                std::string key;
                size_t child_idx;
            };
            std::vector<Frame> stack;

            if (!function.entry_block.empty() &&
                function.basic_blocks.contains(function.entry_block)) {
                stack.push_back({function.entry_block, 0});
                visited.insert(function.entry_block);
            }

            while (!stack.empty()) {
                auto &top = stack.back();
                auto it = succs.find(top.key);
                if (it != succs.end() && top.child_idx < it->second.size()) {
                    const auto &child = it->second[top.child_idx];
                    ++top.child_idx;
                    if (visited.insert(child).second) {
                        stack.push_back({child, 0});
                    }
                } else {
                    post_order.push_back(top.key);
                    stack.pop_back();
                }
            }

            // Reverse for RPO
            std::reverse(post_order.begin(), post_order.end());

            // Append unreachable blocks sorted by address then numeric index.
            // Keys have the form "ram:HEXADDR:NUM:suffix"; lexicographic sort
            // would mis-order "...:10:..." before "...:2:..." so we parse the
            // numeric fields for a correct comparison.
            if (post_order.size() < function.basic_blocks.size()) {
                std::vector<std::string> unreachable;
                for (const auto &[key, _] : function.basic_blocks) {
                    if (!visited.contains(key)) unreachable.push_back(key);
                }
                auto parse_key = [](const std::string &k)
                    -> std::pair<uint64_t, uint64_t> {
                    // "ram:HEXADDR:NUM:suffix"
                    auto p1 = k.find(':');
                    if (p1 == std::string::npos) return {0, 0};
                    auto p2 = k.find(':', p1 + 1);
                    if (p2 == std::string::npos) return {0, 0};
                    auto p3 = k.find(':', p2 + 1);
                    uint64_t addr = 0, idx = 0;
                    try { addr = std::stoull(k.substr(p1 + 1, p2 - p1 - 1), nullptr, 16); }
                    catch (...) {}
                    if (p3 != std::string::npos) {
                        try { idx = std::stoull(k.substr(p2 + 1, p3 - p2 - 1)); }
                        catch (...) {}
                    }
                    return {addr, idx};
                };
                std::sort(unreachable.begin(), unreachable.end(),
                    [&](const std::string &a, const std::string &b) {
                        return parse_key(a) < parse_key(b);
                    });
                post_order.insert(post_order.end(), unreachable.begin(), unreachable.end());
            }

            return post_order;
        }

        /// Parse "ram:HEXADDR:NUM:basic" → HEXADDR as uint64.
        std::optional<uint64_t> parse_block_addr(const std::string &key) {
            auto p1 = key.find(':');
            if (p1 == std::string::npos) return std::nullopt;
            auto p2 = key.find(':', p1 + 1);
            if (p2 == std::string::npos) return std::nullopt;
            auto hex_str = key.substr(p1 + 1, p2 - p1 - 1);
            if (hex_str.empty()) return std::nullopt;
            try {
                return std::stoull(hex_str, nullptr, 16);
            } catch (...) {
                return std::nullopt;
            }
        }

        /// Find the terminal operation (BRANCH/CBRANCH/BRANCHIND/RETURN) in a block.
        /// Assumes the terminal is the last entry in ordered_operations, which
        /// holds for P-Code serialization (Ghidra always places the branch last).
        /// If non-terminal ops follow the branch, the terminal won't be found
        /// and the block will be treated as a fallthrough.
        const ghidra::Operation *find_terminal_op(const ghidra::BasicBlock &block) {
            if (block.ordered_operations.empty()) return nullptr;
            const auto &last_key = block.ordered_operations.back();
            if (!block.operations.contains(last_key)) return nullptr;
            const auto &op = block.operations.at(last_key);
            using M = ghidra::Mnemonic;
            if (op.mnemonic == M::OP_BRANCH || op.mnemonic == M::OP_CBRANCH
                || op.mnemonic == M::OP_BRANCHIND || op.mnemonic == M::OP_RETURN) {
                return &op;
            }
            return nullptr;
        }

    } // anonymous namespace

    CGraph BuildCGraph(FunctionBuilder &builder, clang::ASTContext &ctx) {
        const auto &func = builder.get_function();
        const auto &labels = builder.get_labels();

        // Compute RPO order
        auto rpo = compute_rpo_from_function(func);
        if (rpo.empty()) return {};

        // Map block keys to node indices
        std::unordered_map<std::string, size_t> key_to_index;
        for (size_t i = 0; i < rpo.size(); ++i) {
            key_to_index[rpo[i]] = i;
        }

        CGraph g;
        g.nodes.resize(rpo.size());
        g.entry = 0;

        // Build each CNode
        for (size_t i = 0; i < rpo.size(); ++i) {
            const auto &key = rpo[i];
            if (!func.basic_blocks.contains(key)) continue;
            const auto &block = func.basic_blocks.at(key);

            auto &node = g.nodes[i];
            node.id = i;

            // Set label (skip entry block)
            if (!block.is_entry_block) {
                node.label = LabelNameFromKey(key);
                node.original_label = node.label;
            }

            // Build stmts (non-terminal operations)
            node.stmts = builder.create_block_stmts(ctx, block);

            // Process terminal operation to determine edges
            const auto *term = find_terminal_op(block);
            if (!term) {
                // No terminal: fallthrough to next block in RPO
                if (i + 1 < rpo.size()) {
                    node.succs.push_back(i + 1);
                    node.edge_flags.push_back(0);
                }
                continue;
            }

            using M = ghidra::Mnemonic;

            if (term->mnemonic == M::OP_BRANCH) {
                // Unconditional branch
                if (term->target_block && key_to_index.contains(*term->target_block)) {
                    node.succs.push_back(key_to_index[*term->target_block]);
                    node.edge_flags.push_back(0);
                }

                // Build terminal GotoStmt for goto reconstruction
                if (term->target_block && labels.contains(*term->target_block)) {
                    auto loc = SourceLocation(ctx.getSourceManager(), term->key);
                    auto tgt_loc = SourceLocation(ctx.getSourceManager(), *term->target_block);
                    node.terminal = new (ctx) clang::GotoStmt(
                        labels.at(*term->target_block), loc, tgt_loc);
                }

            } else if (term->mnemonic == M::OP_CBRANCH) {
                // Conditional branch: succs[0]=not_taken, succs[1]=taken
                //
                size_t not_taken = CNode::kNone;
                size_t taken = CNode::kNone;

                if (term->not_taken_block && key_to_index.contains(*term->not_taken_block)) {
                    not_taken = key_to_index[*term->not_taken_block];
                } else if (i + 1 < rpo.size()) {
                    // Fallthrough for not-taken
                    not_taken = i + 1;
                }

                if (term->taken_block && key_to_index.contains(*term->taken_block)) {
                    taken = key_to_index[*term->taken_block];
                }

                if (not_taken != CNode::kNone) {
                    node.succs.push_back(not_taken);
                    node.edge_flags.push_back(0);
                }
                if (taken != CNode::kNone) {
                    node.succs.push_back(taken);
                    node.edge_flags.push_back(0);
                }

                node.is_conditional = (node.succs.size() == 2);
                node.branch_cond = builder.create_branch_condition(ctx, *term);

                // Build terminal IfStmt for goto reconstruction
                if (node.branch_cond && node.succs.size() == 2) {
                    auto loc = SourceLocation(ctx.getSourceManager(), term->key);
                    clang::Stmt *taken_stmt = nullptr;
                    clang::Stmt *not_taken_stmt = nullptr;

                    if (term->taken_block && labels.contains(*term->taken_block)) {
                        auto ll = SourceLocation(ctx.getSourceManager(), *term->taken_block);
                        taken_stmt = new (ctx) clang::GotoStmt(
                            labels.at(*term->taken_block), loc, ll);
                    } else {
                        taken_stmt = new (ctx) clang::NullStmt(loc, false);
                    }

                    if (term->not_taken_block && labels.contains(*term->not_taken_block)) {
                        auto ll = SourceLocation(ctx.getSourceManager(), *term->not_taken_block);
                        not_taken_stmt = new (ctx) clang::GotoStmt(
                            labels.at(*term->not_taken_block), loc, ll);
                    } else {
                        not_taken_stmt = new (ctx) clang::NullStmt(loc, false);
                    }

                    node.terminal = clang::IfStmt::Create(
                        ctx, loc, clang::IfStatementKind::Ordinary, nullptr, nullptr,
                        node.branch_cond, node.branch_cond->getBeginLoc(),
                        taken_stmt->getBeginLoc(), taken_stmt,
                        not_taken_stmt->getBeginLoc(), not_taken_stmt);
                }

            } else if (term->mnemonic == M::OP_BRANCHIND) {
                // Indirect branch — always build a switch.
                //
                // Priority 1: switch_cases present — recovered case
                //   values with a discriminant from switch_input or inputs[0].
                //
                // Priority 2: successor_blocks only — address-based jump table
                //   where each successor block address becomes a case value
                //   and the discriminant is the BRANCHIND input cast to uintptr.
                //
                // Both paths populate node.switch_cases and node.branch_cond
                // so FoldSwitch can produce a proper SSwitch.

                if (!term->switch_cases.empty()) {
                    // --- Priority 1: explicit switch_cases ---
                    node.branch_cond = builder.create_switch_discriminant(ctx, *term);

                    std::unordered_set<size_t> seen_succs;
                    for (const auto &sc : term->switch_cases) {
                        if (!key_to_index.contains(sc.target_block)) continue;
                        size_t target_idx = key_to_index[sc.target_block];
                        if (seen_succs.insert(target_idx).second) {
                            node.succs.push_back(target_idx);
                            node.edge_flags.push_back(0);
                        }
                        // Map to succ index
                        size_t succ_idx = 0;
                        for (size_t si = 0; si < node.succs.size(); ++si) {
                            if (node.succs[si] == target_idx) {
                                succ_idx = si;
                                break;
                            }
                        }
                        node.switch_cases.push_back(SwitchCaseEntry{
                            sc.value, succ_idx, sc.has_exit, /*is_default=*/false});
                    }
                    // Fallback edge (default arm)
                    if (term->fallback_block && key_to_index.contains(*term->fallback_block)) {
                        size_t fb_idx = key_to_index[*term->fallback_block];
                        if (seen_succs.insert(fb_idx).second) {
                            node.succs.push_back(fb_idx);
                            node.edge_flags.push_back(0);
                        }
                        // Find succ index for fallback
                        size_t fb_succ_idx = 0;
                        for (size_t si = 0; si < node.succs.size(); ++si) {
                            if (node.succs[si] == fb_idx) {
                                fb_succ_idx = si;
                                break;
                            }
                        }
                        node.switch_cases.push_back(SwitchCaseEntry{
                            0, fb_succ_idx, /*has_exit=*/false, /*is_default=*/true});
                    }

                } else if (!term->successor_blocks.empty()) {
                    // --- Priority 2: successor_blocks as jump table ---
                    // Discriminant is inputs[0] cast to uintptr_t.
                    // Case values are the hex addresses from block keys.
                    auto loc = SourceLocation(ctx.getSourceManager(), term->key);
                    auto disc_type = ctx.getUIntPtrType();

                    // Build discriminant from inputs[0]
                    clang::Expr *disc = builder.create_switch_discriminant(ctx, *term);
                    if (disc) {
                        // Cast to uintptr_t if needed
                        if (!ctx.hasSameUnqualifiedType(disc->getType(), disc_type)) {
                            disc = builder.create_cast(ctx, disc, disc_type, loc);
                        }
                    }
                    node.branch_cond = disc;

                    std::unordered_set<size_t> seen_succs;
                    for (const auto &block_key : term->successor_blocks) {
                        if (!key_to_index.contains(block_key)) continue;
                        size_t target_idx = key_to_index[block_key];
                        if (seen_succs.insert(target_idx).second) {
                            node.succs.push_back(target_idx);
                            node.edge_flags.push_back(0);
                        }
                        // Parse block address from key for case value
                        auto addr = parse_block_addr(block_key);
                        if (!addr) continue;

                        size_t succ_idx = 0;
                        for (size_t si = 0; si < node.succs.size(); ++si) {
                            if (node.succs[si] == target_idx) {
                                succ_idx = si;
                                break;
                            }
                        }
                        node.switch_cases.push_back(SwitchCaseEntry{
                            static_cast<int64_t>(*addr), succ_idx, false});
                    }
                }

            } else if (term->mnemonic == M::OP_RETURN) {
                // Return: no outgoing edges.
                // The corresponding ReturnStmt has already been emitted into
                // node.stmts by create_block_stmts, so there is nothing to do
                // here other than leave the successor list empty.
            }
        }

        // Build predecessor lists
        for (size_t i = 0; i < g.nodes.size(); ++i) {
            for (size_t s : g.nodes[i].succs) {
                if (s < g.nodes.size()) {
                    g.nodes[s].preds.push_back(i);
                }
            }
        }

        return g;
    }

} // namespace patchestry::ast
