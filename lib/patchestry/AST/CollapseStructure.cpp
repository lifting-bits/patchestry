/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/CollapseStructure.hpp>
#include <patchestry/AST/DomTree.hpp>
#include <patchestry/AST/LoopInfo.hpp>
#include <patchestry/Util/Log.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>

namespace patchestry::ast {

    // -----------------------------------------------------------------------
    // detail:: namespace — CGraph method definitions and graph builders
    // -----------------------------------------------------------------------
    namespace detail {

        size_t CGraph::collapseNodes(const std::vector<size_t> &ids, SNode *snode) {
            // Pick the first id as the representative
            size_t rep = ids[0];
            nodes[rep].structured = snode;

            // Collect external in/out edges
            std::unordered_set<size_t> idset(ids.begin(), ids.end());
            std::vector<size_t> ext_preds, ext_succs;
            std::vector<uint32_t> ext_succ_flags;

            for (size_t nid : ids) {
                for (size_t p : nodes[nid].preds) {
                    if (idset.count(p) == 0) ext_preds.push_back(p);
                }
            }
            // Use the last node's outgoing edges as the representative out edges
            size_t last = ids.back();
            for (size_t i = 0; i < nodes[last].succs.size(); ++i) {
                size_t s = nodes[last].succs[i];
                if (idset.count(s) == 0) {
                    ext_succs.push_back(s);
                    ext_succ_flags.push_back(nodes[last].edge_flags[i]);
                }
            }

            // Mark all as collapsed except rep
            for (size_t nid : ids) {
                if (nid != rep) nodes[nid].collapsed = true;
            }

            // Replace edges: remove old, add new through rep
            for (size_t nid : ids) {
                // Remove all edges involving collapsed nodes
                for (size_t s : nodes[nid].succs) {
                    if (idset.count(s) == 0) {
                        auto &p = nodes[s].preds;
                        p.erase(std::remove(p.begin(), p.end(), nid), p.end());
                    }
                }
                for (size_t p : nodes[nid].preds) {
                    if (idset.count(p) == 0) {
                        auto &ss = nodes[p].succs;
                        for (size_t i = 0; i < ss.size(); ++i) {
                            if (ss[i] == nid) {
                                ss[i] = rep; // redirect to rep
                            }
                        }
                    }
                }
            }

            // Set rep's edges to the external edges
            nodes[rep].succs = ext_succs;
            nodes[rep].edge_flags = ext_succ_flags;

            // Deduplicate preds
            std::sort(ext_preds.begin(), ext_preds.end());
            ext_preds.erase(std::unique(ext_preds.begin(), ext_preds.end()), ext_preds.end());
            nodes[rep].preds = ext_preds;

            // Add rep to succs' pred lists
            for (size_t s : ext_succs) {
                auto &p = nodes[s].preds;
                if (std::find(p.begin(), p.end(), rep) == p.end()) {
                    p.push_back(rep);
                }
            }

            nodes[rep].is_conditional = !ext_succs.empty() && ext_succs.size() == 2;
            nodes[rep].stmts.clear();
            nodes[rep].branch_cond = nullptr;

            return rep;
        }

        // Build the collapse graph from the Cfg
        CGraph buildCGraph(const Cfg &cfg) {
            CGraph g;
            g.entry = cfg.entry;
            g.nodes.resize(cfg.blocks.size());

            for (size_t i = 0; i < cfg.blocks.size(); ++i) {
                auto &cb = cfg.blocks[i];
                auto &cn = g.nodes[i];
                cn.id = i;
                cn.stmts = cb.stmts;
                cn.branch_cond = cb.branch_cond;
                cn.is_conditional = cb.is_conditional;
                cn.succs = cb.succs;
                cn.switch_cases = cb.switch_cases;
                cn.edge_flags.resize(cb.succs.size(), 0);

                // Assert Ghidra convention: 2-successor conditional blocks have
                // succs[0]=false/fallthrough, succs[1]=true/taken
                assert((!cb.is_conditional || cb.succs.size() != 2 ||
                        (cb.succs[0] == cb.fallthrough_succ && cb.succs[1] == cb.taken_succ))
                       && "CfgBlock edge polarity must be false-first (Ghidra convention)");
            }

            // Build predecessor lists
            for (size_t i = 0; i < g.nodes.size(); ++i) {
                for (size_t s : g.nodes[i].succs) {
                    g.nodes[s].preds.push_back(i);
                }
            }

            return g;
        }

        // Detect back-edges using DFS
        void markBackEdges(CGraph &g) {
            enum Color { WHITE, GRAY, BLACK };
            std::vector<Color> color(g.nodes.size(), WHITE);

            std::function<void(size_t)> dfs = [&](size_t u) {
                color[u] = GRAY;
                auto &nd = g.node(u);
                for (size_t i = 0; i < nd.succs.size(); ++i) {
                    size_t v = nd.succs[i];
                    if (color[v] == GRAY) {
                        nd.edge_flags[i] |= CNode::F_BACK;
                    } else if (color[v] == WHITE) {
                        dfs(v);
                    }
                }
                color[u] = BLACK;
            };

            dfs(g.entry);
        }

    } // namespace detail

    // -----------------------------------------------------------------------
    // Anonymous namespace — rule functions and internal helpers
    // -----------------------------------------------------------------------
    namespace {

        using detail::CNode;
        using detail::CGraph;

        // ---------------------------------------------------------------
        // SNode construction helpers
        // ---------------------------------------------------------------

        SNode *leafFromNode(const CNode &n, SNodeFactory &factory) {
            if (n.structured) return n.structured;
            auto *block = factory.make<SBlock>();
            for (auto *s : n.stmts) block->addStmt(s);
            return block;
        }

        // ---------------------------------------------------------------
        // Pattern-matching collapse rules (adapted from Ghidra)
        // ---------------------------------------------------------------

        // Rule: Sequential blocks (A->B chain)
        bool ruleBlockCat(CGraph &g, size_t id, SNodeFactory &factory) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 1) return false;
            if (bl.isSwitchOut()) return false;
            if (!bl.isDecisionOut(0)) return false;

            size_t next_id = bl.succs[0];
            if (next_id == id) return false;  // no self-loop
            auto &next = g.node(next_id);
            if (next.collapsed) return false;
            if (next.sizeIn() != 1) return false;

            // Build a sequence
            auto *seq = factory.make<SSeq>();
            seq->addChild(leafFromNode(bl, factory));

            // Extend the chain
            std::vector<size_t> chain = {id, next_id};
            size_t cur = next_id;
            while (g.node(cur).sizeOut() == 1 && g.node(cur).isDecisionOut(0)) {
                size_t nxt = g.node(cur).succs[0];
                if (nxt == id) break;
                auto &nxtNode = g.node(nxt);
                if (nxtNode.collapsed || nxtNode.sizeIn() != 1) break;
                if (nxtNode.isSwitchOut()) break;
                chain.push_back(nxt);
                cur = nxt;
            }

            for (size_t i = 1; i < chain.size(); ++i) {
                seq->addChild(leafFromNode(g.node(chain[i]), factory));
            }

            g.collapseNodes(chain, seq);
            return true;
        }

        // Rule: If without else (proper if)
        bool ruleBlockProperIf(CGraph &g, size_t id, SNodeFactory &factory,
                               clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 2) return false;
            if (bl.isSwitchOut()) return false;
            if (bl.isGotoOut(0) || bl.isGotoOut(1)) return false;

            for (size_t i = 0; i < 2; ++i) {
                size_t clause_id = bl.succs[i];
                if (clause_id == id) continue;
                auto &clause = g.node(clause_id);
                if (clause.collapsed) continue;
                if (clause.sizeIn() != 1 || clause.sizeOut() != 1) continue;
                if (!bl.isDecisionOut(i)) continue;
                if (clause.isGotoOut(0)) continue;

                size_t exit_id = clause.succs[0];
                if (exit_id != bl.succs[1 - i]) continue;

                // Build the if
                clang::Expr *cond = bl.branch_cond;
                if (!cond) {
                    // Create a placeholder true literal
                    cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                }

                // If the taken branch is succs[0] (false branch in Ghidra convention),
                // we may need to negate
                SNode *clause_body = leafFromNode(clause, factory);
                auto *if_node = factory.make<SIfThenElse>(cond, clause_body, nullptr);

                g.collapseNodes({id, clause_id}, if_node);
                return true;
            }
            return false;
        }

        // Rule: If-else
        bool ruleBlockIfElse(CGraph &g, size_t id, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 2) return false;
            if (bl.isSwitchOut()) return false;
            if (!bl.isDecisionOut(0) || !bl.isDecisionOut(1)) return false;

            size_t tc_id = bl.succs[1];  // true clause (Ghidra: out[1])
            size_t fc_id = bl.succs[0];  // false clause (Ghidra: out[0])
            auto &tc = g.node(tc_id);
            auto &fc = g.node(fc_id);

            if (tc.collapsed || fc.collapsed) return false;
            if (tc.sizeIn() != 1 || fc.sizeIn() != 1) return false;
            if (tc.sizeOut() != 1 || fc.sizeOut() != 1) return false;
            if (tc.succs[0] != fc.succs[0]) return false;  // must exit to same block
            if (tc.succs[0] == id) return false;  // no loops
            if (tc.isGotoOut(0) || fc.isGotoOut(0)) return false;

            clang::Expr *cond = bl.branch_cond;
            if (!cond) {
                cond = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
            }

            auto *then_body = leafFromNode(tc, factory);
            auto *else_body = leafFromNode(fc, factory);
            auto *if_node = factory.make<SIfThenElse>(cond, then_body, else_body);

            g.collapseNodes({id, tc_id, fc_id}, if_node);
            return true;
        }

        // Rule: While-do loop
        bool ruleBlockWhileDo(CGraph &g, size_t id, SNodeFactory &factory,
                              clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 2) return false;
            if (bl.isSwitchOut()) return false;
            if (bl.isGotoOut(0) || bl.isGotoOut(1)) return false;

            for (size_t i = 0; i < 2; ++i) {
                size_t clause_id = bl.succs[i];
                if (clause_id == id) continue;
                auto &clause = g.node(clause_id);
                if (clause.collapsed) continue;
                if (clause.sizeIn() != 1 || clause.sizeOut() != 1) continue;
                if (clause.succs[0] != id) continue;  // must loop back

                clang::Expr *cond = bl.branch_cond;
                if (!cond) {
                    cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                }

                auto *body = leafFromNode(clause, factory);
                auto *while_node = factory.make<SWhile>(cond, body);

                g.collapseNodes({id, clause_id}, while_node);
                return true;
            }
            return false;
        }

        // Rule: Do-while loop (single block looping to itself)
        bool ruleBlockDoWhile(CGraph &g, size_t id, SNodeFactory &factory,
                              clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 2) return false;
            if (bl.isSwitchOut()) return false;
            if (bl.isGotoOut(0) || bl.isGotoOut(1)) return false;

            for (size_t i = 0; i < 2; ++i) {
                if (bl.succs[i] != id) continue;  // must loop to self

                clang::Expr *cond = bl.branch_cond;
                if (!cond) {
                    cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                }

                auto *body = leafFromNode(bl, factory);
                auto *dowhile_node = factory.make<SDoWhile>(body, cond);

                // Remove the self-edge, keep the exit edge
                size_t exit_id = bl.succs[1 - i];
                g.removeEdge(id, id);
                bl.structured = dowhile_node;
                bl.succs = {exit_id};
                bl.edge_flags = {0};
                bl.is_conditional = false;
                bl.branch_cond = nullptr;
                return true;
            }
            return false;
        }

        // Rule: Infinite loop (single out to self)
        bool ruleBlockInfLoop(CGraph &g, size_t id, SNodeFactory &factory,
                              clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 1) return false;
            if (bl.isGotoOut(0)) return false;
            if (bl.succs[0] != id) return false;  // must loop to self

            auto *body = leafFromNode(bl, factory);
            auto *true_lit = clang::IntegerLiteral::Create(
                ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
            auto *loop = factory.make<SWhile>(true_lit, body);

            g.removeEdge(id, id);
            bl.structured = loop;
            bl.succs.clear();
            bl.edge_flags.clear();
            bl.preds.erase(std::remove(bl.preds.begin(), bl.preds.end(), id), bl.preds.end());
            return true;
        }

        // Rule: If with no exit (clause has zero out edges)
        bool ruleBlockIfNoExit(CGraph &g, size_t id, SNodeFactory &factory,
                               clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || bl.sizeOut() != 2) return false;
            if (bl.isSwitchOut()) return false;
            if (bl.isGotoOut(0) || bl.isGotoOut(1)) return false;

            for (size_t i = 0; i < 2; ++i) {
                size_t clause_id = bl.succs[i];
                if (clause_id == id) continue;
                auto &clause = g.node(clause_id);
                if (clause.collapsed) continue;
                if (clause.sizeIn() != 1 || clause.sizeOut() != 0) continue;
                if (!bl.isDecisionOut(i)) continue;

                clang::Expr *cond = bl.branch_cond;
                if (!cond) {
                    cond = clang::IntegerLiteral::Create(
                        ctx, llvm::APInt(32, 1), ctx.IntTy, clang::SourceLocation());
                }

                auto *clause_body = leafFromNode(clause, factory);
                auto *if_node = factory.make<SIfThenElse>(cond, clause_body, nullptr);

                g.collapseNodes({id, clause_id}, if_node);
                return true;
            }
            return false;
        }

        // Rule: Switch statement
        bool ruleBlockSwitch(CGraph &g, size_t id, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
            auto &bl = g.node(id);
            if (bl.collapsed || !bl.isSwitchOut()) return false;

            // Find exit block: look for a successor with sizeIn > 1 or sizeOut > 1
            size_t exit_id = std::numeric_limits<size_t>::max();
            for (size_t s : bl.succs) {
                auto &sn = g.node(s);
                if (sn.collapsed) continue;
                if (s == id || sn.sizeIn() > 1 || sn.sizeOut() > 1) {
                    exit_id = s;
                    break;
                }
            }

            // Validate: each case must have sizeIn==1 and either exit to exit_id or have no exit
            for (size_t s : bl.succs) {
                if (s == exit_id) continue;
                auto &sn = g.node(s);
                if (sn.collapsed) return false;
                if (sn.sizeIn() != 1) return false;
                if (sn.sizeOut() == 1) {
                    if (exit_id != std::numeric_limits<size_t>::max() && sn.succs[0] != exit_id)
                        return false;
                    if (sn.isGotoOut(0)) return false;
                } else if (sn.sizeOut() > 1) {
                    return false;
                }
            }

            // Build the switch SNode
            clang::Expr *disc = bl.branch_cond;
            if (!disc) {
                disc = clang::IntegerLiteral::Create(
                    ctx, llvm::APInt(32, 0), ctx.IntTy, clang::SourceLocation());
            }
            auto *sw = factory.make<SSwitch>(disc);

            std::vector<size_t> collapse_ids = {id};
            for (size_t s : bl.succs) {
                if (s == exit_id) continue;
                collapse_ids.push_back(s);
                // TODO: extract actual case values from switch metadata
                sw->addCase(nullptr, leafFromNode(g.node(s), factory));
            }

            g.collapseNodes(collapse_ids, sw);
            return true;
        }

        // Rule: Mark goto edges
        bool ruleBlockGoto(CGraph &g, size_t id, SNodeFactory &factory) {
            auto &bl = g.node(id);
            if (bl.collapsed) return false;

            for (size_t i = 0; i < bl.succs.size(); ++i) {
                if (bl.isGotoOut(i)) {
                    // Wrap in a goto SNode
                    auto *body = leafFromNode(bl, factory);
                    auto *seq = factory.make<SSeq>();
                    seq->addChild(body);

                    std::string target_label = "block_" + std::to_string(bl.succs[i]);
                    seq->addChild(factory.make<SGoto>(factory.intern(target_label)));

                    bl.structured = seq;
                    // Remove the goto edge
                    g.removeEdge(id, bl.succs[i]);
                    return true;
                }
            }
            return false;
        }

        // ---------------------------------------------------------------
        // Goto selection heuristic
        // ---------------------------------------------------------------

        // When collapsing is stuck, pick the "best" edge to mark as goto.
        // Simple heuristic: pick the edge from the block with most out-edges,
        // preferring non-back edges.
        bool selectAndMarkGoto(CGraph &g) {
            size_t best_from = std::numeric_limits<size_t>::max();
            size_t best_edge = 0;
            size_t best_score = 0;

            for (auto &n : g.nodes) {
                if (n.collapsed) continue;
                if (n.sizeOut() < 2) continue;
                for (size_t i = 0; i < n.succs.size(); ++i) {
                    if (n.isGotoOut(i)) continue;
                    size_t score = n.sizeOut();
                    if (n.isBackEdge(i)) score += 100;  // prefer marking back-edges
                    if (score > best_score) {
                        best_score = score;
                        best_from = n.id;
                        best_edge = i;
                    }
                }
            }

            if (best_from == std::numeric_limits<size_t>::max()) return false;

            g.node(best_from).setGoto(best_edge);
            return true;
        }

        // ---------------------------------------------------------------
        // Main collapse loop
        // ---------------------------------------------------------------

        size_t collapseInternal(CGraph &g, SNodeFactory &factory, clang::ASTContext &ctx) {
            bool change;
            size_t isolated_count;

            do {
                do {
                    change = false;
                    isolated_count = 0;
                    for (auto &n : g.nodes) {
                        if (n.collapsed) continue;
                        if (n.sizeIn() == 0 && n.sizeOut() == 0) {
                            isolated_count++;
                            continue;
                        }

                        if (ruleBlockGoto(g, n.id, factory)) { change = true; continue; }
                        if (ruleBlockCat(g, n.id, factory)) { change = true; continue; }
                        if (ruleBlockProperIf(g, n.id, factory, ctx)) { change = true; continue; }
                        if (ruleBlockIfElse(g, n.id, factory, ctx)) { change = true; continue; }
                        if (ruleBlockWhileDo(g, n.id, factory, ctx)) { change = true; continue; }
                        if (ruleBlockDoWhile(g, n.id, factory, ctx)) { change = true; continue; }
                        if (ruleBlockInfLoop(g, n.id, factory, ctx)) { change = true; continue; }
                        if (ruleBlockSwitch(g, n.id, factory, ctx)) { change = true; continue; }
                    }
                } while (change);

                // Try IfNoExit as fallback (Ghidra applies this only when stuck)
                change = false;
                for (auto &n : g.nodes) {
                    if (n.collapsed) continue;
                    if (ruleBlockIfNoExit(g, n.id, factory, ctx)) {
                        change = true;
                        break;
                    }
                }
            } while (change);

            return isolated_count;
        }

    } // anonymous namespace

    // ---------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------

    SNode *collapseStructure(const Cfg &cfg, SNodeFactory &factory,
                             clang::ASTContext &ctx) {
        if (cfg.blocks.empty()) {
            return factory.make<SSeq>();
        }

        // 1. Build the collapse graph
        detail::CGraph g = detail::buildCGraph(cfg);

        // 2. Mark back-edges
        detail::markBackEdges(g);

        // 3. Main collapse loop
        size_t isolated = collapseInternal(g, factory, ctx);

        // 4. When stuck, select gotos and retry
        size_t max_iterations = g.nodes.size() * 4;  // safety bound
        size_t iter = 0;
        while (isolated < g.activeCount() && iter < max_iterations) {
            if (!selectAndMarkGoto(g)) {
                LOG(WARNING) << "CollapseStructure: could not select goto, "
                             << g.activeCount() - isolated << " blocks remaining\n";
                break;
            }
            isolated = collapseInternal(g, factory, ctx);
            ++iter;
        }

        // 5. Collect the final structured tree
        // Find the root (entry node or its collapsed representative)
        SNode *root = nullptr;
        for (auto &n : g.nodes) {
            if (n.collapsed) continue;
            if (n.structured) {
                if (!root) {
                    root = n.structured;
                } else {
                    // Multiple uncollapsed nodes -- wrap in sequence
                    auto *seq = root->dyn_cast<SSeq>();
                    if (!seq) {
                        seq = factory.make<SSeq>();
                        seq->addChild(root);
                        root = seq;
                    }
                    seq->addChild(n.structured);
                }
            } else {
                // Uncollapsed leaf -- add as block
                auto *block = leafFromNode(n, factory);
                if (!root) {
                    root = block;
                } else {
                    auto *seq = root->dyn_cast<SSeq>();
                    if (!seq) {
                        seq = factory.make<SSeq>();
                        seq->addChild(root);
                        root = seq;
                    }
                    seq->addChild(block);
                }
            }
        }

        if (!root) root = factory.make<SSeq>();
        return root;
    }

} // namespace patchestry::ast
