/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/DomTree.hpp>

#include <algorithm>
#include <stack>
#include <unordered_set>

namespace patchestry::ast {

    // Compute iterative RPO (reverse post-order) traversal
    static std::vector< size_t > computeRPO(
        size_t entry, size_t n,
        const std::vector< std::vector< size_t > > &succs
    ) {
        std::vector< size_t > post_order;
        std::vector< bool > visited(n, false);
        std::stack< std::pair< size_t, size_t > > stk;

        stk.push({entry, 0});
        visited[entry] = true;

        while (!stk.empty()) {
            auto &[block, idx] = stk.top();
            if (idx < succs[block].size()) {
                size_t next = succs[block][idx];
                ++idx;
                if (!visited[next]) {
                    visited[next] = true;
                    stk.push({next, 0});
                }
            } else {
                post_order.push_back(block);
                stk.pop();
            }
        }

        std::reverse(post_order.begin(), post_order.end());

        // Add unreachable nodes at the end
        for (size_t i = 0; i < n; ++i) {
            if (!visited[i]) post_order.push_back(i);
        }

        return post_order;
    }

    size_t DomTree::intersect(size_t b1, size_t b2) const {
        while (b1 != b2) {
            while (rpo_num_[b1] > rpo_num_[b2]) {
                b1 = idom_[b1];
                if (b1 == UNDEF) return UNDEF;
            }
            while (rpo_num_[b2] > rpo_num_[b1]) {
                b2 = idom_[b2];
                if (b2 == UNDEF) return UNDEF;
            }
        }
        return b1;
    }

    // Cooper-Harvey-Kennedy iterative dominator algorithm
    static DomTree buildDomImpl(
        size_t entry, size_t n,
        const std::vector< std::vector< size_t > > &succs,
        const std::vector< std::vector< size_t > > &preds
    ) {
        DomTree tree;
        tree.rpo_order_ = computeRPO(entry, n, succs);
        tree.entry_ = entry;

        tree.idom_.assign(n, DomTree::UNDEF);
        tree.rpo_num_.assign(n, DomTree::UNDEF);

        for (size_t i = 0; i < tree.rpo_order_.size(); ++i) {
            tree.rpo_num_[tree.rpo_order_[i]] = i;
        }

        tree.idom_[entry] = entry;

        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t rpo_idx = 0; rpo_idx < tree.rpo_order_.size(); ++rpo_idx) {
                size_t b = tree.rpo_order_[rpo_idx];
                if (b == entry) continue;

                size_t new_idom = DomTree::UNDEF;
                for (size_t p : preds[b]) {
                    if (tree.idom_[p] == DomTree::UNDEF) continue;
                    if (new_idom == DomTree::UNDEF) {
                        new_idom = p;
                    } else {
                        new_idom = tree.intersect(new_idom, p);
                    }
                }

                if (new_idom != DomTree::UNDEF && tree.idom_[b] != new_idom) {
                    tree.idom_[b] = new_idom;
                    changed = true;
                }
            }
        }

        return tree;
    }

    DomTree DomTree::buildDom(const Cfg &cfg) {
        size_t n = cfg.blocks.size();
        if (n == 0) return DomTree();

        std::vector< std::vector< size_t > > succs(n), preds(n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t s : cfg.blocks[i].succs) {
                succs[i].push_back(s);
                preds[s].push_back(i);
            }
        }

        return buildDomImpl(cfg.entry, n, succs, preds);
    }

    DomTree DomTree::buildPostDom(const Cfg &cfg) {
        size_t n = cfg.blocks.size();
        if (n == 0) return DomTree();

        // Add virtual exit node at index n
        size_t virt_exit = n;
        size_t total = n + 1;

        // Reverse edges + add edges from exit blocks to virtual exit
        std::vector< std::vector< size_t > > rev_succs(total), rev_preds(total);
        for (size_t i = 0; i < n; ++i) {
            if (cfg.blocks[i].succs.empty()) {
                // Exit block: add edge virtual_exit → i in reverse graph
                rev_succs[virt_exit].push_back(i);
                rev_preds[i].push_back(virt_exit);
            }
            for (size_t s : cfg.blocks[i].succs) {
                // Reverse: s → i
                rev_succs[s].push_back(i);
                rev_preds[i].push_back(s);
            }
        }

        return buildDomImpl(virt_exit, total, rev_succs, rev_preds);
    }

    bool DomTree::dominates(size_t a, size_t b) const {
        if (a == b) return true;
        if (b >= idom_.size() || a >= idom_.size()) return false;
        if (idom_[b] == UNDEF) return false;

        size_t x = b;
        unsigned guard = 0;
        while (x != entry_ && x != UNDEF && guard < 2048) {
            x = idom_[x];
            if (x == a) return true;
            if (x == UNDEF) return false;
            ++guard;
        }
        return x == a;
    }

    std::vector< size_t > DomTree::dominanceFrontier(size_t block,
                                                      const Cfg &cfg) const {
        std::unordered_set< size_t > frontier;
        size_t n = cfg.blocks.size();

        for (size_t b = 0; b < n; ++b) {
            if (cfg.blocks[b].succs.size() < 2) continue;
            for (size_t s : cfg.blocks[b].succs) {
                size_t runner = b;
                while (runner != UNDEF && runner != idom(s)) {
                    if (runner == block) {
                        frontier.insert(s);
                    }
                    runner = idom(runner);
                    if (runner == entry_) break;
                }
            }
        }

        return std::vector< size_t >(frontier.begin(), frontier.end());
    }

} // namespace patchestry::ast
