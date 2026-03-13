/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/LoopInfo.hpp>

#include <algorithm>
#include <queue>
#include <unordered_set>

namespace patchestry::ast {

    const NaturalLoop *LoopInfo::LoopFor(size_t block) const {
        const NaturalLoop *best = nullptr;
        for (const auto &loop : loops) {
            for (size_t b : loop.body) {
                if (b == block) {
                    // Prefer innermost (smallest body)
                    if (!best || loop.body.size() < best->body.size()) {
                        best = &loop;
                    }
                }
            }
        }
        return best;
    }

    LoopInfo DetectLoops(const Cfg &cfg, const DomTree &dom) {
        LoopInfo info;
        size_t n = cfg.blocks.size();

        // Find back-edges: edge tail→header where header dominates tail
        struct BackEdge {
            size_t tail;
            size_t header;
        };
        std::vector< BackEdge > back_edges;

        for (size_t i = 0; i < n; ++i) {
            for (size_t s : cfg.blocks[i].succs) {
                if (dom.Dominates(s, i)) {
                    back_edges.push_back({i, s});
                }
            }
        }

        // Predecessor map — pure graph topology, built once.
        std::vector< std::vector< size_t > > preds(n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t s : cfg.blocks[i].succs) {
                preds[s].push_back(i);
            }
        }

        // For each back-edge, collect the natural loop body via reverse BFS
        for (const auto &be : back_edges) {
            NaturalLoop loop;
            loop.header = be.header;
            loop.back_edges.push_back(be.tail);

            std::unordered_set< size_t > body_set;
            body_set.insert(be.header);

            std::queue< size_t > worklist;
            if (be.tail != be.header) {
                body_set.insert(be.tail);
                worklist.push(be.tail);
            }

            while (!worklist.empty()) {
                size_t b = worklist.front();
                worklist.pop();
                for (size_t p : preds[b]) {
                    if (body_set.insert(p).second) {
                        worklist.push(p);
                    }
                }
            }

            loop.body.assign(body_set.begin(), body_set.end());
            std::sort(loop.body.begin(), loop.body.end());

            // Find exit blocks: successors of body blocks that are not in the body
            std::unordered_set< size_t > exit_set;
            for (size_t b : loop.body) {
                for (size_t s : cfg.blocks[b].succs) {
                    if (body_set.find(s) == body_set.end()) {
                        exit_set.insert(s);
                    }
                }
            }
            loop.exits.assign(exit_set.begin(), exit_set.end());
            std::sort(loop.exits.begin(), loop.exits.end());

            // Check if this loop should be merged with an existing one (same header)
            bool merged = false;
            for (auto &existing : info.loops) {
                if (existing.header == loop.header) {
                    // Merge: add the new back-edge and body blocks
                    existing.back_edges.push_back(be.tail);
                    for (size_t b : loop.body) {
                        if (std::find(existing.body.begin(), existing.body.end(), b)
                            == existing.body.end()) {
                            existing.body.push_back(b);
                        }
                    }
                    std::sort(existing.body.begin(), existing.body.end());
                    // Recompute exits
                    std::unordered_set< size_t > merged_body(
                        existing.body.begin(), existing.body.end());
                    existing.exits.clear();
                    for (size_t b : existing.body) {
                        for (size_t s : cfg.blocks[b].succs) {
                            if (merged_body.find(s) == merged_body.end()) {
                                existing.exits.push_back(s);
                            }
                        }
                    }
                    std::sort(existing.exits.begin(), existing.exits.end());
                    existing.exits.erase(
                        std::unique(existing.exits.begin(), existing.exits.end()),
                        existing.exits.end());
                    merged = true;
                    break;
                }
            }

            if (!merged) {
                info.loops.push_back(std::move(loop));
            }
        }

        // Establish nesting using indices: loop i is the parent of loop j if
        // loops[i].body ⊃ loops[j].body and loops[i] is the tightest such enclosure.
        // Using indices avoids dangling-pointer hazards from vector reallocation.
        for (size_t i = 0; i < info.loops.size(); ++i) {
            for (size_t j = 0; j < info.loops.size(); ++j) {
                if (i == j) continue;
                const auto &outer = info.loops[i];
                auto &inner       = info.loops[j];
                if (inner.body.size() >= outer.body.size()) continue;

                // Check if inner is fully contained in outer
                std::unordered_set< size_t > outer_body(
                    outer.body.begin(), outer.body.end());
                bool contained = true;
                for (size_t b : inner.body) {
                    if (outer_body.find(b) == outer_body.end()) {
                        contained = false;
                        break;
                    }
                }
                if (contained) {
                    // inner is nested in outer — keep the tightest (smallest) parent
                    if (inner.parent_idx == kNoLoopParent
                        || info.loops[inner.parent_idx].body.size() > outer.body.size())
                    {
                        inner.parent_idx = i;
                    }
                }
            }
        }

        // Build children index lists from parent indices
        for (size_t j = 0; j < info.loops.size(); ++j) {
            size_t parent = info.loops[j].parent_idx;
            if (parent != kNoLoopParent) {
                info.loops[parent].children_idx.push_back(j);
            }
        }

        return info;
    }

} // namespace patchestry::ast
