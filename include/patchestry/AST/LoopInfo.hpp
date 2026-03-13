/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <climits>
#include <cstddef>
#include <vector>

#include <patchestry/AST/CfgBuilder.hpp>
#include <patchestry/AST/DomTree.hpp>

namespace patchestry::ast {

    // Sentinel value meaning "no parent" for NaturalLoop::parent_idx.
    inline constexpr size_t kNoLoopParent = SIZE_MAX;

    struct NaturalLoop {
        size_t header;                    // loop header block index
        std::vector< size_t > body;       // all blocks in the loop (including header)
        std::vector< size_t > back_edges; // tail blocks with back-edges to header
        std::vector< size_t >
            exits; // blocks outside the loop that are successors of body blocks
        size_t parent_idx =
            kNoLoopParent; // index of enclosing loop in LoopInfo::loops, or kNoLoopParent
        std::vector< size_t >
            children_idx; // indices of directly nested loops in LoopInfo::loops
    };

    struct LoopInfo {
        std::vector< NaturalLoop > loops;  // all natural loops, outermost first

        // Find the innermost loop containing block b, or nullptr
        const NaturalLoop *LoopFor(size_t block) const;

        // Return the parent loop of the given loop, or nullptr if outermost
        const NaturalLoop *ParentOf(const NaturalLoop &loop) const {
            if (loop.parent_idx == kNoLoopParent) {
                return nullptr;
            }
            return &loops[loop.parent_idx];
        }

        bool Empty() const { return loops.empty(); }
    };

    // Detect all natural loops in the CFG using dominator tree back-edge classification.
    LoopInfo DetectLoops(const Cfg &cfg, const DomTree &dom);

} // namespace patchestry::ast
