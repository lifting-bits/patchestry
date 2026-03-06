/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <vector>

#include <patchestry/AST/CfgBuilder.hpp>
#include <patchestry/AST/DomTree.hpp>

namespace patchestry::ast {

    struct NaturalLoop {
        size_t header;                     // loop header block index
        std::vector< size_t > body;        // all blocks in the loop (including header)
        std::vector< size_t > back_edges;  // tail blocks with back-edges to header
        std::vector< size_t > exits;       // blocks outside the loop that are successors of body blocks
        NaturalLoop *parent = nullptr;     // enclosing loop (nullptr if outermost)
        std::vector< NaturalLoop * > children; // nested loops
    };

    struct LoopInfo {
        std::vector< NaturalLoop > loops;  // all natural loops, outermost first

        // Find the innermost loop containing block b, or nullptr
        const NaturalLoop *loopFor(size_t block) const;

        bool empty() const { return loops.empty(); }
    };

    // Detect all natural loops in the CFG using dominator tree back-edge classification.
    LoopInfo detectLoops(const Cfg &cfg, const DomTree &dom);

} // namespace patchestry::ast
