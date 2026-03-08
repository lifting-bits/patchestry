/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include <patchestry/AST/CfgBuilder.hpp>

namespace patchestry::ast {

    // Dominator/post-dominator tree using Cooper-Harvey-Kennedy algorithm.
    class DomTree
    {
      public:
        static constexpr size_t UNDEF = std::numeric_limits< size_t >::max();

        // Build dominator tree from CFG
        static DomTree buildDom(const Cfg &cfg);

        // Build post-dominator tree from CFG (reverse edges, virtual exit)
        static DomTree buildPostDom(const Cfg &cfg);

        // Returns the immediate dominator of block b
        size_t idom(size_t b) const {
            return (b < idom_.size()) ? idom_[b] : UNDEF;
        }

        // Returns true if a dominates b
        bool dominates(size_t a, size_t b) const;

        // Returns the RPO number for block b
        size_t rpoNum(size_t b) const {
            return (b < rpo_num_.size()) ? rpo_num_[b] : UNDEF;
        }

        size_t entry() const { return entry_; }
        size_t blockCount() const { return idom_.size(); }

        // Compute dominance frontier for a block
        std::vector< size_t > dominanceFrontier(size_t block,
                                                 const Cfg &cfg) const;

        // Iterative intersect for the CHK algorithm
        size_t intersect(size_t b1, size_t b2) const;

        DomTree() = default;

        std::vector< size_t > idom_;
        std::vector< size_t > rpo_num_;
        std::vector< size_t > rpo_order_; // blocks in RPO order
        size_t entry_ = 0;
    };

} // namespace patchestry::ast
