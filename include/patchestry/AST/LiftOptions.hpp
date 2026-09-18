/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

namespace patchestry::ast {

    /// Knobs for lifting a P-Code program into a Clang translation unit.
    struct LiftOptions
    {
        /// Debug goto baseline: skip the SNode structuring/cleanup chain and
        /// emit the raw flat CGraph.  Set via `--emit-flat-baseline`.
        bool emit_flat_baseline = false;
        /// Run the Clang-AST post-emission cleanup pipeline.  Flip off via
        /// `--clang-ast-cleanup=false` to bisect structuring drift.
        bool clang_ast_cleanup  = true;
        /// Dump DOT graphs at phase boundaries.  Set via `--emit-dot-cfg`.
        bool emit_dot_cfg       = false;
    };

} // namespace patchestry::ast
