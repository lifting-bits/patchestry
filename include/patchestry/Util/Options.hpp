/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace patchestry {

    struct Options
    {
        bool emit_cir                       = false;
        bool emit_mlir                      = false;
        bool emit_llvm                      = false;
        bool verbose                        = false;
        // Debug goto baseline: skip the SNode structuring/cleanup chain
        // and emit raw flat CGraph. Set via --emit-flat-baseline.
        bool emit_flat_baseline             = false;
        // Flip off via --clang-ast-cleanup=false to bisect structuring drift.
        bool clang_ast_cleanup              = true;

        std::string output_file;
        std::string input_file;

        bool print_tu = false;

        bool emit_dot_cfg = false;

        // Fail the lift when a P-Code fact cannot be reproduced faithfully
        // instead of warning about it. Set via --strict.
        bool strict = false;

        // Write the binding manifest (globals and functions with their
        // original addresses) to this file. Set via --emit-bindings.
        std::string bindings_file;
    };

} // namespace patchestry
