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
        bool emit_asm                       = false;
        bool emit_obj                       = false;
        bool verbose                        = false;
        bool use_structuring_pass           = false;
        bool verify_no_node_loss            = false;
        bool structuring_improvement_report = false;
        // Run the Clang-AST post-emission cleanup pipeline in
        // CleanupPrettyPrint.
        // Default on; flip off via --clang-ast-cleanup=false to
        // compare the structured C output without the AST-layer
        // passes (useful for bisecting structuring drift).
        bool clang_ast_cleanup              = true;

        std::string output_file;
        std::string input_file;

        bool print_tu = false;

        bool emit_dot_cfg = false;
    };

} // namespace patchestry
