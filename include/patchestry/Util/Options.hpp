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

        // `-from-c`: compile a marked C translation unit (LLM-written or
        // hand-written) instead of lifting JSON.  `-input` may still be
        // given: it supplies the target and the model for `-validate-pcode`.
        std::string from_c_file;
        // Ghidra language id override for `-from-c` (`ARM:LE:32:Cortex`).
        std::string target_lang;
        // `-from-c`: fail when a marker-listed function is missing or bodyless.
        bool strict_symbols       = true;
        // `-validate-pcode`: check the parsed C against the JSON P-Code model.
        bool validate_pcode       = false;
        // `-validate-pcode=report`: write the report but do not fail on it.
        bool validate_report_only = false;
    };

} // namespace patchestry
