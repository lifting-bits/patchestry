/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

namespace patchestry {

    struct Options
    {
        bool emit_cir  = false;
        bool emit_mlir = false;
        bool emit_llvm = false;
        bool emit_asm  = false;
        bool emit_obj  = false;
        bool verbose   = false;

        std::string output_file;
        std::string input_file;

        bool print_tu = false;

        std::vector< std::string > pipelines = {};
    };

} // namespace patchestry
