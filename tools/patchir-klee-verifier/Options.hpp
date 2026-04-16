/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/Support/CommandLine.h>

#include <string>

namespace patchestry::klee_verifier {

    extern llvm::cl::opt< std::string > input_filename;
    extern llvm::cl::opt< std::string > output_filename;
    extern llvm::cl::opt< std::string > target_function;
    extern llvm::cl::opt< bool > emit_ll;
    extern llvm::cl::opt< std::string > model_library;
    extern llvm::cl::opt< bool > verbose;
    extern llvm::cl::opt< unsigned > symbolic_ptr_size;
    extern llvm::cl::opt< unsigned > klee_init_max_depth;
    extern llvm::cl::opt< unsigned > klee_init_array_expand_limit;
    extern llvm::cl::opt< bool > strict_contracts;

} // namespace patchestry::klee_verifier
