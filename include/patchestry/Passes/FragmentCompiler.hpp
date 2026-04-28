/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

namespace patchestry::passes::fragment_expr {

    // `name` is without the leading `$`; `c_type` is the spelling clang
    // will see in the wrapper's parameter list (e.g. "uint32_t", "int32_t *").
    struct CaptureBinding
    {
        std::string name;
        std::string c_type;
    };

    struct FragmentResult
    {
        std::string func_name;
        std::optional< std::string > module_text;
        std::string error;
    };

    // Compile a rewrite-mode fragment by wrapping it in a synthesised C
    // function and invoking clang/clangCIR. `arch` follows the Ghidra
    // `lang` convention ("ARM:LE:32:default"). Identical inputs hit a
    // process-scoped content-keyed cache.
    FragmentResult compile_fragment(
        llvm::StringRef fragment, llvm::ArrayRef< CaptureBinding > captures,
        llvm::StringRef arch, llvm::StringRef return_c_type = "int32_t",
        llvm::StringRef extra_decls = ""
    );

    // Compile a `stmt:` body: auto-converts `return X;` / `return;`
    // to `__patchestry_return(...)` markers, substitutes `$NAME`,
    // and wraps in a void function with a `noreturn`-marked extern
    // for the marker (typed by `enclosing_return_c_type`). Cache
    // key includes the enclosing return type.
    FragmentResult compile_stmt_fragment(
        llvm::StringRef body, llvm::ArrayRef< CaptureBinding > captures,
        llvm::StringRef arch, llvm::StringRef enclosing_return_c_type = "void",
        llvm::StringRef extra_decls = ""
    );

} // namespace patchestry::passes::fragment_expr
