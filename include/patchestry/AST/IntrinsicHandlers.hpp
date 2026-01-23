/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <clang/AST/Stmt.h>

namespace patchestry::ast {
    class OpBuilder;
}

namespace patchestry::ghidra {
    struct Function;
    struct Operation;
}

namespace clang {
    class ASTContext;
}

namespace patchestry::ast {

    // Handler function signature for CALLOTHER intrinsics
    using IntrinsicHandler = std::pair< clang::Stmt *, bool > (*)(
        OpBuilder &, clang::ASTContext &, const ghidra::Function &, const ghidra::Operation &
    );

    // Returns the map of intrinsic name -> handler function
    const std::unordered_map< std::string, IntrinsicHandler > &get_intrinsic_handlers();

    // Parse intrinsic name from label (strips type suffix like _void, _int, _uint8_t)
    std::string parse_intrinsic_name(std::string_view label);

} // namespace patchestry::ast
