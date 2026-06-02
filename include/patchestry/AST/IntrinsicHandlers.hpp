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
#include <clang/AST/Type.h>

namespace patchestry::ast {
    class OpBuilder;
}

namespace patchestry::ghidra {
    struct Function;
    struct Operation;
} // namespace patchestry::ghidra

namespace clang {
    class ASTContext;
}

namespace patchestry::ast {

    // Handler function signature for CALLOTHER intrinsics
    using IntrinsicHandler = std::pair< clang::Stmt *, bool > (*)(
        OpBuilder &, clang::ASTContext &, const ghidra::Function &, const ghidra::Operation &,
        const std::string &
    );

    // Returns the map of intrinsic name -> handler function
    const std::unordered_map< std::string, IntrinsicHandler > &get_intrinsic_handlers();

    // Parse intrinsic name from label (strips type suffix like _void, _int, _uint8_t)
    // and applies arch-specific normalization (e.g. AArch64 ldadd -> atomic_fetch_add_*).
    // When `arch` is empty or unrecognized, every registered arch normalizer is tried
    // in turn to preserve backward compatibility with inputs that lack an architecture
    // tag.
    std::string parse_intrinsic_name(std::string_view arch, std::string_view label);

    // int->float conversion userops (e.g. ARM NEON VectorSignedToFloat) are
    // bare `define pcodeop`s with no return type, so Ghidra types their result
    // `undefined<N>`. These resolve it to a real float type. Table in
    // IntrinsicHandlers.cpp; matches on name alone, so gate on is_intrinsic.

    // True if `name` is a known int->float conversion userop.
    bool IsFloatReturningUserop(std::string_view name);

    // Float return type by result width: 32->float, 64->double; null otherwise.
    clang::QualType ResolveUseropFloatReturn(
        clang::ASTContext &ctx, std::string_view name, uint64_t size_bits
    );

} // namespace patchestry::ast
