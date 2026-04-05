/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <string_view>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceLocation.h>

namespace patchestry::ast {

    /// Maximum depth for cascading goto-elimination passes.
    /// Each pass may expose new goto→next-label adjacencies; this
    /// bounds the iteration to prevent runaway on pathological inputs.
    inline constexpr int kMaxGotoEliminationPasses = 8;

    /// Ensure expr is a prvalue (insert lvalue-to-rvalue conversion if needed).
    clang::Expr *EnsureRValue(clang::ASTContext &ctx, clang::Expr *expr);

    /// Create !(expr) using clang::UnaryOperator (logical not).
    /// Wraps in ParenExpr so the pretty-printer emits !(a == b).
    clang::Expr *NegateExpr(clang::ASTContext &ctx, clang::Expr *expr);
    
    clang::SourceLocation SourceLocation(clang::SourceManager &sm, std::string key);

    clang::QualType
    GetTypeFromSize(clang::ASTContext &ctx, unsigned bit_size, bool is_signed, bool is_integer);

    std::string LabelNameFromKey(std::string key);

    std::string SanitizeKeyToIdent(std::string_view key);

    clang::CastKind GetCastKind(
        clang::ASTContext &ctx, const clang::QualType &from_type, const clang::QualType &to_type
    );

    bool
    ShouldReinterpretCast(const clang::QualType &from_type, const clang::QualType &to_type);

} // namespace patchestry::ast
