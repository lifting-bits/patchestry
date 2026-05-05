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

    /// Deep-clone a Clang Expr tree to produce an independent copy.
    /// Used to break Expr* aliasing when the same condition needs to
    /// appear in multiple places in the AST — CIR lowering and some
    /// emitter paths require tree-unique Expr* nodes.  Supports the
    /// expression kinds the structuring pipeline builds (integer/bool
    /// literals, DeclRef, Binary/Unary operator, ImplicitCast,
    /// CStyleCast, ParenExpr, ArraySubscript, MemberExpr, CallExpr,
    /// ConditionalOperator).  For unknown kinds, falls back to wrapping
    /// the original in a ParenExpr (still forces a fresh node while
    /// leaving the subtree shared — acceptable because subtree sharing
    /// is then handled by downstream ClangEmitter cloning).
    clang::Expr *CloneExpr(clang::ASTContext &ctx, clang::Expr *expr);
    
    clang::SourceLocation SourceLocation(clang::SourceManager &sm, std::string key);

    /// Returns a valid SourceLocation backed by a virtual "<patchestry-virtual>"
    /// buffer in the given context's SourceManager.  Idempotent: subsequent
    /// calls with the same SourceManager reuse the same FileID via
    /// FileManager::getVirtualFileRef + SourceManager::translateFile, so the
    /// SM owns the location's lifetime and there is no cross-SM aliasing.
    /// Use this anywhere a default-constructed clang::SourceLocation() would
    /// otherwise be passed to an AST node constructor — CIRGen asserts every
    /// location is valid, and synthetic locations satisfy that invariant
    /// without claiming any real source mapping.
    clang::SourceLocation VirtualLoc(clang::ASTContext &ctx);

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
