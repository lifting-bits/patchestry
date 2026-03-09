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
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceLocation.h>

namespace patchestry::ast {
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
