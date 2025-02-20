/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceLocation.h>

namespace patchestry::ast {
    clang::SourceLocation sourceLocation(clang::SourceManager &sm, std::string key);

    clang::QualType
    getTypeFromSize(clang::ASTContext &ctx, unsigned bit_size, bool is_signed, bool is_integer);

    std::string labelNameFromKey(std::string key);

    clang::CastKind getCastKind(
        clang::ASTContext &ctx, const clang::QualType &from_type, const clang::QualType &to_type
    );

    bool
    shouldReinterpretCast(const clang::QualType &from_type, const clang::QualType &to_type);

} // namespace patchestry::ast
