/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/AST/ASTContext.h>

namespace patchestry::ast {
    clang::SourceLocation source_location_from_key(clang::ASTContext &ctx, std::string key);

    clang::QualType get_type_for_size(
        clang::ASTContext &ctx, unsigned bit_size, bool is_signed, bool is_integer
    );

    std::string label_name_from_key(std::string key);

} // namespace patchestry::ast
