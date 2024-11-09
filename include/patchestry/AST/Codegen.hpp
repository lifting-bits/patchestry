/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/AST/ASTContext.h>

namespace llvm {
    class raw_fd_ostream;
}

namespace patchestry::ast {
    class CodeGenerator
    {
      public:
        CodeGenerator() = default;

        CodeGenerator(const CodeGenerator &)                = default;
        CodeGenerator &operator=(const CodeGenerator &)     = default;
        CodeGenerator(CodeGenerator &&) noexcept            = default;
        CodeGenerator &operator=(CodeGenerator &&) noexcept = default;

        virtual ~CodeGenerator() {}

        void generate_source_ir(clang::ASTContext &ctx, llvm::raw_fd_ostream &os);
    };

} // namespace patchestry::ast
