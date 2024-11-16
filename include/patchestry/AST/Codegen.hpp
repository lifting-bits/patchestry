/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/Frontend/CompilerInstance.h>
#include <vast/Frontend/FrontendAction.hpp>
#include <vast/Frontend/Options.hpp>

namespace llvm {
    class raw_fd_ostream;
}

namespace patchestry::ast {
    class CodeGenerator
    {
      public:
        explicit CodeGenerator(clang::CompilerInstance &ci) : opts(vast::cc::options(ci)) {}

        CodeGenerator(const CodeGenerator &)                = delete;
        CodeGenerator &operator=(const CodeGenerator &)     = delete;
        CodeGenerator(CodeGenerator &&) noexcept            = delete;
        CodeGenerator &operator=(CodeGenerator &&) noexcept = delete;

        virtual ~CodeGenerator() {}

        void generate_source_ir(clang::ASTContext &ctx, llvm::raw_fd_ostream &os);

      private:
        vast::cc::action_options opts;
    };

} // namespace patchestry::ast
