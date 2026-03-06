/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include <patchestry/AST/SNode.hpp>
#include <patchestry/Util/Options.hpp>

#include <clang/AST/ASTContext.h>

namespace patchestry::ast {

    // Base class for passes that operate on SNode trees
    class SNodePass
    {
      public:
        virtual ~SNodePass() = default;
        virtual std::string_view name() const = 0;

        // Returns true if the tree was modified
        virtual bool run(SNode *root, SNodeFactory &factory,
                         clang::ASTContext &ctx) = 0;
    };

    // Manages and runs a sequence of SNode passes
    class SNodePassManager
    {
      public:
        void addPass(std::unique_ptr< SNodePass > pass);

        // Run all passes on the given SNode tree.
        // If verbose & print_tu, dumps pseudo-C, DOT, and JSON after each pass.
        bool run(SNode *root, SNodeFactory &factory,
                 clang::ASTContext &ctx, const patchestry::Options &options);

        size_t passCount() const { return passes_.size(); }

      private:
        std::vector< std::unique_ptr< SNodePass > > passes_;
    };

} // namespace patchestry::ast
