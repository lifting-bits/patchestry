/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/ASTPassManager.hpp>

namespace patchestry::ast {

    void ASTPassManager::add_pass(std::unique_ptr< ASTPass > pass) {
        passes.emplace_back(std::move(pass));
    }

    bool ASTPassManager::run(clang::ASTContext &ctx, const patchestry::Options &options) {
        for (const auto &pass : passes) {
            if (!pass->run(ctx, options)) {
                return false;
            }
        }
        return true;
    }

} // namespace patchestry::ast
