/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/ASTPassManager.hpp>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

namespace patchestry::ast {

    void ASTPassManager::add_pass(std::unique_ptr< ASTPass > pass) {
        passes.emplace_back(std::move(pass));
    }

    bool ASTPassManager::run(clang::ASTContext &ctx, const patchestry::Options &options) {
        unsigned pass_index = 0;
        for (const auto &pass : passes) {
            if (!pass->run(ctx, options)) {
                return false;
            }
            ++pass_index;

            if (options.verbose && options.print_tu && !options.output_file.empty()) {
                std::string dump_path = options.output_file + "_pass_"
                    + (pass_index < 10 ? "0" : "") + std::to_string(pass_index)
                    + "_" + pass->name() + ".c";
                std::error_code ec;
                llvm::raw_fd_ostream out(dump_path, ec, llvm::sys::fs::OF_Text);
                if (!ec) {
                    ctx.getTranslationUnitDecl()->print(out, ctx.getPrintingPolicy(), 0);
                }
            }
        }
        return true;
    }

} // namespace patchestry::ast
