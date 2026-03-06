/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/SNodePass.hpp>
#include <patchestry/AST/SNodeDebug.hpp>
#include <patchestry/Util/Log.hpp>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

namespace patchestry::ast {

    void SNodePassManager::addPass(std::unique_ptr< SNodePass > pass) {
        passes_.emplace_back(std::move(pass));
    }

    bool SNodePassManager::run(SNode *root, SNodeFactory &factory,
                               clang::ASTContext &ctx,
                               const patchestry::Options &options) {
        unsigned pass_index = 0;

        for (const auto &pass : passes_) {
            ++pass_index;
            LOG(INFO) << "Running SNode pass " << pass_index << ": "
                      << pass->name() << "\n";

            bool modified = pass->run(root, factory, ctx);

            if (options.verbose) {
                LOG(INFO) << "  Pass " << pass->name()
                          << (modified ? " modified" : " no change") << "\n";
            }

            // Dump debug output after each pass when verbose + print_tu
            if (options.verbose && options.print_tu && !options.output_file.empty()) {
                std::string prefix = options.output_file + "_spass_"
                    + (pass_index < 10 ? "0" : "") + std::to_string(pass_index)
                    + "_" + std::string(pass->name());

                // Pseudo-C dump
                {
                    std::error_code ec;
                    llvm::raw_fd_ostream out(prefix + ".c", ec,
                                             llvm::sys::fs::OF_Text);
                    if (!ec) {
                        out << "// SNode pass " << pass_index << ": "
                            << pass->name() << "\n\n";
                        printPseudoC(root, out, &ctx);
                    }
                }

                // DOT dump
                {
                    std::error_code ec;
                    llvm::raw_fd_ostream out(prefix + ".dot", ec,
                                             llvm::sys::fs::OF_Text);
                    if (!ec) {
                        emitDOT(root, out);
                    }
                }

                // JSON dump
                {
                    std::error_code ec;
                    llvm::raw_fd_ostream out(prefix + ".json", ec,
                                             llvm::sys::fs::OF_Text);
                    if (!ec) {
                        llvm::json::OStream jos(out, 2);
                        emitJSON(root, jos, &ctx);
                        out << "\n";
                    }
                }
            }
        }

        return true;
    }

} // namespace patchestry::ast
