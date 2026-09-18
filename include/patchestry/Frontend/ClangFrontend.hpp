/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/Frontend/CompilerInstance.h>
#include <llvm/ADT/STLFunctionalExtras.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace patchestry::frontend {

    /// How to set up a clang frontend.
    struct FrontendConfig
    {
        /// Target triple, typically derived from a Ghidra language id.
        std::string triple;
    };

    /// LLVM target triple for a Ghidra language id (`arch:endianness:size:variant`)
    /// and architecture name.  Empty, after logging, when the id is malformed.
    std::string ghidraLangToTriple(const std::string &arch, const std::string &lang);

    /// Clang's resource directory, from `CLANG_RESOURCE_DIR` or
    /// `clang --print-resource-dir`.
    std::optional< std::string > getClangResourceDir();

    /// Include paths for patchestry headers and clang builtins.  System and
    /// SDK paths are deliberately excluded: the sources are cross-compiled for
    /// the firmware target and must not see host libc headers.
    std::vector< std::string > getPatchestryIncludePaths();

    /// A CompilerInstance ready for `clang::ParseAST` on `filename`, with the
    /// patchestry diagnostic client installed.  Null (after logging) on setup
    /// failure.
    std::unique_ptr< clang::CompilerInstance >
    createCompilerInstance(const std::string &filename, const FrontendConfig &config);

    /// Builds the ASTConsumer for a synthetic unit once the CompilerInstance
    /// has an ASTContext.  Called before Sema exists.
    using ConsumerFactory =
        llvm::function_ref< std::unique_ptr< clang::ASTConsumer >(clang::CompilerInstance &) >;

    /// A CompilerInstance whose main file is an in-memory placeholder, for
    /// building a translation unit programmatically instead of parsing one
    /// (the P-Code lifter).  Installs the consumer from `make_consumer`, then
    /// creates Sema, so the returned instance is fully wired.  The decompiler's
    /// codegen options are applied (no optimization, no strict return/enum
    /// assumptions).  Null (after logging) on setup failure.
    std::unique_ptr< clang::CompilerInstance > createSyntheticCompilerInstance(
        const FrontendConfig &config, ConsumerFactory make_consumer
    );

} // namespace patchestry::frontend
