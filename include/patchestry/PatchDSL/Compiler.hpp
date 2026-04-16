/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <patchestry/PatchDSL/AST.hpp>

namespace patchestry::patchdsl {

    struct CompilerOptions {
        // Additional search roots for imports (the -I flag).
        std::vector< std::string > import_paths;
    };

    /// Parses a `.patch` file into an AST. Returns an llvm::Error on any
    /// parse or type-check failure.
    llvm::Expected< std::unique_ptr< AST > >
    ParseFile(llvm::StringRef path, const CompilerOptions &opts = {});

} // namespace patchestry::patchdsl
