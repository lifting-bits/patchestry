/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <patchestry/PatchDSL/AST.hpp>
#include <patchestry/YAML/ConfigurationFile.hpp>

namespace patchestry::patchdsl {

    /// Lower a parsed AST to the downstream `passes::Configuration` consumed
    /// by InstrumentationPass. Returns an error if the AST contains
    /// constructs the Phase 3 lowering cannot yet represent.
    llvm::Expected< patchestry::passes::Configuration >
    Lower(const AST &ast, llvm::StringRef source_path);

} // namespace patchestry::patchdsl
