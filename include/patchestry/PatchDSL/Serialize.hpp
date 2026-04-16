/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <mlir/IR/MLIRContext.h>

#include <patchestry/YAML/ConfigurationFile.hpp>

namespace patchestry::patchdsl {

    /// Serializes a Configuration as MLIR bytecode to `out_path`. The
    /// bytecode payload is a single empty `builtin.module` whose top-level
    /// attribute dictionary carries the whole Configuration under
    /// `patchestry.dsl.*` keys.
    llvm::Error WritePatchmod(
        const patchestry::passes::Configuration &cfg,
        llvm::StringRef out_path,
        mlir::MLIRContext &context
    );

    /// Reads a `.patchmod` written by WritePatchmod and reconstructs the
    /// Configuration by unpacking the `patchestry.dsl.*` attributes.
    llvm::Expected< patchestry::passes::Configuration >
    ReadPatchmod(llvm::StringRef in_path, mlir::MLIRContext &context);

} // namespace patchestry::patchdsl
