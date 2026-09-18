/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <patchestry/AST/LiftOptions.hpp>
#include <patchestry/AST/TranslationUnit.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>

namespace patchestry::ast {

    /// Lift a deserialized P-Code program into a Clang translation unit for
    /// the program's target.  The unit is returned even when lifting reported
    /// diagnostics errors, so callers can still print it; check
    /// `TranslationUnit::has_errors` before lowering.  Null (after logging)
    /// only when the clang frontend could not be set up.
    std::unique_ptr< TranslationUnit >
    LiftProgram(ghidra::Program &program, const LiftOptions &options);

} // namespace patchestry::ast
