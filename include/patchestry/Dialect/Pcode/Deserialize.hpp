/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/Util/Common.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <llvm/Support/JSON.h>

namespace patchestry::pc
{
    mlir::OwningOpRef< mlir::ModuleOp > deserialize(llvm::json::Value const &json, mcontext_t *mctx);
} // namespace patchestry::pc
