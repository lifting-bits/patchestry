/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>

namespace patchestry {
    using mcontext_t = mlir::MLIRContext;

    using mlir_operation = mlir::Operation *;
    using mlir_type = mlir::Type;
    using mlir_value = mlir::Value;

    using mlir_builder = mlir::OpBuilder;

    using loc_t = mlir::Location;

} // namespace patchestry
