/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Util/Warnings.hpp"

PATCHESTRY_RELAX_WARNINGS
#include <mlir/IR/OpImplementation.h>
PATCHESTRY_UNRELAX_WARNINGS

#include "patchestry/Dialect/Pcode/PcodeOps.hpp"

PATCHESTRY_RELAX_WARNINGS
#define GET_OP_CLASSES
#include "patchestry/Dialect/Pcode/Pcode.cpp.inc"

auto patchestry::pc::ConstOp::fold([[maybe_unused]] FoldAdaptor adaptor) -> mlir::OpFoldResult {
    return mlir::UnitAttr::get(this->getContext());
}

PATCHESTRY_UNRELAX_WARNINGS
