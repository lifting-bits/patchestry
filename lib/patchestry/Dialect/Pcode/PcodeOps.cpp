/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <mlir/IR/OpImplementation.h>

#include "patchestry/Dialect/Pcode/PcodeOps.hpp"

#define GET_OP_CLASSES
#include "patchestry/Dialect/Pcode/Pcode.cpp.inc"

auto patchestry::pc::ConstOp::fold([[maybe_unused]] FoldAdaptor adaptor) -> mlir::OpFoldResult {
    return {};
}
