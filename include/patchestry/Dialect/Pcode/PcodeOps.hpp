/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/RegionKindInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

#include "patchestry/Dialect/Pcode/PcodeDialect.hpp"

#define GET_OP_CLASSES
#include "patchestry/Dialect/Pcode/Pcode.h.inc"
