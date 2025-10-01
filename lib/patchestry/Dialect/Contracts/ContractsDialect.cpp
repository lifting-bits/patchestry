/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Dialect/Contracts/ContractsDialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// Pull in enum definitions
#include "contracts/ContractsEnums.cpp.inc"

// Pull in attribute definitions
#define GET_ATTRDEF_CLASSES
#include "contracts/ContractsAttrs.cpp.inc"

using namespace mlir;
using namespace contracts;

ContractsDialect::ContractsDialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, mlir::TypeID::get< ContractsDialect >()) {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "contracts/ContractsAttrs.h.inc"
        >();
}

void ContractsDialect::initialize() {
    // Nothing else for attributes-only initial bring-up
}
