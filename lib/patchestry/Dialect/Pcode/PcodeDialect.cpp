/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Dialect/Pcode/PcodeDialect.hpp"
#include "patchestry/Dialect/Pcode/PcodeOps.hpp"

namespace patchestry::pc {
    void PcodeDialect::initialize() {
        registerTypes();
        addOperations<
#define GET_OP_LIST
#include "patchestry/Dialect/Pcode/Pcode.cpp.inc"
            >();
    }

    using DialectParser  = mlir::AsmParser;
    using DialectPrinter = mlir::AsmPrinter;

} // namespace patchestry::pc

#include "patchestry/Dialect/Pcode/PcodeDialect.cpp.inc"
