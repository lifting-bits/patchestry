/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Util/Warnings.hpp"

PATCHESTRY_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
PATCHESTRY_UNRELAX_WARNINGS

#include "patchestry/Dialect/Pcode/PcodeDialect.hpp"
#include "patchestry/Dialect/Pcode/PcodeTypes.hpp"

namespace patchestry::pc {
    void PcodeDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "patchestry/Dialect/Pcode/PcodeTypes.cpp.inc"
        >();
    }
}

#define GET_TYPEDEF_CLASSES
#include "patchestry/Dialect/Pcode/PcodeTypes.cpp.inc"
