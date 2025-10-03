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
    initialize();
}

void ContractsDialect::initialize() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "contracts/ContractsAttrs.cpp.inc"
        >();
}

mlir::Attribute ContractsDialect::parseAttribute(mlir::DialectAsmParser &parser, mlir::Type type) const {
    mlir::StringRef mnemonic;
    mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &mnemonic, type, attr);
    if (parseResult.has_value())
        return attr;
    parser.emitError(parser.getNameLoc(), "unknown contracts attribute: ") << mnemonic;
    return {};
}

void ContractsDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter &os) const {
    if (mlir::succeeded(generatedAttributePrinter(attr, os)))
        return;
    os << "<unknown contracts attribute>";
}
