/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Dialect/Contracts/ContractsDialect.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace contracts;

Attribute TargetAttr::parse(AsmParser &p, Type) {
    if (failed(p.parseLess())) {
        return {};
    }

    StringRef kw;
    if (failed(p.parseKeyword(&kw))) {
        return {};
    }

    TargetKind kind = TargetKind::ReturnValue;
    uint64_t idx    = 0;
    FlatSymbolRefAttr sym;

    if (kw == "return_value") {
        kind = TargetKind::ReturnValue;
    } else if (kw == "symbol") {
        kind = TargetKind::Symbol;
        if (failed(p.parseOptionalKeyword("@"))) {
            return (p.emitError(p.getCurrentLocation())
                    << "expected '@' followed by symbol name"),
                   Attribute();
        }

        StringAttr symName;
        if (failed(p.parseSymbolName(symName))) {
            return {};
        }
        sym = FlatSymbolRefAttr::get(symName);
    } else if (kw.consume_front("arg")) {
        kind = TargetKind::Arg;
        if (failed(p.parseInteger(idx))) {
            return {};
        }
    } else {
        p.emitError(p.getCurrentLocation()) << "expected 'return_value', 'symbol', or 'arg<N>'";
        return {};
    }

    if (failed(p.parseGreater())) {
        return {};
    }

    return TargetAttr::get(
        p.getContext(), kind, (kind == TargetKind::Arg) ? idx : 0,
        (kind == TargetKind::Symbol) ? sym : FlatSymbolRefAttr()
    );
}

void TargetAttr::print(AsmPrinter &printer) const {
    printer << "<";
    switch (getKind()) {
        case TargetKind::ReturnValue:
            printer << "return_value";
            break;
        case TargetKind::Symbol:
            printer << "symbol ";
            if (getSymbol()) {
                printer.printAttributeWithoutType(getSymbol());
            }
            break;
        case TargetKind::Arg:
            printer << "arg";
            if (getIndex() >= 0) {
                printer << getIndex();
            }
            break;
    }
    printer << ">";
}
