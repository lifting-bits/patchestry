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
    } else if (kw.starts_with("arg")) {
        kind = TargetKind::Arg;
        auto numStr = kw.drop_front(3); // Skip "arg"
        if (numStr.getAsInteger(10, idx)) {
            p.emitError(p.getCurrentLocation()) << "Invalid argument index";
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

Attribute PredicateAttr::parse(AsmParser &p, Type) {
    if (failed(p.parseLess())) {
        return {};
    }

    StringRef kindStr;
    if (failed(p.parseKeyword(&kindStr))) {
        return {};
    }

    PredicateKind kind;
    if (kindStr == "nonnull") {
        kind = PredicateKind::nonnull;
    } else if (kindStr == "relation") {
        kind = PredicateKind::relation;
    } else if (kindStr == "alignment") {
        kind = PredicateKind::alignment;
    } else if (kindStr == "expr") {
        kind = PredicateKind::expr;
    } else if (kindStr == "range") {
        kind = PredicateKind::range;
    } else {
        p.emitError(p.getCurrentLocation())
            << "expected 'nonnull', 'relation', 'alignment', 'expr', or 'range'";
        return {};
    }

    TargetAttr target;
    RelationKind relation = RelationKind::none;
    Attribute value;
    ContractAlignmentAttr align;
    StringAttr expr;
    ContractRangeAttr range;

    // Parse optional fields
    while (succeeded(p.parseOptionalComma())) {
        StringRef fieldName;
        if (failed(p.parseKeyword(&fieldName))) {
            return {};
        }

        if (failed(p.parseEqual())) {
            return {};
        }

        if (fieldName == "target") {
            if (failed(p.parseAttribute(target))) {
                return {};
            }
        } else if (fieldName == "relation") {
            StringRef relStr;
            if (failed(p.parseKeyword(&relStr))) {
                return {};
            }
            if (relStr == "none") {
                relation = RelationKind::none;
            } else if (relStr == "lt") {
                relation = RelationKind::lt;
            } else if (relStr == "lte") {
                relation = RelationKind::lte;
            } else if (relStr == "gt") {
                relation = RelationKind::gt;
            } else if (relStr == "gte") {
                relation = RelationKind::gte;
            } else if (relStr == "eq") {
                relation = RelationKind::eq;
            } else if (relStr == "neq") {
                relation = RelationKind::neq;
            } else {
                p.emitError(p.getCurrentLocation()) << "invalid relation kind";
                return {};
            }
        } else if (fieldName == "value") {
            if (failed(p.parseAttribute(value))) {
                return {};
            }
        } else if (fieldName == "align") {
            if (failed(p.parseAttribute(align))) {
                return {};
            }
        } else if (fieldName == "expr") {
            if (failed(p.parseAttribute(expr))) {
                return {};
            }
        } else if (fieldName == "range") {
            if (failed(p.parseAttribute(range))) {
                return {};
            }
        } else {
            p.emitError(p.getCurrentLocation()) << "unknown field: " << fieldName;
            return {};
        }
    }

    if (failed(p.parseGreater())) {
        return {};
    }

    return PredicateAttr::get(p.getContext(), kind, target, relation, value, align, expr, range);
}

void PredicateAttr::print(AsmPrinter &printer) const {
    printer << "<" << stringifyPredicateKind(getKind());

    if (getTarget()) {
        printer << ", target = ";
        printer.printAttribute(getTarget());
    }

    if (getRelation() != RelationKind::none) {
        printer << ", relation = " << stringifyRelationKind(getRelation());
    }

    if (getValue()) {
        printer << ", value = ";
        printer.printAttribute(getValue());
    }

    if (getAlign()) {
        printer << ", align = ";
        printer.printAttribute(getAlign());
    }

    if (getExpr()) {
        printer << ", expr = ";
        printer.printAttribute(getExpr());
    }

    if (getRange()) {
        printer << ", range = ";
        printer.printAttribute(getRange());
    }

    printer << ">";
}

Attribute StaticContractAttr::parse(AsmParser &p, Type) {
    if (failed(p.parseLess())) {
        return {};
    }

    // Parse "pre"
    if (failed(p.parseKeyword("pre"))) {
        return {};
    }
    if (failed(p.parseEqual())) {
        return {};
    }

    // Parse preconditions array
    SmallVector<Attribute> preconditions;
    if (failed(p.parseLSquare())) {
        return {};
    }

    // Parse optional preconditions
    if (failed(p.parseOptionalRSquare())) {
        do {
            Attribute precond;
            if (failed(p.parseAttribute(precond))) {
                return {};
            }
            preconditions.push_back(precond);
        } while (succeeded(p.parseOptionalComma()));

        if (failed(p.parseRSquare())) {
            return {};
        }
    }

    // Parse comma separator
    if (failed(p.parseComma())) {
        return {};
    }

    // Parse "post"
    if (failed(p.parseKeyword("post"))) {
        return {};
    }
    if (failed(p.parseEqual())) {
        return {};
    }

    // Parse postconditions array
    SmallVector<Attribute> postconditions;
    if (failed(p.parseLSquare())) {
        return {};
    }

    // Parse optional postconditions
    if (failed(p.parseOptionalRSquare())) {
        do {
            Attribute postcond;
            if (failed(p.parseAttribute(postcond))) {
                return {};
            }
            postconditions.push_back(postcond);
        } while (succeeded(p.parseOptionalComma()));

        if (failed(p.parseRSquare())) {
            return {};
        }
    }

    if (failed(p.parseGreater())) {
        return {};
    }

    return StaticContractAttr::get(p.getContext(), preconditions, postconditions);
}

void StaticContractAttr::print(AsmPrinter &printer) const {
    printer << "<pre = [";

    bool first = true;
    for (auto pre : getPreconditions()) {
        if (!first) {
            printer << ", ";
        }
        printer.printAttribute(pre);
        first = false;
    }

    printer << "], post = [";

    first = true;
    for (auto post : getPostconditions()) {
        if (!first) {
            printer << ", ";
        }
        printer.printAttribute(post);
        first = false;
    }

    printer << "]>";
}
