/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Dialect/Contracts/ContractsDialect.hpp"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace contracts;

// Verify: exactly one of rhsVar or rhsConst
static LogicalResult verifyCmpClauseAttr(
    CmpKind pred, VarRefAttr lhs, VarRefAttr rhsVar, ConstIntAttr rhsConst,
    function_ref< InFlightDiagnostic() > emitError
) {
    bool hasVar   = static_cast< bool >(rhsVar);
    bool hasConst = static_cast< bool >(rhsConst);
    if (hasVar == hasConst) {
        return emitError() << "`cmp` needs exactly one of rhsVar or rhsConst";
    }
    return success();
}

LogicalResult cmpAttr::verify(
    function_ref< InFlightDiagnostic() > emitError, CmpKind pred, VarRefAttr lhs,
    VarRefAttr rhsVar, ConstIntAttr rhsConst
) {
    return verifyCmpClauseAttr(pred, lhs, rhsVar, rhsConst, emitError);
}

LogicalResult staticAttr::verify(
    function_ref< InFlightDiagnostic() > emitError, mlir::StringAttr message,
    mlir::Attribute expr
) {
    if (!llvm::isa< cmpAttr, all_ofAttr, any_ofAttr >(expr)) {
        return emitError() << "expr must be cmp/all_of/any_of";
    }
    if (auto c = llvm::dyn_cast< cmpAttr >(expr)) {
        if (failed(verifyCmpClauseAttr(
                c.getPred(), c.getLhs(), c.getRhsVar(), c.getRhsConst(), emitError
            )))
        {
            return failure();
        }
    }
    return success();
}
