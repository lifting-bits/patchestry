/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>

namespace contracts {
    class ContractsDialect : public mlir::Dialect
    {
      public:
        explicit ContractsDialect(mlir::MLIRContext *context);

        static constexpr mlir::StringRef getDialectNamespace() { return "contracts"; }

        void initialize();
    };
} // namespace contracts

// Pull in the dialect definition.
#include "patchestry/Dialect/Contracts/ContractsDialect.h.inc"

// Pull in enum definitions
#include "patchestry/Dialect/Contracts/contracts/ContractsEnums.h.inc"

// Forward declarations for attributes (before including the generated code)
namespace contracts {
    class varrefAttr;
    class iconstAttr;
    class cmpAttr;
    class all_ofAttr;
    class any_ofAttr;
    class staticAttr;

    // Provide aliases matching the generated C++ accessor names.
    using VarRefAttr         = varrefAttr;
    using ConstIntAttr       = iconstAttr;
    using CmpClauseAttr      = cmpAttr;
    using AllOfAttr          = all_ofAttr;
    using AnyOfAttr          = any_ofAttr;
    using StaticContractAttr = staticAttr;
} // namespace contracts

// Pull in attribute definitions
#define GET_ATTRDEF_CLASSES
#include "patchestry/Dialect/Contracts/contracts/ContractsAttrs.h.inc"
