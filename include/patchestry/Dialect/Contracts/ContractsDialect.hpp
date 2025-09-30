/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BuiltinAttributes.h>

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
#include "contracts/ContractsDialect.h.inc"

// Pull in enum definitions
#include "contracts/ContractsEnums.h.inc"

// Forward declarations for attributes (before including the generated code)
namespace contracts {
    class varrefAttr;
    class iconstAttr;
    class cmpAttr;
    class all_ofAttr;
    class any_ofAttr;
    class staticAttr;

    // Type aliases to match what the generated code expects
    using VarRefAttr = varrefAttr;
    using ConstIntAttr = iconstAttr;
    using CmpClauseAttr = cmpAttr;
    using AllOfAttr = all_ofAttr;
    using AnyOfAttr = any_ofAttr;
    using StaticContractAttr = staticAttr;
}

// Pull in attribute definitions
#define GET_ATTRDEF_CLASSES
#include "contracts/ContractsAttrs.h.inc"