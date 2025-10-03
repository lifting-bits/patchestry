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

        // Attribute parsing and printing hooks
        mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser, mlir::Type type) const override;
        void printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter &os) const override;
    };
} // namespace contracts

// Pull in the dialect definition.
#include "patchestry/Dialect/Contracts/contracts/ContractsDialect.h.inc"

// Pull in enum definitions
#include "patchestry/Dialect/Contracts/contracts/ContractsEnums.h.inc"

// Forward declarations now handled by generated code with correct cppClassName
namespace contracts {
} // namespace contracts

// Pull in attribute definitions
#define GET_ATTRDEF_CLASSES
#include "patchestry/Dialect/Contracts/contracts/ContractsAttrs.h.inc"
