/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace patchestry::passes {

    class OperandNameResolver
    {
      public:
        explicit OperandNameResolver(mlir::Operation *root_op);

        std::string get_operand_name(mlir::Value operand) const;

        std::vector< std::string > get_all_operand_names(mlir::Operation *op) const;

        // Check if an operand matches a name pattern
        bool operand_matches_pattern(mlir::Value operand, const std::string &pattern) const;

        void dump_value_names(void);

      private:
        void build_value_names(mlir::Operation *op);

        std::string extract_name_from_defining_op(mlir::Value value) const;

        std::string extract_name_from_global_operation(mlir::Operation *op) const;

        // Map from value to name
        llvm::DenseMap< mlir::Value, std::string > value_names;
        mlir::Operation *root_op;
    };
} // namespace patchestry::passes
