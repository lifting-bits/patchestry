/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <memory>
#include <unordered_map>

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>

namespace patchestry::codegen {

    class PassManagerBuilder
    {
      public:
        explicit PassManagerBuilder(mlir::MLIRContext *context) : mctx(context) {
            pm = std::make_unique< mlir::PassManager >(context);
        }

        static std::vector< std::string > list_vast_passes(void); // NOLINT

        void add_passes(const std::vector< std::string > &steps);

        std::unique_ptr< mlir::PassManager > build() { return std::move(pm); }

      private:
        void build_operation_map(const std::vector< std::string > &steps);

        mlir::MLIRContext *mctx;
        std::unique_ptr< mlir::PassManager > pm;
        std::unordered_map< std::string, std::string > operation_names;
    };
} // namespace patchestry::codegen
