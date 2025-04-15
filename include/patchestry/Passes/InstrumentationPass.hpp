/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <mlir/IR/Value.h>
#include <string>

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <clang/Frontend/CompilerInstance.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassOptions.h>
#include <mlir/Support/LLVM.h>

#include <patchestry/Passes/PatchSpec.hpp>

namespace patchestry::passes {

    void registerInstrumentationPasses(std::string spec_file);

    std::unique_ptr< mlir::Pass > createInstrumentationPass(std::string spec_file);

    class InstrumentationPass
        : public mlir::PassWrapper< InstrumentationPass, mlir::OperationPass< mlir::ModuleOp > >
    {
        std::string spec_file;
        std::optional< PatchSpec > config;

      public:
        explicit InstrumentationPass(std::string spec);

        void runOnOperation() final;

        void instrument_function_calls(cir::FuncOp func);

      private:
        void prepare_call_arguments(
            mlir::OpBuilder &builder, cir::CallOp op, cir::FuncOp patch_func,
            const PatchOperation &patch, llvm::SmallVector< mlir::Value > &args
        );

        void apply_before_patch(
            cir::CallOp op, const PatchOperation &patch, mlir::ModuleOp patch_module
        );
        void apply_after_patch(
            cir::CallOp op, const PatchOperation &patch, mlir::ModuleOp patch_module
        );
        void wrap_around_patch(
            cir::CallOp op, const PatchOperation &patch, mlir::ModuleOp patch_module
        );

        mlir::OwningOpRef< mlir::ModuleOp >
        load_patch_module(mlir::MLIRContext &ctx, const std::string &patch_string);

        mlir::LogicalResult merge_module_symbol(
            mlir::ModuleOp dest, mlir::ModuleOp src, const std::string &symbol_name
        );
    };

} // namespace patchestry::passes
