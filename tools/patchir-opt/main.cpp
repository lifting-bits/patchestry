/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Passes/InstrumentationPass.hpp>
#include <patchestry/Util/Log.hpp>
#include <patchestry/Util/Options.hpp>

int main(int argc, char **argv) {
    mlir::LLVM::registerLLVMPasses();

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert< cir::CIRDialect >();

    mlir::registerAllToLLVMIRTranslations(registry);

    patchestry::passes::registerInstrumentationPasses();

    return failed(mlir::MlirOptMain(argc, argv, "Patch IR Instrumentation Driver\n", registry));
}
