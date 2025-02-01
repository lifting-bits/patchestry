/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#define VAST_ENABLE_EXCEPTIONS
#include <vast/Util/Warnings.hpp>

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <mlir/IR/Attributes.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

#define GAP_ENABLE_COROUTINES
#include <vast/Conversion/Passes.hpp>
#include <vast/Dialect/Dialects.hpp>
#include <vast/Dialect/HighLevel/Passes.hpp>
#include <vast/Dialect/LowLevel/Passes.hpp>

#include <patchestry/Codegen/Codegen.hpp>

namespace patchestry::codegen {
    MLIRRegistry::MLIRRegistry(mlir::DialectRegistry &registry) {
        vast::registerAllDialects(registry);
        mlir::registerAllDialects(registry);
        vast::registerConversionPasses();
        mlir::LLVM::registerLLVMPasses();
        vast::hl::registerHighLevelPasses();
    }

    CodegenInitializer::CodegenInitializer(int /*unused*/)
        : registry_initializer(registry), ctx(registry, mlir::MLIRContext::Threading::ENABLED) {
        ctx.disableMultithreading();
        ctx.loadAllAvailableDialects();
        ctx.enableMultithreading();
    }

    CodegenInitializer::~CodegenInitializer(void) { ctx.disableMultithreading(); }
} // namespace patchestry::codegen
