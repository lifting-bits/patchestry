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
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>

VAST_UNRELAX_WARNINGS

#define GAP_ENABLE_COROUTINES

#include <vast/CodeGen/AttrVisitorProxy.hpp>
#include <vast/CodeGen/CodeGenBuilder.hpp>
#include <vast/CodeGen/CodeGenDriver.hpp>
#include <vast/CodeGen/DefaultCodeGenPolicy.hpp>
#include <vast/CodeGen/DefaultMetaGenerator.hpp>
#include <vast/Conversion/Passes.hpp>
#include <vast/Dialect/Dialects.hpp>
#include <vast/Dialect/HighLevel/Passes.hpp>
#include <vast/Dialect/LowLevel/Passes.hpp>
#include <vast/Dialect/Meta/MetaAttributes.hpp>

#include <patchestry/Codegen/Codegen.hpp>

namespace patchestry::codegen {

    MLIRRegistryInitializer::MLIRRegistryInitializer(mlir::DialectRegistry &registry) {
        mlir::LLVM::registerLLVMPasses();
        vast::hl::registerHighLevelPasses();
        vast::registerConversionPasses();
        vast::registerAllDialects(registry);
        mlir::registerAllDialects(registry);
    }

    CodegenInitializer::CodegenInitializer(int /*unused*/)
        : registry_initializer(registry), ctx(registry, mlir::MLIRContext::Threading::ENABLED) {
        ctx.disableMultithreading();
        ctx.loadAllAvailableDialects();
        ctx.enableMultithreading();
    }

    CodegenInitializer::~CodegenInitializer(void) { ctx.disableMultithreading(); }

} // namespace patchestry::codegen
