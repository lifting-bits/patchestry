/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <system_error>

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Dialect/Contracts/ContractsDialect.hpp>
#include <patchestry/Passes/InstrumentationPass.hpp>
#include <patchestry/Util/Log.hpp>
#include <patchestry/Util/Options.hpp>

namespace patchestry::cl {
    namespace cl = llvm::cl;

    static cl::OptionCategory category("Patch IR Instrumentation Options"
    ); // NOLINT(cert-err58-cpp)

    const cl::opt< std::string > input_filename( // NOLINT(cert-err58-cpp)
        llvm::cl::Positional, llvm::cl::desc("<input CIR file to patch>"), llvm::cl::init("-"),
        cl::cat(category)
    );

    const cl::opt< std::string > output_filename( // NOLINT(cert-err58-cpp)
        "o", llvm::cl::desc("Output filename for the patched CIR"), llvm::cl::value_desc("filename"),
        llvm::cl::init("-"), cl::cat(category)
    );

    const cl::opt< std::string > spec_filename( // NOLINT(cert-err58-cpp)
        "spec", llvm::cl::desc("Specification file for patch placement (in YAML)"),
        llvm::cl::value_desc("filename"), llvm::cl::cat(category)
    );

    const cl::opt< bool > enable_instrumentation( // NOLINT(cert-err58-cpp)
        "enable-instrumentation", llvm::cl::desc("Enable instrumentation passes"),
        llvm::cl::init(true), llvm::cl::cat(category)
    );

    const cl::opt< bool > enable_inlining( // NOLINT(cert-err58-cpp)
        "enable-inlining", llvm::cl::desc("Enable inlining of patch functions"),
        llvm::cl::init(false), llvm::cl::cat(category)
    );

    const cl::opt< std::string > patch_map_file( // NOLINT(cert-err58-cpp)
        "emit-patch-map",
        llvm::cl::desc("Write a JSON patch-location map (applied patch -> binary "
                       "address) to this file"),
        llvm::cl::value_desc("filename"), llvm::cl::init(""), cl::cat(category)
    );

} // namespace patchestry::cl

using namespace patchestry::cl;

namespace patchestry::instrumentation {

    static mlir::LogicalResult run(mlir::MLIRContext &context) {
        auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(input_filename.getValue());
        if (auto err = file_or_err.getError()) {
            LOG(ERROR) << "Error opening file: " << input_filename << "\n";
            return mlir::failure();
        }

        llvm::SourceMgr sm;
        sm.AddNewSourceBuffer(std::move(*file_or_err), llvm::SMLoc());
        auto module = mlir::parseSourceFile< mlir::ModuleOp >(sm, &context);
        if (!module) {
            LOG(ERROR) << "Error parsing mlir module\n";
            return mlir::failure();
        }

        // Defer failure return until after writing output so the rolled-back
        // CIR is still inspectable; exit code stays non-zero. #244
        bool pass_failed = false;
        if (enable_instrumentation.getValue()) {
            patchestry::passes::InstrumentationOptions inline_options = {
                enable_inlining.getValue(), patch_map_file.getValue()
            };
            mlir::PassManager pm(&context);
            pm.addPass(patchestry::passes::CreateInstrumentationPass(
                spec_filename.getValue(), inline_options
            ));
            if (mlir::failed(pm.run(*module))) {
                LOG(ERROR) << "Failed to run instrumentation passes\n";
                pass_failed = true;
            }
        }

        std::error_code ec;
        llvm::raw_fd_ostream os(output_filename, ec, llvm::sys::fs::OF_None);
        if (ec) {
            if (ec.value() == ENOENT) {
                LOG(ERROR) << "Error: Cannot open " << output_filename << " - parent directory does not exist\n";
            } else {
                LOG(ERROR) << "Error opening " << output_filename << ": " << ec.message() << "\n";
            }
            return llvm::failure();
        }
        // Print with debug info enabled so the Ghidra address-key MLIR
        // locations (e.g. loc("ram:00022ff8:28:0")) survive into the patched
        // CIR text and downstream into the lowered LLVM IR. Mirrors the decomp
        // serializer (lib/patchestry/Codegen/Serializer.cpp).
        auto print_flags = mlir::OpPrintingFlags();
        print_flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
        module->print(os, print_flags);
        os.flush();
        return pass_failed ? mlir::failure() : mlir::success();
    }

} // namespace patchestry::instrumentation

int main(int argc, char **argv) {
    llvm::InitLLVM llvm_init(argc, argv);

    llvm::cl::HideUnrelatedOptions(patchestry::cl::category);
    llvm::cl::ParseCommandLineOptions(argc, argv, "Patch IR Instrumentation Driver");

    if (enable_instrumentation.getValue() && spec_filename.getValue().empty()) {
        LOG(ERROR) << "--spec is required when --enable-instrumentation is set\n";
        return EXIT_FAILURE;
    }

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert< mlir::DLTIDialect, mlir::func::FuncDialect >();

    registry.insert< cir::CIRDialect >();
    registry.insert< ::contracts::ContractsDialect >();

    mlir::MLIRContext context;
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    return mlir::failed(patchestry::instrumentation::run(context)) ? EXIT_FAILURE
                                                                   : EXIT_SUCCESS;
}
