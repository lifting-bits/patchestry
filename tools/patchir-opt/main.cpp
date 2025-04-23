/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <string>
#include <system_error>

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <patchestry/Codegen/Codegen.hpp>
#include <patchestry/Passes/InstrumentationPass.hpp>
#include <patchestry/Util/Log.hpp>
#include <patchestry/Util/Options.hpp>

namespace patchestry::cl {
    namespace cl = llvm::cl;

    static cl::OptionCategory category("Patch IR Instrumentation Options"
    ); // NOLINT(cert-err58-cpp)

    const cl::opt< std::string > input_filename( // NOLINT(cert-err58-cpp)
        llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
        cl::cat(category)
    );

    const cl::opt< std::string > output_filename( // NOLINT(cert-err58-cpp)
        "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
        llvm::cl::init("-"), cl::cat(category)
    );

    const cl::opt< std::string > spec_filename( // NOLINT(cert-err58-cpp)
        "spec", llvm::cl::desc("Specification file for patches"),
        llvm::cl::value_desc("filename"), llvm::cl::cat(category)
    );

    const cl::opt< bool > enable_instrumentation( // NOLINT(cert-err58-cpp)
        "enable-instrumentation", llvm::cl::desc("Enable instrumentation passes"),
        llvm::cl::init(true), llvm::cl::cat(category)
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

        if (enable_instrumentation.getValue()) {
            mlir::PassManager pm(&context);
            pm.addPass(patchestry::passes::createInstrumentationPass(spec_filename.getValue()));
            if (mlir::failed(pm.run(*module))) {
                LOG(ERROR) << "Failed to run instrumentation passes\n";
                return mlir::failure();
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
        module->print(os);
        os.flush();
        return mlir::success();
    }

} // namespace patchestry::instrumentation

int main(int argc, char **argv) {
    llvm::InitLLVM llvm_init(argc, argv);

    llvm::cl::HideUnrelatedOptions(patchestry::cl::category);
    llvm::cl::ParseCommandLineOptions(argc, argv, "Patch IR Instrumentation Driver");

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert< mlir::DLTIDialect, mlir::func::FuncDialect >();

    registry.insert< cir::CIRDialect >();

    mlir::MLIRContext context(registry);

    return mlir::failed(patchestry::instrumentation::run(context)) ? EXIT_FAILURE
                                                                   : EXIT_SUCCESS;
}
