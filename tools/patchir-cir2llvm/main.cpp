/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <cstdlib>
#include <system_error>

#include <clang/Basic/TargetInfo.h>
#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <clang/CIR/LowerToLLVM.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Target/LLVMIR/Import.h>

#include <patchestry/Util/Log.hpp>

namespace {

    const llvm::cl::opt< std::string > input_filename( // NOLINT(cert-err58-cpp)
        llvm::cl::Positional, llvm::cl::desc("<input filename>"), llvm::cl::init("-")
    );

    const llvm::cl::opt< std::string > output_filename( // NOLINT(cert-err58-cpp)
        "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
        llvm::cl::init("-")
    );

    const llvm::cl::opt< std::string > target_triple( // NOLINT(cert-err58-cpp)
        "target", llvm::cl::desc("Specify a target triple when compiling"), llvm::cl::init("")
    );

    const llvm::cl::opt< bool > emit_ll( // NOLINT(cert-err58-cpp)
        "S", llvm::cl::desc("Emit LLVM IR Representation"), llvm::cl::init(false)
    );

} // namespace

namespace cir {
    namespace direct {
        extern void registerCIRDialectTranslation(mlir::DialectRegistry &registry);
    }
} // namespace cir

namespace {
    std::string prepareModuleTriple(mlir::ModuleOp &module) {
        if (!target_triple.empty()) {
            module->setAttr(
                cir::CIRDialect::getTripleAttrName(),
                mlir::StringAttr::get(module.getContext(), target_triple.getValue())
            );
            return target_triple.getValue();
        }
        return {};
    }

    llvm::LogicalResult prepareModuleDataLayout(mlir::ModuleOp module, llvm::StringRef tt) {
        auto *context = module.getContext();

        llvm::Triple triple(tt);
        clang::TargetOptions target_options;
        target_options.Triple = tt;
        // FIXME: AllocateTarget is a big deal. Better make it a global state.
        auto target_info = clang::targets::AllocateTarget(llvm::Triple(tt), target_options);
        if (!target_info) {
            module.emitError() << "error: invalid target triple '" << tt << "'\n";
            return llvm::failure();
        }
        std::string layout_string = target_info->getDataLayoutString();

        context->loadDialect< mlir::DLTIDialect, mlir::LLVM::LLVMDialect >();
        mlir::DataLayoutSpecInterface dl_spec =
            mlir::translateDataLayout(llvm::DataLayout(layout_string), context);
        module->setAttr(
            mlir::DLTIDialect::kDataLayoutAttrName, mlir::cast< mlir::Attribute >(dl_spec)
        );

        return llvm::success();
    }

    llvm::LogicalResult prepareModuleForTranslation(mlir::ModuleOp module) {
        auto mod_triple =
            module->getAttrOfType< mlir::StringAttr >(cir::CIRDialect::getTripleAttrName());
        auto data_layout       = module->getAttr(mlir::DLTIDialect::kDataLayoutAttrName);
        bool has_target_option = target_triple.getNumOccurrences() > 0;

        if (!has_target_option && mod_triple && data_layout) {
            return llvm::success();
        }

        std::string triple;
        if (!has_target_option && mod_triple) {
            triple = mod_triple.getValue();
        } else {
            triple = prepareModuleTriple(module);
        }

        return prepareModuleDataLayout(module, triple);
    }

    llvm::LogicalResult
    writeBitcodeToFile(llvm::Module &module, llvm::StringRef output_filename) {
        std::error_code ec;
        llvm::raw_fd_ostream os(output_filename, ec, llvm::sys::fs::OF_None);
        if (ec) {
            LOG(ERROR) << "Error opening " << output_filename << ": " << ec.message() << "\n";
            return llvm::failure();
        }
        if (emit_ll) {
            module.print(os, nullptr);
        } else {
            llvm::WriteBitcodeToFile(module, os);
        }
        return llvm::success();
    }
} // namespace

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);
    llvm::cl::ParseCommandLineOptions(argc, argv, "patchir-cir2llvm driver\n");

    mlir::DialectRegistry registry;
    registry.insert< mlir::DLTIDialect, mlir::func::FuncDialect >();

    mlir::registerAllDialects(registry);
    mlir::registerAllToLLVMIRTranslations(registry);

    registry.insert< cir::CIRDialect >();
    cir::direct::registerCIRDialectTranslation(registry);

    mlir::MLIRContext context(registry);
    auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(input_filename);
    if (auto err = file_or_err.getError()) {
        LOG(ERROR) << "Error opening file: " << input_filename << "\n";
        return EXIT_FAILURE;
    }

    auto module =
        mlir::parseSourceString< mlir::ModuleOp >(file_or_err.get()->getBuffer(), &context);
    if (!module) {
        LOG(ERROR) << "Error parsing mlir module\n";
        return EXIT_FAILURE;
    }

    if (mlir::failed(prepareModuleForTranslation(module.get()))) {
        return EXIT_FAILURE;
    }

    llvm::LLVMContext llvm_context;
    auto llvm_module = cir::direct::lowerDirectlyFromCIRToLLVMIR(*module, llvm_context);
    if (!llvm_module) {
        LOG(ERROR) << "Failed to lower cir to llvm\n";
        return EXIT_FAILURE;
    }
    auto err = writeBitcodeToFile(*llvm_module, output_filename);
    if (mlir::failed(err)) {
        LOG(ERROR) << "Failed to write bitcode to file\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
