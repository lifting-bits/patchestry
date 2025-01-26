/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <fstream>
#include <sstream>

#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <vast/Util/Common.hpp>

#include <patchestry/Codegen/Serializer.hpp>

namespace patchestry::codegen {

    bool Serializer::serializeToFile(vast::mlir_module mod, const std::string &filename) {
        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile) {
            return false;
        }

        std::string module_string = Serializer::convertModuleToString(mod);
        outfile << module_string;
        outfile.close();
        return true;
    }

    vast::mlir_module
    Serializer::deserializeFromFile(mlir::MLIRContext *mctx, const std::string &filename) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile) {
            return nullptr;
        }

        std::stringstream buffer;
        buffer << infile.rdbuf();
        std::string module_string = buffer.str();
        return Serializer::parseModuleFromString(mctx, module_string);
    }

    std::string Serializer::convertModuleToString(vast::mlir_module mod) {
        std::string module_string;
        llvm::raw_string_ostream os(module_string);
        auto flags = mlir::OpPrintingFlags();
        flags.enableDebugInfo(true, false);
        mod.print(os, flags);
        return module_string;
    }

    mlir::ModuleOp Serializer::parseModuleFromString(
        mlir::MLIRContext *mctx, const std::string &module_string
    ) {
        llvm::SourceMgr sm;
        llvm::SMDiagnostic error;
        (void) mctx;
        (void) module_string;

        return nullptr;

        // Parse the module
        // return mlir::parseSourceString< vast::owning_mlir_module_ref >(module_string, mctx);
    }
} // namespace patchestry::codegen
