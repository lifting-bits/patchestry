/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <fstream>

#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <patchestry/Codegen/Serializer.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::codegen {

    bool Serializer::serializeToFile(mlir::ModuleOp mod, const std::string &filename) {
        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile) {
            return false;
        }

        std::string module_string = Serializer::convertModuleToString(mod);
        outfile << module_string;
        outfile.close();
        return true;
    }

    mlir::ModuleOp Serializer::
        deserializeFromFile(mlir::MLIRContext * /*unused*/, const std::string & /*unused*/) {
        UNIMPLEMENTED("not implemented"); // NOLINT
        return {};
    }

    std::string Serializer::convertModuleToString(mlir::ModuleOp mod) {
        std::string module_string;
        llvm::raw_string_ostream os(module_string);
        auto flags = mlir::OpPrintingFlags();
        flags.enableDebugInfo(true, false);
        mod.print(os, flags);
        return module_string;
    }
} // namespace patchestry::codegen
