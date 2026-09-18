/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <system_error>

#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <patchestry/Codegen/Serializer.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::codegen {

    namespace {
        template< typename Print >
        bool writeFile(const std::string &filename, Print print) {
            std::error_code ec;
            llvm::raw_fd_ostream out(filename, ec, llvm::sys::fs::OF_None);
            if (ec) {
                LOG(ERROR) << "Failed to write output '" << filename << "': " << ec.message()
                           << "\n";
                return false;
            }
            print(out);
            out.close();
            if (out.has_error()) {
                LOG(ERROR) << "Failed to write output '" << filename
                           << "': " << out.error().message() << "\n";
                out.clear_error();
                return false;
            }
            return true;
        }
    } // namespace

    bool Serializer::SerializeToFile(mlir::ModuleOp mod, const std::string &filename) {
        return writeFile(filename, [&](llvm::raw_ostream &out) {
            auto flags = mlir::OpPrintingFlags();
            flags.enableDebugInfo(true, false);
            mod.print(out, flags);
        });
    }

    bool Serializer::SerializeToFile(llvm::Module *mod, const std::string &filename) {
        return writeFile(filename, [&](llvm::raw_ostream &out) { mod->print(out, nullptr); });
    }

    mlir::ModuleOp Serializer::
        DeserializeFromFile(mlir::MLIRContext * /*unused*/, const std::string & /*unused*/) {
        LOG_FATAL("Serializer::DeserializeFromFile is not implemented.");
        return {};
    }

    std::string Serializer::ConvertModuleToString(mlir::ModuleOp mod) {
        std::string module_string;
        llvm::raw_string_ostream os(module_string);
        auto flags = mlir::OpPrintingFlags();
        flags.enableDebugInfo(true, false);
        mod.print(os, flags);
        return module_string;
    }
} // namespace patchestry::codegen
