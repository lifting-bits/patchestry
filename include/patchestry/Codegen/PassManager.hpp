/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include "patchestry/Util/Log.hpp"
#include <llvm/Support/LogicalResult.h>
#include <memory>

#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <unordered_map>

namespace patchestry::codegen {

    inline std::string to_string(mlir::Pass *pass) {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        pass->printAsTextualPipeline(os);
        return os.str();
    }

    class PassManagerBuilder
    {
      public:
        explicit PassManagerBuilder(mlir::MLIRContext *context) : mctx(context) {
            pm = std::make_unique< mlir::PassManager >(context);
        }

        void build_operation_map(const std::vector< std::string > &anchors) {
            mlir::PassManager parser_pm(mctx);
            for (const auto &anchor : anchors) {
                if (llvm::failed(mlir::parsePassPipeline(anchor, parser_pm))) {
                    LOG(ERROR) << "Failed to parse anchor name";
                }
            }

            for (auto &p : parser_pm.getPasses()) {
                operation_names.emplace(
                    std::pair< std::string, std::string >(to_string(&p), p.getOpName()->str())
                );
            }
        }

        void add_passes(const std::vector< std::string > &anchors) {
            build_operation_map(anchors);
            for (const auto &step : anchors) {
                auto operation_name = operation_names.at(step);
                llvm::errs() << "Operation name for step: " << step << " -> " << operation_name
                             << "\n";
                if (operation_name == "core.module") {
                    auto &nested_pm = pm->nest(operation_name);
                    if (failed(mlir::parsePassPipeline(step, nested_pm))) {
                        llvm::errs() << "Failed to parse pipeline " << step << " for op "
                                     << operation_name << "\n";
                    }
                } else if (operation_name == "builtin.module") {
                    if (failed(mlir::parsePassPipeline(step, *pm))) {
                        llvm::errs() << "Failed to parse pipeline " << step << " for op "
                                     << operation_name << "\n";
                    }
                }
            }
        }

        std::unique_ptr< mlir::PassManager > build() { return std::move(pm); }

      private:
        mlir::MLIRContext *mctx;
        std::unique_ptr< mlir::PassManager > pm;
        std::unordered_map< std::string, std::string > operation_names;
    };
} // namespace patchestry::codegen
