/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/Codegen/PassManager.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::codegen {
    namespace {

        inline std::string to_string(mlir::Pass *pass) { // NOLINT
            std::string buffer;
            llvm::raw_string_ostream os(buffer);
            pass->printAsTextualPipeline(os);
            return os.str();
        }
    } // namespace

    std::vector< std::string > PassManagerBuilder::list_vast_passes(void) {
        return { "vast-hl-splice-trailing-scopes",
                 "vast-hl-to-hl-builtin",
                 "vast-hl-ude",
                 "vast-hl-dce",
                 "vast-hl-lower-elaborated-types",
                 "vast-hl-lower-typedefs",
                 "vast-hl-lower-enum-refs",
                 "vast-hl-lower-enum-decls",
                 "vast-hl-lower-types",
                 "vast-hl-to-ll-func",
                 "vast-hl-to-ll-cf",
                 "vast-hl-to-ll-geps",
                 "vast-vars-to-cells",
                 "vast-refs-to-ssa",
                 "vast-evict-static-locals",
                 "vast-strip-param-lvalues",
                 "vast-lower-value-categories",
                 "vast-hl-to-lazy-regions",
                 "vast-emit-abi",
                 "vast-lower-abi",
                 "vast-irs-to-llvm",
                 "vast-core-to-llvm" };
    }

    void PassManagerBuilder::add_passes(const std::vector< std::string > &steps) {
        build_operation_map(steps);
        for (const auto &step : steps) {
            auto operation_name = operation_names.at(step);
            LOG(INFO) << "Operation name for step: " << step << " -> " << operation_name
                      << "\n";
            if (operation_name == "core.module") {
                auto &nested_pm = pm->nest(operation_name);
                if (failed(mlir::parsePassPipeline(step, nested_pm))) {
                    LOG(ERROR) << "Failed to parse pipeline " << step << " for op "
                               << operation_name << "\n";
                }
            } else if (operation_name == "builtin.module") {
                if (failed(mlir::parsePassPipeline(step, *pm))) {
                    LOG(ERROR) << "Failed to parse pipeline " << step << " for op "
                               << operation_name << "\n";
                }
            }
        }
    }

    void PassManagerBuilder::build_operation_map(const std::vector< std::string > &steps) {
        mlir::PassManager parser_pm(mctx);
        for (const auto &step : steps) {
            if (llvm::failed(mlir::parsePassPipeline(step, parser_pm))) {
                LOG(ERROR) << "Failed to parse anchor name";
            }
        }

        for (auto &p : parser_pm.getPasses()) {
            operation_names.emplace(
                std::pair< std::string, std::string >(to_string(&p), p.getOpName()->str())
            );
        }
    }

} // namespace patchestry::codegen
