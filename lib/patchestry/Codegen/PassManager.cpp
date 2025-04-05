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

    void PassManagerBuilder::add_passes(const std::vector< std::string > &steps) {
        build_operation_map(steps);
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
