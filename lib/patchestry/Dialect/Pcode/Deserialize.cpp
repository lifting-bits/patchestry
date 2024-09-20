/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/Dialect/Pcode/Deserialize.hpp>
#include <patchestry/Dialect/Pcode/PcodeOps.hpp>

namespace patchestry::pc {

    mlir::OwningOpRef< mlir::ModuleOp > deserialize(const json_obj &json, mcontext_t *mctx) {
        // FIXME: use implicit module creation
        auto loc = mlir::UnknownLoc::get(mctx);
        auto mod = mlir::OwningOpRef< mlir::ModuleOp >(mlir::ModuleOp::create(loc));

        deserializer des(mod.get());
        des.process(json);

        return mod;
    }

    void deserializer::process(const json_obj &json) {
        if (auto *functions = json.getArray("functions")) {
            if (functions->empty()) {
                mlir::emitError(bld.getUnknownLoc(), "No function to process!");
                return;
            }

            for (const auto &func_json : *functions) {
                if (const auto *func_obj = func_json.getAsObject()) {
                    process_function(*func_obj);
                } else {
                    mlir::emitError(bld.getUnknownLoc(), "Failed to get function object from json input!");
                }
            }
        } else {
            mlir::emitError(bld.getUnknownLoc(), "Key `functions` is missing or not an array in the JSON input!");
        }
    }

    void deserializer::process_function(const json_obj &json) {
        if (!json.getString("name")) {
            mlir::emitError(bld.getUnknownLoc(), "Function JSON missing 'name' field.");
            return;
        }

        auto _ = insertion_guard(bld);
        auto fn = bld.create< pc::FuncOp >(
            bld.getUnknownLoc(),
            json.getString("name").value()
        );

        bld.setInsertionPointToStart(bld.createBlock(&fn.getBlocks()));
        if (auto blocks = json.getArray("basic_blocks")) {
            for (const auto &block : *blocks) {
                process_block(*block.getAsObject());
            }
        }
    }

    void deserializer::process_block(const json_obj &json) {
        if (!json.getString("label")) {
            mlir::emitError(bld.getUnknownLoc(), "Block JSON missing 'label' field.");
            return;
        }

        auto _ = insertion_guard(bld);
        auto block = bld.create< pc::BlockOp >(
            bld.getUnknownLoc(),
            json.getString("label").value()
        );

        bld.createBlock(&block.getInstructions());

        const auto *insts = json.getArray("instructions");
        if (insts == nullptr) {
            mlir::emitError(bld.getUnknownLoc(), "Block JSON missing 'instructions' field.");
            return;
        }

        for (const auto &inst : *insts) {
            process_instruction(*inst.getAsObject());
        }
    }

    void deserializer::process_instruction(const json_obj &json) {

    }

} // namespace patchestry::pc
