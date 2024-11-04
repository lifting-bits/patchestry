/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/Ghidra/PcodeTranslation.hpp>

#include <patchestry/Dialect/Pcode/PcodeDialect.hpp>
#include <patchestry/Dialect/Pcode/Deserialize.hpp>

#include <patchestry/Util/Common.hpp>

#include <mlir/IR/OwningOpRef.h>
#include <mlir/Tools/mlir-translate/Translation.h>

#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

namespace patchestry::ghidra {

    static mlir::OwningOpRef< mlir_operation > deserialize(
        const llvm::MemoryBuffer *buffer, mcontext_t *mctx
    ) {
        mctx->loadAllAvailableDialects();

        auto json = llvm::json::parse(buffer->getBuffer());
        if (!json) {
            mlir::emitError(mlir::UnknownLoc::get(mctx), "failed to parse PCode JSON: ") << toString(json.takeError());
        }
        return pc::deserialize(*json->getAsObject(), mctx);
    }

    void register_pcode_translation() {
        mlir::TranslateToMLIRRegistration(
            "deserialize-pcode", "translate Ghidra Pcode JSON into Patchestry's Pcode dialect",
            [] (llvm::SourceMgr &smgr, mcontext_t *mctx) {
                assert(smgr.getNumBuffers() == 1 && "expected one buffer");
                return deserialize(smgr.getMemoryBuffer(smgr.getMainFileID()), mctx);
            },
            [] (mlir::DialectRegistry &registry) {
                registry.insert< patchestry::pc::PcodeDialect >();
            }
        );
    }

} // namespace patchestry::ghidra
