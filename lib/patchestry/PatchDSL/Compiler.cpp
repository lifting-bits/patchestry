/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/PatchDSL/Compiler.hpp>

#include <memory>
#include <system_error>

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>

#include "Parser.hpp"

namespace patchestry::patchdsl {

    llvm::Expected< std::unique_ptr< AST > >
    ParseFile(llvm::StringRef path, const CompilerOptions &opts) {
        (void)opts;  // import_paths are resolved later, in Phase 7.

        auto buf_or = llvm::MemoryBuffer::getFile(path);
        if (std::error_code ec = buf_or.getError()) {
            return llvm::createStringError(
                ec, "failed to read `%s`: %s", path.str().c_str(), ec.message().c_str()
            );
        }
        auto buf = std::move(*buf_or);
        return ParseSource(buf->getBuffer(), path);
    }

} // namespace patchestry::patchdsl
