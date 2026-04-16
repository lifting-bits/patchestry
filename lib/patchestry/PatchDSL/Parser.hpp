/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <patchestry/PatchDSL/AST.hpp>

namespace patchestry::patchdsl {

    llvm::Expected< std::unique_ptr< AST > >
    ParseSource(llvm::StringRef source, llvm::StringRef filename);

} // namespace patchestry::patchdsl
