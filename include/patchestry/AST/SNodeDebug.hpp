/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/SNode.hpp>

#include <llvm/Support/raw_ostream.h>

namespace patchestry::ast {

    // Emit Graphviz DOT representation of SNode tree
    void EmitDot(const SNode *node, llvm::raw_ostream &os);

} // namespace patchestry::ast
