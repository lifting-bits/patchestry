/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/SNode.hpp>
#include <patchestry/AST/CfgBuilder.hpp>

namespace patchestry::ast {

    // Convert a CFG into an initial SNode tree.
    // The result is a flat SSeq of SLabel(SBlock) with SGoto/SIfThenElse for branches.
    SNode *buildSNodeTree(const Cfg &cfg, SNodeFactory &factory);

} // namespace patchestry::ast
