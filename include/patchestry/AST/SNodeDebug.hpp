/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <patchestry/AST/SNode.hpp>

#include <clang/AST/ASTContext.h>

#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

namespace patchestry::ast {

    // Print SNode tree as indented pseudo-C
    void printPseudoC(
        const SNode *node, llvm::raw_ostream &os,
        clang::ASTContext *ctx = nullptr, unsigned indent = 0
    );

    // Emit Graphviz DOT representation of SNode tree
    void emitDOT(const SNode *node, llvm::raw_ostream &os);

    // Emit JSON representation of SNode tree
    void emitJSON(const SNode *node, llvm::json::OStream &jos,
                  clang::ASTContext *ctx = nullptr);

} // namespace patchestry::ast
