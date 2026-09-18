/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/Ghidra/PcodeOperations.hpp>

#include <string>
#include <unordered_map>

namespace patchestry::ast {

    /// Options for PrintTranslationUnit.
    struct TUPrintOptions
    {
        /// Ghidra language id (`ARM:LE:32:Cortex`) and architecture for the
        /// `// patchestry:tu` header line.  Either may be empty.
        std::string lang_id;
        std::string arch;
        /// Emit the header and `// patchestry:function-begin/end <key>`
        /// markers around every definition.  Off prints plain C with the same
        /// ordering.
        bool emit_markers = true;
    };

    /// Maps an emitted definition to the Ghidra function it was lifted from.
    /// Definitions missing from the map print without a key or comment.
    using DefinitionMap =
        std::unordered_map< const clang::FunctionDecl *, const ghidra::Function * >;

    /// Print the translation unit as re-parseable C.
    ///
    /// Differences from `TranslationUnitDecl::print`:
    ///  - top-level declarations are emitted in dependency order (tag forward
    ///    declarations, type definitions, globals, prototypes, definitions)
    ///    instead of AST insertion order;
    ///  - integer literals narrower than `int` print without the MS-only
    ///    `i8`/`Ui16` suffixes, and NaN/Inf floats print as `__builtin_nan*`;
    ///  - each definition is wrapped in `patchestry:function-*` markers and
    ///    preceded by its `Function::comment`, when present.
    ///
    /// Call ReparenthesizeForPrint on every body first: the printer relies on
    /// ParenExpr nodes to reproduce the AST's precedence.
    void PrintTranslationUnit(
        llvm::raw_ostream &os, clang::ASTContext &ctx, const DefinitionMap &defs,
        const TUPrintOptions &opts
    );

    /// Insert ParenExpr nodes wherever clang's StmtPrinter would otherwise
    /// print a sub-expression at the wrong precedence.  The lifter builds
    /// expressions directly over their operands, with no ParenExpr, so
    /// `*(T *)((int)p + 10)` printed as `*(T *)(int)p + 10`.  ParenExpr is
    /// transparent to CIRGen, so the lowered IR is unchanged.  Idempotent.
    void ReparenthesizeForPrint(clang::ASTContext &ctx, clang::Stmt *body);

} // namespace patchestry::ast
