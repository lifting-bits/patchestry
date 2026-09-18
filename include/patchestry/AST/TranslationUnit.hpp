/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <clang/AST/ASTContext.h>
#include <clang/Basic/CodeGenOptions.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/CompilerInstance.h>

namespace patchestry::ast {

    /// The handoff between an AST source and the AST-level stages (printing,
    /// lowering).  Source-agnostic: the P-Code lifter produces one, and so can
    /// anything else that fills a Clang translation unit.  Owns the
    /// CompilerInstance that owns the AST, Sema, target and diagnostics.
    struct TranslationUnit
    {
        std::unique_ptr< clang::CompilerInstance > ci;

        clang::ASTContext &context() { return ci->getASTContext(); }

        clang::DiagnosticsEngine &diagnostics() { return ci->getDiagnostics(); }

        const clang::CodeGenOptions &codegen_options() { return ci->getCodeGenOpts(); }

        /// True when the source reported an error while building the unit.
        bool has_errors() { return ci->getDiagnostics().hasErrorOccurred(); }
    };

} // namespace patchestry::ast
