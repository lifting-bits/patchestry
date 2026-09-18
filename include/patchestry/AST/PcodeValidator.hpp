/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/Ghidra/PcodeOperations.hpp>

#include <string>
#include <vector>

namespace patchestry::ast {

    /// One `// patchestry:function-begin` marker from a printed translation
    /// unit: the P-Code function key, the C name and the linker symbol.
    struct MarkerEntry
    {
        std::string key;
        std::string name;
        std::string symbol;
    };

    struct ValidationFlag
    {
        enum Severity { kWarning, kCritical };

        std::string code;
        Severity severity = kWarning;
        std::string detail;
    };

    struct FunctionValidation
    {
        std::string key;
        std::string name;
        std::vector< ValidationFlag > flags;

        bool HasCritical() const;
        bool HasWarning() const;
        /// "pass", "warn" or "fail".
        llvm::StringRef Verdict() const;
    };

    struct ValidationReport
    {
        std::vector< FunctionValidation > functions;
        unsigned failed = 0;
        unsigned warned = 0;

        bool HasCritical() const { return failed > 0; }
    };

    struct ValidationOptions
    {
        /// Fraction of P-Code conditions that may be missing before
        /// COND_DEFICIT is raised.
        double cond_warn_ratio  = 0.30;
        /// Fractions of P-Code STOREs that may be missing before
        /// STORE_DEFICIT is raised as a warning, then as critical.
        double store_warn_ratio = 0.20;
        double store_crit_ratio = 0.50;
    };

    /// The function definition named `name` in the translation unit, or null.
    const clang::FunctionDecl *
    FindFunctionDefinition(clang::ASTContext &ctx, llvm::StringRef name);

    /// Check every marker-listed function of a parsed C translation unit
    /// against the P-Code model it was lifted from: signature, calls, string
    /// literals, globals, returns, stores, conditions and switch cases.  This
    /// is a fact-preservation check, not an equivalence proof.
    ValidationReport ValidateAgainstPcode(
        clang::ASTContext &ctx, const ghidra::Program &program,
        const std::vector< MarkerEntry > &markers, const ValidationOptions &options
    );

    /// Write the report as JSON.
    void WriteValidationReport(llvm::raw_ostream &os, const ValidationReport &report);

} // namespace patchestry::ast
