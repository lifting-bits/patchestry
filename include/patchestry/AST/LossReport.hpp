/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceLocation.h>
#include <llvm/ADT/StringRef.h>

namespace patchestry::ast {

    /// Reports places where the lifter could not reproduce a P-Code fact
    /// faithfully (a dropped operation, a synthesized argument, a fallback
    /// type).  The lift keeps going either way; what changes is the severity:
    /// a warning by default, an error under `-strict`, which makes
    /// `TranslationUnit::has_errors()` true and stops the driver before
    /// lowering.
    ///
    /// `category` is a stable kebab-case tag (`op-dropped`, ...) that tests
    /// and reports match on; `message` is free text for the reader.
    class LossReporter
    {
      public:
        LossReporter(clang::DiagnosticsEngine &diags, bool strict)
            : diags(&diags), is_strict(strict) {}

        bool strict() const { return is_strict; }

        void report(
            clang::SourceLocation loc, llvm::StringRef category, llvm::StringRef message
        ) const {
            const unsigned diag_id = is_strict
                ? diags->getCustomDiagID(
                      clang::DiagnosticsEngine::Error, "lossy lift [%0]: %1")
                : diags->getCustomDiagID(
                      clang::DiagnosticsEngine::Warning, "lossy lift [%0]: %1");
            diags->Report(loc, diag_id) << category << message;
        }

      private:
        clang::DiagnosticsEngine *diags;
        bool is_strict;
    };

} // namespace patchestry::ast
