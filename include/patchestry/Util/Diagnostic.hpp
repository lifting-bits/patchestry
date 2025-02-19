/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticOptions.h>

#include <patchestry/Util/Log.hpp>

namespace patchestry {

    class DiagnosticClient : public clang::DiagnosticConsumer
    {
      public:
        void HandleDiagnostic(
            clang::DiagnosticsEngine::Level level, const clang::Diagnostic &info
        ) override {
            llvm::SmallString< 100 > message;
            info.FormatDiagnostic(message);
            switch (level) {
                case clang::DiagnosticsEngine::Note:
                    LOG(INFO) << "Diag Note: " << message << "\n";
                    break;
                case clang::DiagnosticsEngine::Warning:
                    LOG(WARNING) << "Diag Warning: " << message << "\n";
                    break;
                case clang::DiagnosticsEngine::Error:
                    LOG(ERROR) << "Diag Error: " << message << "\n";
                    break;
                case clang::DiagnosticsEngine::Fatal:
                    LOG(ERROR) << "Diag Fatal: " << message << "\n";
                    break;
                default:
                    LOG(INFO) << "Diag info: " << message << "\n";
                    break;
            }
        }
    };
} // namespace patchestry
