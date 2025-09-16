/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <string>

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/SourceManager.h>

#include <patchestry/Util/Log.hpp>

namespace patchestry {

    class DiagnosticClient : public clang::DiagnosticConsumer
    {
      public:
        DiagnosticClient()
            : last_note_was_previous_definition(false)
            , last_error_location()
            , last_error_message() {}

        void HandleDiagnostic(
            clang::DiagnosticsEngine::Level level, const clang::Diagnostic &info
        ) override {
            llvm::SmallString< 256 > message;
            info.FormatDiagnostic(message);

            // Get source location information
            clang::SourceLocation loc = info.getLocation();
            std::string location_info = getLocationString(info, loc);

            // Check if this is a "previous definition" note
            bool is_previous_definition =
                (level == clang::DiagnosticsEngine::Note
                 && (message.str().find("previous") != std::string::npos));

            // Get additional context for specific diagnostic types
            std::string additional_info = getAdditionalDiagnosticInfo(info);

            // If this is a redefinition error, store context for the next note
            if (level == clang::DiagnosticsEngine::Error
                && (message.str().find("redefinition") != std::string::npos
                    || message.str().find("conflicting") != std::string::npos))
            {
                last_error_location = location_info;
                last_error_message  = message.str();
            }

            // Format the complete diagnostic message
            std::string fullMessage = formatDiagnosticMessage(
                level, std::string(message.str()), location_info, additional_info
            );

            // For previous definition notes, add connection to the error
            if (is_previous_definition && !last_error_location.empty()) {
                fullMessage +=
                    "\n Related to error at " + last_error_location + ": " + last_error_message;
                last_error_location.clear();
                last_error_message.clear();
            }

            switch (level) {
                case clang::DiagnosticsEngine::Note:
                    LOG(INFO) << "Diag Note: " << fullMessage << "\n";
                    break;
                case clang::DiagnosticsEngine::Warning:
                    LOG(WARNING) << "Diag Warning: " << fullMessage << "\n";
                    break;
                case clang::DiagnosticsEngine::Error:
                    LOG(ERROR) << "Diag Error: " << fullMessage << "\n";
                    break;
                case clang::DiagnosticsEngine::Fatal:
                    LOG(ERROR) << "Diag Fatal: " << fullMessage << "\n";
                    break;
                default:
                    LOG(INFO) << "Diag info: " << fullMessage << "\n";
                    break;
            }

            last_note_was_previous_definition = is_previous_definition;
        }

      private:
        std::string
        getLocationString(const clang::Diagnostic &info, clang::SourceLocation loc) {
            if (loc.isInvalid()) {
                return "[unknown location]";
            }

            const clang::SourceManager &sm  = info.getSourceManager();
            clang::PresumedLoc presumed_loc = sm.getPresumedLoc(loc);

            if (presumed_loc.isInvalid()) {
                return "[invalid location]";
            }

            std::string filename = presumed_loc.getFilename();
            unsigned line        = presumed_loc.getLine();
            unsigned column      = presumed_loc.getColumn();

            // Extract just the filename without full path for cleaner output
            size_t last_slash = filename.find_last_of('/');
            if (last_slash != std::string::npos && last_slash + 1 < filename.length()) {
                filename = filename.substr(last_slash + 1);
            }

            return filename + ":" + std::to_string(line) + ":" + std::to_string(column);
        }

        std::string getAdditionalDiagnosticInfo(const clang::Diagnostic &info) {
            std::string additional;

            // Get diagnostic ID to provide context-specific information
            unsigned diag_id = info.getID();

            // Check for ranges that might indicate previous definitions or related locations
            if (info.getNumRanges() > 0) {
                additional +=
                    " [with " + std::to_string(info.getNumRanges()) + " source range(s)]";
            }

            // Check for fix-it hints
            if (info.getNumFixItHints() > 0) {
                additional += " [" + std::to_string(info.getNumFixItHints())
                    + " fix-it suggestion(s) available]";
            }

            // For redefinition errors, try to get more context
            const clang::DiagnosticsEngine *engine = info.getDiags();
            if (engine) {
                llvm::StringRef diag_text = engine->getDiagnosticIDs()->getDescription(diag_id);

                if (diag_text.contains("redefinition") || diag_text.contains("previous")
                    || diag_text.contains("conflicting") || diag_text.contains("duplicate"))
                {
                    additional += " [This appears to be a redefinition/conflict issue]";
                }

                if (diag_text.contains("include") || diag_text.contains("header")
                    || diag_text.contains("file"))
                {
                    additional += " [File/include related issue]";

                    // For include issues, suggest common solutions
                    if (diag_text.contains("not found")) {
                        additional +=
                            " [Suggestion: Check include paths and header availability]";
                    }
                }
            }

            return additional;
        }

        std::string formatDiagnosticMessage(
            clang::DiagnosticsEngine::Level level, const std::string &message,
            const std::string &location, const std::string &additional
        ) {
            std::string levelStr;
            switch (level) {
                case clang::DiagnosticsEngine::Note:
                    levelStr = "NOTE";
                    break;
                case clang::DiagnosticsEngine::Warning:
                    levelStr = "WARNING";
                    break;
                case clang::DiagnosticsEngine::Error:
                    levelStr = "ERROR";
                    break;
                case clang::DiagnosticsEngine::Fatal:
                    levelStr = "FATAL";
                    break;
                default:
                    levelStr = "INFO";
                    break;
            }

            std::string result = "[" + levelStr + "] " + location + ": " + message;
            if (!additional.empty()) {
                result += additional;
            }

            return result;
        }

      private:
        bool last_note_was_previous_definition{ false };
        std::string last_error_location{ "" };
        std::string last_error_message{ "" };
    };
} // namespace patchestry
