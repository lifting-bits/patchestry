/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

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
            : lastNoteWasPreviousDefinition(false), lastErrorLocation(), lastErrorMessage() {}

      public:
        void HandleDiagnostic(
            clang::DiagnosticsEngine::Level level, const clang::Diagnostic &info
        ) override {
            llvm::SmallString< 256 > message;
            info.FormatDiagnostic(message);

            // Get source location information
            clang::SourceLocation loc = info.getLocation();
            std::string locationInfo  = getLocationString(info, loc);

            // Check if this is a "previous definition" note
            bool isPreviousDefinition =
                (level == clang::DiagnosticsEngine::Note
                 && (message.str().find("previous") != std::string::npos));

            // Get additional context for specific diagnostic types
            std::string additionalInfo = getAdditionalDiagnosticInfo(info);

            // If this is a redefinition error, store context for the next note
            if (level == clang::DiagnosticsEngine::Error
                && (message.str().find("redefinition") != std::string::npos
                    || message.str().find("conflicting") != std::string::npos))
            {
                lastErrorLocation = locationInfo;
                lastErrorMessage  = message.str();
            }

            // Format the complete diagnostic message
            std::string fullMessage = formatDiagnosticMessage(
                level, std::string(message.str()), locationInfo, additionalInfo
            );

            // For previous definition notes, add connection to the error
            if (isPreviousDefinition && !lastErrorLocation.empty()) {
                fullMessage +=
                    "\n Related to error at " + lastErrorLocation + ": " + lastErrorMessage;
                lastErrorLocation.clear();
                lastErrorMessage.clear();
            }

            switch (level) {
                case clang::DiagnosticsEngine::Note:
                    if (isPreviousDefinition) {
                        LOG(INFO) << fullMessage << "\n";
                    } else {
                        LOG(INFO) << fullMessage << "\n";
                    }
                    break;
                case clang::DiagnosticsEngine::Warning:
                    LOG(WARNING) << fullMessage << "\n";
                    break;
                case clang::DiagnosticsEngine::Error:
                    LOG(ERROR) << fullMessage << "\n";
                    break;
                case clang::DiagnosticsEngine::Fatal:
                    LOG(ERROR) << fullMessage << "\n";
                    break;
                default:
                    LOG(INFO) << fullMessage << "\n";
                    break;
            }

            lastNoteWasPreviousDefinition = isPreviousDefinition;
        }

      private:
        std::string
        getLocationString(const clang::Diagnostic &info, clang::SourceLocation loc) {
            if (loc.isInvalid()) {
                return "[unknown location]";
            }

            const clang::SourceManager &sm = info.getSourceManager();
            clang::PresumedLoc presumedLoc = sm.getPresumedLoc(loc);

            if (presumedLoc.isInvalid()) {
                return "[invalid location]";
            }

            std::string filename = presumedLoc.getFilename();
            unsigned line        = presumedLoc.getLine();
            unsigned column      = presumedLoc.getColumn();

            // Extract just the filename without full path for cleaner output
            size_t lastSlash = filename.find_last_of('/');
            if (lastSlash != std::string::npos && lastSlash + 1 < filename.length()) {
                filename = filename.substr(lastSlash + 1);
            }

            return filename + ":" + std::to_string(line) + ":" + std::to_string(column);
        }

        std::string getAdditionalDiagnosticInfo(const clang::Diagnostic &info) {
            std::string additional;

            // Get diagnostic ID to provide context-specific information
            unsigned diagID = info.getID();

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
                llvm::StringRef diagText = engine->getDiagnosticIDs()->getDescription(diagID);

                if (diagText.contains("redefinition") || diagText.contains("previous")
                    || diagText.contains("conflicting") || diagText.contains("duplicate"))
                {
                    additional += " [This appears to be a redefinition/conflict issue]";
                }

                if (diagText.contains("include") || diagText.contains("header")
                    || diagText.contains("file"))
                {
                    additional += " [File/include related issue]";

                    // For include issues, suggest common solutions
                    if (diagText.contains("not found")) {
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

        // Member variables to track diagnostic context
        bool lastNoteWasPreviousDefinition;
        std::string lastErrorLocation;
        std::string lastErrorMessage;
    };
} // namespace patchestry
