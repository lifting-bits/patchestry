/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

enum LogLevel { DEBUG, INFO, WARNING, ERROR, FATAL };

namespace patchestry::logging {
    // Min level a LOG() must meet to emit (glog-style). Default DEBUG keeps the
    // historical "print everything" behavior for tools that don't opt in;
    // patchir-decomp raises it to WARNING (INFO off) unless --verbose.
    inline LogLevel &MinLogLevel() {
        static LogLevel level = DEBUG;
        return level;
    }
    // Sink for suppressed levels.
    inline llvm::raw_ostream &NullLogStream() {
        static llvm::raw_null_ostream stream;
        return stream;
    }
} // namespace patchestry::logging

#define LOG(level) \
    (((level) < ::patchestry::logging::MinLogLevel()) \
         ? ::patchestry::logging::NullLogStream() \
         : ((((level) == DEBUG)     ? llvm::outs() << "[DEBUG] " \
              : ((level) == INFO)    ? llvm::outs() << "[INFO] " \
              : ((level) == WARNING) ? llvm::outs() << "[WARNING] " \
              : ((level) == FATAL)   ? llvm::errs() << "[FATAL] " \
              : ((level) == ERROR)   ? llvm::errs() << "[ERROR] " \
                                     : llvm::outs()) \
            << "(" << __FILE__ << ":" << __LINE__ << ") "))

#define LOG_FATAL(...) \
    do { \
        LOG(FATAL) << llvm::formatv(__VA_ARGS__); \
        llvm::report_fatal_error("fatal error in patchestry", false); \
    } while (0)

#define LOG_FATAL_IF(cond, ...) \
    do { \
        if (cond) { \
            LOG(FATAL) << llvm::formatv(__VA_ARGS__); \
            llvm::report_fatal_error("fatal error in patchestry", false); \
        } \
    } while (0)

#define UNIMPLEMENTED(...) \
    do { \
        LOG(ERROR) << llvm::formatv(__VA_ARGS__); \
        llvm_unreachable(nullptr); \
    } while (0)

#define UNREACHABLE(...) \
    do { \
        LOG(ERROR) << llvm::formatv(__VA_ARGS__); \
        llvm_unreachable(nullptr); \
    } while (0)
