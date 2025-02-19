/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

enum LogLevel { INFO, WARNING, ERROR };

#define LOG(level) \
    (((level) == INFO)          ? llvm::outs() << "[INFO] " \
         : ((level) == WARNING) ? llvm::outs() << "[WARNING] " \
         : ((level) == ERROR)   ? llvm::errs() << "[ERROR] " \
                                : llvm::outs() \
    ) << "(" \
      << __FILE__ << ":" << __LINE__ << ") "

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
