/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/Support/raw_ostream.h>

#define ENABLE_DEBUG

enum LogLevel { INFO, WARNING, ERROR };

#define LOG(level) \
    (((level) == INFO)          ? llvm::outs() << "[INFO] " \
         : ((level) == WARNING) ? llvm::outs() << "[WARNING] " \
         : ((level) == ERROR)   ? llvm::errs() << "[ERROR] " \
                                : llvm::outs() \
    ) << "(" \
      << __FILE__ << ":" << __LINE__ << ") "
