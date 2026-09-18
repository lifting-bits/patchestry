/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <optional>
#include <string>

namespace patchestry::ghidra {

    /// Preserve the patch compiler's variant mapping, or use the lifter's
    /// architecture-only mapping. Both retain the existing Linux/ARM hard-float ABI.
    enum class VariantPolicy { Ignore, Preserve };

    /// Translate a Ghidra language id to an LLVM triple. The optional architecture
    /// overrides the language id's architecture (as in a lifted Program).
    /// Returns an empty string after logging malformed language ids.
    std::string targetTriple(
        const std::string &lang, VariantPolicy policy,
        std::optional< std::string > architecture = std::nullopt
    );

} // namespace patchestry::ghidra
