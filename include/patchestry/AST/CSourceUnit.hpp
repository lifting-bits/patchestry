/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>

#include <patchestry/AST/TranslationUnit.hpp>
#include <patchestry/Ghidra/JsonDeserialize.hpp>

namespace patchestry::ast {

    /// How to re-enter a C translation unit that carries `patchestry:`
    /// markers (written by the LLM stage or by hand).
    struct CSourceOptions
    {
        /// The C file.
        std::string path;
        /// Ghidra language id override (`ARM:LE:32:Cortex`).  Empty: use the
        /// `patchestry:tu` header, else the program's language.
        std::string target_lang;
    };

    /// The second AST source next to the P-Code lifter: parse `options.path`
    /// for the decompiler's target and hand it over as a TranslationUnit with
    /// the file's `patchestry:` markers.  `program`, when given, supplies the
    /// target fallback.  Null (after logging) when the file cannot be read, no
    /// target can be resolved, the frontend fails to set up, or parsing
    /// reports errors.
    std::unique_ptr< TranslationUnit >
    ParseCTranslationUnit(const CSourceOptions &options, const ghidra::Program *program);

} // namespace patchestry::ast
