/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace patchestry::patchdsl {

    struct SourceSpan {
        std::uint32_t line = 0;
        std::uint32_t col  = 0;
    };

    struct Parameter {
        std::string name;
        std::string type;
    };

    struct PatchSignature {
        std::string name;
        std::vector< Parameter > params;
        std::string return_type;
        SourceSpan span;
    };

    struct TargetNode {
        std::string binary;
        std::string arch;
        std::optional< std::string > variant;
        SourceSpan span;
    };

    struct MetadataNode {
        std::string name;
        std::string description;
        std::string version;
        std::string author;
        std::string created;
        std::optional< TargetNode > target;
        SourceSpan span;
    };

    struct ImportNode {
        std::string path;
        std::string alias;
        // Populated for C-file imports (`foo.c`).
        // Empty for library-file imports (Phase 7).
        std::vector< PatchSignature > signatures;
        bool is_library = false;
        SourceSpan span;
    };

    struct InlinePatchNode {
        std::string name;
        std::vector< Parameter > params;
        std::string return_type;
        std::string body;
        SourceSpan span;
    };

    // Argument in a call: $X, $...XS, literal, bare ident, or &$X.
    struct CallArg {
        enum class Kind : std::uint8_t {
            CAPTURE,           // $X
            VARIADIC_CAPTURE,  // $...XS
            INT_LITERAL,       // 512
            STRING_LITERAL,    // "foo"
            BARE_IDENT,        // bl_spi_mode (global symbol)
            ADDRESS_OF         // &$X or &ident
        };
        Kind kind = Kind::CAPTURE;
        std::string text;
        std::unique_ptr< CallArg > inner;  // for ADDRESS_OF
        SourceSpan span;
    };

    struct CallExpr {
        std::vector< std::string > namespace_path;  // ["sys"] or ["usb", "native"]
        std::string function_name;
        std::vector< CallArg > args;
        SourceSpan span;
    };

    struct ClauseNode {
        enum class Kind : std::uint8_t {
            PATTERN,
            PATTERN_EITHER,
            PATTERN_INSIDE,
            CAPTURE_PATTERN,
            CAPTURE_COMPARISON,
            CAPTURE_TAINT,
            WHERE,
            DESCRIPTION,
            ID
        };
        Kind kind;
        // Opaque body text: the stuff after the `:` up to the next clause,
        // action, or `}` at the same indent. For DESCRIPTION / ID, this is
        // the string literal value.
        std::string body;
        SourceSpan span;
    };

    struct ActionNode {
        enum class Kind : std::uint8_t {
            REWRITE,
            CALL,
            REMOVE,
            ASSERT,
            INSERT_BEFORE,
            INSERT_AFTER,
            INSERT_AT_ENTRY,
            INSERT_AT_EXIT
        };
        Kind kind;
        // For CALL (and INSERT_* whose body is `call: ...`): the structured call.
        std::optional< CallExpr > call_expr;
        // For REWRITE and INSERT_* with a plain body: the opaque body text.
        std::string body;
        // For ASSERT: the predicate text (opaque).
        std::string predicate;
        SourceSpan span;
    };

    struct RuleNode {
        std::string name;
        std::vector< ClauseNode > clauses;
        std::vector< ActionNode > actions;
        SourceSpan span;
    };

    struct ContractClauseNode {
        enum class Kind : std::uint8_t {
            REQUIRES,
            ENSURES,
            INVARIANT,
            ATTRIBUTES
        };
        Kind kind;
        // For REQUIRES / ENSURES / INVARIANT: the predicate text (opaque).
        // For ATTRIBUTES: comma-separated attribute names (stripped of `[ ]`).
        std::string body;
        SourceSpan span;
    };

    struct ContractNode {
        std::string name;
        std::vector< ClauseNode > clauses;
        std::vector< ContractClauseNode > contract_clauses;
        SourceSpan span;
    };

    struct AST {
        std::optional< MetadataNode > metadata;
        std::vector< ImportNode > imports;
        std::vector< InlinePatchNode > inline_patches;
        std::vector< RuleNode > rules;
        std::vector< ContractNode > contracts;
    };

} // namespace patchestry::patchdsl
