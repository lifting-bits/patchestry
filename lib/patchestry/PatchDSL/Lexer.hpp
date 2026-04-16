/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <llvm/ADT/StringRef.h>

#include <patchestry/PatchDSL/AST.hpp>

namespace patchestry::patchdsl {

    enum class TokenKind : std::uint8_t {
        // Structural punctuation
        LBRACE, RBRACE, LBRACKET, RBRACKET, LPAREN, RPAREN,
        COMMA, COLON, SEMICOLON, ARROW, COLON_COLON, AMPERSAND, STAR,
        AT_RETURN,

        // Literals
        IDENT,
        STRING_LIT,
        INT_LIT,
        CAPTURE,           // $X
        VARIADIC_CAPTURE,  // $...XS
        STRINGIFIED,       // #FN

        // Top-level keywords
        KW_METADATA, KW_TARGET, KW_IMPORT, KW_AS,
        KW_PATCH, KW_RULE, KW_CONTRACT,

        // Clause keywords
        KW_PATTERN, KW_PATTERN_EITHER, KW_PATTERN_INSIDE,
        KW_CAPTURE_PATTERN, KW_CAPTURE_COMPARISON, KW_CAPTURE_TAINT,
        KW_WHERE, KW_DESCRIPTION, KW_ID,

        // Action keywords
        KW_REWRITE, KW_CALL, KW_INSERT, KW_REMOVE, KW_ASSERT,
        KW_BEFORE, KW_AFTER, KW_AT_ENTRY, KW_AT_EXIT,

        // Contract clause keywords
        KW_REQUIRES, KW_ENSURES, KW_INVARIANT, KW_ATTRIBUTES,

        // Special
        END_OF_FILE,
        INVALID
    };

    struct Token {
        TokenKind kind = TokenKind::INVALID;
        std::string text;
        SourceSpan span;
    };

    /// Streaming lexer. The parser drives it by calling peek/consume;
    /// when the parser enters a code-block body context, it calls
    /// readCodeBlockBody() to switch to raw-character mode for that one
    /// body, then resumes token-mode afterward.
    class Lexer {
      public:
        explicit Lexer(llvm::StringRef source);

        /// Returns the next token without consuming it.
        const Token &peek();

        /// Consumes and returns the next token.
        Token consume();

        /// Position of the token peek() would return (1-based line / col).
        SourceSpan currentSpan() const { return current_span_; }

        /// After a `:` has been consumed, read the body of a code-block
        /// clause/action. Two shapes:
        ///   1. `|` on the same line, then an indented block. Reads all
        ///      lines indented strictly more than `base_indent`.
        ///   2. Otherwise, reads the rest of the current line.
        /// Returns the body with leading/trailing whitespace trimmed, and
        /// repositions the cursor just past the body.
        std::string readCodeBlockBody(std::uint32_t base_indent);

        /// Returns the column (1-based) of the token peek() would return.
        std::uint32_t peekColumn();

      private:
        llvm::StringRef source_;
        std::size_t offset_ = 0;
        std::uint32_t line_ = 1;
        std::uint32_t col_  = 1;

        bool have_peeked_ = false;
        Token peeked_;
        SourceSpan current_span_;

        void skipWhitespaceAndComments();
        Token scanOne();
        Token makeToken(TokenKind k, std::string text, SourceSpan span);
        TokenKind classifyIdent(llvm::StringRef ident) const;
    };

} // namespace patchestry::patchdsl
