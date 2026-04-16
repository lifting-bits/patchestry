/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "Lexer.hpp"

#include <cassert>
#include <cctype>
#include <cstring>
#include <unordered_map>

namespace patchestry::patchdsl {

    namespace {

        using KeywordMap = std::unordered_map< std::string, TokenKind >;

        const KeywordMap &keywords() {
            static const KeywordMap table = {
                { "metadata",            TokenKind::KW_METADATA },
                { "target",              TokenKind::KW_TARGET },
                { "import",              TokenKind::KW_IMPORT },
                { "as",                  TokenKind::KW_AS },
                { "patch",               TokenKind::KW_PATCH },
                { "rule",                TokenKind::KW_RULE },
                { "contract",            TokenKind::KW_CONTRACT },
                { "pattern",             TokenKind::KW_PATTERN },
                { "pattern-either",      TokenKind::KW_PATTERN_EITHER },
                { "pattern-inside",      TokenKind::KW_PATTERN_INSIDE },
                { "capture-pattern",     TokenKind::KW_CAPTURE_PATTERN },
                { "capture-comparison",  TokenKind::KW_CAPTURE_COMPARISON },
                { "capture-taint",       TokenKind::KW_CAPTURE_TAINT },
                { "where",               TokenKind::KW_WHERE },
                { "description",         TokenKind::KW_DESCRIPTION },
                { "id",                  TokenKind::KW_ID },
                { "rewrite",             TokenKind::KW_REWRITE },
                { "call",                TokenKind::KW_CALL },
                { "insert",              TokenKind::KW_INSERT },
                { "remove",              TokenKind::KW_REMOVE },
                { "assert",              TokenKind::KW_ASSERT },
                { "before",              TokenKind::KW_BEFORE },
                { "after",               TokenKind::KW_AFTER },
                { "at_entry",            TokenKind::KW_AT_ENTRY },
                { "at_exit",             TokenKind::KW_AT_EXIT },
                { "requires",            TokenKind::KW_REQUIRES },
                { "ensures",             TokenKind::KW_ENSURES },
                { "invariant",           TokenKind::KW_INVARIANT },
                { "attributes",          TokenKind::KW_ATTRIBUTES },
            };
            return table;
        }

        bool isIdentStart(char c) {
            return std::isalpha(static_cast< unsigned char >(c)) || c == '_';
        }
        bool isIdentChar(char c) {
            return std::isalnum(static_cast< unsigned char >(c)) || c == '_' || c == '-';
        }

        void rtrim(std::string &s) {
            while (!s.empty() && std::isspace(static_cast< unsigned char >(s.back()))) {
                s.pop_back();
            }
        }
        void ltrim(std::string &s) {
            std::size_t i = 0;
            while (i < s.size() && std::isspace(static_cast< unsigned char >(s[i]))) {
                ++i;
            }
            if (i != 0) {
                s.erase(0, i);
            }
        }

    } // namespace

    Lexer::Lexer(llvm::StringRef source) : source_(source) {}

    Token Lexer::makeToken(TokenKind k, std::string text, SourceSpan span) {
        Token t;
        t.kind = k;
        t.text = std::move(text);
        t.span = span;
        return t;
    }

    TokenKind Lexer::classifyIdent(llvm::StringRef ident) const {
        auto it = keywords().find(std::string(ident));
        return it == keywords().end() ? TokenKind::IDENT : it->second;
    }

    void Lexer::skipWhitespaceAndComments() {
        while (offset_ < source_.size()) {
            char c = source_[offset_];
            if (c == ' ' || c == '\t' || c == '\r') {
                ++offset_;
                ++col_;
            } else if (c == '\n') {
                ++offset_;
                ++line_;
                col_ = 1;
            } else if (c == '/' && offset_ + 1 < source_.size() && source_[offset_ + 1] == '/') {
                while (offset_ < source_.size() && source_[offset_] != '\n') {
                    ++offset_;
                }
            } else {
                return;
            }
        }
    }

    Token Lexer::scanOne() {
        skipWhitespaceAndComments();
        SourceSpan span{ line_, col_ };
        if (offset_ >= source_.size()) {
            return makeToken(TokenKind::END_OF_FILE, "", span);
        }
        char c = source_[offset_];

        // Single-character punctuation.
        switch (c) {
            case '{':
                ++offset_; ++col_;
                return makeToken(TokenKind::LBRACE, "{", span);
            case '}':
                ++offset_; ++col_;
                return makeToken(TokenKind::RBRACE, "}", span);
            case '[':
                ++offset_; ++col_;
                return makeToken(TokenKind::LBRACKET, "[", span);
            case ']':
                ++offset_; ++col_;
                return makeToken(TokenKind::RBRACKET, "]", span);
            case '(':
                ++offset_; ++col_;
                return makeToken(TokenKind::LPAREN, "(", span);
            case ')':
                ++offset_; ++col_;
                return makeToken(TokenKind::RPAREN, ")", span);
            case ',':
                ++offset_; ++col_;
                return makeToken(TokenKind::COMMA, ",", span);
            case ';':
                ++offset_; ++col_;
                return makeToken(TokenKind::SEMICOLON, ";", span);
            case '&':
                ++offset_; ++col_;
                return makeToken(TokenKind::AMPERSAND, "&", span);
            case '*':
                ++offset_; ++col_;
                return makeToken(TokenKind::STAR, "*", span);
        }

        // Colon or ::
        if (c == ':') {
            if (offset_ + 1 < source_.size() && source_[offset_ + 1] == ':') {
                offset_ += 2;
                col_ += 2;
                return makeToken(TokenKind::COLON_COLON, "::", span);
            }
            ++offset_; ++col_;
            return makeToken(TokenKind::COLON, ":", span);
        }

        // Arrow
        if (c == '-' && offset_ + 1 < source_.size() && source_[offset_ + 1] == '>') {
            offset_ += 2;
            col_ += 2;
            return makeToken(TokenKind::ARROW, "->", span);
        }

        // String literal.
        if (c == '"') {
            ++offset_; ++col_;
            std::string text;
            while (offset_ < source_.size() && source_[offset_] != '"') {
                if (source_[offset_] == '\\' && offset_ + 1 < source_.size()) {
                    char next = source_[offset_ + 1];
                    switch (next) {
                        case 'n': text.push_back('\n'); break;
                        case 't': text.push_back('\t'); break;
                        case '"': text.push_back('"'); break;
                        case '\\': text.push_back('\\'); break;
                        default: text.push_back(next); break;
                    }
                    offset_ += 2;
                    col_ += 2;
                } else if (source_[offset_] == '\n') {
                    // Unterminated string.
                    return makeToken(TokenKind::INVALID, "unterminated string", span);
                } else {
                    text.push_back(source_[offset_]);
                    ++offset_; ++col_;
                }
            }
            if (offset_ >= source_.size()) {
                return makeToken(TokenKind::INVALID, "unterminated string", span);
            }
            ++offset_; ++col_;  // closing "
            return makeToken(TokenKind::STRING_LIT, std::move(text), span);
        }

        // Integer literal.
        if (std::isdigit(static_cast< unsigned char >(c))) {
            std::size_t start = offset_;
            // Support 0x, 0X hex literals.
            if (c == '0' && offset_ + 1 < source_.size()
                && (source_[offset_ + 1] == 'x' || source_[offset_ + 1] == 'X')) {
                offset_ += 2;
                col_ += 2;
                while (offset_ < source_.size()
                       && std::isxdigit(static_cast< unsigned char >(source_[offset_]))) {
                    ++offset_; ++col_;
                }
            } else {
                while (offset_ < source_.size()
                       && std::isdigit(static_cast< unsigned char >(source_[offset_]))) {
                    ++offset_; ++col_;
                }
            }
            return makeToken(
                TokenKind::INT_LIT,
                std::string(source_.data() + start, offset_ - start),
                span
            );
        }

        // Capture: $X or $...XS
        if (c == '$') {
            ++offset_; ++col_;
            if (offset_ + 2 < source_.size() && source_[offset_] == '.'
                && source_[offset_ + 1] == '.' && source_[offset_ + 2] == '.') {
                offset_ += 3;
                col_ += 3;
                std::string name;
                while (offset_ < source_.size() && isIdentChar(source_[offset_])) {
                    name.push_back(source_[offset_]);
                    ++offset_; ++col_;
                }
                return makeToken(TokenKind::VARIADIC_CAPTURE, std::move(name), span);
            }
            std::string name;
            while (offset_ < source_.size() && isIdentChar(source_[offset_])) {
                name.push_back(source_[offset_]);
                ++offset_; ++col_;
            }
            if (name.empty()) {
                return makeToken(TokenKind::INVALID, "bare `$` with no name", span);
            }
            return makeToken(TokenKind::CAPTURE, std::move(name), span);
        }

        // Stringified capture: #FN
        if (c == '#') {
            ++offset_; ++col_;
            std::string name;
            while (offset_ < source_.size() && isIdentChar(source_[offset_])) {
                name.push_back(source_[offset_]);
                ++offset_; ++col_;
            }
            if (name.empty()) {
                return makeToken(TokenKind::INVALID, "bare `#` with no name", span);
            }
            return makeToken(TokenKind::STRINGIFIED, std::move(name), span);
        }

        // @return pseudo-capture
        if (c == '@') {
            ++offset_; ++col_;
            std::string name;
            while (offset_ < source_.size() && isIdentChar(source_[offset_])) {
                name.push_back(source_[offset_]);
                ++offset_; ++col_;
            }
            if (name == "return") {
                return makeToken(TokenKind::AT_RETURN, "@return", span);
            }
            return makeToken(TokenKind::INVALID, "unknown @-identifier @" + name, span);
        }

        // Identifier or keyword (hyphens allowed for keywords like
        // `pattern-inside`).
        if (isIdentStart(c)) {
            std::size_t start = offset_;
            while (offset_ < source_.size() && isIdentChar(source_[offset_])) {
                ++offset_; ++col_;
            }
            std::string ident(source_.data() + start, offset_ - start);
            TokenKind k = classifyIdent(ident);
            return makeToken(k, std::move(ident), span);
        }

        // Unknown byte.
        char bad = c;
        ++offset_; ++col_;
        std::string msg = "unexpected character `";
        msg.push_back(bad);
        msg.push_back('`');
        return makeToken(TokenKind::INVALID, std::move(msg), span);
    }

    const Token &Lexer::peek() {
        if (!have_peeked_) {
            peeked_ = scanOne();
            have_peeked_ = true;
            current_span_ = peeked_.span;
        }
        return peeked_;
    }

    Token Lexer::consume() {
        if (!have_peeked_) {
            peek();
        }
        Token t = std::move(peeked_);
        have_peeked_ = false;
        return t;
    }

    std::uint32_t Lexer::peekColumn() {
        peek();
        return peeked_.span.col;
    }

    std::string Lexer::readCodeBlockBody(std::uint32_t base_indent) {
        // Contract: caller must not have an outstanding peek. The normal
        // parser flow consumes `:` and then calls this method, which
        // satisfies the invariant. Violating it would feed a body from
        // the wrong offset.
        assert(!have_peeked_ && "readCodeBlockBody called with a pending peek");
        (void)base_indent;

        // Skip spaces/tabs (but not newlines) so we can detect the `|`.
        while (offset_ < source_.size()
               && (source_[offset_] == ' ' || source_[offset_] == '\t')) {
            ++offset_; ++col_;
        }

        if (offset_ < source_.size() && source_[offset_] == '|') {
            // Indented-block form: `|` then newline, then lines indented
            // more than the base_indent column.
            ++offset_; ++col_;
            // Skip the rest of this line.
            while (offset_ < source_.size() && source_[offset_] != '\n') {
                ++offset_;
            }
            if (offset_ < source_.size()) {
                ++offset_; ++line_; col_ = 1;
            }
            std::string body;
            while (offset_ < source_.size()) {
                // Measure leading spaces of this line.
                std::size_t line_start = offset_;
                std::uint32_t indent = 0;
                while (offset_ < source_.size()
                       && (source_[offset_] == ' ' || source_[offset_] == '\t')) {
                    ++offset_;
                    ++indent;
                }
                // Blank line → include as-is and continue.
                if (offset_ < source_.size() && source_[offset_] == '\n') {
                    body.push_back('\n');
                    ++offset_; ++line_; col_ = 1;
                    continue;
                }
                // Dedented below base_indent → end of block.
                if (indent <= base_indent) {
                    offset_ = line_start;
                    col_ = 1;
                    break;
                }
                // Include the line (preserving original spacing relative
                // to base_indent).
                while (offset_ < source_.size() && source_[offset_] != '\n') {
                    body.push_back(source_[offset_]);
                    ++offset_; ++col_;
                }
                body.push_back('\n');
                if (offset_ < source_.size()) {
                    ++offset_; ++line_; col_ = 1;
                }
            }
            rtrim(body);
            return body;
        }

        // Single-line form: read to end of line, stripping trailing comments.
        std::string body;
        while (offset_ < source_.size() && source_[offset_] != '\n') {
            // Line comment — stop before `//`.
            if (source_[offset_] == '/' && offset_ + 1 < source_.size()
                && source_[offset_ + 1] == '/') {
                break;
            }
            body.push_back(source_[offset_]);
            ++offset_; ++col_;
        }
        rtrim(body);
        ltrim(body);
        return body;
    }

} // namespace patchestry::patchdsl
