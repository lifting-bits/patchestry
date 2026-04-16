/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "Parser.hpp"
#include "Lexer.hpp"

#include <memory>
#include <string>
#include <utility>

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>

namespace patchestry::patchdsl {

    namespace {

        class ParseError : public llvm::ErrorInfo< ParseError > {
          public:
            static char ID;

            ParseError(std::string filename, SourceSpan span, std::string message)
                : filename_(std::move(filename)), span_(span), message_(std::move(message)) {}

            void log(llvm::raw_ostream &os) const override {
                os << filename_ << ":" << span_.line << ":" << span_.col << ": error: "
                   << message_;
            }

            std::error_code convertToErrorCode() const override {
                return llvm::inconvertibleErrorCode();
            }

          private:
            std::string filename_;
            SourceSpan span_;
            std::string message_;
        };

        char ParseError::ID = 0;

        class Parser {
          public:
            Parser(llvm::StringRef source, llvm::StringRef filename)
                : lexer_(source), filename_(filename) {}

            llvm::Expected< std::unique_ptr< AST > > parseFile();

          private:
            Lexer lexer_;
            std::string filename_;

            llvm::Error errorAt(SourceSpan span, std::string message) {
                return llvm::make_error< ParseError >(filename_, span, std::move(message));
            }

            llvm::Error expect(TokenKind kind, const char *what) {
                const Token &t = lexer_.peek();
                if (t.kind != kind) {
                    return errorAt(t.span, std::string("expected ") + what);
                }
                lexer_.consume();
                return llvm::Error::success();
            }

            llvm::Expected< std::string > expectString(const char *what) {
                const Token &t = lexer_.peek();
                if (t.kind != TokenKind::STRING_LIT) {
                    return errorAt(t.span, std::string("expected ") + what);
                }
                std::string v = t.text;
                lexer_.consume();
                return v;
            }

            llvm::Expected< std::string > expectIdent(const char *what) {
                const Token &t = lexer_.peek();
                if (t.kind != TokenKind::IDENT) {
                    return errorAt(t.span, std::string("expected ") + what);
                }
                std::string v = t.text;
                lexer_.consume();
                return v;
            }

            llvm::Error parseMetadata(AST &ast);
            llvm::Error parseTargetBlock(MetadataNode &meta);
            llvm::Error parseImport(AST &ast);
            llvm::Error parsePatchSignature(std::vector< PatchSignature > &out);
            llvm::Error parseTypeSpec(std::string &out);
            llvm::Error parseParams(std::vector< Parameter > &out);
            llvm::Error parseRule(AST &ast);
            llvm::Error parseContract(AST &ast);
            llvm::Error parseRuleBodyItem(RuleNode &rule);
            llvm::Error parseContractBodyItem(ContractNode &contract);

            llvm::Expected< ClauseNode > parseClause();
            llvm::Expected< ActionNode > parseAction();
            llvm::Expected< ContractClauseNode > parseContractClause();
            llvm::Expected< CallExpr > parseCallExpr();
            llvm::Expected< CallArg > parseCallArg();

            bool atClauseStart() const;
            bool atActionStart() const;
            bool atContractClauseStart() const;
        };

        bool isClauseKeyword(TokenKind k) {
            switch (k) {
                case TokenKind::KW_PATTERN:
                case TokenKind::KW_PATTERN_EITHER:
                case TokenKind::KW_PATTERN_INSIDE:
                case TokenKind::KW_CAPTURE_PATTERN:
                case TokenKind::KW_CAPTURE_COMPARISON:
                case TokenKind::KW_CAPTURE_TAINT:
                case TokenKind::KW_WHERE:
                case TokenKind::KW_DESCRIPTION:
                case TokenKind::KW_ID:
                    return true;
                default:
                    return false;
            }
        }

        bool isActionKeyword(TokenKind k) {
            switch (k) {
                case TokenKind::KW_REWRITE:
                case TokenKind::KW_CALL:
                case TokenKind::KW_INSERT:
                case TokenKind::KW_REMOVE:
                case TokenKind::KW_ASSERT:
                    return true;
                default:
                    return false;
            }
        }

        bool isContractClauseKeyword(TokenKind k) {
            switch (k) {
                case TokenKind::KW_REQUIRES:
                case TokenKind::KW_ENSURES:
                case TokenKind::KW_INVARIANT:
                case TokenKind::KW_ATTRIBUTES:
                    return true;
                default:
                    return false;
            }
        }

        bool Parser::atClauseStart() const {
            return isClauseKeyword(const_cast< Parser * >(this)->lexer_.peek().kind);
        }
        bool Parser::atActionStart() const {
            return isActionKeyword(const_cast< Parser * >(this)->lexer_.peek().kind);
        }
        bool Parser::atContractClauseStart() const {
            return isContractClauseKeyword(const_cast< Parser * >(this)->lexer_.peek().kind);
        }

        llvm::Expected< std::unique_ptr< AST > > Parser::parseFile() {
            auto ast = std::make_unique< AST >();
            while (true) {
                const Token &t = lexer_.peek();
                switch (t.kind) {
                    case TokenKind::END_OF_FILE:
                        return ast;
                    case TokenKind::KW_METADATA:
                        if (auto err = parseMetadata(*ast)) {
                            return std::move(err);
                        }
                        break;
                    case TokenKind::KW_IMPORT:
                        if (auto err = parseImport(*ast)) {
                            return std::move(err);
                        }
                        break;
                    case TokenKind::KW_RULE:
                        if (auto err = parseRule(*ast)) {
                            return std::move(err);
                        }
                        break;
                    case TokenKind::KW_CONTRACT:
                        if (auto err = parseContract(*ast)) {
                            return std::move(err);
                        }
                        break;
                    case TokenKind::KW_PATCH:
                        return errorAt(
                            t.span,
                            "inline `patch fn(...) { ... }` helpers are not supported in v1"
                        );
                    case TokenKind::INVALID:
                        return errorAt(t.span, t.text);
                    default:
                        return errorAt(
                            t.span,
                            "unexpected top-level token `" + t.text
                                + "` (expected metadata / import / rule / contract)"
                        );
                }
            }
        }

        llvm::Error Parser::parseMetadata(AST &ast) {
            Token kw = lexer_.consume();  // `metadata`
            if (ast.metadata) {
                return errorAt(kw.span, "duplicate `metadata` block");
            }
            MetadataNode meta;
            meta.span = kw.span;
            if (auto err = expect(TokenKind::LBRACE, "`{` after `metadata`")) {
                return err;
            }
            while (lexer_.peek().kind != TokenKind::RBRACE) {
                if (lexer_.peek().kind == TokenKind::END_OF_FILE) {
                    return errorAt(lexer_.peek().span, "unterminated metadata block");
                }
                if (lexer_.peek().kind == TokenKind::KW_TARGET
                    && lexer_.peek().text == "target") {
                    // Disambiguate `target` the keyword from a hypothetical
                    // user field — only call parseTargetBlock if the next
                    // token is the LBRACE that starts a nested block.
                    // (Lookahead via a single peek is enough: our lexer
                    // doesn't backtrack, but `target { ... }` always has
                    // `{` as the next token after `target`.)
                    if (auto err = parseTargetBlock(meta)) {
                        return err;
                    }
                    continue;
                }
                // Metadata field names: we accept IDENT plus any keyword
                // token — keywords like `description` / `id` double as
                // field names in this block context.
                const Token &peeked = lexer_.peek();
                if (peeked.kind == TokenKind::END_OF_FILE
                    || peeked.kind == TokenKind::LBRACE
                    || peeked.kind == TokenKind::RBRACE) {
                    return errorAt(
                        peeked.span,
                        "expected metadata field name, got `" + peeked.text + "`"
                    );
                }
                Token key = lexer_.consume();
                if (auto err = expect(TokenKind::COLON, "`:` after metadata field name")) {
                    return err;
                }
                auto val_or = expectString("string value for metadata field");
                if (!val_or) {
                    return val_or.takeError();
                }
                std::string value = std::move(*val_or);
                if (key.text == "name") {
                    meta.name = value;
                } else if (key.text == "description") {
                    meta.description = value;
                } else if (key.text == "version") {
                    meta.version = value;
                } else if (key.text == "author") {
                    meta.author = value;
                } else if (key.text == "created") {
                    meta.created = value;
                } else {
                    return errorAt(
                        key.span,
                        "unknown metadata field `" + key.text + "`"
                    );
                }
            }
            lexer_.consume();  // `}`
            if (!meta.target) {
                return errorAt(
                    meta.span,
                    "metadata block must contain a `target { ... }` sub-block"
                );
            }
            ast.metadata = std::move(meta);
            return llvm::Error::success();
        }

        llvm::Error Parser::parseTargetBlock(MetadataNode &meta) {
            Token kw = lexer_.consume();  // `target`
            TargetNode t;
            t.span = kw.span;
            if (auto err = expect(TokenKind::LBRACE, "`{` after `target`")) {
                return err;
            }
            while (lexer_.peek().kind != TokenKind::RBRACE) {
                if (lexer_.peek().kind == TokenKind::END_OF_FILE) {
                    return errorAt(lexer_.peek().span, "unterminated target block");
                }
                if (lexer_.peek().kind != TokenKind::IDENT) {
                    return errorAt(lexer_.peek().span, "expected target field name");
                }
                Token key = lexer_.consume();
                if (auto err = expect(TokenKind::COLON, "`:` after target field name")) {
                    return err;
                }
                auto val_or = expectString("string value for target field");
                if (!val_or) {
                    return val_or.takeError();
                }
                if (key.text == "binary") {
                    t.binary = std::move(*val_or);
                } else if (key.text == "arch") {
                    t.arch = std::move(*val_or);
                } else if (key.text == "variant") {
                    t.variant = std::move(*val_or);
                } else {
                    return errorAt(key.span, "unknown target field `" + key.text + "`");
                }
            }
            lexer_.consume();  // `}`
            if (t.binary.empty()) {
                return errorAt(t.span, "target block must set `binary`");
            }
            if (t.arch.empty()) {
                return errorAt(t.span, "target block must set `arch`");
            }
            meta.target = std::move(t);
            return llvm::Error::success();
        }

        llvm::Error Parser::parseImport(AST &ast) {
            Token kw = lexer_.consume();  // `import`
            ImportNode imp;
            imp.span = kw.span;
            auto path_or = expectString("import path string");
            if (!path_or) {
                return path_or.takeError();
            }
            imp.path = std::move(*path_or);
            if (auto err = expect(TokenKind::KW_AS, "`as` after import path")) {
                return err;
            }
            auto alias_or = expectIdent("import alias identifier");
            if (!alias_or) {
                return alias_or.takeError();
            }
            imp.alias = std::move(*alias_or);

            bool path_ends_patch = imp.path.size() >= 6
                && imp.path.compare(imp.path.size() - 6, 6, ".patch") == 0;
            bool path_ends_c = imp.path.size() >= 2
                && imp.path.compare(imp.path.size() - 2, 2, ".c") == 0;

            if (lexer_.peek().kind == TokenKind::LBRACE) {
                if (path_ends_patch) {
                    return errorAt(
                        lexer_.peek().span,
                        "library-file import (`.patch`) must not carry a signature block"
                    );
                }
                lexer_.consume();  // `{`
                while (lexer_.peek().kind != TokenKind::RBRACE) {
                    if (lexer_.peek().kind == TokenKind::END_OF_FILE) {
                        return errorAt(lexer_.peek().span, "unterminated import block");
                    }
                    if (auto err = parsePatchSignature(imp.signatures)) {
                        return err;
                    }
                }
                lexer_.consume();  // `}`
                imp.is_library = false;
            } else if (path_ends_patch) {
                // Library-file import: no block, deferred to Phase 7.
                imp.is_library = true;
                return errorAt(
                    imp.span,
                    "library-file imports (`.patch`) are scheduled for v1.7; v1 supports only `.c` imports with inline signatures"
                );
            } else {
                // Bare C-file import — v1 requires signatures.
                return errorAt(
                    imp.span,
                    path_ends_c
                        ? "C-file import requires an inline `{ patch sig; ... }` block"
                        : "import path must end in `.c` (v1 supports only C-file imports)"
                );
            }

            ast.imports.push_back(std::move(imp));
            return llvm::Error::success();
        }

        llvm::Error Parser::parsePatchSignature(std::vector< PatchSignature > &out) {
            if (lexer_.peek().kind != TokenKind::KW_PATCH) {
                return errorAt(
                    lexer_.peek().span,
                    "expected `patch NAME(...) -> T;` signature"
                );
            }
            Token kw = lexer_.consume();  // `patch`
            PatchSignature sig;
            sig.span = kw.span;
            auto name_or = expectIdent("patch function name");
            if (!name_or) {
                return name_or.takeError();
            }
            sig.name = std::move(*name_or);
            if (auto err = expect(TokenKind::LPAREN, "`(` after patch name")) {
                return err;
            }
            if (lexer_.peek().kind != TokenKind::RPAREN) {
                if (auto err = parseParams(sig.params)) {
                    return err;
                }
            }
            if (auto err = expect(TokenKind::RPAREN, "`)` after patch parameters")) {
                return err;
            }
            if (auto err = expect(TokenKind::ARROW, "`->` before return type")) {
                return err;
            }
            if (auto err = parseTypeSpec(sig.return_type)) {
                return err;
            }
            if (auto err = expect(TokenKind::SEMICOLON, "`;` after patch signature")) {
                return err;
            }
            out.push_back(std::move(sig));
            return llvm::Error::success();
        }

        llvm::Error Parser::parseParams(std::vector< Parameter > &out) {
            while (true) {
                auto name_or = expectIdent("parameter name");
                if (!name_or) {
                    return name_or.takeError();
                }
                Parameter p;
                p.name = std::move(*name_or);
                if (auto err = expect(TokenKind::COLON, "`:` after parameter name")) {
                    return err;
                }
                if (auto err = parseTypeSpec(p.type)) {
                    return err;
                }
                out.push_back(std::move(p));
                if (lexer_.peek().kind != TokenKind::COMMA) {
                    return llvm::Error::success();
                }
                lexer_.consume();  // `,`
            }
        }

        // A type spec in Phase 2 is a compact token sequence: optional `*`
        // (for pointer-to) followed by an identifier, optionally followed
        // by more `*`s and identifiers. e.g. `*char`, `size_t`, `uint16_t`.
        // We accept whatever the lexer gives us between the start and the
        // next `,` / `)` / `;` and stringify it.
        llvm::Error Parser::parseTypeSpec(std::string &out) {
            std::string text;
            const Token &first = lexer_.peek();
            SourceSpan first_span = first.span;
            bool saw_any = false;
            while (true) {
                const Token &t = lexer_.peek();
                if (t.kind == TokenKind::COMMA || t.kind == TokenKind::RPAREN
                    || t.kind == TokenKind::SEMICOLON || t.kind == TokenKind::LBRACE
                    || t.kind == TokenKind::END_OF_FILE) {
                    break;
                }
                if (t.kind == TokenKind::STAR) {
                    text.push_back('*');
                    lexer_.consume();
                    saw_any = true;
                    continue;
                }
                if (t.kind == TokenKind::IDENT) {
                    if (saw_any && !text.empty() && text.back() != '*' && text.back() != ' ') {
                        text.push_back(' ');
                    }
                    text += t.text;
                    lexer_.consume();
                    saw_any = true;
                    continue;
                }
                return errorAt(
                    t.span,
                    "unexpected token `" + t.text + "` in type specifier"
                );
            }
            if (!saw_any) {
                return errorAt(first_span, "expected type specifier");
            }
            out = std::move(text);
            return llvm::Error::success();
        }

        llvm::Error Parser::parseRule(AST &ast) {
            Token kw = lexer_.consume();  // `rule`
            RuleNode rule;
            rule.span = kw.span;
            auto name_or = expectIdent("rule name");
            if (!name_or) {
                return name_or.takeError();
            }
            rule.name = std::move(*name_or);
            if (auto err = expect(TokenKind::LBRACE, "`{` after rule name")) {
                return err;
            }
            while (lexer_.peek().kind != TokenKind::RBRACE) {
                if (lexer_.peek().kind == TokenKind::END_OF_FILE) {
                    return errorAt(lexer_.peek().span, "unterminated rule block");
                }
                if (auto err = parseRuleBodyItem(rule)) {
                    return err;
                }
            }
            lexer_.consume();  // `}`
            if (rule.actions.empty()) {
                return errorAt(
                    rule.span,
                    "rule `" + rule.name + "` must contain at least one action"
                );
            }
            ast.rules.push_back(std::move(rule));
            return llvm::Error::success();
        }

        llvm::Error Parser::parseRuleBodyItem(RuleNode &rule) {
            TokenKind k = lexer_.peek().kind;
            if (isClauseKeyword(k)) {
                auto c_or = parseClause();
                if (!c_or) {
                    return c_or.takeError();
                }
                rule.clauses.push_back(std::move(*c_or));
                return llvm::Error::success();
            }
            if (isActionKeyword(k)) {
                auto a_or = parseAction();
                if (!a_or) {
                    return a_or.takeError();
                }
                rule.actions.push_back(std::move(*a_or));
                return llvm::Error::success();
            }
            const Token &bad = lexer_.peek();
            return errorAt(
                bad.span,
                "unexpected token `" + bad.text + "` inside rule body"
            );
        }

        llvm::Expected< ClauseNode > Parser::parseClause() {
            Token kw = lexer_.consume();
            ClauseNode c;
            c.span = kw.span;
            switch (kw.kind) {
                case TokenKind::KW_PATTERN:             c.kind = ClauseNode::Kind::PATTERN; break;
                case TokenKind::KW_PATTERN_EITHER:      c.kind = ClauseNode::Kind::PATTERN_EITHER; break;
                case TokenKind::KW_PATTERN_INSIDE:      c.kind = ClauseNode::Kind::PATTERN_INSIDE; break;
                case TokenKind::KW_CAPTURE_PATTERN:     c.kind = ClauseNode::Kind::CAPTURE_PATTERN; break;
                case TokenKind::KW_CAPTURE_COMPARISON:  c.kind = ClauseNode::Kind::CAPTURE_COMPARISON; break;
                case TokenKind::KW_CAPTURE_TAINT:       c.kind = ClauseNode::Kind::CAPTURE_TAINT; break;
                case TokenKind::KW_WHERE:               c.kind = ClauseNode::Kind::WHERE; break;
                case TokenKind::KW_DESCRIPTION:         c.kind = ClauseNode::Kind::DESCRIPTION; break;
                case TokenKind::KW_ID:                  c.kind = ClauseNode::Kind::ID; break;
                default:
                    return errorAt(kw.span, "internal: non-clause keyword in parseClause");
            }
            if (auto err = expect(TokenKind::COLON, "`:` after clause keyword")) {
                return std::move(err);
            }
            // For description: / id: we expect a string literal; otherwise
            // the body is opaque text.
            if (c.kind == ClauseNode::Kind::DESCRIPTION || c.kind == ClauseNode::Kind::ID) {
                auto v_or = expectString("string value");
                if (!v_or) {
                    return v_or.takeError();
                }
                c.body = std::move(*v_or);
                return c;
            }
            c.body = lexer_.readCodeBlockBody(kw.span.col - 1);
            if (c.body.empty()) {
                return errorAt(kw.span, "clause body must not be empty");
            }
            return c;
        }

        llvm::Expected< ActionNode > Parser::parseAction() {
            Token kw = lexer_.consume();
            ActionNode a;
            a.span = kw.span;
            switch (kw.kind) {
                case TokenKind::KW_REWRITE: {
                    a.kind = ActionNode::Kind::REWRITE;
                    if (auto err = expect(TokenKind::COLON, "`:` after `rewrite`")) {
                        return std::move(err);
                    }
                    a.body = lexer_.readCodeBlockBody(kw.span.col - 1);
                    if (a.body.empty()) {
                        return errorAt(kw.span, "rewrite body must not be empty");
                    }
                    return a;
                }
                case TokenKind::KW_CALL: {
                    a.kind = ActionNode::Kind::CALL;
                    if (auto err = expect(TokenKind::COLON, "`:` after `call`")) {
                        return std::move(err);
                    }
                    auto call_or = parseCallExpr();
                    if (!call_or) {
                        return call_or.takeError();
                    }
                    a.call_expr = std::move(*call_or);
                    return a;
                }
                case TokenKind::KW_REMOVE: {
                    a.kind = ActionNode::Kind::REMOVE;
                    if (lexer_.peek().kind == TokenKind::COLON) {
                        lexer_.consume();
                        a.body = lexer_.readCodeBlockBody(kw.span.col - 1);
                    }
                    return a;
                }
                case TokenKind::KW_ASSERT: {
                    a.kind = ActionNode::Kind::ASSERT;
                    if (auto err = expect(TokenKind::COLON, "`:` after `assert`")) {
                        return std::move(err);
                    }
                    a.predicate = lexer_.readCodeBlockBody(kw.span.col - 1);
                    if (a.predicate.empty()) {
                        return errorAt(kw.span, "assert predicate must not be empty");
                    }
                    return a;
                }
                case TokenKind::KW_INSERT: {
                    // insert [before|after|at_entry|at_exit] (: <body> | : call: CALL)
                    const Token &pos_tok = lexer_.peek();
                    switch (pos_tok.kind) {
                        case TokenKind::KW_BEFORE:
                            a.kind = ActionNode::Kind::INSERT_BEFORE; break;
                        case TokenKind::KW_AFTER:
                            a.kind = ActionNode::Kind::INSERT_AFTER; break;
                        case TokenKind::KW_AT_ENTRY:
                            a.kind = ActionNode::Kind::INSERT_AT_ENTRY; break;
                        case TokenKind::KW_AT_EXIT:
                            a.kind = ActionNode::Kind::INSERT_AT_EXIT; break;
                        default:
                            return errorAt(
                                pos_tok.span,
                                "expected `before` / `after` / `at_entry` / `at_exit` after `insert`"
                            );
                    }
                    lexer_.consume();
                    if (auto err = expect(TokenKind::COLON, "`:` after insert position")) {
                        return std::move(err);
                    }
                    // Check for `call: …` sub-form.
                    if (lexer_.peek().kind == TokenKind::KW_CALL) {
                        lexer_.consume();  // `call`
                        if (auto err = expect(TokenKind::COLON, "`:` after inner `call`")) {
                            return std::move(err);
                        }
                        auto call_or = parseCallExpr();
                        if (!call_or) {
                            return call_or.takeError();
                        }
                        a.call_expr = std::move(*call_or);
                        return a;
                    }
                    a.body = lexer_.readCodeBlockBody(kw.span.col - 1);
                    if (a.body.empty()) {
                        return errorAt(kw.span, "insert body must not be empty");
                    }
                    return a;
                }
                default:
                    return errorAt(kw.span, "internal: non-action keyword in parseAction");
            }
        }

        llvm::Expected< CallExpr > Parser::parseCallExpr() {
            CallExpr call;
            call.span = lexer_.peek().span;
            auto first_or = expectIdent("callee namespace or function name");
            if (!first_or) {
                return first_or.takeError();
            }
            std::string current = std::move(*first_or);
            while (lexer_.peek().kind == TokenKind::COLON_COLON) {
                lexer_.consume();  // `::`
                call.namespace_path.push_back(std::move(current));
                auto next_or = expectIdent("identifier after `::`");
                if (!next_or) {
                    return next_or.takeError();
                }
                current = std::move(*next_or);
            }
            call.function_name = std::move(current);
            if (auto err = expect(TokenKind::LPAREN, "`(` after callee")) {
                return std::move(err);
            }
            if (lexer_.peek().kind != TokenKind::RPAREN) {
                while (true) {
                    auto arg_or = parseCallArg();
                    if (!arg_or) {
                        return arg_or.takeError();
                    }
                    call.args.push_back(std::move(*arg_or));
                    if (lexer_.peek().kind != TokenKind::COMMA) {
                        break;
                    }
                    lexer_.consume();  // `,`
                }
            }
            if (auto err = expect(TokenKind::RPAREN, "`)` after call arguments")) {
                return std::move(err);
            }
            return call;
        }

        llvm::Expected< CallArg > Parser::parseCallArg() {
            const Token &t = lexer_.peek();
            CallArg arg;
            arg.span = t.span;
            switch (t.kind) {
                case TokenKind::CAPTURE:
                    arg.kind = CallArg::Kind::CAPTURE;
                    arg.text = t.text;
                    lexer_.consume();
                    return arg;
                case TokenKind::VARIADIC_CAPTURE:
                    arg.kind = CallArg::Kind::VARIADIC_CAPTURE;
                    arg.text = t.text;
                    lexer_.consume();
                    return arg;
                case TokenKind::INT_LIT:
                    arg.kind = CallArg::Kind::INT_LITERAL;
                    arg.text = t.text;
                    lexer_.consume();
                    return arg;
                case TokenKind::STRING_LIT:
                    arg.kind = CallArg::Kind::STRING_LITERAL;
                    arg.text = t.text;
                    lexer_.consume();
                    return arg;
                case TokenKind::IDENT:
                    arg.kind = CallArg::Kind::BARE_IDENT;
                    arg.text = t.text;
                    lexer_.consume();
                    return arg;
                case TokenKind::AMPERSAND: {
                    arg.kind = CallArg::Kind::ADDRESS_OF;
                    lexer_.consume();
                    auto inner_or = parseCallArg();
                    if (!inner_or) {
                        return inner_or.takeError();
                    }
                    arg.inner = std::make_unique< CallArg >(std::move(*inner_or));
                    return arg;
                }
                default:
                    return errorAt(t.span, "expected call argument (capture, literal, or &expr)");
            }
        }

        llvm::Error Parser::parseContract(AST &ast) {
            Token kw = lexer_.consume();  // `contract`
            ContractNode c;
            c.span = kw.span;
            auto name_or = expectIdent("contract name");
            if (!name_or) {
                return name_or.takeError();
            }
            c.name = std::move(*name_or);
            if (auto err = expect(TokenKind::LBRACE, "`{` after contract name")) {
                return err;
            }
            while (lexer_.peek().kind != TokenKind::RBRACE) {
                if (lexer_.peek().kind == TokenKind::END_OF_FILE) {
                    return errorAt(lexer_.peek().span, "unterminated contract block");
                }
                if (auto err = parseContractBodyItem(c)) {
                    return err;
                }
            }
            lexer_.consume();  // `}`
            ast.contracts.push_back(std::move(c));
            return llvm::Error::success();
        }

        llvm::Error Parser::parseContractBodyItem(ContractNode &c) {
            TokenKind k = lexer_.peek().kind;
            if (isClauseKeyword(k)) {
                auto clause_or = parseClause();
                if (!clause_or) {
                    return clause_or.takeError();
                }
                c.clauses.push_back(std::move(*clause_or));
                return llvm::Error::success();
            }
            if (isContractClauseKeyword(k)) {
                auto cc_or = parseContractClause();
                if (!cc_or) {
                    return cc_or.takeError();
                }
                c.contract_clauses.push_back(std::move(*cc_or));
                return llvm::Error::success();
            }
            const Token &bad = lexer_.peek();
            return errorAt(
                bad.span,
                "unexpected token `" + bad.text + "` inside contract body"
            );
        }

        llvm::Expected< ContractClauseNode > Parser::parseContractClause() {
            Token kw = lexer_.consume();
            ContractClauseNode cc;
            cc.span = kw.span;
            switch (kw.kind) {
                case TokenKind::KW_REQUIRES:   cc.kind = ContractClauseNode::Kind::REQUIRES; break;
                case TokenKind::KW_ENSURES:    cc.kind = ContractClauseNode::Kind::ENSURES; break;
                case TokenKind::KW_INVARIANT:  cc.kind = ContractClauseNode::Kind::INVARIANT; break;
                case TokenKind::KW_ATTRIBUTES: cc.kind = ContractClauseNode::Kind::ATTRIBUTES; break;
                default:
                    return errorAt(kw.span, "internal: non-contract-clause keyword");
            }
            if (auto err = expect(TokenKind::COLON, "`:` after contract clause keyword")) {
                return std::move(err);
            }
            cc.body = lexer_.readCodeBlockBody(kw.span.col - 1);
            if (cc.body.empty()) {
                return errorAt(kw.span, "contract clause body must not be empty");
            }
            return cc;
        }

    } // namespace

    llvm::Expected< std::unique_ptr< AST > >
    ParseSource(llvm::StringRef source, llvm::StringRef filename) {
        Parser parser(source, filename);
        return parser.parseFile();
    }

} // namespace patchestry::patchdsl
