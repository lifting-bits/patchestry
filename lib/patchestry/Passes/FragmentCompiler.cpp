/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/Passes/FragmentCompiler.hpp>

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/StringRef.h>

// Defined in Compiler.cpp.
namespace patchestry::passes {

    std::optional< std::string > emitModuleAsStringFromSource( // NOLINT
        const std::string &source, const std::string &virtual_name,
        const std::string &lang
    );

} // namespace patchestry::passes

namespace patchestry::passes::fragment_expr {

    namespace {

        constexpr bool is_ident_start(char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                || c == '_';
        }

        constexpr bool is_ident_continue(char c) {
            return is_ident_start(c) || (c >= '0' && c <= '9');
        }

        // Rewrite `$NAME` → `(*__cap_NAME)` in code positions only —
        // string / char literals and comments are passed through.
        // Parens guard precedence (e.g. `$P->f` → `(*__cap_P)->f`).
        std::string substitute_metavars(llvm::StringRef fragment) {
            std::string out;
            out.reserve(fragment.size() + 16);

            enum class Mode {
                Code,         // identifier substitution active
                StringLit,    // inside "..."
                CharLit,      // inside '...'
                LineComment,  // // ... \n
                BlockComment, // /* ... */
            };
            Mode mode = Mode::Code;

            for (std::size_t i = 0; i < fragment.size();) {
                char c = fragment[i];

                switch (mode) {
                    case Mode::StringLit:
                    case Mode::CharLit: {
                        out.push_back(c);
                        // Pass `\?` through atomically so a backslashed
                        // quote can't close the literal early.
                        if (c == '\\' && i + 1 < fragment.size()) {
                            out.push_back(fragment[i + 1]);
                            i += 2;
                            continue;
                        }
                        char closer = (mode == Mode::StringLit) ? '"' : '\'';
                        if (c == closer) {
                            mode = Mode::Code;
                        }
                        i++;
                        continue;
                    }
                    case Mode::LineComment: {
                        out.push_back(c);
                        if (c == '\n') {
                            mode = Mode::Code;
                        }
                        i++;
                        continue;
                    }
                    case Mode::BlockComment: {
                        out.push_back(c);
                        if (c == '*' && i + 1 < fragment.size()
                            && fragment[i + 1] == '/')
                        {
                            out.push_back('/');
                            i += 2;
                            mode = Mode::Code;
                            continue;
                        }
                        i++;
                        continue;
                    }
                    case Mode::Code:
                        break;
                }

                if (c == '"') {
                    mode = Mode::StringLit;
                    out.push_back(c);
                    i++;
                    continue;
                }
                if (c == '\'') {
                    mode = Mode::CharLit;
                    out.push_back(c);
                    i++;
                    continue;
                }
                if (c == '/' && i + 1 < fragment.size()) {
                    char next = fragment[i + 1];
                    if (next == '/') {
                        mode = Mode::LineComment;
                        out.append("//");
                        i += 2;
                        continue;
                    }
                    if (next == '*') {
                        mode = Mode::BlockComment;
                        out.append("/*");
                        i += 2;
                        continue;
                    }
                }
                if (c != '$') {
                    out.push_back(c);
                    i++;
                    continue;
                }

                std::size_t j = i + 1;
                if (j >= fragment.size() || !is_ident_start(fragment[j])) {
                    out.push_back('$'); // stray, defer to clang
                    i++;
                    continue;
                }
                out.append("(*__cap_");
                while (j < fragment.size() && is_ident_continue(fragment[j])) {
                    out.push_back(fragment[j]);
                    j++;
                }
                out.push_back(')');
                i = j;
            }
            return out;
        }

        // Next `;` at depth 0, skipping strings / char literals /
        // comments / paren-bracket-brace pairs. body.size() if absent.
        std::size_t find_top_level_semicolon(
            llvm::StringRef body, std::size_t start
        ) {
            int depth = 0;
            enum class M { Code, S, C, LC, BC };
            M m = M::Code;
            for (std::size_t k = start; k < body.size();) {
                char cc = body[k];
                if (m == M::S || m == M::C) {
                    if (cc == '\\' && k + 1 < body.size()) {
                        k += 2;
                        continue;
                    }
                    if (cc == ((m == M::S) ? '"' : '\'')) {
                        m = M::Code;
                    }
                    k++;
                    continue;
                }
                if (m == M::LC) {
                    if (cc == '\n') {
                        m = M::Code;
                    }
                    k++;
                    continue;
                }
                if (m == M::BC) {
                    if (cc == '*' && k + 1 < body.size()
                        && body[k + 1] == '/')
                    {
                        m = M::Code;
                        k += 2;
                        continue;
                    }
                    k++;
                    continue;
                }
                if (cc == '"') {
                    m = M::S;
                    k++;
                    continue;
                }
                if (cc == '\'') {
                    m = M::C;
                    k++;
                    continue;
                }
                if (cc == '/' && k + 1 < body.size()) {
                    char n = body[k + 1];
                    if (n == '/') {
                        m = M::LC;
                        k += 2;
                        continue;
                    }
                    if (n == '*') {
                        m = M::BC;
                        k += 2;
                        continue;
                    }
                }
                if (cc == '(' || cc == '[' || cc == '{') {
                    depth++;
                    k++;
                    continue;
                }
                if (cc == ')' || cc == ']' || cc == '}') {
                    depth--;
                    k++;
                    continue;
                }
                if (cc == ';' && depth == 0) {
                    return k;
                }
                k++;
            }
            return body.size();
        }

        // Rewrite each `return X;` / `return;` in a stmt-form body
        // to `__patchestry_return(X);` / `__patchestry_return();`.
        // String / char / comment contents pass through. The inliner
        // turns the marker calls back into real CFG ops at clone time.
        std::string preprocess_return_markers(llvm::StringRef body) {
            std::string out;
            out.reserve(body.size() + 32);

            enum class Mode {
                Code,
                StringLit,
                CharLit,
                LineComment,
                BlockComment,
            };
            Mode mode = Mode::Code;

            constexpr llvm::StringRef kReturnKw = "return";

            for (std::size_t i = 0; i < body.size();) {
                char c = body[i];

                switch (mode) {
                    case Mode::StringLit:
                    case Mode::CharLit: {
                        out.push_back(c);
                        if (c == '\\' && i + 1 < body.size()) {
                            out.push_back(body[i + 1]);
                            i += 2;
                            continue;
                        }
                        char closer = (mode == Mode::StringLit) ? '"' : '\'';
                        if (c == closer) {
                            mode = Mode::Code;
                        }
                        i++;
                        continue;
                    }
                    case Mode::LineComment: {
                        out.push_back(c);
                        if (c == '\n') {
                            mode = Mode::Code;
                        }
                        i++;
                        continue;
                    }
                    case Mode::BlockComment: {
                        out.push_back(c);
                        if (c == '*' && i + 1 < body.size()
                            && body[i + 1] == '/')
                        {
                            out.push_back('/');
                            i += 2;
                            mode = Mode::Code;
                            continue;
                        }
                        i++;
                        continue;
                    }
                    case Mode::Code:
                        break;
                }

                if (c == '"') {
                    mode = Mode::StringLit;
                    out.push_back(c);
                    i++;
                    continue;
                }
                if (c == '\'') {
                    mode = Mode::CharLit;
                    out.push_back(c);
                    i++;
                    continue;
                }
                if (c == '/' && i + 1 < body.size()) {
                    char next = body[i + 1];
                    if (next == '/') {
                        mode = Mode::LineComment;
                        out.append("//");
                        i += 2;
                        continue;
                    }
                    if (next == '*') {
                        mode = Mode::BlockComment;
                        out.append("/*");
                        i += 2;
                        continue;
                    }
                }

                // Match `return` keyword at a word boundary. The
                // preceding char must not be ident-continue (so
                // `myreturn` / `returns` don't trigger), and the
                // following char must not be ident-continue either
                // (so `returned` doesn't either).
                bool at_word_start =
                    (i == 0) || !is_ident_continue(body[i - 1]);
                if (at_word_start && c == 'r'
                    && i + kReturnKw.size() <= body.size()
                    && body.substr(i, kReturnKw.size()) == kReturnKw
                    && (i + kReturnKw.size() == body.size()
                        || !is_ident_continue(body[i + kReturnKw.size()])))
                {
                    std::size_t after_kw = i + kReturnKw.size();
                    // Skip whitespace between `return` and the value
                    // (or trailing `;`).
                    std::size_t value_start = after_kw;
                    while (value_start < body.size()
                           && (body[value_start] == ' '
                               || body[value_start] == '\t'
                               || body[value_start] == '\n'
                               || body[value_start] == '\r'))
                    {
                        value_start++;
                    }
                    if (value_start < body.size()
                        && body[value_start] == ';')
                    {
                        // `return;` (no value).
                        out.append("__patchestry_return()");
                        i = value_start; // emit the `;` next iter.
                        continue;
                    }
                    std::size_t semi_pos =
                        find_top_level_semicolon(body, value_start);
                    if (semi_pos == body.size()) {
                        // Unterminated `return …` — emit the keyword
                        // unchanged so clang surfaces the parse error
                        // with its full diagnostic context, rather
                        // than hiding it behind a marker call.
                        out.append(kReturnKw.str());
                        i = after_kw;
                        continue;
                    }
                    llvm::StringRef expr = body.substr(
                        value_start, semi_pos - value_start
                    );
                    expr = expr.trim();
                    out.append("__patchestry_return(");
                    out.append(expr.str());
                    out.push_back(')');
                    i = semi_pos; // emit the `;` next iter.
                    continue;
                }

                out.push_back(c);
                i++;
            }
            return out;
        }

        // Reject expr-form bodies that aren't a single C expression.
        // Catches leading control-flow keywords (if/while/for/...,
        // return/break/continue/goto) and any top-level `;` or `{`.
        // Statement-form bodies belong on the dedicated `stmt:` key.
        bool looks_like_statement(llvm::StringRef fragment) {
            llvm::StringRef trimmed = fragment.ltrim();
            if (trimmed.empty()) {
                return false;
            }
            if (trimmed.front() == '{') {
                return true;
            }
            static constexpr std::array< llvm::StringRef, 9 > kws = {
                "if", "while", "for", "do", "switch",
                "return", "break", "continue", "goto",
            };
            for (llvm::StringRef kw : kws) {
                if (!trimmed.starts_with(kw)) {
                    continue;
                }
                if (trimmed.size() == kw.size()) {
                    return true;
                }
                char next = trimmed[kw.size()];
                bool ident_continuation = (next >= 'a' && next <= 'z')
                    || (next >= 'A' && next <= 'Z')
                    || (next >= '0' && next <= '9') || next == '_';
                if (!ident_continuation) {
                    return true;
                }
            }
            return find_top_level_semicolon(fragment, 0) != fragment.size();
        }

        std::string make_wrapper_name(
            llvm::StringRef substituted, llvm::ArrayRef< CaptureBinding > captures
        ) {
            llvm::hash_code h = llvm::hash_value(substituted);
            for (const auto &cap : captures) {
                h = llvm::hash_combine(
                    h, llvm::hash_value(cap.name), llvm::hash_value(cap.c_type)
                );
            }
            std::ostringstream os;
            os << "__rw_h" << std::hex << static_cast< std::uint64_t >(h);
            return os.str();
        }

        // Distinct `__rwst_h…` prefix vs expr-form's `__rw_h…`. Hash
        // folds in the enclosing return type so two call sites with
        // different enclosing fns don't collide on a wrapper.
        std::string make_stmt_wrapper_name(
            llvm::StringRef substituted,
            llvm::ArrayRef< CaptureBinding > captures,
            llvm::StringRef enclosing_return_c_type
        ) {
            llvm::hash_code h = llvm::hash_value(substituted);
            for (const auto &cap : captures) {
                h = llvm::hash_combine(
                    h, llvm::hash_value(cap.name), llvm::hash_value(cap.c_type)
                );
            }
            h = llvm::hash_combine(
                h, llvm::hash_value(enclosing_return_c_type)
            );
            std::ostringstream os;
            os << "__rwst_h" << std::hex << static_cast< std::uint64_t >(h);
            return os.str();
        }

        std::string synthesise_wrapper(
            llvm::StringRef fragment, llvm::ArrayRef< CaptureBinding > captures,
            llvm::StringRef wrapper_name, llvm::StringRef return_type,
            llvm::StringRef extra_decls
        ) {
            std::ostringstream os;
            // Inline typedefs instead of <stdint.h> so the wrapper
            // doesn't need a clang resource-dir / sysroot. Widths track
            // createTargetTriple's choice of ABI (ARM:LE:32 etc.).
            os << "typedef signed char int8_t;\n";
            os << "typedef short int16_t;\n";
            os << "typedef int int32_t;\n";
            os << "typedef long long int64_t;\n";
            os << "typedef unsigned char uint8_t;\n";
            os << "typedef unsigned short uint16_t;\n";
            os << "typedef unsigned int uint32_t;\n";
            os << "typedef unsigned long long uint64_t;\n";
            // ConvertCirTypesToCTypes spells cir::BoolType as `bool`.
            os << "typedef _Bool bool;\n";
            if (!extra_decls.empty()) {
                os << extra_decls.str();
                if (extra_decls.back() != '\n') {
                    os << "\n";
                }
            }
            // `__attribute__((used))` keeps clang from dropping the
            // unreferenced static (which would emit an empty module).
            // Captures take T*: `rewriteWithExpression` provides the
            // address (source alloca or a materialised temp).
            os << "__attribute__((used)) static " << return_type.str() << " "
               << wrapper_name.str() << "(";
            bool first = true;
            for (const auto &cap : captures) {
                if (!first) {
                    os << ", ";
                }
                first = false;
                os << cap.c_type << " *__cap_" << cap.name;
            }
            if (captures.empty()) {
                os << "void";
            }
            os << ") {\n";
            std::string body = substitute_metavars(fragment);
            if (return_type == "void") {
                os << "    " << body;
                if (!body.empty() && body.back() != ';' && body.back() != '}') {
                    os << ";";
                }
                os << "\n";
            } else {
                os << "    return " << body << ";\n";
            }
            os << "}\n";
            return os.str();
        }

        // Stmt-form wrapper: void-returning, with a `noreturn`
        // marker extern in the prelude typed by the enclosing fn's
        // return type (so clang type-checks each `return X;` site).
        std::string synthesise_stmt_wrapper(
            llvm::StringRef preprocessed_body,
            llvm::ArrayRef< CaptureBinding > captures,
            llvm::StringRef wrapper_name,
            llvm::StringRef enclosing_return_c_type,
            llvm::StringRef extra_decls
        ) {
            std::ostringstream os;
            os << "typedef signed char int8_t;\n";
            os << "typedef short int16_t;\n";
            os << "typedef int int32_t;\n";
            os << "typedef long long int64_t;\n";
            os << "typedef unsigned char uint8_t;\n";
            os << "typedef unsigned short uint16_t;\n";
            os << "typedef unsigned int uint32_t;\n";
            os << "typedef unsigned long long uint64_t;\n";
            os << "typedef _Bool bool;\n";
            if (!extra_decls.empty()) {
                os << extra_decls.str();
                if (extra_decls.back() != '\n') {
                    os << "\n";
                }
            }
            os << "extern void __patchestry_return("
               << (enclosing_return_c_type == "void"
                       ? "void"
                       : enclosing_return_c_type.str())
               << ") __attribute__((noreturn));\n";
            os << "__attribute__((used)) static void "
               << wrapper_name.str() << "(";
            bool first = true;
            for (const auto &cap : captures) {
                if (!first) {
                    os << ", ";
                }
                first = false;
                os << cap.c_type << " *__cap_" << cap.name;
            }
            if (captures.empty()) {
                os << "void";
            }
            os << ") {\n    " << preprocessed_body.str();
            if (!preprocessed_body.empty()
                && preprocessed_body.back() != ';'
                && preprocessed_body.back() != '}')
            {
                os << ";";
            }
            os << "\n}\n";
            return os.str();
        }

        std::uint64_t make_cache_key(
            llvm::StringRef substituted_source,
            llvm::ArrayRef< CaptureBinding > captures, llvm::StringRef arch,
            llvm::StringRef return_c_type
        ) {
            llvm::hash_code h = llvm::hash_value(substituted_source);
            for (const auto &cap : captures) {
                h = llvm::hash_combine(
                    h, llvm::hash_value(cap.name), llvm::hash_value(cap.c_type)
                );
            }
            h = llvm::hash_combine(
                h, llvm::hash_value(arch), llvm::hash_value(return_c_type)
            );
            return static_cast< std::uint64_t >(h);
        }

        struct FragmentCache
        {
            std::unordered_map< std::uint64_t,
                                std::pair< std::string, std::string > >
                entries; // key → (wrapper_name, module_text)
            std::mutex mu;
        };

        FragmentCache &global_cache() {
            static FragmentCache cache;
            return cache;
        }

    } // namespace

    FragmentResult compile_fragment(
        llvm::StringRef fragment, llvm::ArrayRef< CaptureBinding > captures,
        llvm::StringRef arch, llvm::StringRef return_c_type, llvm::StringRef extra_decls
    ) {
        FragmentResult result;

        // Strict expr: rule. Statement-shaped bodies (control-flow keyword,
        // or any top-level `;`) belong on the dedicated `stmt:` key.
        if (looks_like_statement(fragment)) {
            result.error =
                "rewrite-mode 'expr:' must be a single C expression "
                "with no trailing ';' and no leading control-flow "
                "keyword. Use 'stmt:' for statement bodies "
                "(multi-statement, control flow, or local "
                "declarations).";
            return result;
        }

        std::string substituted = substitute_metavars(fragment);
        std::string wrapper_name = make_wrapper_name(substituted, captures);
        // Folds extra_decls into the key so two matched modules with
        // different symbol sets don't share an entry.
        std::uint64_t key = llvm::hash_combine(
            make_cache_key(substituted, captures, arch, return_c_type),
            llvm::hash_value(extra_decls)
        );

        FragmentCache &cache = global_cache();
        {
            std::lock_guard< std::mutex > lock(cache.mu);
            auto it = cache.entries.find(key);
            if (it != cache.entries.end()) {
                result.module_text = it->second.second;
                result.func_name   = it->second.first;
                return result;
            }
        }

        std::string source = synthesise_wrapper(
            fragment, captures, wrapper_name, return_c_type, extra_decls
        );

        std::string virtual_name = "rewrite_fragment.c";
        std::string lang         = arch.str();

        auto module_text = emitModuleAsStringFromSource(source, virtual_name, lang);
        if (!module_text.has_value()) {
            result.error = "clang frontend rejected the synthesised wrapper; "
                           "check the fragment for syntax errors or undeclared identifiers";
            return result;
        }

        {
            std::lock_guard< std::mutex > lock(cache.mu);
            cache.entries.emplace(
                key, std::make_pair(wrapper_name, *module_text)
            );
        }

        result.module_text = std::move(module_text);
        result.func_name   = std::move(wrapper_name);
        return result;
    }

    FragmentResult compile_stmt_fragment(
        llvm::StringRef body, llvm::ArrayRef< CaptureBinding > captures,
        llvm::StringRef arch, llvm::StringRef enclosing_return_c_type,
        llvm::StringRef extra_decls
    ) {
        FragmentResult result;

        // Marker preprocess before metavar substitute so `$NAME`
        // refs inside a `return …;` end up inside the marker call.
        std::string preprocessed = preprocess_return_markers(body);
        std::string substituted  = substitute_metavars(preprocessed);
        std::string wrapper_name = make_stmt_wrapper_name(
            substituted, captures, enclosing_return_c_type
        );
        std::uint64_t key = llvm::hash_combine(
            make_cache_key(substituted, captures, arch, enclosing_return_c_type),
            llvm::hash_value(extra_decls)
        );

        FragmentCache &cache = global_cache();
        {
            std::lock_guard< std::mutex > lock(cache.mu);
            auto it = cache.entries.find(key);
            if (it != cache.entries.end()) {
                result.module_text = it->second.second;
                result.func_name   = it->second.first;
                return result;
            }
        }

        std::string source = synthesise_stmt_wrapper(
            substituted, captures, wrapper_name, enclosing_return_c_type,
            extra_decls
        );

        std::string virtual_name = "rewrite_stmt_fragment.c";
        std::string lang         = arch.str();

        auto module_text = emitModuleAsStringFromSource(source, virtual_name, lang);
        if (!module_text.has_value()) {
            result.error =
                "clang frontend rejected the synthesised stmt-form wrapper; "
                "check the body for syntax errors, undeclared identifiers, "
                "or a `return` whose value type doesn't match the enclosing "
                "function's return type";
            return result;
        }

        {
            std::lock_guard< std::mutex > lock(cache.mu);
            cache.entries.emplace(
                key, std::make_pair(wrapper_name, *module_text)
            );
        }

        result.module_text = std::move(module_text);
        result.func_name   = std::move(wrapper_name);
        return result;
    }

} // namespace patchestry::passes::fragment_expr
