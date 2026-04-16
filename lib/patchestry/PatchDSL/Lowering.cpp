/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "Lowering.hpp"

#include <cctype>
#include <cstdlib>
#include <sstream>
#include <unordered_map>
#include <utility>

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Path.h>

namespace patchestry::patchdsl {

    namespace passes   = patchestry::passes;
    namespace patch    = patchestry::passes::patch;
    namespace contract = patchestry::passes::contract;

    namespace {

        llvm::Error errorAt(llvm::StringRef path, SourceSpan span, std::string message) {
            return llvm::createStringError(
                std::errc::invalid_argument,
                "%s:%u:%u: error: %s",
                path.str().c_str(),
                span.line,
                span.col,
                message.c_str()
            );
        }

        // Convert the DSL's `__` namespace separator to `::` for the
        // downstream `function_name` field (legacy convention).
        std::string translateFnName(llvm::StringRef dsl_name) {
            std::string out;
            out.reserve(dsl_name.size());
            for (std::size_t i = 0; i < dsl_name.size();) {
                if (i + 1 < dsl_name.size() && dsl_name[i] == '_'
                    && dsl_name[i + 1] == '_') {
                    out.append("::");
                    i += 2;
                } else {
                    out.push_back(dsl_name[i]);
                    ++i;
                }
            }
            return out;
        }

        // Scan a pattern body (from `pattern:` / `pattern-inside:`) and
        // extract the captures found inside `( ... )` in source order.
        // Simple heuristic sufficient for Phase 3 fixtures.
        std::vector< std::string > captureOrderIn(llvm::StringRef body) {
            std::vector< std::string > captures;
            // Restrict to the first matched `(...)` run to avoid picking
            // up captures from surrounding `pattern-inside:` scopes.
            auto lparen = body.find('(');
            if (lparen == llvm::StringRef::npos) {
                return captures;
            }
            int depth = 0;
            std::size_t end = lparen;
            for (std::size_t i = lparen; i < body.size(); ++i) {
                if (body[i] == '(') {
                    ++depth;
                } else if (body[i] == ')') {
                    --depth;
                    if (depth == 0) {
                        end = i;
                        break;
                    }
                }
            }
            llvm::StringRef inside = body.substr(lparen + 1, end - lparen - 1);
            for (std::size_t i = 0; i < inside.size(); ++i) {
                if (inside[i] != '$') {
                    continue;
                }
                ++i;
                // Skip variadic-capture ellipsis.
                if (i + 2 < inside.size() && inside[i] == '.' && inside[i + 1] == '.'
                    && inside[i + 2] == '.') {
                    i += 3;
                }
                std::string name;
                while (i < inside.size()
                       && (std::isalnum(static_cast< unsigned char >(inside[i]))
                           || inside[i] == '_')) {
                    name.push_back(inside[i]);
                    ++i;
                }
                if (!name.empty()) {
                    captures.push_back(std::move(name));
                }
                --i;  // undo the outer ++i
            }
            return captures;
        }

        // Extract the callee identifier from a simple function-call pattern
        // such as `system($CMD)` or `mkdir($PATH, $...REST)`. Returns
        // empty if the pattern is not a plain call.
        std::string patternCallee(llvm::StringRef body) {
            std::size_t lparen = body.find('(');
            if (lparen == llvm::StringRef::npos) {
                return {};
            }
            llvm::StringRef head = body.substr(0, lparen).rtrim();
            // Trim any leading `$X =` prefix (return-value capture form).
            auto eq = head.find('=');
            if (eq != llvm::StringRef::npos) {
                head = head.substr(eq + 1).ltrim();
            }
            // Take the last identifier token of `head` (handles `ns::fn`
            // although Phase 3 patterns don't use that).
            std::size_t start = head.size();
            while (start > 0) {
                char c = head[start - 1];
                if (std::isalnum(static_cast< unsigned char >(c)) || c == '_') {
                    --start;
                } else {
                    break;
                }
            }
            return head.substr(start).str();
        }

        // Extract the enclosing function name from a `pattern-inside` body
        // like `$RET create_port(...) { ... }`. Returns empty on failure.
        std::string enclosingFnName(llvm::StringRef body) {
            // Strip a leading `$X ` (return-value capture).
            if (!body.empty() && body[0] == '$') {
                std::size_t i = 1;
                while (i < body.size()
                       && (std::isalnum(static_cast< unsigned char >(body[i]))
                           || body[i] == '_')) {
                    ++i;
                }
                body = body.substr(i).ltrim();
            }
            return patternCallee(body);
        }

        // Pull captures referenced inside $RET create_port(..., $MSG, ...) {...}
        // — needed by the `apply_at_entrypoint` path for arguments like
        // `source: variable`.
        std::vector< std::string > enclosingParamCaptures(llvm::StringRef body) {
            // Skip the optional leading return-value capture.
            if (!body.empty() && body[0] == '$') {
                std::size_t i = 1;
                while (i < body.size()
                       && (std::isalnum(static_cast< unsigned char >(body[i]))
                           || body[i] == '_')) {
                    ++i;
                }
                body = body.substr(i).ltrim();
            }
            return captureOrderIn(body);
        }

        llvm::Expected< patch::PatchSpec >
        lowerSignature(const PatchSignature &sig, llvm::StringRef code_file) {
            patch::PatchSpec ps;
            ps.name          = sig.name;
            ps.code_file     = code_file.str();
            ps.function_name = translateFnName(sig.name);
            for (auto const &p : sig.params) {
                passes::Parameter lp;
                lp.name = p.name;
                lp.type = p.type;
                ps.parameters.push_back(lp);
            }
            return ps;
        }

        struct ImportIndex {
            // alias → (code_file, signature-name → sig pointer)
            std::unordered_map<
                std::string,
                std::pair< std::string,
                           std::unordered_map< std::string, const PatchSignature * > >
            > by_alias;
        };

        ImportIndex indexImports(const std::vector< ImportNode > &imports) {
            ImportIndex idx;
            for (auto const &imp : imports) {
                auto &entry = idx.by_alias[imp.alias];
                entry.first = imp.path;
                for (auto const &sig : imp.signatures) {
                    entry.second[sig.name] = &sig;
                }
            }
            return idx;
        }

        struct LoweredArgs {
            std::vector< patch::ArgumentSource > args;
        };

        llvm::Expected< LoweredArgs > lowerCallArgs(
            llvm::StringRef path,
            const RuleNode &rule,
            const CallExpr &call,
            const std::vector< passes::Parameter > &sig_params,
            const std::vector< std::string > &pattern_captures,
            const std::vector< std::string > &enclosing_captures
        ) {
            LoweredArgs out;
            if (sig_params.size() != call.args.size()) {
                return errorAt(
                    path,
                    call.span,
                    llvm::formatv(
                        "`call:` passes {0} argument(s) but imported signature `{1}` "
                        "expects {2}",
                        call.args.size(),
                        call.function_name,
                        sig_params.size()
                    ).str()
                );
            }
            for (std::size_t i = 0; i < call.args.size(); ++i) {
                const CallArg &a   = call.args[i];
                patch::ArgumentSource as;
                as.name = sig_params[i].name;

                // Peel any &…
                const CallArg *cur = &a;
                if (cur->kind == CallArg::Kind::ADDRESS_OF) {
                    as.is_reference = true;
                    cur = cur->inner.get();
                }
                switch (cur->kind) {
                    case CallArg::Kind::CAPTURE: {
                        // Locate the capture in the rule's pattern to
                        // determine its operand index.
                        auto it = std::find(
                            pattern_captures.begin(), pattern_captures.end(), cur->text
                        );
                        if (it != pattern_captures.end()) {
                            as.source = passes::ArgumentSourceType::OPERAND;
                            as.index  = static_cast< unsigned >(
                                std::distance(pattern_captures.begin(), it)
                            );
                            break;
                        }
                        // Fallback: match captures on the enclosing
                        // `pattern-inside:` (variable source).
                        auto eit = std::find(
                            enclosing_captures.begin(), enclosing_captures.end(),
                            cur->text
                        );
                        if (eit != enclosing_captures.end()) {
                            as.source = passes::ArgumentSourceType::VARIABLE;
                            as.symbol = cur->text;
                            break;
                        }
                        return errorAt(
                            path, cur->span,
                            "capture `$" + cur->text + "` is not bound by any pattern in rule `"
                                + rule.name + "`"
                        );
                    }
                    case CallArg::Kind::VARIADIC_CAPTURE:
                        return errorAt(
                            path, cur->span,
                            "variadic captures (`$...` in call arguments) are a v2 feature"
                        );
                    case CallArg::Kind::INT_LITERAL:
                    case CallArg::Kind::STRING_LITERAL:
                        as.source = passes::ArgumentSourceType::CONSTANT;
                        as.value  = cur->text;
                        break;
                    case CallArg::Kind::BARE_IDENT:
                        // Treat as a global-symbol reference.
                        as.source = passes::ArgumentSourceType::SYMBOL;
                        as.symbol = cur->text;
                        break;
                    case CallArg::Kind::ADDRESS_OF:
                        return errorAt(
                            path, cur->span, "nested `&` not supported"
                        );
                }
                out.args.push_back(std::move(as));
            }
            return out;
        }

        llvm::Expected< patch::MetaPatchConfig >
        lowerRule(
            llvm::StringRef path,
            const RuleNode &rule,
            const ImportIndex &imports
        ) {
            // Find the primary `pattern:` clause and any
            // `pattern-inside:` clauses. Multiple `pattern-inside:`
            // clauses are allowed but we only use the first for function
            // context in Phase 3.
            const ClauseNode *pat         = nullptr;
            const ClauseNode *pat_inside  = nullptr;
            std::string description;
            for (auto const &c : rule.clauses) {
                switch (c.kind) {
                    case ClauseNode::Kind::PATTERN:
                        if (pat) {
                            return errorAt(
                                path, c.span,
                                "multiple `pattern:` clauses in one rule are not supported in v1"
                            );
                        }
                        pat = &c;
                        break;
                    case ClauseNode::Kind::PATTERN_INSIDE:
                        if (!pat_inside) {
                            pat_inside = &c;
                        }
                        break;
                    case ClauseNode::Kind::DESCRIPTION:
                        description = c.body;
                        break;
                    case ClauseNode::Kind::ID:
                    case ClauseNode::Kind::WHERE:
                    case ClauseNode::Kind::PATTERN_EITHER:
                    case ClauseNode::Kind::CAPTURE_PATTERN:
                    case ClauseNode::Kind::CAPTURE_COMPARISON:
                    case ClauseNode::Kind::CAPTURE_TAINT:
                        // Recognized but deferred to a later phase.
                        break;
                }
            }
            if (!pat) {
                return errorAt(
                    path, rule.span,
                    "rule `" + rule.name + "` must contain a `pattern:` clause"
                );
            }

            std::vector< std::string > pattern_captures    = captureOrderIn(pat->body);
            std::string callee                              = patternCallee(pat->body);
            std::string enclosing_fn;
            std::vector< std::string > enclosing_captures;
            if (pat_inside) {
                enclosing_fn        = enclosingFnName(pat_inside->body);
                enclosing_captures  = enclosingParamCaptures(pat_inside->body);
            }

            patch::MetaPatchConfig meta;
            meta.name        = rule.name;
            meta.description = description;

            patch::PatchAction action;
            action.action_id   = rule.name + "/0";
            action.description = description;

            patch::MatchConfig match;
            match.name = callee;
            match.kind = passes::MatchKind::FUNCTION;
            if (!enclosing_fn.empty()) {
                passes::FunctionContext fc;
                fc.name = enclosing_fn;
                match.function_context.push_back(fc);
            }
            action.match.push_back(std::move(match));

            // Lower each action in the rule.
            for (auto const &a : rule.actions) {
                patch::Action la;
                const CallExpr *call = nullptr;
                switch (a.kind) {
                    case ActionNode::Kind::CALL:
                        la.mode = passes::PatchInfoMode::REPLACE;
                        call    = a.call_expr ? &*a.call_expr : nullptr;
                        break;
                    case ActionNode::Kind::INSERT_BEFORE:
                        if (!a.call_expr) {
                            return errorAt(
                                path, a.span,
                                "`insert before:` without a nested `call:` is a v2 feature"
                            );
                        }
                        la.mode = passes::PatchInfoMode::APPLY_BEFORE;
                        call    = &*a.call_expr;
                        break;
                    case ActionNode::Kind::INSERT_AFTER:
                        if (!a.call_expr) {
                            return errorAt(
                                path, a.span,
                                "`insert after:` without a nested `call:` is a v2 feature"
                            );
                        }
                        la.mode = passes::PatchInfoMode::APPLY_AFTER;
                        call    = &*a.call_expr;
                        break;
                    case ActionNode::Kind::INSERT_AT_ENTRY:
                        return errorAt(
                            path, a.span,
                            "`insert at_entry:` should be expressed as a `contract` block "
                            "(APPLY_AT_ENTRYPOINT is a contract mode)"
                        );
                    case ActionNode::Kind::INSERT_AT_EXIT:
                    case ActionNode::Kind::REWRITE:
                    case ActionNode::Kind::REMOVE:
                    case ActionNode::Kind::ASSERT:
                        return errorAt(
                            path, a.span,
                            "this action kind is deferred to v2"
                        );
                }
                if (!call) {
                    return errorAt(
                        path, a.span,
                        "action is missing a callee expression"
                    );
                }

                // Resolve the import alias + sig lookup.
                if (call->namespace_path.size() != 1) {
                    return errorAt(
                        path, call->span,
                        "v1 call targets must be `ns::fn(...)`; nested namespaces are v2"
                    );
                }
                const std::string &alias = call->namespace_path[0];
                auto ait = imports.by_alias.find(alias);
                if (ait == imports.by_alias.end()) {
                    return errorAt(
                        path, call->span,
                        "call target references unknown import alias `" + alias + "`"
                    );
                }
                auto sit = ait->second.second.find(call->function_name);
                if (sit == ait->second.second.end()) {
                    return errorAt(
                        path, call->span,
                        "import `" + alias + "` has no signature `" + call->function_name
                            + "`"
                    );
                }
                const PatchSignature &sig = *sit->second;

                la.patch_id = sig.name;

                // Build sig-parameter Parameter list for argument lowering.
                std::vector< passes::Parameter > sig_params;
                for (auto const &p : sig.params) {
                    passes::Parameter pp;
                    pp.name = p.name;
                    pp.type = p.type;
                    sig_params.push_back(pp);
                }
                auto args_or = lowerCallArgs(
                    path, rule, *call, sig_params, pattern_captures, enclosing_captures
                );
                if (!args_or) {
                    return args_or.takeError();
                }
                la.arguments = std::move(args_or->args);
                action.action.push_back(std::move(la));
            }
            meta.patch_actions.push_back(std::move(action));
            return meta;
        }

    } // namespace

    llvm::Expected< passes::Configuration >
    Lower(const AST &ast, llvm::StringRef source_path) {
        passes::Configuration cfg;
        cfg.api_version = "patchestry.io/v1";
        if (ast.metadata) {
            cfg.metadata.name        = ast.metadata->name;
            cfg.metadata.description = ast.metadata->description;
            cfg.metadata.version     = ast.metadata->version;
            cfg.metadata.author      = ast.metadata->author;
            cfg.metadata.created     = ast.metadata->created;
            if (ast.metadata->target) {
                cfg.target.binary = ast.metadata->target->binary;
                cfg.target.arch   = ast.metadata->target->arch;
            }
        }

        // Resolve import paths relative to the .patch source file's
        // directory so the .patchmod carries absolute code_file paths.
        std::string base_dir = llvm::sys::path::parent_path(source_path).str();

        // Lower imports into the flat Library.patches list.
        for (auto const &imp : ast.imports) {
            std::string resolved_path = imp.path;
            if (!llvm::sys::path::is_absolute(imp.path) && !base_dir.empty()) {
                llvm::SmallVector< char > abs;
                abs.assign(base_dir.begin(), base_dir.end());
                llvm::sys::path::append(abs, imp.path);
                llvm::sys::path::remove_dots(abs);
                resolved_path = std::string(abs.data(), abs.size());
            }
            for (auto const &sig : imp.signatures) {
                auto ps_or = lowerSignature(sig, resolved_path);
                if (!ps_or) {
                    return ps_or.takeError();
                }
                cfg.libraries.patches.push_back(std::move(*ps_or));
            }
        }

        ImportIndex import_index = indexImports(ast.imports);

        for (auto const &rule : ast.rules) {
            auto meta_or = lowerRule(source_path, rule, import_index);
            if (!meta_or) {
                return meta_or.takeError();
            }
            cfg.meta_patches.push_back(std::move(*meta_or));
        }

        if (!ast.contracts.empty()) {
            return errorAt(
                source_path,
                ast.contracts.front().span,
                "contract block lowering is a Phase 3 extension (v1.3+); "
                "v1 Phase 3 supports rule/import/metadata only"
            );
        }

        return cfg;
    }

} // namespace patchestry::patchdsl
