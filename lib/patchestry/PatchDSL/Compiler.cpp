/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/PatchDSL/Compiler.hpp>

#include <memory>
#include <sstream>
#include <system_error>
#include <utility>

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>

#include "Lowering.hpp"
#include "Parser.hpp"

namespace patchestry::patchdsl {

    llvm::Expected< std::unique_ptr< AST > >
    ParseFile(llvm::StringRef path, const CompilerOptions &opts) {
        (void)opts;  // import paths are resolved later, in Phase 7.

        auto buf_or = llvm::MemoryBuffer::getFile(path);
        if (std::error_code ec = buf_or.getError()) {
            return llvm::createStringError(
                ec, "failed to read `%s`: %s", path.str().c_str(), ec.message().c_str()
            );
        }
        auto buf = std::move(*buf_or);
        return ParseSource(buf->getBuffer(), path);
    }

    llvm::Expected< patchestry::passes::Configuration >
    CompileFile(llvm::StringRef path, const CompilerOptions &opts) {
        auto ast_or = ParseFile(path, opts);
        if (!ast_or) {
            return ast_or.takeError();
        }
        return Lower(**ast_or, path);
    }

    namespace {

        namespace passes = patchestry::passes;
        namespace patch  = patchestry::passes::patch;

        const char *modeToString(passes::PatchInfoMode m) {
            switch (m) {
                case passes::PatchInfoMode::APPLY_BEFORE: return "apply_before";
                case passes::PatchInfoMode::APPLY_AFTER:  return "apply_after";
                case passes::PatchInfoMode::REPLACE:      return "replace";
                case passes::PatchInfoMode::NONE:         return "none";
            }
            return "none";
        }

        const char *matchKindToString(passes::MatchKind k) {
            switch (k) {
                case passes::MatchKind::OPERATION: return "operation";
                case passes::MatchKind::FUNCTION:  return "function";
                case passes::MatchKind::NONE:      return "none";
            }
            return "none";
        }

        const char *sourceToString(passes::ArgumentSourceType s) {
            switch (s) {
                case passes::ArgumentSourceType::OPERAND:      return "operand";
                case passes::ArgumentSourceType::VARIABLE:     return "variable";
                case passes::ArgumentSourceType::SYMBOL:       return "symbol";
                case passes::ArgumentSourceType::CONSTANT:     return "constant";
                case passes::ArgumentSourceType::RETURN_VALUE: return "return_value";
            }
            return "unknown";
        }

        void emitScalar(std::ostringstream &os, llvm::StringRef s) {
            // Quote if it looks ambiguous (contains ':' or starts with a
            // non-alnum that YAML might reinterpret).
            bool needs_quotes = s.empty() || s.find(':') != llvm::StringRef::npos
                || s.find('#') != llvm::StringRef::npos
                || s.find(' ') != llvm::StringRef::npos;
            if (needs_quotes) {
                os << '"';
                for (char c : s) {
                    if (c == '"') {
                        os << "\\\"";
                    } else if (c == '\\') {
                        os << "\\\\";
                    } else {
                        os << c;
                    }
                }
                os << '"';
            } else {
                os << s.str();
            }
        }

    } // namespace

    std::string ConfigurationToYAML(const passes::Configuration &cfg) {
        std::ostringstream os;
        os << "apiVersion: ";
        emitScalar(os, cfg.api_version);
        os << "\n";

        os << "metadata:\n";
        os << "  name: ";        emitScalar(os, cfg.metadata.name);        os << "\n";
        os << "  description: "; emitScalar(os, cfg.metadata.description); os << "\n";
        os << "  version: ";     emitScalar(os, cfg.metadata.version);     os << "\n";
        os << "  author: ";      emitScalar(os, cfg.metadata.author);      os << "\n";
        os << "  created: ";     emitScalar(os, cfg.metadata.created);     os << "\n";

        os << "target:\n";
        os << "  binary: "; emitScalar(os, cfg.target.binary); os << "\n";
        os << "  arch: ";   emitScalar(os, cfg.target.arch);   os << "\n";

        if (!cfg.libraries.patches.empty()) {
            os << "libraries:\n";
            os << "  patches:\n";
            for (auto const &p : cfg.libraries.patches) {
                os << "    - name: ";          emitScalar(os, p.name);          os << "\n";
                os << "      code_file: ";     emitScalar(os, p.code_file);     os << "\n";
                os << "      function_name: "; emitScalar(os, p.function_name); os << "\n";
                if (!p.parameters.empty()) {
                    os << "      parameters:\n";
                    for (auto const &param : p.parameters) {
                        os << "        - name: "; emitScalar(os, param.name); os << "\n";
                        os << "          type: "; emitScalar(os, param.type); os << "\n";
                    }
                }
            }
        }

        if (!cfg.meta_patches.empty()) {
            os << "meta_patches:\n";
            for (auto const &mp : cfg.meta_patches) {
                os << "  - name: ";        emitScalar(os, mp.name);        os << "\n";
                os << "    description: "; emitScalar(os, mp.description); os << "\n";
                os << "    patch_actions:\n";
                for (auto const &pa : mp.patch_actions) {
                    os << "      - id: ";          emitScalar(os, pa.action_id);   os << "\n";
                    os << "        description: "; emitScalar(os, pa.description); os << "\n";
                    os << "        match:\n";
                    for (auto const &m : pa.match) {
                        os << "          - name: "; emitScalar(os, m.name); os << "\n";
                        os << "            kind: " << matchKindToString(m.kind) << "\n";
                        if (!m.function_context.empty()) {
                            os << "            function_context:\n";
                            for (auto const &fc : m.function_context) {
                                os << "              - name: ";
                                emitScalar(os, fc.name);
                                os << "\n";
                            }
                        }
                    }
                    os << "        action:\n";
                    for (auto const &ac : pa.action) {
                        os << "          - mode: " << modeToString(ac.mode) << "\n";
                        os << "            patch_id: ";
                        emitScalar(os, ac.patch_id);
                        os << "\n";
                        if (!ac.arguments.empty()) {
                            os << "            arguments:\n";
                            for (auto const &arg : ac.arguments) {
                                os << "              - name: ";
                                emitScalar(os, arg.name);
                                os << "\n";
                                os << "                source: "
                                   << sourceToString(arg.source) << "\n";
                                if (arg.index) {
                                    os << "                index: " << *arg.index
                                       << "\n";
                                }
                                if (arg.symbol) {
                                    os << "                symbol: ";
                                    emitScalar(os, *arg.symbol);
                                    os << "\n";
                                }
                                if (arg.value) {
                                    os << "                value: ";
                                    emitScalar(os, *arg.value);
                                    os << "\n";
                                }
                                if (arg.is_reference && *arg.is_reference) {
                                    os << "                is_reference: true\n";
                                }
                            }
                        }
                    }
                }
            }
        }
        return os.str();
    }

} // namespace patchestry::patchdsl
