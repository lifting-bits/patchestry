/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "PredicateParser.hpp"

#include <exception>
#include <limits>
#include <map>

#include <llvm/ADT/StringRef.h>

#include <patchestry/Util/Log.hpp>

namespace patchestry::klee_verifier {

    namespace {

        // Parse a contract-target string of the form `Arg(N)` or `ReturnValue`.
        //
        // On success, `target` is set to the canonical string form and `index`
        // to the parsed argument index (0 for `ReturnValue`). On any failure,
        // *both* out-parameters are reset to an unambiguously-invalid state
        // (`target` empty, `index = 0`) so a caller that accidentally ignores
        // the return value cannot silently reinterpret a malformed target as
        // `Arg(0)`. Downstream code then sees an empty target and its
        // PK_Range/PK_Nonnull/etc. dispatch naturally leaves `pred.kind`
        // at `PK_Unknown`, which `parseContractSection` drops.
        bool parseTarget(const std::string &target_str, std::string &target, unsigned &index) {
            target.clear();
            index = 0;

            if (target_str.substr(0, 4) == "Arg(") {
                auto end_pos = target_str.find(')');
                if (end_pos == std::string::npos) {
                    LOG(WARNING) << "malformed target '" << target_str
                                 << "': missing ')'\n";
                    return false;
                }
                std::string index_str = target_str.substr(4, end_pos - 4);
                if (index_str.empty()) {
                    LOG(WARNING) << "empty argument index in '" << target_str << "'\n";
                    return false;
                }
                try {
                    unsigned long parsed = std::stoul(index_str);
                    if (parsed > std::numeric_limits< unsigned >::max()) {
                        LOG(WARNING) << "argument index " << parsed
                                     << " exceeds maximum\n";
                        return false;
                    }
                    index = static_cast< unsigned >(parsed);
                } catch (const std::exception &e) {
                    LOG(WARNING) << "failed to parse argument index '"
                                 << index_str << "': " << e.what() << "\n";
                    return false;
                }
                target = target_str;
                return true;
            }

            if (target_str == "ReturnValue") {
                target = target_str;
                return true;
            }

            return false;
        }

        size_t findMatchingBracket(const std::string &str, size_t start) {
            unsigned depth = 1;
            for (size_t i = start; i < str.length(); ++i) {
                if (str[i] == '[') {
                    depth++;
                } else if (str[i] == ']') {
                    depth--;
                    if (depth == 0)
                        return i;
                }
            }
            return std::string::npos;
        }

        std::map< std::string, std::string >
        parseKeyValues(const std::string &pred_str) {
            std::map< std::string, std::string > kv;
            size_t pos = 0;

            while (pos < pred_str.length()) {
                size_t eq_pos = pred_str.find('=', pos);
                if (eq_pos == std::string::npos)
                    break;

                std::string key = pred_str.substr(pos, eq_pos - pos);
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);

                pos            = eq_pos + 1;
                std::string value;

                if (pos < pred_str.length() && pred_str[pos] == '"') {
                    pos++;
                    size_t end_quote = pred_str.find('"', pos);
                    if (end_quote != std::string::npos) {
                        value = pred_str.substr(pos, end_quote - pos);
                        pos   = end_quote + 1;
                    }
                } else if (pos < pred_str.length() && pred_str[pos] == '[') {
                    // Bracketed value — find matching ']'
                    size_t bracket_end = findMatchingBracket(pred_str, pos + 1);
                    if (bracket_end != std::string::npos) {
                        // Include content between brackets (exclusive)
                        value = pred_str.substr(pos + 1, bracket_end - pos - 1);
                        pos   = bracket_end + 1;
                    } else {
                        value = pred_str.substr(pos);
                        pos   = pred_str.length();
                    }
                } else {
                    size_t next_delim = pred_str.find_first_of(",;]}", pos);
                    if (next_delim != std::string::npos) {
                        value = pred_str.substr(pos, next_delim - pos);
                        pos   = next_delim;
                    } else {
                        value = pred_str.substr(pos);
                        pos   = pred_str.length();
                    }
                }

                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                kv[key] = value;

                if (pos < pred_str.length() && (pred_str[pos] == ',' || pred_str[pos] == ';'))
                    pos++;
                while (pos < pred_str.length() && (pred_str[pos] == ' ' || pred_str[pos] == '\t'))
                    pos++;
            }

            return kv;
        }

        ParsedPredicate kvToPredicate(const std::map< std::string, std::string > &kv) {
            ParsedPredicate pred;

            auto kind_it = kv.find("kind");
            if (kind_it == kv.end()) {
                LOG(WARNING) << "predicate missing 'kind' key\n";
                return pred;
            }

            std::string kind_str = kind_it->second;

            auto target_it = kv.find("target");
            if (target_it != kv.end()) {
                if (!parseTarget(target_it->second, pred.target, pred.arg_index)) {
                    LOG(WARNING) << "malformed target '" << target_it->second
                                 << "' — skipping predicate\n";
                    return pred; // kind remains PK_Unknown → predicate is skipped
                }
            }

            if (kind_str == "nonnull") {
                pred.kind = PK_Nonnull;
            } else if (kind_str == "relation") {
                auto rel_it = kv.find("relation");
                auto val_it = kv.find("value");

                if (rel_it != kv.end() && val_it != kv.end()
                    && !val_it->second.empty())
                {
                    std::string rel = rel_it->second;
                    try {
                        pred.constant = std::stoll(val_it->second);

                        if (rel == "neq")
                            pred.kind = PK_RelNeqArgConst;
                        else if (rel == "eq")
                            pred.kind = PK_RelEqArgConst;
                        else if (rel == "lt")
                            pred.kind = PK_RelLtArgConst;
                        else if (rel == "lte")
                            pred.kind = PK_RelLeArgConst;
                        else if (rel == "gt")
                            pred.kind = PK_RelGtArgConst;
                        else if (rel == "gte")
                            pred.kind = PK_RelGeArgConst;
                    } catch (const std::exception &e) {
                        LOG(WARNING) << "failed to parse relation value '"
                                     << val_it->second << "': " << e.what() << "\n";
                    }
                }
            } else if (kind_str == "range") {
                // Tentatively classify by target — we'll revert to PK_Unknown
                // below if min/max parsing fails or the `range` field is
                // missing/empty, so a typo like `range=[min=oops,max=10]`
                // cannot ship as a [0, parsed-max] silently-narrower bound.
                PredicateKind tentative = PK_Unknown;
                if (pred.target == "ReturnValue") {
                    tentative = PK_RangeRet;
                } else if (pred.target.substr(0, 3) == "Arg") {
                    tentative = PK_RangeArg;
                }

                auto range_it = kv.find("range");
                if (tentative != PK_Unknown && range_it != kv.end()) {
                    std::string range_str = range_it->second;
                    size_t min_pos = range_str.find("min=");
                    size_t max_pos = range_str.find("max=");

                    bool min_ok = false;
                    bool max_ok = false;

                    if (min_pos != std::string::npos) {
                        min_pos += 4;
                        size_t min_end = range_str.find_first_of(",]", min_pos);
                        std::string min_str = range_str.substr(min_pos, min_end - min_pos);
                        if (min_str.empty()) {
                            LOG(WARNING) << "empty range min value\n";
                        } else {
                            try {
                                pred.min_val = std::stoll(min_str);
                                min_ok = true;
                            } catch (const std::exception &e) {
                                LOG(WARNING) << "failed to parse range min: " << e.what() << "\n";
                            }
                        }
                    }

                    if (max_pos != std::string::npos) {
                        max_pos += 4;
                        size_t max_end = range_str.find_first_of(",]", max_pos);
                        std::string max_str = range_str.substr(max_pos, max_end - max_pos);
                        if (max_str.empty()) {
                            LOG(WARNING) << "empty range max value\n";
                        } else {
                            try {
                                pred.max_val = std::stoll(max_str);
                                max_ok = true;
                            } catch (const std::exception &e) {
                                LOG(WARNING) << "failed to parse range max: " << e.what() << "\n";
                            }
                        }
                    }

                    // Require BOTH bounds: a one-sided range that silently
                    // defaults the other side to 0 is almost certainly a
                    // typo, and "Arg(0) ∈ [0, parsed-max]" is strictly
                    // weaker than any intended full bound.
                    if (min_ok && max_ok && pred.min_val <= pred.max_val) {
                        pred.kind = tentative;
                    } else if (min_ok != max_ok) {
                        LOG(WARNING) << "range predicate requires both 'min' and "
                                        "'max'; got only one — dropping\n";
                    } else if (min_ok && max_ok) {
                        LOG(WARNING) << "range predicate has min > max ("
                                     << pred.min_val << " > " << pred.max_val
                                     << ") — dropping\n";
                    }
                }
            } else if (kind_str == "alignment") {
                auto align_it = kv.find("align");
                if (align_it == kv.end()) {
                    LOG(WARNING) << "alignment predicate missing 'align' key\n";
                } else if (align_it->second.empty()) {
                    LOG(WARNING) << "empty alignment value\n";
                } else {
                    try {
                        pred.alignment = std::stoull(align_it->second);
                        // Alignment 0 is meaningless (it would emit `x & -1 == 0`
                        // which is trivially true and wastes a constraint slot);
                        // treat it the same as a parse failure.
                        if (pred.alignment != 0) {
                            pred.kind = PK_Alignment;
                        } else {
                            LOG(WARNING) << "alignment predicate has align=0 — dropping\n";
                        }
                    } catch (const std::exception &e) {
                        LOG(WARNING) << "failed to parse alignment '"
                                     << align_it->second << "': " << e.what() << "\n";
                    }
                }
            } else {
                LOG(WARNING) << "unknown predicate kind '" << kind_str << "'\n";
            }

            // Every predicate kind we support needs a target to have any
            // effect at codegen (emitKleePredicate returns without emitting
            // when target is empty). A missing `target=` key or one that
            // `parseTarget` rejected leaves pred.target empty; without this
            // gate those predicates would parse "successfully" here, be
            // silently dropped at codegen, and evade the strict_contracts
            // accounting — exactly the failure mode the flag was designed to
            // surface. Demote to PK_Unknown so parseContractSection counts
            // them as dropped.
            if (pred.kind != PK_Unknown && pred.target.empty()) {
                LOG(WARNING) << "predicate of kind '" << kind_str
                             << "' has no target — dropping\n";
                pred.kind = PK_Unknown;
            }

            return pred;
        }

        // Parses one section (either "preconditions" or "postconditions") of a
        // contract string. Appends successfully parsed predicates to `preds` and
        // increments `dropped` for every `{...}` block whose contents fail to
        // produce a valid predicate, so the caller can surface silent drops.
        void parseContractSection(
            const std::string &contract_str, llvm::StringRef section_key,
            bool is_precondition, std::vector< ParsedPredicate > &preds,
            unsigned &dropped
        ) {
            std::string key_eq = (section_key + "=[").str();
            size_t start_key = contract_str.find(key_eq);
            if (start_key == std::string::npos)
                return;
            size_t section_start = start_key + key_eq.length();
            size_t section_end   = findMatchingBracket(contract_str, section_start);
            if (section_end == std::string::npos)
                return;

            std::string section = contract_str.substr(section_start, section_end - section_start);
            size_t pos = 0;
            while (pos < section.length()) {
                size_t start = section.find('{', pos);
                if (start == std::string::npos)
                    break;
                start++;
                size_t end = section.find('}', start);
                if (end == std::string::npos)
                    break;
                std::string pred_str = section.substr(start, end - start);
                auto kv              = parseKeyValues(pred_str);
                auto pred            = kvToPredicate(kv);
                if (pred.kind != PK_Unknown) {
                    pred.is_precondition = is_precondition;
                    preds.push_back(pred);
                } else {
                    dropped++;
                    LOG(WARNING) << section_key << " predicate dropped: {"
                                 << pred_str << "}\n";
                }
                pos = end + 1;
            }
        }

    } // namespace

    std::vector< ParsedPredicate >
    parseStaticContractText(const std::string &contract_str, unsigned &dropped) {
        std::vector< ParsedPredicate > preds;
        parseContractSection(contract_str, "preconditions", true, preds, dropped);
        parseContractSection(contract_str, "postconditions", false, preds, dropped);
        return preds;
    }

} // namespace patchestry::klee_verifier
