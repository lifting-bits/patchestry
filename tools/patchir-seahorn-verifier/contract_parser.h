/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/IR/Instruction.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace {
    // =========================================================================
    // Data Structures for Contract Parsing
    // =========================================================================

    /// Maximum argument index allowed in target specifications
    constexpr unsigned MAX_ARGUMENT_INDEX = 256;

    // =========================================================================
    // Parsing Constants
    // =========================================================================

    /// Length of "Arg(" prefix in target specifications
    constexpr size_t ARG_PREFIX_LENGTH = 4;

    /// Length of "min=" field in range specifications
    constexpr size_t MIN_FIELD_LENGTH = 4;

    /// Length of "max=" field in range specifications
    constexpr size_t MAX_FIELD_LENGTH = 4;

    /// Prefix for argument target specifications (e.g., "Arg(0)")
    constexpr std::string_view ARG_PREFIX = "Arg(";

    /// Target specification for return value
    constexpr std::string_view RETURN_VALUE_TARGET = "ReturnValue";

    /// Prefix for preconditions section in contract strings
    constexpr std::string_view PRECONDITIONS_PREFIX = "preconditions=[";

    /// Prefix for postconditions section in contract strings
    constexpr std::string_view POSTCONDITIONS_PREFIX = "postconditions=[";

    /// Supported predicate types for static contract verification
    ///
    /// EXTENSIBILITY NOTE: To add new predicate types:
    /// 1. Add new enum value below
    /// 2. Add entry to getPredicateKindName() in contract_parser.h
    /// 3. Add parsing logic in kvToPredicate() in contract_parser.h
    /// 4. Add code generation in main.cpp::injectPredicates()
    /// 5. Update documentation in README.md and EXTENDING.md
    enum PredicateKind {
        PK_Unknown,         // Invalid/unsupported predicate
        PK_Nonnull,         // target != null
        PK_RelNeqArgConst,  // arg[i] != constant
        PK_RelEqArgConst,   // arg[i] == constant
        PK_RelLtArgConst,   // arg[i] < constant
        PK_RelLeArgConst,   // arg[i] <= constant
        PK_RelGtArgConst,   // arg[i] > constant
        PK_RelGeArgConst,   // arg[i] >= constant
        PK_RangeRet,        // min <= return_value <= max
        PK_RangeArg,        // min <= arg[i] <= max
        PK_Alignment        // (ptrtoint(ptr) % align) == 0
    };

    /// Structure to hold parsed predicate information
    ///
    /// EXTENSIBILITY: When adding new predicate types, add fields here for the
    /// new predicate's parameters. Use a union or std::variant for complex types
    /// to avoid bloating the structure.
    ///
    /// Example for future buffer bounds predicate:
    ///   unsigned buffer_arg_index = 0;  // buffer pointer argument
    ///   unsigned size_arg_index = 0;    // size argument
    ///   size_t min_access = 0;          // minimum required access size
    struct ParsedPredicate {
        PredicateKind kind = PK_Unknown;
        std::string target;              // "Arg(N)" or "ReturnValue"
        unsigned arg_index = 0;          // Parsed from target

        // Fields for relation predicates
        int64_t constant = 0;

        // Fields for range predicates
        int64_t min_val = 0;
        int64_t max_val = 0;

        // Fields for alignment predicates
        uint64_t alignment = 0;

        bool is_precondition = true;     // true=precondition, false=postcondition
    };

    /// Result type for parse operations with error reporting
    template< typename T >
    struct ParseResult {
        std::optional< T > value;
        std::string error_message;

        bool ok() const { return value.has_value(); }

        static ParseResult success(T val) {
            return ParseResult{ std::optional< T >(std::move(val)), "" };
        }

        static ParseResult error(std::string msg) {
            return ParseResult{ std::nullopt, std::move(msg) };
        }
    };

    /// Source location information for error reporting
    struct ContractLocation {
        std::string function_name;
        unsigned instruction_index;
        std::string contract_text;
    };

    /// Work item for contract processing
    struct ContractWorkItem {
        llvm::Instruction *inst = nullptr;
        std::string function_name;
        std::string contract_text;
        unsigned instruction_index = 0;
    };

    // =========================================================================
    // ContractParser Class
    // =========================================================================

    /// Parser for static contract metadata strings
    ///
    /// This class provides robust parsing of contract metadata embedded in LLVM IR.
    /// It converts string-based contract specifications into structured ParsedPredicate objects.
    ///
    /// Usage:
    ///   ContractParser parser(verbose_mode);
    ///   std::vector<std::string> errors;
    ///   auto predicates = parser.parseStaticContractText(contract_text, &errors);
    class ContractParser {
    public:
        /// Constructor
        /// @param verbose Enable verbose logging during parsing
        inline explicit ContractParser(bool verbose) : verbose_(verbose) {}

        /// Parse a complete contract metadata string
        /// @param contract_str The contract metadata string (e.g., "preconditions=[...], postconditions=[...]")
        /// @param errors Optional output vector to collect parse errors
        /// @return Vector of successfully parsed predicates
        inline std::vector< ParsedPredicate > parseStaticContractText(
            const std::string &contract_str,
            std::vector< std::string > *errors = nullptr
        ) const {
            std::vector< ParsedPredicate > preds;

            // Edge case: empty contract string
            if (contract_str.empty()) {
                if (errors) {
                    errors->push_back("Contract string is empty");
                }
                return preds;  // Return empty vector
            }

            constexpr std::string_view PRECONDITIONS_PREFIX = "preconditions=[";
            constexpr std::string_view POSTCONDITIONS_PREFIX = "postconditions=[";

            // Find preconditions section
            size_t pre_start = contract_str.find(PRECONDITIONS_PREFIX);
            if (pre_start != std::string::npos) {
                pre_start += PRECONDITIONS_PREFIX.length();
                size_t pre_end = findMatchingClosingSquareBracket(contract_str, pre_start);
                if (pre_end != std::string::npos) {
                    std::string pre_section =
                        contract_str.substr(pre_start, pre_end - pre_start);

                    // Parse individual preconditions
                    size_t pos = 0;
                    unsigned pred_index = 0;
                    while (pos < pre_section.length()) {
                        size_t start = pre_section.find('{', pos);
                        if (start == std::string::npos)
                            break;
                        start++;

                        size_t end = findMatchingClosingBrace(pre_section, start);
                        if (end == std::string::npos) {
                            if (errors) {
                                errors->push_back(
                                    "Malformed precondition: missing closing brace"
                                );
                            }
                            break;
                        }

                        std::string pred_str = pre_section.substr(start, end - start);
                        auto kv = parseKeyValues(pred_str);
                        auto pred_result = kvToPredicate(kv);

                        if (pred_result.ok()) {
                            auto pred = *pred_result.value;
                            pred.is_precondition = true;
                            preds.push_back(pred);

                            if (verbose_) {
                                llvm::outs() << "  Parsed precondition #" << pred_index
                                             << ": kind=" << pred.kind << "\n";
                            }
                        } else {
                            if (errors) {
                                errors->push_back(
                                    "Precondition #" + std::to_string(pred_index) +
                                    ": " + pred_result.error_message
                                );
                            }
                            if (verbose_) {
                                llvm::errs() << "  Failed to parse precondition #"
                                             << pred_index << ": "
                                             << pred_result.error_message << "\n";
                            }
                        }

                        pred_index++;
                        pos = end + 1;
                    }
                }
            }

            // Find postconditions section
            size_t post_start = contract_str.find(POSTCONDITIONS_PREFIX);
            if (post_start != std::string::npos) {
                post_start += POSTCONDITIONS_PREFIX.length();
                size_t post_end = findMatchingClosingSquareBracket(contract_str, post_start);
                if (post_end != std::string::npos) {
                    std::string post_section =
                        contract_str.substr(post_start, post_end - post_start);

                    size_t pos = 0;
                    unsigned pred_index = 0;
                    while (pos < post_section.length()) {
                        size_t start = post_section.find('{', pos);
                        if (start == std::string::npos)
                            break;
                        start++;

                        size_t end = findMatchingClosingBrace(post_section, start);
                        if (end == std::string::npos) {
                            if (errors) {
                                errors->push_back(
                                    "Malformed postcondition: missing closing brace"
                                );
                            }
                            break;
                        }

                        std::string pred_str = post_section.substr(start, end - start);
                        auto kv = parseKeyValues(pred_str);
                        auto pred_result = kvToPredicate(kv);

                        if (pred_result.ok()) {
                            auto pred = *pred_result.value;
                            pred.is_precondition = false;
                            preds.push_back(pred);

                            if (verbose_) {
                                llvm::outs() << "  Parsed postcondition #" << pred_index
                                             << ": kind=" << pred.kind << "\n";
                            }
                        } else {
                            if (errors) {
                                errors->push_back(
                                    "Postcondition #" + std::to_string(pred_index) +
                                    ": " + pred_result.error_message
                                );
                            }
                            if (verbose_) {
                                llvm::errs() << "  Failed to parse postcondition #"
                                             << pred_index << ": "
                                             << pred_result.error_message << "\n";
                            }
                        }

                        pred_index++;
                        pos = end + 1;
                    }
                }
            }

            return preds;
        }

        /// Get human-readable name for a predicate kind
        inline static const char *getPredicateKindName(PredicateKind kind) {
            switch (kind) {
                case PK_Unknown: return "unknown";
                case PK_Nonnull: return "nonnull";
                case PK_RelNeqArgConst: return "relation(neq)";
                case PK_RelEqArgConst: return "relation(eq)";
                case PK_RelLtArgConst: return "relation(lt)";
                case PK_RelLeArgConst: return "relation(lte)";
                case PK_RelGtArgConst: return "relation(gt)";
                case PK_RelGeArgConst: return "relation(gte)";
                case PK_RangeRet: return "range(return)";
                case PK_RangeArg: return "range(arg)";
                case PK_Alignment: return "alignment";
                default: return "invalid";
            }
        }

        /// Get comma-separated list of supported predicate kinds
        inline static std::string getSupportedPredicateKinds() {
            return "nonnull, relation, range, alignment";
        }

    private:
        bool verbose_;

        /// Safe string trimming that handles edge cases
        /// @param s String to trim (modified in-place)
        inline static void trimString(std::string &s) {
            auto first = s.find_first_not_of(" \t");
            if (first == std::string::npos) {
                s.clear();
                return;
            }
            auto last = s.find_last_not_of(" \t");
            s = s.substr(first, last - first + 1);
        }

        /// Parse a target specification like "Arg(0)" or "ReturnValue"
        ///
        /// Validates and extracts the argument index from target specifications.
        /// For "Arg(N)", returns N (where N must be in range [0, MAX_ARGUMENT_INDEX]).
        /// For "ReturnValue", returns 0 as a sentinel value.
        ///
        /// @param target_str Input string to parse (e.g., "Arg(0)", "Arg(42)", "ReturnValue")
        /// @param target Output parameter to store the original target string
        /// @return ParseResult with argument index on success, or error message on failure
        ///
        /// @note Edge cases:
        ///   - "Arg()" returns error for empty index
        ///   - "Arg(999)" returns error if index > MAX_ARGUMENT_INDEX
        ///   - "Arg(abc)" returns error for non-numeric index
        ///   - "Arg(0" returns error for missing closing parenthesis
        inline static ParseResult< unsigned > parseTarget(
            const std::string &target_str,
            std::string &target
        ) {
            target = target_str;

            // Parse Arg(N)
            if (target_str.size() >= ARG_PREFIX_LENGTH &&
                target_str.substr(0, ARG_PREFIX_LENGTH) == ARG_PREFIX) {
                auto end_pos = target_str.find(')');
                if (end_pos == std::string::npos) {
                    return ParseResult< unsigned >::error(
                        "Missing closing parenthesis in target '" + target_str + "'"
                    );
                }

                if (end_pos <= ARG_PREFIX_LENGTH) {
                    return ParseResult< unsigned >::error(
                        "Empty argument index in target '" + target_str + "'"
                    );
                }

                std::string index_str = target_str.substr(ARG_PREFIX_LENGTH, end_pos - ARG_PREFIX_LENGTH);

                try {
                    unsigned long val = std::stoul(index_str);

                    // Validate reasonable argument index
                    if (val > MAX_ARGUMENT_INDEX) {
                        return ParseResult< unsigned >::error(
                            "Argument index too large: " + std::to_string(val) + " (max " +
                            std::to_string(MAX_ARGUMENT_INDEX) + ")"
                        );
                    }

                    return ParseResult< unsigned >::success(static_cast< unsigned >(val));

                } catch (const std::invalid_argument &) {
                    return ParseResult< unsigned >::error(
                        "Invalid argument index '" + index_str + "': not a valid number"
                    );
                } catch (const std::out_of_range &) {
                    return ParseResult< unsigned >::error(
                        "Argument index out of range: " + index_str
                    );
                }
            }

            // Parse ReturnValue
            if (target_str == RETURN_VALUE_TARGET) {
                return ParseResult< unsigned >::success(0);
            }

            return ParseResult< unsigned >::error(
                "Unknown target specification: '" + target_str +
                "' (expected 'Arg(N)' or 'ReturnValue')"
            );
        }

        /// Find the matching closing bracket, accounting for nested structures
        ///
        /// When parsing contract strings like "preconditions=[{...}]" where
        /// the content may have nested brackets like "range=[min=0, max=100]",
        /// a simple find(']') will incorrectly match a ']' inside nested structures.
        /// This function properly tracks nesting depth (both [] and {}) to find
        /// the correct matching closing bracket.
        ///
        /// @param section The string to search in
        /// @param start Starting position (should point to first character after opening '[')
        /// @return Position of matching ']', or std::string::npos if not found
        ///
        /// @example
        ///   findMatchingClosingSquareBracket("[{range=[min=0, max=100]}]", 1)
        ///   => returns last ']' position
        inline static size_t findMatchingClosingSquareBracket(
            const std::string &section,
            size_t start
        ) {
            size_t pos = start;
            int depth = 0;          // Track nesting depth for all bracket types
            bool in_quotes = false; // Track if we're inside quoted strings

            while (pos < section.length()) {
                char c = section[pos];

                // Handle quoted strings - skip bracket counting inside quotes
                if (c == '"' && (pos == start || section[pos - 1] != '\\')) {
                    in_quotes = !in_quotes;
                }

                // Count all bracket types outside of quoted strings
                if (!in_quotes) {
                    if (c == '[' || c == '{') {
                        depth++;
                    } else if (c == ']') {
                        if (depth == 0) {
                            // Found the matching closing bracket!
                            return pos;
                        }
                        depth--;
                    } else if (c == '}') {
                        depth--;
                    }
                }

                pos++;
            }

            // Not found - reached end of string without finding matching ']'
            return std::string::npos;
        }

        /// Find the matching closing brace for a predicate, accounting for nested brackets
        ///
        /// When parsing contract strings like "{kind=range, range=[min=0, max=100]}",
        /// a simple find('}') will incorrectly match the '}' inside the range value.
        /// This function properly tracks square bracket nesting depth to find the
        /// correct matching closing brace.
        ///
        /// @param section The section string to search in (preconditions or postconditions)
        /// @param start Starting position (should point to first character after opening '{')
        /// @return Position of matching '}', or std::string::npos if not found
        ///
        /// @example
        ///   findMatchingClosingBrace("kind=range, range=[min=0, max=100]}", 0)
        ///   => returns 53 (position of final '}')
        inline static size_t findMatchingClosingBrace(
            const std::string &section,
            size_t start
        ) {
            size_t pos = start;
            int bracket_depth = 0;  // Track square bracket nesting: [ increases, ] decreases
            bool in_quotes = false; // Track if we're inside quoted strings

            while (pos < section.length()) {
                char c = section[pos];

                // Handle quoted strings - skip bracket counting inside quotes
                // Check for unescaped quote character
                if (c == '"' && (pos == start || section[pos - 1] != '\\')) {
                    in_quotes = !in_quotes;
                }

                // Only count brackets outside of quoted strings
                if (!in_quotes) {
                    if (c == '[') {
                        bracket_depth++;
                    } else if (c == ']') {
                        bracket_depth--;
                    } else if (c == '}' && bracket_depth == 0) {
                        // Found the matching closing brace!
                        return pos;
                    }
                }

                pos++;
            }

            // Not found - reached end of string without finding matching '}'
            return std::string::npos;
        }

        /// Parse key-value pairs from a predicate string
        ///
        /// Extracts key=value pairs from a comma/semicolon-separated string.
        /// Handles both quoted and unquoted values, and trims whitespace from keys and values.
        ///
        /// @param pred_str Predicate string fragment (e.g., "kind=nonnull, target=Arg(0)")
        /// @return Map of key-value pairs with whitespace trimmed
        ///
        /// @note Parsing details:
        ///   - Keys and values are trimmed of leading/trailing whitespace
        ///   - Quoted values: value="quoted string with, special chars"
        ///   - Unquoted values: terminated by comma, semicolon, bracket, or end of string
        ///   - Empty keys or values are preserved in the map
        ///
        /// @example
        ///   parseKeyValues("kind=nonnull, target=Arg(0)")
        ///   => {{"kind", "nonnull"}, {"target", "Arg(0)"}}
        inline static std::map< std::string, std::string > parseKeyValues(
            const std::string &pred_str
        ) {
            std::map< std::string, std::string > kv;
            size_t pos = 0;

            while (pos < pred_str.length()) {
                // Find next key
                size_t eq_pos = pred_str.find('=', pos);
                if (eq_pos == std::string::npos)
                    break;

                std::string key = pred_str.substr(pos, eq_pos - pos);
                // Trim whitespace
                trimString(key);

                // Find value (could be quoted or not)
                pos            = eq_pos + 1;
                std::string value;

                if (pos < pred_str.length() && pred_str[pos] == '"') {
                    // Quoted value
                    pos++;
                    size_t end_quote = pred_str.find('"', pos);
                    if (end_quote != std::string::npos) {
                        value = pred_str.substr(pos, end_quote - pos);
                        pos   = end_quote + 1;
                    }
                } else {
                    // Unquoted value - find next delimiter, accounting for nested brackets
                    // Need to skip over nested [...] structures in values like range=[min=0, max=100]
                    size_t value_start = pos;
                    int bracket_depth = 0;

                    while (pos < pred_str.length()) {
                        char c = pred_str[pos];

                        if (c == '[') {
                            bracket_depth++;
                        } else if (c == ']') {
                            bracket_depth--;
                        } else if (bracket_depth == 0 && (c == ',' || c == ';' || c == ']' || c == '}')) {
                            // Found a delimiter at depth 0
                            break;
                        }

                        pos++;
                    }

                    value = pred_str.substr(value_start, pos - value_start);
                }

                // Trim whitespace from value
                trimString(value);

                kv[key] = value;

                // Skip delimiter
                if (pos < pred_str.length() && (pred_str[pos] == ',' || pred_str[pos] == ';'))
                    pos++;
                while (pos < pred_str.length() && (pred_str[pos] == ' ' || pred_str[pos] == '\t'))
                    pos++;
            }

            return kv;
        }

        /// Convert key-value map to a structured predicate
        ///
        /// Validates and converts a parsed key-value map into a strongly-typed ParsedPredicate.
        /// Performs comprehensive validation of all required and optional fields.
        ///
        /// @param kv Map of key-value pairs from parseKeyValues()
        /// @return ParseResult with ParsedPredicate on success, or detailed error message on failure
        ///
        /// @note Required fields:
        ///   - kind: Predicate type (nonnull, relation, range, alignment)
        ///
        /// @note Optional fields (depending on kind):
        ///   - target: Arg(N) or ReturnValue (required for most kinds)
        ///   - relation: Comparison operator for relation predicates (eq, neq, lt, lte, gt, gte)
        ///   - value: Integer constant for relation predicates
        ///   - range: Range specification [min=X, max=Y] for range predicates
        ///   - align: Alignment value (must be power of 2) for alignment predicates
        ///
        /// @note Validation performed:
        ///   - Argument indices must be in range [0, MAX_ARGUMENT_INDEX]
        ///   - Integer constants must be parseable and in range
        ///   - Alignment values must be powers of 2
        ///   - Range min must be <= max
        inline static ParseResult< ParsedPredicate > kvToPredicate(
            const std::map< std::string, std::string > &kv
        ) {
            ParsedPredicate pred;

            // Require 'kind' field
            auto kind_it = kv.find("kind");
            if (kind_it == kv.end()) {
                return ParseResult< ParsedPredicate >::error(
                    "Missing required 'kind' field in predicate. Supported kinds: " +
                    ContractParser::getSupportedPredicateKinds()
                );
            }

            std::string kind_str = kind_it->second;

            // Parse target if present
            auto target_it = kv.find("target");
            if (target_it != kv.end()) {
                auto target_result = parseTarget(target_it->second, pred.target);
                if (!target_result.ok()) {
                    return ParseResult< ParsedPredicate >::error(
                        "Invalid target: " + target_result.error_message
                    );
                }
                pred.arg_index = *target_result.value;
            }

            // Determine predicate kind
            if (kind_str == "nonnull") {
                pred.kind = PK_Nonnull;

            } else if (kind_str == "relation") {
                // Need to look at relation field
                auto rel_it = kv.find("relation");
                auto val_it = kv.find("value");

                if (rel_it == kv.end()) {
                    return ParseResult< ParsedPredicate >::error(
                        "Missing 'relation' field for relation predicate. "
                        "Expected: relation=<eq|neq|lt|lte|gt|gte>"
                    );
                }
                if (val_it == kv.end()) {
                    return ParseResult< ParsedPredicate >::error(
                        "Missing 'value' field for relation predicate. "
                        "Expected: value=<integer>"
                    );
                }

                std::string rel = rel_it->second;

                try {
                    pred.constant = std::stoll(val_it->second);
                    // Explicit validation that constant fits in int64_t (redundant but clear)
                    if (pred.constant < INT64_MIN || pred.constant > INT64_MAX) {
                        return ParseResult< ParsedPredicate >::error(
                            "Constant value out of valid int64 range: " + val_it->second
                        );
                    }
                } catch (const std::invalid_argument &) {
                    return ParseResult< ParsedPredicate >::error(
                        "Invalid constant value: '" + val_it->second +
                        "' (expected integer)"
                    );
                } catch (const std::out_of_range &) {
                    return ParseResult< ParsedPredicate >::error(
                        "Constant value out of range (must fit in int64): " + val_it->second
                    );
                }

                // Map relation string to predicate kind
                if (rel == "neq") {
                    pred.kind = PK_RelNeqArgConst;
                } else if (rel == "eq") {
                    pred.kind = PK_RelEqArgConst;
                } else if (rel == "lt") {
                    pred.kind = PK_RelLtArgConst;
                } else if (rel == "lte") {
                    pred.kind = PK_RelLeArgConst;
                } else if (rel == "gt") {
                    pred.kind = PK_RelGtArgConst;
                } else if (rel == "gte") {
                    pred.kind = PK_RelGeArgConst;
                } else {
                    return ParseResult< ParsedPredicate >::error(
                        "Unknown relation: '" + rel +
                        "' (expected: eq, neq, lt, lte, gt, gte)"
                    );
                }

            } else if (kind_str == "range") {
                // Validate target
                if (pred.target == RETURN_VALUE_TARGET) {
                    pred.kind = PK_RangeRet;
                } else if (pred.target.size() >= 3 && pred.target.substr(0, 3) == ARG_PREFIX.substr(0, 3)) {
                    pred.kind = PK_RangeArg;
                } else {
                    return ParseResult< ParsedPredicate >::error(
                        std::string("Range predicate requires target to be Arg(N) or ") +
                        std::string(RETURN_VALUE_TARGET)
                    );
                }

                auto range_it = kv.find("range");
                if (range_it == kv.end()) {
                    return ParseResult< ParsedPredicate >::error(
                        "Missing 'range' field for range predicate"
                    );
                }

                std::string range_str = range_it->second;
                size_t min_pos = range_str.find("min=");
                size_t max_pos = range_str.find("max=");

                if (min_pos == std::string::npos || max_pos == std::string::npos) {
                    return ParseResult< ParsedPredicate >::error(
                        "Range must specify both min= and max=: '" + range_str + "'"
                    );
                }

                try {
                    min_pos += MIN_FIELD_LENGTH;
                    size_t min_end = range_str.find_first_of(",]", min_pos);
                    if (min_end == std::string::npos) {
                        min_end = range_str.length();  // Use rest of string
                    }
                    std::string min_str = range_str.substr(min_pos, min_end - min_pos);
                    trimString(min_str);  // Trim whitespace before parsing
                    pred.min_val = std::stoll(min_str);

                    max_pos += MAX_FIELD_LENGTH;
                    size_t max_end = range_str.find_first_of(",]", max_pos);
                    if (max_end == std::string::npos) {
                        max_end = range_str.length();  // Use rest of string
                    }
                    std::string max_str = range_str.substr(max_pos, max_end - max_pos);
                    trimString(max_str);  // Trim whitespace before parsing
                    pred.max_val = std::stoll(max_str);

                } catch (const std::exception &e) {
                    return ParseResult< ParsedPredicate >::error(
                        "Failed to parse range values in '" + range_str + "': " +
                        e.what()
                    );
                }

                // Validate range
                if (pred.min_val > pred.max_val) {
                    return ParseResult< ParsedPredicate >::error(
                        "Invalid range: min (" + std::to_string(pred.min_val) +
                        ") > max (" + std::to_string(pred.max_val) + ")"
                    );
                }

            } else if (kind_str == "alignment") {
                pred.kind = PK_Alignment;
                auto align_it = kv.find("align");

                if (align_it == kv.end()) {
                    return ParseResult< ParsedPredicate >::error(
                        "Missing 'align' field for alignment predicate"
                    );
                }

                try {
                    pred.alignment = std::stoull(align_it->second);
                } catch (const std::invalid_argument &) {
                    return ParseResult< ParsedPredicate >::error(
                        "Invalid alignment value: '" + align_it->second +
                        "' (expected positive integer)"
                    );
                } catch (const std::out_of_range &) {
                    return ParseResult< ParsedPredicate >::error(
                        "Alignment value out of range: " + align_it->second
                    );
                }

                // Validate alignment is power of 2
                if (pred.alignment == 0 ||
                    (pred.alignment & (pred.alignment - 1)) != 0) {
                    return ParseResult< ParsedPredicate >::error(
                        "Alignment must be a power of 2, got: " +
                        std::to_string(pred.alignment)
                    );
                }

            } else {
                return ParseResult< ParsedPredicate >::error(
                    "Unknown predicate kind: '" + kind_str +
                    "' (supported: " + getSupportedPredicateKinds() + ")"
                );
            }

            return ParseResult< ParsedPredicate >::success(pred);
        }
    };

} // anonymous namespace
