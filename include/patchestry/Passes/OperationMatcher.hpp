/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <regex>
#include <string>
#include <vector>

#include <llvm/ADT/StringMap.h>

#include <patchestry/YAML/BaseSpec.hpp>
#include <patchestry/YAML/ContractSpec.hpp>
#include <patchestry/YAML/PatchSpec.hpp>

// Forward declarations
namespace mlir {
    class Operation;
    class Type;
    class Value;
} // namespace mlir

namespace cir {
    class FuncOp;
    class CallOp;
} // namespace cir

namespace patchestry::passes {

    /**
     * @brief Provides matching capabilities for MLIR operations.
     *
     * The OperationMatcher class encapsulates logic for determining whether
     * an operation should be instrumented based on specifications. It supports
     * various matching criteria including operation names, function context,
     * argument patterns, and variable patterns.
     */
    class OperationMatcher
    {
      public:
        enum Mode : uint8_t { OPERATION, FUNCTION };

        /**
         * @brief Checks if an operation matches the given contract specification.
         *
         * This is the main entry point for contract matching. It evaluates all
         * matching criteria in the contract specification against the given operation.
         *
         * @param op The operation to evaluate
         * @param func The function containing the operation
         * @param spec The contract specification to match against
         * @return true if the operation matches the specification
         */
        static bool contract_action_matches(
            mlir::Operation *op, cir::FuncOp func, const contract::ContractAction &spec,
            Mode mode
        );

        /**
         * @brief Checks if an operation matches the given patch specification and, on
         *        success, binds named captures.
         *
         * This is the main entry point for patch matching. It evaluates the full
         * match criteria (name / kind / context / operand / symbol / capture index
         * bounds). Captures declared under `match.captures:` are resolved against
         * the matched op and stored in `captures_out` keyed by capture name; the
         * resulting map is read later by `handle_capture_argument` when a
         * `source: capture` argument is built.
         *
         * @param op The operation to evaluate.
         * @param func The function containing `op`.
         * @param spec The patch-action specification to match against.
         * @param mode OPERATION or FUNCTION — chooses the sub-matcher.
         * @param captures_out Output map populated on successful match. The caller
         *        is responsible for clearing `captures_out` before reuse across
         *        call sites; this function only *inserts/overwrites* entries for
         *        names declared in `match.captures`. On `false` return the
         *        contents of `captures_out` are unspecified: a capture index may
         *        have been out of range *after* the base match already succeeded,
         *        leaving partial bindings behind — treat the map as invalid.
         * @return true if the operation matches the specification and all
         *         declared captures resolve.
         */
        static bool patch_action_matches(
            mlir::Operation *op, cir::FuncOp func, const patch::PatchAction &spec, Mode mode,
            llvm::StringMap< mlir::Value > &captures_out
        );

        /**
         * @brief Extracts the called function name from a call operation.
         *
         * Used by patch / contract dispatch code to report which callee a match
         * fired on. Returns an empty string if the callee can't be resolved
         * (e.g. indirect calls whose callee SSA value has no source name).
         *
         * @param call_op The call operation
         * @return The name of the called function, or empty string on failure.
         */
        static std::string extract_callee_name(cir::CallOp call_op); // NOLINT

      private:
        /**
         * @brief Match-only primitive. Internal helper for the capture-populating
         *        overload above. Does **not** bind captures — external callers
         *        should always use the five-argument overload so that
         *        `source: capture` arguments resolve correctly.
         */
        static bool patch_action_matches(
            mlir::Operation *op, cir::FuncOp func, const patch::PatchAction &spec, Mode mode
        );

        /**
         * @brief Checks if an operation name matches the specified operation pattern.
         */
        static bool matches_operation_name( // NOLINT
            mlir::Operation *op, const std::string &operation_pattern
        );

        /**
         * @brief Checks if a function matches the specified function context criteria.
         */
        static bool matches_function_context( // NOLINT
            cir::FuncOp func, const std::vector< FunctionContext > &function_context
        );

        /**
         * @brief Checks if operation arguments match the specified argument patterns.
         */
        static bool matches_arguments( // NOLINT
            mlir::Operation *op, const std::vector< ArgumentMatch > &argument_matches
        );

        /**
         * @brief Checks if operation operands match the specified operand patterns.
         */
        static bool matches_operands( // NOLINT
            mlir::Operation *op, const std::vector< OperandMatch > &operand_matches
        );

        /**
         * @brief Checks if operation variables match the specified variable patterns.
         */
        static bool matches_variables( // NOLINT
            mlir::Operation *op, const std::vector< VariableMatch > &variable_matches
        );

        /**
         * @brief Checks if symbols match the specified symbol patterns.
         */
        static bool matches_symbols( // NOLINT
            mlir::Operation *op, const std::vector< SymbolMatch > &symbol_matches
        );

        /**
         * @brief Operation-kind sub-matcher (used by the capture-populating overload
         *        when `mode == OPERATION`).
         */
        static bool patch_action_matches_operation( // NOLINT
            mlir::Operation *op, cir::FuncOp func, const patch::MatchConfig &match
        );

        /**
         * @brief Function-call sub-matcher (used by the capture-populating overload
         *        when `mode == FUNCTION`).
         */
        static bool patch_action_matches_function_call( // NOLINT
            mlir::Operation *op, cir::FuncOp func, const patch::MatchConfig &match
        );


        /**
         * @brief Performs pattern matching with support for regex.
         *
         * @param text The text to match against
         * @param pattern The pattern (plain string or regex in /pattern/ format)
         * @return true if the text matches the pattern
         */
        static bool
        matches_pattern(const std::string &text, const std::string &pattern); // NOLINT

        /**
         * @brief Checks if a type matches the specified type pattern.
         *
         * @param type The MLIR type to check
         * @param type_pattern The type pattern to match against
         * @return true if the type matches the pattern
         */
        static bool matches_type(mlir::Type type, const std::string &type_pattern); // NOLINT

        /**
         * @brief Extracts the variable name from an operation's operands or results.
         *
         * @param op The operation to extract variable names from
         * @param index The operand/result index
         * @return The variable name if available, empty string otherwise
         */
        static std::string extract_variable_name(mlir::Operation *op, unsigned index); // NOLINT

        /**
         * @brief Extracts the name from an SSA value.
         *
         * @param value The SSA value to extract name from
         * @return The variable name if available, empty string otherwise
         */
        static std::string extract_ssa_value_name(mlir::Value value); // NOLINT

        /**
         * @brief Extracts the type from an SSA value, following the definition chain.
         *
         * @param value The SSA value to extract type from
         * @return The variable type, following through operations that may modify the original
         * type
         */
        static mlir::Type extract_ssa_value_type(mlir::Value value); // NOLINT

        /**
         * @brief Extracts symbol name from operation attributes.
         *
         * @param op The operation to extract symbol name from
         * @return The symbol name if available, empty string otherwise
         */
        static std::string extract_symbol_name(mlir::Operation *op); // NOLINT

        /**
         * @brief Extracts variable-related attributes from an operation.
         *
         * @param op The operation to extract variable attributes from
         * @return The variable name if available, empty string otherwise
         */
        static std::string extract_variable_attribute(mlir::Operation *op); // NOLINT

        /**
         * @brief Converts an MLIR type to its string representation.
         *
         * @param type The MLIR type to convert
         * @return String representation of the type
         */
        static std::string type_to_string(mlir::Type type); // NOLINT
    };

} // namespace patchestry::passes
