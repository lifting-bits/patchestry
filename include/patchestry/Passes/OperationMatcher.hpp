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

#include "PatchSpec.hpp"

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
     * an operation should be instrumented based on patch specifications. It supports
     * various matching criteria including operation names, function context,
     * argument patterns, and variable patterns.
     */
    class OperationMatcher
    {
      public:
        enum Mode : uint8_t { OPERATION, FUNCTION };

        /**
         * @brief Checks if an operation matches the given patch specification.
         *
         * This is the main entry point for operation matching. It evaluates all
         * matching criteria in the patch specification against the given operation.
         *
         * @param op The operation to evaluate
         * @param func The function containing the operation
         * @param spec The patch specification to match against
         * @return true if the operation matches the specification
         */
        static bool
        matches(mlir::Operation *op, cir::FuncOp func, const PatchAction &spec, Mode mode);

        /**
         * @brief Checks if an operation name matches the specified operation pattern.
         *
         * @param op The operation to check
         * @param operation_pattern The operation name pattern to match
         * @return true if the operation name matches
         */
        static bool matches_operation_name( // NOLINT
            mlir::Operation *op, const std::string &operation_pattern
        );

        /**
         * @brief Checks if a function matches the specified function context criteria.
         *
         * @param func The function to evaluate
         * @param function_context The function context criteria to match against
         * @return true if the function matches the criteria
         */
        static bool matches_function_context( // NOLINT
            cir::FuncOp func, const std::vector< FunctionContext > &function_context
        );

        /**
         * @brief Checks if operation arguments match the specified argument patterns.
         *
         * @param op The operation whose arguments to check
         * @param argument_matches The argument match criteria
         * @return true if the arguments match the criteria
         */
        static bool matches_arguments( // NOLINT
            mlir::Operation *op, const std::vector< ArgumentMatch > &argument_matches
        );

        /**
         * @brief Checks if operation operands match the specified operand patterns.
         *
         * @param op The operation whose operands to check
         * @param operand_matches The operand match criteria
         * @return true if the operands match the criteria
         */
        static bool matches_operands( // NOLINT
            mlir::Operation *op, const std::vector< OperandMatch > &operand_matches
        );

        /**
         * @brief Checks if operation variables match the specified variable patterns.
         *
         * This checks variables used or defined by the operation against the
         * variable match criteria.
         *
         * @param op The operation to check
         * @param variable_matches The variable match criteria
         * @return true if the variables match the criteria
         */
        static bool matches_variables( // NOLINT
            mlir::Operation *op, const std::vector< VariableMatch > &variable_matches
        );

        /**
         * @brief Checks if symbols match the specified symbol patterns.
         *
         * @param op The operation to check
         * @param symbol_matches The symbol match criteria
         * @return true if the symbols match the criteria
         */
        static bool matches_symbols( // NOLINT
            mlir::Operation *op, const std::vector< SymbolMatch > &symbol_matches
        );

        /**
         * @brief Matches operation-based criteria.
         *
         * @param op The operation to check
         * @param func The function containing the operation
         * @param match The operation match criteria
         * @return true if the operation matches the criteria
         */
        static bool matches_operation( // NOLINT
            mlir::Operation *op, cir::FuncOp func, const MatchConfig &match
        );

        /**
         * @brief Matches function call-based criteria.
         *
         * @param op The operation to check (should be a call operation)
         * @param func The function containing the call
         * @param match The function match criteria
         * @return true if the function call matches the criteria
         */
        static bool matches_function_call( // NOLINT
            mlir::Operation *op, cir::FuncOp func, const MatchConfig &match
        );

        /**
         * @brief Extracts the called function name from a call operation.
         *
         * @param call_op The call operation
         * @return The name of the called function
         */
        static std::string extract_callee_name(cir::CallOp call_op); // NOLINT

      private:
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
