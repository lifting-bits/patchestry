/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <string>

#include <patchestry/Passes/PatchSpec.hpp>

// Forward declarations to minimize header dependencies
namespace mlir {
    class Operation;
    class Type;
    class Value;
} // namespace mlir

namespace patchestry::passes {

    /**
     * @brief A class for matching MLIR operations against patch specifications.
     *
     * The OperationMatcher class provides functionality to determine if a given
     * MLIR operation matches the criteria specified in patch specifications.
     * It supports various matching criteria including operation types, symbols,
     * function names, arguments, and variables.
     */
    class OperationMatcher
    {
        const PatchConfiguration &config_;

      public:
        /**
         * @brief Construct a new OperationMatcher object.
         *
         * @param config The patch configuration containing the patch specifications
         */
        explicit OperationMatcher(const PatchConfiguration &config);

        /**
         * @brief Find all patch specifications that match the given operation.
         *
         * @param op The MLIR operation to match against
         * @return std::vector<const PatchSpec*> Vector of pointers to matching patch
         * specifications
         */
        std::vector< const PatchSpec * > find_matching_spec(mlir::Operation *op) const;

        /**
         * @brief Check if a specific operation matches a patch specification.
         *
         * @param op The MLIR operation to check
         * @param spec The patch specification to match against
         * @return true if the operation matches the specification, false otherwise
         */
        bool matches_operation(mlir::Operation *op, const PatchSpec &spec) const;

      private:
        /**
         * @brief Check if an operation matches a specific kind (e.g., "function", "call",
         * "load").
         *
         * @param op The MLIR operation to check
         * @param kind The kind string to match against
         * @return true if the operation matches the kind, false otherwise
         */
        bool matchesKind(mlir::Operation *op, const std::string &kind) const;

        /**
         * @brief Check if an operation matches a specific symbol.
         *
         * @param op The MLIR operation to check
         * @param symbol The symbol string to match against
         * @return true if the operation matches the symbol, false otherwise
         */
        bool matchesSymbol(mlir::Operation *op, const std::string &symbol) const;

        /**
         * @brief Check if an operation is within a function with the specified name.
         *
         * @param op The MLIR operation to check
         * @param function_name The function name to match against
         * @return true if the operation is in a matching function, false otherwise
         */
        bool matchesFunctionName(mlir::Operation *op, const std::string &function_name) const;

        /**
         * @brief Check if an operation's arguments match the specified criteria.
         *
         * @param op The MLIR operation to check
         * @param arg_matches Vector of argument match criteria
         * @return true if all argument criteria are satisfied, false otherwise
         */
        bool matchesArguments(
            mlir::Operation *op, const std::vector< ArgumentMatch > &arg_matches
        ) const;

        /**
         * @brief Check if an operation's variables match the specified criteria.
         *
         * @param op The MLIR operation to check
         * @param var_matches Vector of variable match criteria
         * @return true if all variable criteria are satisfied, false otherwise
         */
        bool matchesVariables(
            mlir::Operation *op, const std::vector< VariableMatch > &var_matches
        ) const;

        /**
         * @brief Check if a value matches a pattern (supports both exact and regex matching).
         *
         * @param value The string value to check
         * @param pattern The pattern to match against (regex patterns are enclosed in forward
         * slashes)
         * @return true if the value matches the pattern, false otherwise
         */
        bool matchesPattern(const std::string &value, const std::string &pattern) const;

        /**
         * @brief Get a string representation of an MLIR type.
         *
         * @param type The MLIR type to convert to string
         * @return std::string String representation of the type
         */
        std::string getTypeString(mlir::Type type) const;
    };

} // namespace patchestry::passes
