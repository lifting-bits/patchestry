/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include "patchestry/Util/Warnings.hpp"

PATCHESTRY_RELAX_WARNINGS
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
PATCHESTRY_UNRELAX_WARNINGS

#include <cassert>
#include <string_view>

#include "patchestry/Ghidra/Deserialize.hpp"

// NOLINTBEGIN(readability-identifier-naming)

namespace llvm {
    template<>
    struct DenseMapInfo< std::string >
    {
        static inline auto getEmptyKey() -> std::string { return "<<<EMPTY KEY>>>"; }

        static inline auto getTombstoneKey() -> std::string { return "<<<TOMBSTONE KEY>>>"; }

        static auto getHashValue(const std::string &Val) -> unsigned {
            return static_cast< unsigned >(llvm::hash_value(Val));
        }

        static auto isEqual(const std::string &LHS, const std::string &RHS) -> bool {
            return LHS == RHS;
        }
    };
} // namespace llvm

// NOLINTEND(readability-identifier-naming)

namespace patchestry::ghidra {

    struct mlir_codegen_visitor
    {
        mlir::OpBuilder bld;
        mlir::MLIRContext *ctx;

        using operation_t  = mlir::Operation *;
        using type_t       = mlir::Type;
        using value_t      = mlir::Value;
        using string_view  = std::string_view;
        using values_ref   = llvm::ArrayRef< value_t >;
        using address_t    = std::pair< std::string, int64_t >;
        using memory_t     = llvm::ScopedHashTable< address_t, value_t >;
        using memory_scope = llvm::ScopedHashTableScope< address_t, value_t >;

        memory_t memory;

        explicit mlir_codegen_visitor(mlir::ModuleOp mod) : bld(mod), ctx(bld.getContext()) {
            assert(mod->getNumRegions() > 0 && "Module has no regions.");
            auto &reg = mod->getRegion(0);
            assert(reg.hasOneBlock() && "Region has unexpected blocks.");
            bld.setInsertionPointToStart(&*reg.begin());
        }

        auto visit(const deserialized_t &ref) -> operation_t { return std::visit(*this, ref); }

        auto get_type(const varnode_t &var) -> type_t;

        auto get_type(const std::optional< varnode_t > &var) -> type_t {
            return var ? get_type(*var) : bld.getNoneType();
        }

        auto mk_varnode(const varnode_t &var) -> value_t;
        auto mk_pcode(string_view mnemonic, type_t result, values_ref inputs) -> operation_t;
        auto mk_inst(string_view mnemonic) -> operation_t;
        auto mk_block(string_view label) -> operation_t;
        auto mk_func(string_view name) -> operation_t;

        auto operator()([[maybe_unused]] const auto &arg) -> operation_t {
            assert(false && "Unexpected ghidra type.");
            return nullptr;
        }

        auto operator()(const pcode_t &pcode) -> operation_t;
        auto operator()(const instruction_t &inst) -> operation_t;
        auto operator()(const code_block_t &blk) -> operation_t;
        auto operator()(const function_t &func) -> operation_t;
    };
} // namespace patchestry::ghidra
