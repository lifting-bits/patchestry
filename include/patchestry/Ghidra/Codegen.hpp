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

#include "patchestry/Util/Common.hpp"

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
        mlir_builder bld;
        mcontext_t *ctx;

        using string_view  = std::string_view;

        using values_ref   = llvm::ArrayRef< mlir_value >;
        using address_t    = std::pair< std::string, int64_t >;
        using memory_t     = llvm::ScopedHashTable< address_t, mlir_value >;
        using memory_scope = llvm::ScopedHashTableScope< address_t, mlir_value >;

        memory_t memory;

        explicit mlir_codegen_visitor(mlir::ModuleOp mod) : bld(mod), ctx(bld.getContext()) {
            assert(mod->getNumRegions() > 0 && "Module has no regions.");
            auto &reg = mod->getRegion(0);
            assert(reg.hasOneBlock() && "Region has unexpected blocks.");
            bld.setInsertionPointToStart(&*reg.begin());
        }

        auto visit(const deserialized_t &ref) -> mlir_operation { return std::visit(*this, ref); }

        auto get_type(const varnode_t &var) -> mlir_type;

        auto get_type(const std::optional< varnode_t > &var) -> mlir_type {
            return var ? get_type(*var) : bld.getNoneType();
        }

        auto mk_varnode(const varnode_t &var) -> mlir_value;
        auto mk_pcode(string_view mnemonic, mlir_type result, values_ref inputs) -> mlir_operation;
        auto mk_inst(string_view mnemonic) -> mlir_operation;
        auto mk_block(string_view label) -> mlir_operation;
        auto mk_func(string_view name) -> mlir_operation;

        auto operator()([[maybe_unused]] const auto &arg) -> mlir_operation {
            assert(false && "Unexpected ghidra type.");
            return nullptr;
        }

        auto operator()(const pcode_t &pcode) -> mlir_operation;
        auto operator()(const instruction_t &inst) -> mlir_operation;
        auto operator()(const code_block_t &blk) -> mlir_operation;
        auto operator()(const function_t &func) -> mlir_operation;
    };
} // namespace patchestry::ghidra
