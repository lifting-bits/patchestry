/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <cassert>
#include <string_view>

#include "patchestry/Ghidra/Deserialize.hpp"

#include "patchestry/Dialect/Pcode/PcodeOps.hpp"

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

        using translation_map_t = std::unordered_map< std::string_view, mlir::StringAttr >;
        translation_map_t opcode_to_op;

        using string_view  = std::string_view;

        using values_ref   = llvm::ArrayRef< mlir_value >;
        using address_t    = std::pair< std::string, int64_t >;
        using memory_t     = llvm::ScopedHashTable< address_t, mlir_value >;
        using memory_scope = llvm::ScopedHashTableScope< address_t, mlir_value >;

        memory_t memory;

        explicit mlir_codegen_visitor(mlir::ModuleOp mod)
            : bld(mod), opcode_to_op(create_translation_map())
        {
            assert(mod->getNumRegions() > 0 && "Module has no regions.");
            auto &reg = mod->getRegion(0);
            assert(reg.hasOneBlock() && "Region has unexpected blocks.");
            bld.setInsertionPointToStart(&*reg.begin());
        }

        translation_map_t create_translation_map();

        template< typename op_t >
        mlir::StringAttr op_name() {
            return bld.getStringAttr(op_t::getOperationName());
        }

        auto visit(const deserialized_t &ref) -> mlir_operation { return std::visit(*this, ref); }

        auto get_type(const varnode_t &var) -> mlir_type;

        auto get_type(const std::optional< varnode_t > &var) -> mlir_type {
            return var ? get_type(*var) : bld.getNoneType();
        }

        auto mk_varnode(const varnode_t &var) -> mlir_value;
        auto mk_pcode(string_view mnemonic, mlir_type result, values_ref inputs) -> mlir_operation;

        template< typename op_t, typename pcode_t >
        auto mk_with_children(const pcode_t &obj) {
            const mlir::OpBuilder::InsertionGuard guard(bld);
            auto op = bld.create< op_t >(bld.getUnknownLoc(), obj.id());

            if (obj.children().empty()) {
                return op;
            }

            bld.createBlock(&op->getRegion(0));
            for (const auto &child : obj.children()) {
                visit(child);
            }

            return op;
        }

        auto operator()([[maybe_unused]] const auto &arg) -> mlir_operation {
            assert(false && "Unexpected ghidra type.");
            return nullptr;
        }

        auto operator()(const pcode_t &pcode) -> mlir_operation;
        auto operator()(const instruction_t &inst) -> mlir_operation {
            const memory_scope scope(memory);
            return mk_with_children< pc::InstOp >(inst);
        }

        auto operator()(const code_block_t &blk) -> mlir_operation {
            return mk_with_children< pc::BlockOp >(blk);
        }

        auto operator()(const function_t &func) -> mlir_operation {
            return mk_with_children< pc::FuncOp >(func);
        }
    };
} // namespace patchestry::ghidra
