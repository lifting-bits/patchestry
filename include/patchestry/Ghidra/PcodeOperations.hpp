/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <patchestry/Ghidra/Pcode.hpp>
#include <patchestry/Ghidra/PcodeTypes.hpp>

namespace patchestry::ghidra {
    struct Varnode;
    struct Variable;
    struct Operation;
    struct BasicBlock;
    struct FunctionPrototype;
    struct Function;
    struct Program;

} // namespace patchestry::ghidra

namespace patchestry::ghidra {
    // ARM EXC_RETURN magic values occupy 0xFFFFFFE0..0xFFFFFFFF; a constant
    // BRANCHIND target at or above this is an exception return (emitters
    // loud-fail on it). Mirrors util.firmware.InterruptAnalysis.EXC_RETURN_LOW.
    inline constexpr std::uint32_t kArmExcReturnLow = 0xFFFFFFE0U;

    using TypeMap = std::unordered_map< std::string, std::shared_ptr< VarnodeType > >;

    using FunctionMap = std::unordered_map< std::string, Function >;

    using BasicBlockMap = std::unordered_map< std::string, BasicBlock >;

    using VariableMap = std::unordered_map< std::string, Variable >;

    struct Varnode
    {
        enum Kind {
            VARNODE_UNKNOWN = 0,
            VARNODE_GLOBAL,
            VARNODE_LOCAL,
            VARNODE_PARAM,
            VARNODE_FUNCTION,
            VARNODE_TEMPORARY,
            VARNODE_CONSTANT,
            VARNODE_STRING,
            VARNODE_INTRINSIC,
        };

        static Varnode::Kind ConvertToKind(const std::string &kdd) {
            static const std::unordered_map< std::string, Varnode::Kind > kind_map = {
                {   "unknown",   VARNODE_UNKNOWN },
                {    "global",    VARNODE_GLOBAL },
                {     "local",     VARNODE_LOCAL },
                { "parameter",     VARNODE_PARAM },
                {  "function",  VARNODE_FUNCTION },
                { "temporary", VARNODE_TEMPORARY },
                {  "constant",  VARNODE_CONSTANT },
                {    "string",    VARNODE_STRING },
                { "intrinsic", VARNODE_INTRINSIC }
            };

            // if kind is not present in the map, return varnode_unknown
            auto iter = kind_map.find(kdd);
            return iter != kind_map.end() ? iter->second : VARNODE_UNKNOWN;
        }

#ifdef ENABLE_DEBUG
        std::string dump() const {
            std::string result;
            result += "Varnode {\n";

            // Convert kind to string
            std::string kind_str;
            switch (kind) {
                case VARNODE_UNKNOWN:
                    kind_str = "UNKNOWN";
                    break;
                case VARNODE_GLOBAL:
                    kind_str = "GLOBAL";
                    break;
                case VARNODE_LOCAL:
                    kind_str = "LOCAL";
                    break;
                case VARNODE_PARAM:
                    kind_str = "PARAM";
                    break;
                case VARNODE_FUNCTION:
                    kind_str = "FUNCTION";
                    break;
                case VARNODE_TEMPORARY:
                    kind_str = "TEMPORARY";
                    break;
                case VARNODE_CONSTANT:
                    kind_str = "CONSTANT";
                    break;
                case VARNODE_STRING:
                    kind_str = "STRING";
                    break;
                case VARNODE_INTRINSIC:
                    kind_str = "INTRINSIC";
                    break;
            }

            result += "  kind: " + kind_str + "\n";
            result += "  size: " + std::to_string(size) + "\n";
            result += "  type_key: " + type_key + "\n";

            if (operation) { result += "  operation: " + *operation + "\n"; }
            if (function) { result += "  function: " + *function + "\n"; }
            if (value) { result += "  value: " + std::to_string(*value) + "\n"; }
            if (string_value) { result += "  string_value: " + *string_value + "\n"; }
            if (global) { result += "  global: " + *global + "\n"; }

            result += "}";
            return result;
        }
#endif

        Kind kind;
        uint32_t size;
        std::string type_key;

        std::optional< std::string > operation;
        std::optional< std::string > function;
        std::optional< uint32_t > value;
        std::optional< std::string > string_value;
        std::optional< std::string > global;
    };

    struct Variable
    {
        std::string name;
        std::string type;
        uint32_t size;
        std::string key;
    };

    struct OperationTarget

    {
        Varnode::Kind kind;
        std::optional< std::string > function;
        std::optional< std::string > operation;
        std::optional< std::string > global;   // For CALLIND global var targets
        std::optional< std::string > type_key; // Type of the target
        // Literal address for kind:"address" targets (unresolved callee).
        std::optional< std::string > address;
        bool is_noreturn;

        // ARM system-userop classification from the Ghidra UseropClassifier
        // pre-pass (optional; absent on older serializer output).
        // `intrinsic_class`: arch-neutral tag ("irq_mask_set", "sysreg_read",
        // "coproc_load", ...); its presence is the "is classified" signal.
        // `system_register`: decoded register if resolvable ("BASEPRI", "PRIMASK").
        std::optional< std::string > intrinsic_class;
        std::optional< std::string > system_register;
    };

    // One arm of a BRANCHIND switch. `value` is meaningful only when
    // `is_default == false`; for the default arm `target_block` names
    // the `default:` destination.
    struct SwitchCase
    {
        int64_t value;
        std::string target_block;
        bool has_exit = false;
        bool is_default = false;
    };

    struct Operation
    {
        Mnemonic mnemonic;
        std::optional< Varnode > output;
        std::vector< Varnode > inputs;

        std::string key;
        std::string parent_block_key;

        // Parameter/variable declaration
        std::optional< std::string > name;
        std::optional< std::string > type;
        std::optional< uint32_t > index;

        // Call Operation
        std::optional< OperationTarget > target;
        std::optional< bool > has_return_value;

        // CALLOTHER BUILTIN_STRINGDATA: literal string the op produces,
        // resolved by the Ghidra serialiser from the instruction's data
        // references.  Consumed by handle_stringdata.
        std::optional< std::string > string_value;

        // Branch Operation
        std::optional< std::string > target_block;

        // Cond Branch
        std::optional< std::string > taken_block;
        std::optional< std::string > not_taken_block;
        std::optional< Varnode > condition;
        std::optional< std::string > address;

        // Indirect Branch: Ghidra-resolved jump-table successors (may be empty)
        std::vector< std::string > successor_blocks;

        // Switch recovery — all three are independently optional
        std::optional< std::string > fallback_block;  // direct-goto fallback when no case matches
        std::optional< Varnode > switch_input;         // original integer discriminant
        std::vector< SwitchCase > switch_cases;        // case value → target block
    };

    struct BasicBlock
    {
        std::shared_ptr< BasicBlock > parent;
        std::string key;
        std::unordered_map< std::string, Operation > operations;
        std::vector< std::string > ordered_operations;
        bool is_entry_block;
    };

    struct FunctionPrototype
    {
        std::vector< std::string > parameters;
        std::string rttype_key;
        std::string calling_convention;
        bool is_variadic;
        bool is_noreturn;
    };

    // One contiguous range of a function's body.
    struct AddressRange
    {
        std::string start;
        std::string end;
    };

    // Mirror of Ghidra's structured region tree (DecompInterface.structureGraph).
    // Leaves (kind == "plain") name an original basic block; interior nodes
    // carry children. Kinds: PcodeBlock::typeToName.
    struct RegionNode
    {
        std::string kind;
        int ghidra_index = 0;
        // Leaf-only.
        std::optional< std::string > block;
        std::vector< RegionNode > children;
        // Non-empty only for `goto`/`ifgoto`/`multigoto`.
        std::vector< std::string > goto_targets;
        // Optional so older JSON stays valid.
        std::optional< int > goto_type;
        std::optional< std::string > condition_opcode;
    };

    struct Function
    {
        std::string name;          // original (possibly mangled) symbol name
        std::string display_name;  // C-safe identifier (demangled or same as name)
        bool is_intrinsic = false; // true for SLEIGH CALLOTHER userop records
        FunctionPrototype prototype;
        std::string key;
        std::string entry_block;
        std::unordered_map< std::string, BasicBlock > basic_blocks;

        // Optional schema fields; empty on older serializer outputs.
        std::string entry_point;
        std::vector< AddressRange > address_ranges;

        // Interrupt-handler metadata from the Ghidra InterruptAnalysis pass.
        // When set, the function is emitted void(void) with an interrupt
        // attribute. interrupt_kind: "" (Generic/M-profile) or IRQ/FIQ/SWI/
        // ABORT/UNDEF (classic A/R).
        bool is_interrupt = false;
        std::optional< std::string > interrupt_kind;

        // Absent when Ghidra didn't produce one; consumers fall back
        // to CFG-based structuring.
        std::optional< RegionNode > region;
    };

    struct Program
    {
        std::optional< std::string > arch;
        std::optional< std::string > lang;
        std::optional< std::string > format;
        std::unordered_map< std::string, Function > serialized_functions;
        std::unordered_map< std::string, std::shared_ptr< VarnodeType > > serialized_types;
        std::unordered_map< std::string, Variable > serialized_globals;
    };
} // namespace patchestry::ghidra
