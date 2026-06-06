/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/AST/IntrinsicHandlers.hpp>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/OperationKinds.h>
#include <clang/Sema/Sema.h>
#include <llvm/Support/Casting.h>

#include <array>
#include <cctype>
#include <optional>

#include <patchestry/AST/OperationBuilder.hpp>
#include <patchestry/AST/Utils.hpp>
#include <patchestry/Ghidra/PcodeOperations.hpp>
#include <patchestry/Util/Log.hpp>

namespace patchestry::ast {

    namespace {

        bool starts_with(std::string_view value, std::string_view prefix) {
            return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
        }

        bool ends_with(std::string_view value, std::string_view suffix) {
            return value.size() >= suffix.size()
                && value.substr(value.size() - suffix.size()) == suffix;
        }

        std::string to_lower_ascii(std::string_view value) {
            std::string result;
            result.reserve(value.size());
            for (auto c : value) {
                result.push_back(
                    static_cast< char >(std::tolower(static_cast< unsigned char >(c)))
                );
            }
            return result;
        }

        bool is_hex_digit(char c) {
            return std::isxdigit(static_cast< unsigned char >(c)) != 0;
        }

        bool looks_like_serializer_type_id(std::string_view suffix) {
            // Current Ghidra serialization assigns return type IDs as t<hex>
            // (for example, "__aarch64_ldadd4_acq_rel:t0").
            if (suffix.size() < 2 || suffix[0] != 't') { return false; }

            for (auto c : suffix.substr(1)) {
                if (!is_hex_digit(c)) { return false; }
            }

            return true;
        }

        bool looks_like_type_suffix(std::string_view suffix) {
            if (suffix.empty()) { return false; }

            if (looks_like_serializer_type_id(suffix)) { return true; }

            static constexpr std::array< std::string_view, 35 > type_suffixes = {
                "void",    "bool",    "char",     "uchar",    "short",     "ushort",  "int",
                "uint",    "long",    "ulong",    "llong",    "ullong",    "size_t",  "ssize_t",
                "ptr",     "intptr",  "uintptr",  "intptr_t", "uintptr_t", "int8",    "uint8",
                "int16",   "uint16",  "int32",    "uint32",   "int64",     "uint64",  "int8_t",
                "uint8_t", "int16_t", "uint16_t", "int32_t",  "uint32_t",  "int64_t", "uint64_t"
            };

            auto lower_suffix = to_lower_ascii(suffix);
            // Serializer JSON type keys use the type_* namespace; keep that
            // path flexible while matching bare language suffixes exactly.
            if (starts_with(lower_suffix, "type_")) { return true; }
            for (auto type_suffix : type_suffixes) {
                if (lower_suffix == type_suffix) { return true; }
            }
            return false;
        }

        std::string strip_return_type_suffix(std::string_view label) {
            // Primary path: Ghidra's current JSON serialization uses a colon to
            // attach the return-type id, so "name:t0" or "name:int" strips cleanly.
            auto colon_pos = label.rfind(':');
            if (colon_pos != std::string_view::npos) {
                auto suffix = label.substr(colon_pos + 1);
                if (looks_like_type_suffix(suffix)) {
                    return std::string(label.substr(0, colon_pos));
                }
            }

            // Legacy/fallback path: older or alternative emitters glue the type
            // onto the userop name with an underscore (for example
            // "atomic_load_int"). The scan only matches the trailing segment
            // against the type_suffixes table, so a userop whose final
            // underscore-delimited word *coincides* with a type keyword (e.g.
            // a hypothetical "LOCK_LOAD_INT" where "INT" is part of the
            // operation name) will be incorrectly truncated. Callers that emit
            // such names should use the colon form to disambiguate.
            for (auto underscore_pos = label.rfind('_');
                 underscore_pos != std::string_view::npos;
                 underscore_pos = underscore_pos == 0 ? std::string_view::npos
                                                      : label.rfind('_', underscore_pos - 1))
            {
                auto suffix = label.substr(underscore_pos + 1);
                if (looks_like_type_suffix(suffix)) {
                    return std::string(label.substr(0, underscore_pos));
                }
            }

            return std::string(label);
        }

        std::string_view strip_trailing_size(std::string_view value) {
            while (!value.empty()
                   && std::isdigit(static_cast< unsigned char >(value.back())) != 0)
            {
                value.remove_suffix(1);
            }
            return value;
        }

        // === Vocabulary tables ===========================================
        //
        // The normalizer is driven by two data tables, grouped by the source
        // that produces the raw name:
        //
        //   1. Real CALLOTHER userops declared in stock Ghidra 12.0.4 SLEIGH
        //      (Ghidra/Processors/{AARCH64,ARM}/data/languages/*.sinc).
        //   2. LLVM compiler-rt AArch64 outline-atomic helper symbols compiled
        //      into the target binary by `-moutline-atomics`
        //      (compiler-rt/lib/builtins/aarch64/lse.S).
        //
        // Both sources use globally unique names -- the AArch64 and ARM SLEIGH
        // declare overlapping userops (DataMemoryBarrier, ClearExclusiveLocal,
        // ...) with *identical* semantics, and the compiler-rt symbols carry an
        // unambiguous `__aarch64_` prefix -- so a single flat lookup is enough.
        // The `arch` parameter on parse_intrinsic_name is preserved for ABI
        // compatibility with callers and to leave room for a future
        // architecture that genuinely needs disjoint normalization, but it is
        // not consulted today.

        struct UseropMapping
        {
            std::string_view raw;       // Exact userop / symbol name to match
            std::string_view canonical; // Canonical intrinsic name we emit
        };

        // Ghidra SLEIGH CALLOTHER userops with C11 atomic semantics.
        // Sources:
        //   Ghidra/Processors/AARCH64/data/languages/AARCH64instructions.sinc
        //   Ghidra/Processors/ARM/data/languages/{ARMinstructions,
        //     ARMTHUMBinstructions,ARMv8}.sinc
        // (Ghidra 12.0.4).
        //
        // DMB and DSB map to *distinct* canonical names rather than collapsing
        // both onto atomic_thread_fence_seq_cst, because stock Ghidra emits DMB
        // with two arguments and DSB with three; the patchir-decomp pipeline
        // declares each canonical with the fixed arity of its first call site,
        // so unifying them on a single canonical causes the second caller to
        // fail CIR call-shape verification.
        //
        // LOAcquire / LORelease are AArch64-only ordering hooks SLEIGH emits
        // around the LSE instructions' inline P-Code bodies; SpeculationBarrier
        // is the AArch64 SB userop. Including them here also handles any ARM
        // input that happened to surface them (Ghidra's ARM SLEIGH does not,
        // but the broader table makes the lookup arch-independent).
        //
        // ExclusiveAccess / hasExclusiveAccess (ARM) and ExclusiveMonitorPass /
        // ExclusiveMonitorsStatus (AArch64) are LL/SC monitor primitives with
        // no clean C11 mapping; they are intentionally not normalized.
        constexpr std::array< UseropMapping, 7 > sleigh_userops = { {
            { "DataMemoryBarrier",                 "atomic_thread_fence_seq_cst"    },
            { "DataSynchronizationBarrier",        "atomic_data_sync_fence_seq_cst" },
            { "InstructionSynchronizationBarrier", "instruction_sync_fence"         },
            { "ClearExclusiveLocal",               "atomic_clear_exclusive"         },
            { "SpeculationBarrier",                "cpu_speculation_barrier"        },
            { "LOAcquire",                         "atomic_thread_fence_acquire"    },
            { "LORelease",                         "atomic_thread_fence_release"    },
        } };

        // int->float conversion userops whose undefined<N> result we resolve to
        // float/double (see ResolveUseropFloatReturn).
        // Source: Ghidra ARM/data/languages/ARMneon.sinc (lines 1446-1447).
        constexpr std::array< std::string_view, 2 > kFloatReturningUserops = { {
            "VectorSignedToFloat",
            "VectorUnsignedToFloat",
        } };

        // compiler-rt AArch64 outline-atomic helper bases.
        // Source: compiler-rt/lib/builtins/aarch64/lse.S
        // Symbol shape: __aarch64_<base><size><ordering>
        //   base     in {cas, swp, ldadd, ldclr, ldeor, ldset}
        //   size     in {1, 2, 4, 8} (cas also 16 for paired registers)
        //   ordering in {_relax, _acq, _rel, _acq_rel, _sync}
        struct OutlineAtomicBase
        {
            std::string_view base;      // Base token between prefix and size
            std::string_view canonical; // canonical_op in atomic_<canonical>_<order>
        };
        constexpr std::array< OutlineAtomicBase, 6 > compiler_rt_aarch64_bases = { {
            { "cas",   "compare_exchange" },
            { "swp",   "exchange"         },
            { "ldadd", "fetch_add"        },
            { "ldclr", "fetch_clear"      },
            { "ldeor", "fetch_xor"        },
            { "ldset", "fetch_or"         },
        } };

        struct OrderingSuffix
        {
            std::string_view suffix;    // suffix in the binary symbol (with leading _)
            std::string_view canonical; // ordering token in the canonical name
        };
        // Order matters: greedy ends_with must see longer suffixes first so
        // "_acq_rel" wins over "_acq".
        constexpr std::array< OrderingSuffix, 5 > compiler_rt_ordering = { {
            { "_acq_rel", "acq_rel" },
            { "_relax",   "relaxed" },
            { "_acq",     "acquire" },
            { "_rel",     "release" },
            { "_sync",    "seq_cst" },
        } };

        // === Lookups =====================================================

        std::optional< std::string > lookup_sleigh_userop(std::string_view name) {
            for (const auto &entry : sleigh_userops) {
                if (name == entry.raw) {
                    return std::string(entry.canonical);
                }
            }
            return std::nullopt;
        }

        std::optional< std::string >
        match_compiler_rt_aarch64_outline(std::string_view name) {
            constexpr std::string_view prefix = "__aarch64_";
            if (!starts_with(name, prefix)) {
                return std::nullopt;
            }

            std::string_view op = name.substr(prefix.size());
            std::string_view order;

            for (const auto &entry : compiler_rt_ordering) {
                if (ends_with(op, entry.suffix)) {
                    order = entry.canonical;
                    op.remove_suffix(entry.suffix.size());
                    break;
                }
            }

            op = strip_trailing_size(op);

            for (const auto &entry : compiler_rt_aarch64_bases) {
                if (op != entry.base) {
                    continue;
                }
                // Real compiler-rt always emits one of the five ordering
                // suffixes (lse.S sets SUFF for every MODEL). Fall back to
                // seq_cst defensively if the symbol lacks one, since that is
                // the most conservative ordering for an unknown source.
                std::string_view effective_order =
                    order.empty() ? std::string_view("seq_cst") : order;
                return "atomic_" + std::string(entry.canonical) + "_"
                    + std::string(effective_order);
            }
            return std::nullopt;
        }

        std::string
        normalize_intrinsic_name(std::string_view arch, std::string_view name) {
            // The `arch` parameter is preserved on the signature for ABI
            // compatibility with callers and so a future architecture that
            // genuinely needs arch-specific normalization can be added without
            // touching call sites. Today both supported vocabularies (Ghidra
            // SLEIGH userops, compiler-rt outline atomics) use globally unique
            // names, so the lookup does not need to dispatch on it.
            (void) arch;

            if (auto result = lookup_sleigh_userop(name)) {
                return *result;
            }
            if (auto result = match_compiler_rt_aarch64_outline(name)) {
                return *result;
            }
            return std::string(name);
        }

        std::pair< clang::Stmt *, bool > handle_generic_intrinsic_call(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op, const std::string &name
        ) {
            return b.create_intrinsic_call(ctx, fn, op, name);
        }

        std::pair< clang::Stmt *, bool > handle_volatile_read(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op, const std::string &name
        ) {
            (void) name;
            if (op.inputs.empty()) {
                LOG(ERROR) << "volatile_read requires at least one input. key: " << op.key
                           << "\n";
                return {};
            }

            auto op_loc = SourceLocation(ctx.getSourceManager(), op.key);

            // Get address expression
            auto *addr =
                clang::dyn_cast< clang::Expr >(b.create_varnode(ctx, fn, op.inputs[0]));
            if (addr == nullptr) {
                LOG(ERROR) << "Failed to create address expr for volatile_read. key: " << op.key
                           << "\n";
                return {};
            }

            // Result type: output -> op.type -> int. Must stay non-null
            // (getVolatileType(null) would abort).
            clang::QualType result_type = ctx.IntTy;
            clang::QualType from_output;
            if (op.output) { from_output = b.get_varnode_type(ctx, *op.output); }
            if (!from_output.isNull()) {
                result_type = from_output;
            } else if (op.type) {
                auto t = b.type_builder().GetSerializedType(*op.type);
                if (!t.isNull()) { result_type = t; }
            }

            // Cast to volatile pointer and dereference: *(volatile T*)addr
            auto vol_ptr = ctx.getPointerType(ctx.getVolatileType(result_type));
            // Explicit C-style cast (make_cast's implicit cast is dropped by the
            // pretty-printer, losing `volatile` in the emitted source). A global
            // operand is the datum at the address, so take its address:
            // *(volatile T *)&DAT_xxxx (keyed on kind, not pointer-ness).
            if (op.inputs[0].kind == ghidra::Varnode::VARNODE_GLOBAL) {
                auto addr_of = b.sema().CreateBuiltinUnaryOp(op_loc, clang::UO_AddrOf, addr);
                if (addr_of.isInvalid()) {
                    LOG(ERROR) << "volatile access: could not take address of "
                                  "global operand; refusing to emit a wrong MMIO "
                                  "access. key: " << op.key << "\n";
                    return {};
                }
                addr = addr_of.getAs< clang::Expr >();
            }
            auto *cast   = b.make_explicit_cast(ctx, addr, vol_ptr, op_loc);
            if (cast == nullptr) {
                LOG(ERROR) << "volatile_read: volatile-pointer cast failed. key: "
                           << op.key << "\n";
                return {};
            }
            auto deref   = b.sema().CreateBuiltinUnaryOp(op_loc, clang::UO_Deref, cast);

            if (deref.isInvalid()) {
                LOG(ERROR) << "Failed to create dereference for volatile_read. key: " << op.key
                           << "\n";
                return {};
            }

            // If no output, return the deref expression directly
            if (!op.output) { return { deref.getAs< clang::Stmt >(), true }; }

            // Self-referential output (operation points back at this op): return
            // the read as a mergeable value; resolving it as an assignment target
            // would recurse into create_varnode and stack-overflow.
            if (op.output->operation && *op.output->operation == op.key) {
                return { deref.getAs< clang::Stmt >(), true };
            }

            // Assign to output; if it won't resolve to an lvalue, loud-fail and
            // emit just the read rather than assigning through null.
            auto *out = clang::dyn_cast_or_null< clang::Expr >(
                b.create_varnode(ctx, fn, *op.output));
            if (out == nullptr) {
                LOG(ERROR) << "volatile_read output could not be resolved to an "
                              "lvalue; emitting the read without assignment. key: "
                           << op.key << "\n";
                return { deref.getAs< clang::Stmt >(), true };
            }
            return { b.create_assign_operation(ctx, deref.getAs< clang::Expr >(), out, op_loc),
                     false };
        }

        std::pair< clang::Stmt *, bool > handle_volatile_write(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op, const std::string &name
        ) {
            (void) name;
            if (op.inputs.size() < 2) {
                LOG(ERROR) << "volatile_write requires at least two inputs. key: " << op.key
                           << "\n";
                return {};
            }

            auto op_loc = SourceLocation(ctx.getSourceManager(), op.key);

            // Get address and value expressions
            auto *addr =
                clang::dyn_cast< clang::Expr >(b.create_varnode(ctx, fn, op.inputs[0]));
            auto *val = clang::dyn_cast< clang::Expr >(b.create_varnode(ctx, fn, op.inputs[1]));

            if (addr == nullptr || val == nullptr) {
                LOG(ERROR) << "Failed to create expressions for volatile_write. key: " << op.key
                           << "\n";
                return {};
            }

            // Cast to volatile pointer: (volatile T*)addr
            auto vol_ptr = ctx.getPointerType(ctx.getVolatileType(val->getType()));
            // Explicit C-style cast (make_cast's implicit cast is dropped by the
            // pretty-printer, losing `volatile` in the emitted source). A global
            // operand is the datum at the address, so take its address:
            // *(volatile T *)&DAT_xxxx (keyed on kind, not pointer-ness).
            if (op.inputs[0].kind == ghidra::Varnode::VARNODE_GLOBAL) {
                auto addr_of = b.sema().CreateBuiltinUnaryOp(op_loc, clang::UO_AddrOf, addr);
                if (addr_of.isInvalid()) {
                    LOG(ERROR) << "volatile access: could not take address of "
                                  "global operand; refusing to emit a wrong MMIO "
                                  "access. key: " << op.key << "\n";
                    return {};
                }
                addr = addr_of.getAs< clang::Expr >();
            }
            auto *cast   = b.make_explicit_cast(ctx, addr, vol_ptr, op_loc);
            if (cast == nullptr) {
                LOG(ERROR) << "volatile_write: volatile-pointer cast failed. key: "
                           << op.key << "\n";
                return {};
            }

            // Dereference and assign: *(volatile T*)addr = val
            auto deref = b.sema().CreateBuiltinUnaryOp(op_loc, clang::UO_Deref, cast);
            if (deref.isInvalid()) {
                LOG(ERROR) << "Failed to create dereference for volatile_write. key: " << op.key
                           << "\n";
                return {};
            }

            auto assign = b.sema().CreateBuiltinBinOp(
                op_loc, clang::BO_Assign, deref.getAs< clang::Expr >(), val
            );

            if (assign.isInvalid()) {
                LOG(ERROR) << "Failed to create assign for volatile_write. key: " << op.key
                           << "\n";
                return {};
            }

            return { assign.getAs< clang::Stmt >(), false };
        }

        std::pair< clang::Stmt *, bool > handle_builtin_memcpy(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op, const std::string &name
        ) {
            // For memcpy/strncpy/wcsncpy, just use the generic fallback
            // which creates a function call with all inputs as arguments
            return handle_generic_intrinsic_call(b, ctx, fn, op, name);
        }

        void add_atomic_intrinsic_handlers(
            std::unordered_map< std::string, IntrinsicHandler > &handlers,
            std::string_view operation
        ) {
            static constexpr std::array< std::string_view, 5 > orders = { "relaxed", "acquire",
                                                                          "release", "acq_rel",
                                                                          "seq_cst" };

            for (auto order : orders) {
                handlers.emplace(
                    "atomic_" + std::string(operation) + "_" + std::string(order),
                    handle_generic_intrinsic_call
                );
            }
        }

        // Emit op.string_value as a StringLiteral decayed to `const char *`;
        // returning merge_to_next=true so it caches in operation_stmts and
        // downstream temporary references pick it up. Falls back to the
        // registered-decl call when the serialiser couldn't resolve the literal.
        std::pair< clang::Stmt *, bool > handle_stringdata(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op, const std::string &name
        ) {
            (void) name;
            if (!op.string_value || op.string_value->empty()) {
                return b.build_intrinsic_call_against_registered(ctx, fn, op);
            }
            auto op_loc     = SourceLocation(ctx.getSourceManager(), op.key);
            auto char_type  = ctx.CharTy.withConst();
            auto array_size = op.string_value->size() + 1; // null terminator
            auto array_type = ctx.getConstantArrayType(
                char_type, llvm::APInt(32, array_size), nullptr,
                clang::ArraySizeModifier::Normal, 0);
            auto *lit = clang::StringLiteral::Create(
                ctx, *op.string_value,
                clang::StringLiteralKind::Ordinary, /*Pascal=*/false,
                array_type, op_loc);

            auto ptr_type = ctx.getPointerType(char_type);
            auto *decayed = b.make_cast(ctx, lit, ptr_type, op_loc);
            return { decayed ? decayed : lit, true };
        }

        // Opaque exception return (M-profile EXC_RETURN bx, or A/R CPSR-restoring
        // subs pc,lr / ldm^ / rfe). The unstack/mode-switch isn't modelable in
        // P-Code, so emit an extern __patchestry_<name> call no pass folds away.
        std::pair< clang::Stmt *, bool > handle_exception_return(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op, const std::string &name
        ) {
            return b.create_intrinsic_call(ctx, fn, op, "__patchestry_" + name);
        }

        // === ARM system-userop class -> compiler-intrinsic spelling ==========
        //
        // Maps the arch-neutral `intrinsic_class` (assigned by the Ghidra
        // UseropClassifier pre-pass) to the name the compiler understands:
        // CMSIS-Core for Cortex-M special registers and interrupt masking, ACLE
        // builtins for hints and the directly-mappable coprocessor LDC/STC. The
        // read/write/return shape is carried by the serialized op, so once the
        // name is known a plain create_intrinsic_call is enough (as for the
        // atomic_* families). Returns nullopt for any class this layer does not
        // spell, so the caller falls through to the existing dispatch.

        // Cortex-M special registers exposed by CMSIS as __get_<R>()/__set_<R>().
        bool is_known_cmsis_sysreg(std::string_view reg) {
            static constexpr std::array< std::string_view, 12 > regs = {
                "BASEPRI", "BASEPRI_MAX", "PRIMASK", "FAULTMASK", "CONTROL", "IPSR",
                "MSP",     "PSP",         "MSPLIM",  "PSPLIM",     "APSR",    "XPSR"
            };
            for (auto candidate : regs) {
                if (reg == candidate) {
                    return true;
                }
            }
            return false;
        }

        std::optional< std::string >
        arm_canonical_for(std::string_view klass, const std::optional< std::string > &reg) {
            // Interrupt masking (CPSID/CPSIE). Only PRIMASK (CPS i) and FAULTMASK
            // (CPS f, == __*_fault_irq on M) have a CMSIS spelling; other CPS
            // forms (e.g. the A/R abort-mask `a` bit) fall through rather than
            // mis-spell as __disable_irq.
            if (klass == "irq_mask_set" || klass == "irq_mask_clear") {
                bool set = (klass == "irq_mask_set");
                if (reg && *reg == "PRIMASK") {
                    return set ? "__disable_irq" : "__enable_irq";
                }
                if (reg && *reg == "FAULTMASK") {
                    return set ? "__disable_fault_irq" : "__enable_fault_irq";
                }
                return std::nullopt;
            }
            // Named Cortex-M special registers -> CMSIS accessors.
            if (klass == "sysreg_read" && reg && is_known_cmsis_sysreg(*reg)) {
                return "__get_" + *reg;
            }
            if (klass == "sysreg_write" && reg && is_known_cmsis_sysreg(*reg)) {
                return "__set_" + *reg;
            }
            // Hints -> ACLE nullary builtins.
            if (klass == "hint_wfi") { return "__wfi"; }
            if (klass == "hint_wfe") { return "__wfe"; }
            if (klass == "hint_sev") { return "__sev"; }
            if (klass == "hint_yield") { return "__yield"; }
            if (klass == "hint_nop") { return "__nop"; }
            // (coproc LDC/STC are handled in arm_emit_system_intrinsic, which
            // casts the address arg to a pointer — not via this name map.)
            // Fall through (caller keeps existing behavior):
            //   - barrier_* : handled by the C11 atomic-fence mapping.
            //   - coproc_read/write/cdp : handled by emit_arm_coproc (operand
            //     reorder), not here.
            //   - un-named sysreg, trap, mode_switch, query, unknown.
            return std::nullopt;
        }

        // Coprocessor MCR/MRC/CDP: Ghidra's SLEIGH operand order differs from
        // ACLE, which puts opc2 last. Reorder the serialized inputs (the userop
        // index is already stripped) before emitting, reusing create_intrinsic_call
        // for decl synthesis + output assignment. Single-register forms only;
        // the *2 (MCRR/MRRC) classes and odd-arity movefromRt variants fall
        // through (nullopt) rather than emit a wrong-shape ACLE call.
        //   coprocessor_moveto(cpn,op1,op2,Rt,CRn,CRm)    -> __arm_mcr(cpn,op1,Rt,CRn,CRm,op2)
        //   coprocessor_movefromRt(cpn,op1,op2,CRn,CRm)   -> __arm_mrc(cpn,op1,CRn,CRm,op2)
        //   coprocessor_function(cpn,op1,op2,CRd,CRn,CRm) -> __arm_cdp(cpn,op1,CRd,CRn,CRm,op2)
        std::optional< std::pair< clang::Stmt *, bool > > emit_arm_coproc(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op, std::string_view klass
        ) {
            const auto &in = op.inputs;
            ghidra::Operation reordered = op;
            std::string name;
            if (klass == "coproc_write" && in.size() == 6) {
                name             = "__arm_mcr";
                reordered.inputs = { in[0], in[1], in[3], in[4], in[5], in[2] };
            } else if (klass == "coproc_read" && in.size() == 5) {
                name             = "__arm_mrc";
                reordered.inputs = { in[0], in[1], in[3], in[4], in[2] };
            } else if (klass == "coproc_cdp" && in.size() == 6) {
                name             = "__arm_cdp";
                reordered.inputs = { in[0], in[1], in[3], in[4], in[5], in[2] };
            } else {
                return std::nullopt;
            }
            return b.create_intrinsic_call(ctx, fn, reordered, name);
        }

        // === Per-architecture emitter registry =============================
        //
        // The taxonomy (intrinsic_class) is architecture-neutral; each arch
        // plugs in a IntrinsicEmitFn that turns a classified op into a compiler
        // intrinsic. The emitter is selected by the program's processor string
        // (program_arch() == the JSON `architecture`). To add an architecture:
        // write a IntrinsicEmitFn and add one registry row.
        using IntrinsicEmitFn = std::optional< std::pair< clang::Stmt *, bool > > (*)(
            OpBuilder &, clang::ASTContext &, const ghidra::Function &,
            const ghidra::Operation &
        );

        // ARM (32-bit): CMSIS interrupt masking / M-profile special registers,
        // ACLE hints + coprocessor (MCR/MRC/CDP reorder, LDC/STC pointer cast).
        // Precondition: op.target->intrinsic_class is set (the dispatcher guards).
        std::optional< std::pair< clang::Stmt *, bool > > arm_emit_system_intrinsic(
            OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
            const ghidra::Operation &op
        ) {
            const std::string &klass = *op.target->intrinsic_class;
            // Coprocessor MCR/MRC/CDP need an operand reorder vs ACLE.
            if (klass == "coproc_write" || klass == "coproc_read" || klass == "coproc_cdp") {
                return emit_arm_coproc(b, ctx, fn, op, klass);
            }
            // Coprocessor LDC/STC: the address (input 2) is `const void *` in
            // the ACLE prototype, so cast it to keep the call recompilable.
            if (klass == "coproc_load" || klass == "coproc_loadl"
                || klass == "coproc_store" || klass == "coproc_storel")
            {
                const std::string ldc_name = (klass == "coproc_load")  ? "__arm_ldc"
                                           : (klass == "coproc_loadl") ? "__arm_ldcl"
                                           : (klass == "coproc_store") ? "__arm_stc"
                                                                       : "__arm_stcl";
                return b.create_intrinsic_call(ctx, fn, op, ldc_name, /*pointer_arg_index=*/2);
            }
            auto canonical = arm_canonical_for(klass, op.target->system_register);
            if (!canonical) {
                return std::nullopt;
            }
            return b.create_intrinsic_call(ctx, fn, op, *canonical);
        }

        // AArch64: extension point. Returns nullopt (all classes fall through)
        // until an AArch64 emitter is implemented.
        std::optional< std::pair< clang::Stmt *, bool > > aarch64_emit_system_intrinsic(
            OpBuilder &, clang::ASTContext &, const ghidra::Function &,
            const ghidra::Operation &
        ) {
            return std::nullopt;
        }

        // Intrinsics are emitted as extern declarations (create_intrinsic_call ->
        // get_or_create_intrinsic_decl, SC_Extern); the emitted C carries its own
        // extern prototypes rather than #include-ing arch headers, so the emitter
        // carries no header info.
        struct ArchIntrinsicEmitter
        {
            std::string_view arch; // matches program_arch() (case-insensitive)
            IntrinsicEmitFn emit;          // classified op -> compiler intrinsic
        };

        constexpr std::array< ArchIntrinsicEmitter, 2 > arch_intrinsic_emitters = { {
            { "ARM",     &arm_emit_system_intrinsic     },
            { "AARCH64", &aarch64_emit_system_intrinsic },
        } };

        const ArchIntrinsicEmitter *get_intrinsic_emitter(std::string_view arch) {
            auto lowered = to_lower_ascii(arch);
            for (const auto &emitter : arch_intrinsic_emitters) {
                if (to_lower_ascii(emitter.arch) == lowered) {
                    return &emitter;
                }
            }
            return nullptr;
        }

    } // anonymous namespace

    std::string parse_intrinsic_name(std::string_view arch, std::string_view label) {
        return normalize_intrinsic_name(arch, strip_return_type_suffix(label));
    }

    bool IsFloatReturningUserop(std::string_view name) {
        for (auto candidate : kFloatReturningUserops) {
            if (candidate == name) {
                return true;
            }
        }
        return false;
    }

    clang::QualType ResolveUseropFloatReturn(
        clang::ASTContext &ctx, std::string_view name, uint64_t size_bits
    ) {
        if (!IsFloatReturningUserop(name)) {
            return {};
        }
        switch (size_bits) {
            case 32:
                return ctx.FloatTy;
            case 64:
                return ctx.DoubleTy;
            default:
                // No standard float type for this width (e.g. 128-bit Q-reg);
                // keep the raw type via the size-aware path.
                return {};
        }
    }

    const std::unordered_map< std::string, IntrinsicHandler > &get_intrinsic_handlers() {
        static const auto handlers = [] {
            std::unordered_map< std::string, IntrinsicHandler > result = {
                {          "volatile_read",         handle_volatile_read },
                {         "volatile_write",        handle_volatile_write },
                {         "builtin_memcpy",        handle_builtin_memcpy },
                {        "builtin_strncpy",        handle_builtin_memcpy }, // Same impl as memcpy
                {        "builtin_wcsncpy",        handle_builtin_memcpy },
                {             "stringdata",            handle_stringdata },
                // Exception returns (ARM EXC_RETURN / CPSR-restoring return).
                // Emitted opaque so the control-flow side effect is preserved.
                {       "exception_return",      handle_exception_return },
                {  "exception_return_cpsr",      handle_exception_return },
                // Bare-named synchronization primitives that have no C11
                // memory-ordering parameter. ClearExclusiveLocal resets the
                // local exclusive monitor; ISB is a pipeline sync, not a
                // memory fence; SpeculationBarrier blocks speculative
                // execution -- C11 has no equivalent for any of them.
                {   "atomic_clear_exclusive", handle_generic_intrinsic_call },
                {  "instruction_sync_fence", handle_generic_intrinsic_call },
                { "cpu_speculation_barrier", handle_generic_intrinsic_call },
            };

            // Register canonical intrinsic names that the per-arch normalizers
            // can produce. Each entry covers all five C11 memory orderings; the
            // set is exactly the union of:
            //   - compiler-rt outline-atomic bases (fetch_add, fetch_clear,
            //     fetch_xor, fetch_or, exchange, compare_exchange)
            //   - Ghidra SLEIGH barrier mappings (thread_fence, data_sync_fence)
            add_atomic_intrinsic_handlers(result, "fetch_add");
            add_atomic_intrinsic_handlers(result, "fetch_clear");
            add_atomic_intrinsic_handlers(result, "fetch_xor");
            add_atomic_intrinsic_handlers(result, "fetch_or");
            add_atomic_intrinsic_handlers(result, "exchange");
            add_atomic_intrinsic_handlers(result, "compare_exchange");
            add_atomic_intrinsic_handlers(result, "thread_fence");
            add_atomic_intrinsic_handlers(result, "data_sync_fence");
            return result;
        }();
        return handlers;
    }

    std::optional< std::pair< clang::Stmt *, bool > > emit_system_intrinsic(
        OpBuilder &b, clang::ASTContext &ctx, const ghidra::Function &fn,
        const ghidra::Operation &op, std::string_view arch
    ) {
        // Only classified system userops participate; everything else falls
        // through to the existing name-based dispatch.
        if (!op.target || !op.target->intrinsic_class) {
            return std::nullopt;
        }
        // Select the emitter for this program's architecture. Unknown arch (or
        // an arch whose emitter does not cover this class) falls through.
        const auto *emitter = get_intrinsic_emitter(arch);
        if (emitter == nullptr) {
            return std::nullopt;
        }
        return emitter->emit(b, ctx, fn, op);
    }

} // namespace patchestry::ast
