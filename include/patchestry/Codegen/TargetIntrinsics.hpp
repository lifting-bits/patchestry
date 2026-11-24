/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsARM.h>
#include <llvm/IR/Module.h>
#include <llvm/TargetParser/Triple.h>

namespace patchestry::codegen {

    // The lifter spells an ARM coprocessor instruction as a call to the ACLE
    // name: __arm_mrc, __arm_mcr, __arm_cdp.  In <arm_acle.h> those are macros
    // over compiler builtins, which ClangIR does not implement, so in the
    // lifted module they are ordinary external functions.  Nothing defines
    // them: code that calls one does not link, and if it did, the call would
    // not be the instruction.
    //
    // This rewrites each such call to the LLVM intrinsic the builtin stands
    // for (llvm.arm.mrc, llvm.arm.mcr, llvm.arm.cdp), which the ARM backend
    // selects to the instruction.  The operand order of the ACLE function and
    // of the intrinsic is the same.
    //
    // A call is rewritten only when every operand the instruction encodes
    // (coprocessor, opcodes, coprocessor registers) is a constant integer in
    // range.  Any other call is left as it is, so the unresolved name still
    // shows up at link time instead of a wrong instruction being emitted.
    //
    // Only for a module whose triple is 32-bit ARM or Thumb.  Returns the
    // number of calls rewritten.
    inline unsigned lower_arm_coprocessor_calls(llvm::Module &module) {
        const llvm::Triple &triple = module.getTargetTriple();
        if (!triple.isARM() && !triple.isThumb()) {
            return 0;
        }

        struct Entry
        {
            std::string_view name;
            llvm::Intrinsic::ID id;
            // Position of the one operand that is a run-time value (the
            // register written by MCR), or -1.
            int value_index;
            // Upper bound (exclusive) of each encoded operand: coprocessor
            // and coprocessor registers are 4 bits, opc1 is 3 bits for
            // MRC/MCR and 4 for CDP, opc2 is 3 bits.
            std::vector< unsigned > limits;
        };
        const std::array< Entry, 3 > entries = { {
            { "__arm_mrc", llvm::Intrinsic::arm_mrc, -1, { 16, 8, 16, 16, 8 } },
            { "__arm_mcr", llvm::Intrinsic::arm_mcr, 2, { 16, 8, 0, 16, 16, 8 } },
            { "__arm_cdp", llvm::Intrinsic::arm_cdp, -1, { 16, 16, 16, 16, 16, 8 } },
        } };

        unsigned rewritten = 0;
        llvm::Type *i32    = llvm::Type::getInt32Ty(module.getContext());
        for (const auto &entry : entries) {
            llvm::Function *callee = module.getFunction(entry.name);
            if (callee == nullptr || !callee->isDeclaration()) {
                continue;
            }

            llvm::SmallVector< llvm::CallInst *, 8 > calls;
            for (llvm::User *user : callee->users()) {
                auto *call = llvm::dyn_cast< llvm::CallInst >(user);
                if (call != nullptr && call->getCalledFunction() == callee) {
                    calls.push_back(call);
                }
            }

            for (llvm::CallInst *call : calls) {
                if (call->arg_size() != entry.limits.size()) {
                    continue;
                }
                bool is_read = entry.id == llvm::Intrinsic::arm_mrc;
                llvm::Type *result = call->getType();
                if (is_read ? !result->isIntegerTy() : !(result->isVoidTy() || call->use_empty()))
                {
                    continue;
                }

                llvm::IRBuilder<> builder(call);
                llvm::SmallVector< llvm::Value *, 6 > arguments;
                bool encodable = true;
                for (unsigned index = 0; index < call->arg_size() && encodable; ++index) {
                    llvm::Value *argument = call->getArgOperand(index);
                    if (static_cast< int >(index) == entry.value_index) {
                        if (argument->getType()->isPointerTy()) {
                            arguments.push_back(builder.CreatePtrToInt(argument, i32));
                        } else if (argument->getType()->isIntegerTy()) {
                            arguments.push_back(builder.CreateZExtOrTrunc(argument, i32));
                        } else {
                            encodable = false;
                        }
                        continue;
                    }
                    auto *constant = llvm::dyn_cast< llvm::ConstantInt >(argument);
                    if (constant == nullptr || constant->getValue().getActiveBits() > 32
                        || constant->getZExtValue() >= entry.limits[index])
                    {
                        encodable = false;
                        continue;
                    }
                    arguments.push_back(llvm::ConstantInt::get(i32, constant->getZExtValue()));
                }
                if (!encodable) {
                    // Casts built for an MCR value before a later operand
                    // failed are dead; they have no uses and no effect.
                    continue;
                }

                llvm::Function *intrinsic =
                    llvm::Intrinsic::getOrInsertDeclaration(&module, entry.id);
                llvm::CallInst *lowered = builder.CreateCall(intrinsic, arguments);
                lowered->setDebugLoc(call->getDebugLoc());
                if (is_read) {
                    call->replaceAllUsesWith(builder.CreateZExtOrTrunc(lowered, result));
                }
                call->eraseFromParent();
                ++rewritten;
            }

            if (callee->use_empty()) {
                callee->eraseFromParent();
            }
        }
        return rewritten;
    }

} // namespace patchestry::codegen
