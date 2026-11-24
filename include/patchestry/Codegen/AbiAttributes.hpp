/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/TargetParser/Triple.h>

namespace patchestry::codegen {

    // On 32-bit ARM a `_Bool` is passed and returned zero-extended to the
    // whole register, and Clang says so in the IR: `zeroext i1` on the return
    // value and on each such parameter, at the function and at every call.
    // ClangIR leaves the attribute out.  Without it the backend may not
    // assume the upper bits, so it masks the register (`and r0, r0, #1`)
    // after a call and tests bit 0 only, where the original code compares the
    // whole register (`cmp r0, #0`).  A callee that Ghidra typed `_Bool` and
    // that returns 2 then reads as false.
    //
    // This adds `zeroext` to every `i1` return value and `i1` argument of
    // every function and call in the module.  Wider integers are left alone:
    // their extension depends on the signedness of the C type, which the IR
    // no longer has.
    //
    // Only for a module whose triple is 32-bit ARM or Thumb.  Returns the
    // number of attributes added.
    inline unsigned add_bool_extension_attributes(llvm::Module &module) {
        const llvm::Triple &triple = module.getTargetTriple();
        if (!triple.isARM() && !triple.isThumb()) {
            return 0;
        }

        unsigned added = 0;
        auto is_bool   = [](const llvm::Type *type) { return type->isIntegerTy(1); };

        for (llvm::Function &function : module) {
            if (!function.isIntrinsic()) {
                if (is_bool(function.getReturnType())
                    && !function.hasRetAttribute(llvm::Attribute::ZExt))
                {
                    function.addRetAttr(llvm::Attribute::ZExt);
                    ++added;
                }
                for (llvm::Argument &argument : function.args()) {
                    if (is_bool(argument.getType())
                        && !argument.hasAttribute(llvm::Attribute::ZExt))
                    {
                        argument.addAttr(llvm::Attribute::ZExt);
                        ++added;
                    }
                }
            }

            for (llvm::BasicBlock &block : function) {
                for (llvm::Instruction &instruction : block) {
                    auto *call = llvm::dyn_cast< llvm::CallBase >(&instruction);
                    if (call == nullptr) {
                        continue;
                    }
                    const llvm::Function *callee = call->getCalledFunction();
                    if (callee != nullptr && callee->isIntrinsic()) {
                        continue;
                    }
                    // The call's own list: CallBase::hasRetAttr also answers
                    // from the callee, and an indirect call has none.
                    if (is_bool(call->getType())
                        && !call->getAttributes().hasRetAttr(llvm::Attribute::ZExt))
                    {
                        call->addRetAttr(llvm::Attribute::ZExt);
                        ++added;
                    }
                    for (unsigned index = 0; index < call->arg_size(); ++index) {
                        if (is_bool(call->getArgOperand(index)->getType())
                            && !call->getAttributes().hasParamAttr(index, llvm::Attribute::ZExt))
                        {
                            call->addParamAttr(index, llvm::Attribute::ZExt);
                            ++added;
                        }
                    }
                }
            }
        }
        return added;
    }

} // namespace patchestry::codegen
