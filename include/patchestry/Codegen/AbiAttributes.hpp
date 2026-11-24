/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/TargetParser/Triple.h>

namespace patchestry::codegen {

    namespace detail {

        inline bool is_arm32(const llvm::Module &module) {
            const llvm::Triple &triple = module.getTargetTriple();
            return triple.isARM() || triple.isThumb();
        }

        inline bool is_bool(const llvm::Type *type) { return type->isIntegerTy(1); }

        // `zeroext` and `signext` exclude each other; a value that already
        // says how it is extended is left as it is.
        inline bool has_extension(const llvm::AttributeSet &attributes) {
            return attributes.hasAttribute(llvm::Attribute::ZExt)
                || attributes.hasAttribute(llvm::Attribute::SExt);
        }

    } // namespace detail

    // On 32-bit ARM a function returns a `_Bool` in the whole of r0, and the
    // code compiled against it tests the whole register (`cmp r0, #0`).  As
    // an `i1` the result is read as one bit: at -O0 the backend masks it
    // (`and r0, r0, #1`) before storing or testing it, with or without
    // `zeroext`, so a callee that in fact returns 2 reads as false.  Ghidra
    // assigns `bool` from how a result is used, not from what the callee
    // computes, so such callees exist.
    //
    // This gives every `i1`-returning function and call in the module an
    // `i32` result.  A definition returns its `i1` zero-extended, which the
    // ABI requires of it anyway; a call compares the whole result with zero
    // to get its `i1` back.  Functions and calls are rewritten together, so
    // the two always agree on the type.
    //
    // It is done here, on the lowered module, and not in the C types: the
    // rule belongs to this one ABI (on x86-64 a `_Bool` result occupies al
    // and the rest of rax is unspecified), and the C and ClangIR signatures
    // must stay `_Bool` for patches that are matched against them.
    //
    // Only for a module whose triple is 32-bit ARM or Thumb.  Returns the
    // number of functions and calls rewritten.
    inline unsigned widen_bool_results(llvm::Module &module) {
        if (!detail::is_arm32(module)) {
            return 0;
        }

        llvm::LLVMContext &context = module.getContext();
        llvm::Type *wide           = llvm::Type::getInt32Ty(context);
        auto widened_type          = [&](const llvm::FunctionType *type) {
            return llvm::FunctionType::get(wide, type->params(), type->isVarArg());
        };
        auto without_result_extension = [&](llvm::AttributeList attributes) {
            return attributes.removeRetAttribute(context, llvm::Attribute::ZExt)
                .removeRetAttribute(context, llvm::Attribute::SExt);
        };

        unsigned changed = 0;

        std::vector< llvm::Function * > functions;
        for (llvm::Function &function : module) {
            if (!function.isIntrinsic() && detail::is_bool(function.getReturnType())) {
                functions.push_back(&function);
            }
        }
        for (llvm::Function *old_function : functions) {
            auto *function = llvm::Function::Create(
                widened_type(old_function->getFunctionType()), old_function->getLinkage(),
                old_function->getAddressSpace()
            );
            module.getFunctionList().insert(old_function->getIterator(), function);
            function->copyAttributesFrom(old_function);
            function->setAttributes(without_result_extension(function->getAttributes()));
            function->setComdat(old_function->getComdat());
            function->copyMetadata(old_function, 0);
            function->takeName(old_function);
            function->splice(function->begin(), old_function);

            auto new_argument = function->arg_begin();
            for (llvm::Argument &old_argument : old_function->args()) {
                new_argument->takeName(&old_argument);
                old_argument.replaceAllUsesWith(&*new_argument);
                ++new_argument;
            }
            for (llvm::BasicBlock &block : *function) {
                auto *ret = llvm::dyn_cast< llvm::ReturnInst >(block.getTerminator());
                if (ret == nullptr || ret->getReturnValue() == nullptr) {
                    continue;
                }
                ret->setOperand(
                    0, new llvm::ZExtInst(ret->getReturnValue(), wide, "", ret->getIterator())
                );
            }

            old_function->replaceAllUsesWith(function);
            old_function->eraseFromParent();
            ++changed;
        }

        std::vector< llvm::CallInst * > calls;
        for (llvm::Function &function : module) {
            for (llvm::BasicBlock &block : function) {
                for (llvm::Instruction &instruction : block) {
                    auto *call = llvm::dyn_cast< llvm::CallInst >(&instruction);
                    if (call == nullptr || !detail::is_bool(call->getType())
                        || call->isInlineAsm())
                    {
                        continue;
                    }
                    const llvm::Function *callee = call->getCalledFunction();
                    if (callee != nullptr && callee->isIntrinsic()) {
                        continue;
                    }
                    calls.push_back(call);
                }
            }
        }
        for (llvm::CallInst *old_call : calls) {
            std::vector< llvm::Value * > arguments(old_call->arg_begin(), old_call->arg_end());
            llvm::SmallVector< llvm::OperandBundleDef, 1 > bundles;
            old_call->getOperandBundlesAsDefs(bundles);

            auto *call = llvm::CallInst::Create(
                widened_type(old_call->getFunctionType()), old_call->getCalledOperand(),
                arguments, bundles, "", old_call->getIterator()
            );
            call->setCallingConv(old_call->getCallingConv());
            call->setAttributes(without_result_extension(old_call->getAttributes()));
            call->setDebugLoc(old_call->getDebugLoc());
            // A comparison follows the call, so it can no longer be a
            // guaranteed tail call.
            call->setTailCallKind(
                old_call->isMustTailCall() ? llvm::CallInst::TCK_Tail
                                           : old_call->getTailCallKind()
            );

            auto *truth = new llvm::ICmpInst(
                old_call->getIterator(), llvm::ICmpInst::ICMP_NE, call,
                llvm::ConstantInt::get(wide, 0)
            );
            truth->takeName(old_call);
            old_call->replaceAllUsesWith(truth);
            old_call->eraseFromParent();
            ++changed;
        }
        return changed;
    }

    // On 32-bit ARM a `_Bool` is passed zero-extended to the whole register,
    // and Clang says so in the IR with `zeroext i1` on each such parameter,
    // at the function and at every call.  ClangIR leaves the attribute out,
    // and without it the backend may not assume the upper bits.
    //
    // This adds `zeroext` to every `i1` argument, and to any `i1` return
    // value `widen_bool_results` left in place, of every function and call
    // in the module.  A value that already carries `zeroext` or `signext` is
    // left alone.  Wider integers are left alone too: their extension
    // depends on the signedness of the C type, which the IR no longer has.
    //
    // Only for a module whose triple is 32-bit ARM or Thumb.  Returns the
    // number of attributes added.
    inline unsigned add_bool_extension_attributes(llvm::Module &module) {
        if (!detail::is_arm32(module)) {
            return 0;
        }

        unsigned added = 0;

        for (llvm::Function &function : module) {
            if (!function.isIntrinsic()) {
                const llvm::AttributeList attributes = function.getAttributes();
                if (detail::is_bool(function.getReturnType())
                    && !detail::has_extension(attributes.getRetAttrs()))
                {
                    function.addRetAttr(llvm::Attribute::ZExt);
                    ++added;
                }
                for (llvm::Argument &argument : function.args()) {
                    if (detail::is_bool(argument.getType())
                        && !detail::has_extension(attributes.getParamAttrs(argument.getArgNo())))
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
                    const llvm::AttributeList attributes = call->getAttributes();
                    if (detail::is_bool(call->getType())
                        && !detail::has_extension(attributes.getRetAttrs()))
                    {
                        call->addRetAttr(llvm::Attribute::ZExt);
                        ++added;
                    }
                    for (unsigned index = 0; index < call->arg_size(); ++index) {
                        if (detail::is_bool(call->getArgOperand(index)->getType())
                            && !detail::has_extension(attributes.getParamAttrs(index)))
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
