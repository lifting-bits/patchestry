/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.tailcall.arm;

import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.Register;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.scalar.Scalar;

import util.tailcall.TailCallProcessorRules;

// ARM-32 (A32) and Thumb (T32). Recognised prologue shapes:
//   stmdb sp!,{...,lr} / push {...,lr}   callee-saved + LR spill
//   sub sp,sp,#imm                       bare stack reservation
//   mov r12,sp                           frame-pointer setup
public final class ARMTailCallRules implements TailCallProcessorRules {

    public ARMTailCallRules(Language lang) { /* unused */ }

    @Override
    public boolean isFunctionPrologue(Instruction first) {
        if (first == null) return false;
        String m = first.getMnemonicString();
        if (m == null) return false;

        // push: base implicit. stmdb/stmfd: explicit base — require sp.
        if (m.equalsIgnoreCase("push")) {
            return operandsContainsRegister(first, "lr")
                || operandsContainsRegister(first, "r14");
        }
        if (m.equalsIgnoreCase("stmdb") || m.equalsIgnoreCase("stmfd")) {
            boolean baseIsSp = firstOperandIs(first, "sp");
            boolean spillsLr = operandsContainsRegister(first, "lr")
                || operandsContainsRegister(first, "r14");
            return baseIsSp && spillsLr;
        }
        if (m.equalsIgnoreCase("sub")) {
            // `sub sp, rN, #imm` is a context-save / longjmp shim.
            return firstOperandIs(first, "sp")
                && operandReferencesRegister(first, 1, "sp")
                && hasImmediateOperand(first);
        }
        if (m.equalsIgnoreCase("mov")) {
            return firstOperandIs(first, "r12")
                && operandsContainsRegister(first, "sp");
        }
        return false;
    }

    @Override
    public int redZoneSize() { return 0; }

    @Override
    public int thunkInstructionLimit() { return 1; }

    private static boolean operandsContainsRegister(Instruction ins, String regName) {
        for (int i = 0; i < ins.getNumOperands(); ++i) {
            for (Object obj : ins.getOpObjects(i)) {
                if (obj instanceof Register
                        && ((Register) obj).getName().equalsIgnoreCase(regName)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean firstOperandIs(Instruction ins, String regName) {
        if (ins.getNumOperands() == 0) return false;
        for (Object obj : ins.getOpObjects(0)) {
            if (obj instanceof Register
                    && ((Register) obj).getName().equalsIgnoreCase(regName)) {
                return true;
            }
        }
        return false;
    }

    private static boolean operandReferencesRegister(
            Instruction ins, int opIdx, String regName) {
        if (opIdx < 0 || opIdx >= ins.getNumOperands()) return false;
        for (Object obj : ins.getOpObjects(opIdx)) {
            if (obj instanceof Register
                    && ((Register) obj).getName().equalsIgnoreCase(regName)) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasImmediateOperand(Instruction ins) {
        for (int i = 0; i < ins.getNumOperands(); ++i) {
            for (Object obj : ins.getOpObjects(i)) {
                if (obj instanceof Scalar) return true;
            }
        }
        return false;
    }
}
