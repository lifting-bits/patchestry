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

// AArch64 (A64). Recognised prologue shapes:
//   stp xN,xM,[sp,#-N]!   paired callee-saved store with SP pre-dec
//   sub sp,sp,#N          bare stack reservation
//   paciasp / pacibsp     PAC sign-LR
//   bti c|j|jc            BTI landing pad
public final class AArch64TailCallRules implements TailCallProcessorRules {

    public AArch64TailCallRules(Language lang) { /* unused */ }

    @Override
    public boolean isFunctionPrologue(Instruction first) {
        if (first == null) return false;
        String m = first.getMnemonicString();
        if (m == null) return false;

        if (m.equalsIgnoreCase("bti")) return true;
        if (m.toLowerCase().startsWith("pac")) return true;
        if (m.equalsIgnoreCase("stp")) {
            return operandReferencesRegister(first, 2, "sp");
        }
        if (m.equalsIgnoreCase("sub")) {
            return firstOperandIs(first, "sp") && hasImmediateOperand(first);
        }
        return false;
    }

    @Override
    public int redZoneSize() { return 0; }

    @Override
    public int thunkInstructionLimit() { return 1; }

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
