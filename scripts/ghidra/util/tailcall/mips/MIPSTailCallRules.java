/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.tailcall.mips;

import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.Register;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.scalar.Scalar;

import util.tailcall.TailCallProcessorRules;

// MIPS. Recognised prologue shapes: addiu sp,sp,-N; sw|sd ra,N(sp).
public final class MIPSTailCallRules implements TailCallProcessorRules {

    public MIPSTailCallRules(Language lang) { /* unused */ }

    @Override
    public boolean isFunctionPrologue(Instruction first) {
        if (first == null) return false;
        String m = first.getMnemonicString();
        if (m == null) return false;
        String lower = m.toLowerCase();
        if (lower.equals("addiu") || lower.equals("daddiu")) {
            return firstOperandIs(first, "sp") && hasNegativeImmediateOperand(first);
        }
        if (lower.equals("sw") || lower.equals("sd")) {
            return operandReferencesRegister(first, 0, "ra");
        }
        return false;
    }

    @Override
    public int redZoneSize() { return 0; }

    @Override
    public int thunkInstructionLimit() { return 2; }

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

    private static boolean hasNegativeImmediateOperand(Instruction ins) {
        for (int i = 0; i < ins.getNumOperands(); ++i) {
            for (Object obj : ins.getOpObjects(i)) {
                if (obj instanceof Scalar && ((Scalar) obj).getSignedValue() < 0) {
                    return true;
                }
            }
        }
        return false;
    }
}
