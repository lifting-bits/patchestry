/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.tailcall.ppc;

import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.Register;
import ghidra.program.model.listing.Instruction;

import util.tailcall.TailCallProcessorRules;

// PowerPC (32/64). Prologue shapes: mflr | stwu/stdu/stw/std on r1 (sp).
public final class PowerPCTailCallRules implements TailCallProcessorRules {

    public PowerPCTailCallRules(Language lang) { /* unused */ }

    @Override
    public boolean isFunctionPrologue(Instruction first) {
        if (first == null) return false;
        String m = first.getMnemonicString();
        if (m == null) return false;
        String lower = m.toLowerCase();
        // Strong (not perfectly unambiguous) entry signal.
        if (lower.equals("mflr")) return true;
        // Generic stw/std to non-r1 bases aren't prologues.
        if (lower.equals("stwu") || lower.equals("stdu")
                || lower.equals("stw") || lower.equals("std")) {
            return operandReferencesRegister(first, 1, "r1");
        }
        return false;
    }

    @Override
    public int redZoneSize() { return 0; }

    @Override
    public int thunkInstructionLimit() { return 2; }

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
}
