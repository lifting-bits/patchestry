/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.tailcall.ppc;

import ghidra.program.model.lang.Language;
import ghidra.program.model.listing.Instruction;

import util.tailcall.TailCallProcessorRules;

// PowerPC (32 / 64). Recognised prologue shapes: mflr, stwu, stdu,
// stw / std (LR spill).
public final class PowerPCTailCallRules implements TailCallProcessorRules {

    public PowerPCTailCallRules(Language lang) { /* unused */ }

    @Override
    public boolean isFunctionPrologue(Instruction first) {
        if (first == null) return false;
        String m = first.getMnemonicString();
        if (m == null) return false;
        String lower = m.toLowerCase();
        return lower.equals("mflr")
            || lower.equals("stwu")
            || lower.equals("stdu")
            || lower.equals("stw")
            || lower.equals("std");
    }

    @Override
    public int redZoneSize() { return 0; }

    @Override
    public int thunkInstructionLimit() { return 2; }
}
