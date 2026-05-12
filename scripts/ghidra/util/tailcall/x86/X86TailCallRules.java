/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.tailcall.x86;

import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.Register;
import ghidra.program.model.listing.Instruction;

import util.tailcall.TailCallProcessorRules;

// x86 / x86-64. Recognised prologue shapes: push, sub rsp,#imm, endbr*.
// SysV x86-64 has a 128-byte red zone; all other x86 variants do not.
public final class X86TailCallRules implements TailCallProcessorRules {
    private final boolean is64Bit;

    public X86TailCallRules(Language lang) {
        this.is64Bit = lang != null
            && lang.getLanguageDescription() != null
            && lang.getLanguageDescription().getSize() == 64;
    }

    @Override
    public boolean isFunctionPrologue(Instruction first) {
        if (first == null) return false;
        String m = first.getMnemonicString();
        if (m == null) return false;
        String lower = m.toLowerCase();
        if (lower.equals("endbr64") || lower.equals("endbr32")) return true;
        if (lower.equals("push")) return true;
        if (lower.equals("sub")) return firstOperandIsStackPointer(first);
        return false;
    }

    @Override
    public int redZoneSize() { return is64Bit ? 128 : 0; }

    @Override
    public int thunkInstructionLimit() { return 1; }

    private static boolean firstOperandIsStackPointer(Instruction ins) {
        if (ins.getNumOperands() == 0) return false;
        for (Object obj : ins.getOpObjects(0)) {
            if (obj instanceof Register) {
                String name = ((Register) obj).getName().toLowerCase();
                if (name.equals("rsp") || name.equals("esp") || name.equals("sp")) {
                    return true;
                }
            }
        }
        return false;
    }
}
