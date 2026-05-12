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

        // `bti c` / `bti jc`: entry. `bti j`: mid-function pad.
        // Conservative accept if Ghidra's hint encoding is unparseable.
        if (m.equalsIgnoreCase("bti")) {
            if (first.getNumOperands() == 0) return true;
            String hint;
            try {
                hint = first.getDefaultOperandRepresentation(0);
            } catch (Exception e) {
                return true;
            }
            if (hint == null) return true;
            String h = hint.toLowerCase().trim();
            return h.equals("c") || h.equals("jc");
        }
        // Only LR-signing PAC variants are reliable prologue markers.
        String lower = m.toLowerCase();
        if (lower.equals("paciasp") || lower.equals("pacibsp")
                || lower.equals("pacia1716") || lower.equals("pacib1716")) {
            return true;
        }
        if (m.equalsIgnoreCase("stp")) {
            // Prologue stp: sp base AND at least one callee-saved reg.
            if (!operandReferencesRegister(first, 2, "sp")) return false;
            return spilledRegisterIsCalleeSaved(first, 0)
                || spilledRegisterIsCalleeSaved(first, 1);
        }
        if (m.equalsIgnoreCase("sub")) {
            // `sub sp, xN, #imm` is a context-save shim.
            return firstOperandIs(first, "sp")
                && operandReferencesRegister(first, 1, "sp")
                && hasImmediateOperand(first);
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

    // AAPCS64 callee-saved: x19-x28, x29 (fp), x30 (lr), d8-d15.
    private static boolean spilledRegisterIsCalleeSaved(
            Instruction ins, int opIdx) {
        if (opIdx < 0 || opIdx >= ins.getNumOperands()) return false;
        for (Object obj : ins.getOpObjects(opIdx)) {
            if (!(obj instanceof Register)) continue;
            String n = ((Register) obj).getName().toLowerCase();
            // FP/LR (x29/x30) — the canonical prologue pair.
            if (n.equals("x29") || n.equals("x30")
                    || n.equals("fp") || n.equals("lr")
                    || n.equals("w29") || n.equals("w30")) {
                return true;
            }
            // x19-x28 (and w-aliases) — additional callee-saved GPRs.
            if (n.length() >= 3
                    && (n.charAt(0) == 'x' || n.charAt(0) == 'w')) {
                try {
                    int idx = Integer.parseInt(n.substring(1));
                    if (idx >= 19 && idx <= 28) return true;
                } catch (NumberFormatException e) {
                    // fall through
                }
            }
            // d8-d15 — callee-saved FP regs (lower 64 bits).
            if (n.length() >= 2 && n.charAt(0) == 'd') {
                try {
                    int idx = Integer.parseInt(n.substring(1));
                    if (idx >= 8 && idx <= 15) return true;
                } catch (NumberFormatException e) {
                    // fall through
                }
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
