/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.tailcall;

import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.Processor;
import ghidra.program.model.listing.Instruction;

// Per-architecture knowledge for TailCallDetector. Adding a target is one
// class plus one switch case in lookup().
public interface TailCallProcessorRules {

    // True when `first` plausibly begins a function (per-arch prologue idiom).
    boolean isFunctionPrologue(Instruction first);

    // SP-delta tolerance at the branch site (bytes). 0 for most ABIs;
    // 128 for SysV x86-64 (red zone).
    int redZoneSize();

    // Max body size (instructions) treated as a single-instruction thunk.
    int thunkInstructionLimit();

    static TailCallProcessorRules lookup(Language lang) {
        if (lang == null) return null;
        Processor proc = lang.getProcessor();
        if (proc == null) return null;
        switch (proc.toString()) {
            case "ARM":     return new util.tailcall.arm.ARMTailCallRules(lang);
            case "AARCH64": return new util.tailcall.arm.AArch64TailCallRules(lang);
            case "x86":     return new util.tailcall.x86.X86TailCallRules(lang);
            case "PowerPC": return new util.tailcall.ppc.PowerPCTailCallRules(lang);
            case "MIPS":    return new util.tailcall.mips.MIPSTailCallRules(lang);
            case "RISCV":   return new util.tailcall.riscv.RISCVTailCallRules(lang);
            default:        return null;
        }
    }
}
