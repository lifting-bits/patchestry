/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.firmware;

import java.util.HashMap;
import java.util.Map;

/**
 * ARM (32-bit) userop vocabulary. Maps Ghidra 12.0.4 ARM SLEIGH
 * {@code define pcodeop} names (Processors/ARM/data/languages/*.sinc) onto the
 * shared {@link IntrinsicClassifier} taxonomy.
 *
 * <p>Names that carry their identity (the common case) are resolved by a flat
 * map; the named CP15 {@code coproc_movefrom_<X>} / {@code coproc_moveto_<X>}
 * family is resolved by prefix.
 */
public final class Arm32IntrinsicClassifier implements ArchIntrinsicClassifier {

    private static final class Entry {
        final String klass;
        final String register;
        Entry(String klass, String register) { this.klass = klass; this.register = register; }
    }

    private static final Map<String, Entry> NAME_MAP = build();

    private static Map<String, Entry> build() {
        Map<String, Entry> m = new HashMap<>();

        // Interrupt masking (CPSID/CPSIE). PRIMASK == `i`, FAULTMASK == `f`.
        m.put("disableIRQinterrupts",  new Entry("irq_mask_set",   "PRIMASK"));
        m.put("enableIRQinterrupts",   new Entry("irq_mask_clear", "PRIMASK"));
        m.put("disableFIQinterrupts",  new Entry("irq_mask_set",   "FAULTMASK"));
        m.put("enableFIQinterrupts",   new Entry("irq_mask_clear", "FAULTMASK"));
        // A/R abort-mask `a` bit: classified but no CMSIS spelling (C++ falls through).
        m.put("disableDataAbortInterrupts", new Entry("irq_mask_set",   null));
        m.put("enableDataAbortInterrupts",  new Entry("irq_mask_clear", null));

        // Cortex-M special registers -> CMSIS __get_<R>()/__set_<R>().
        m.put("getBasePriority",             new Entry("sysreg_read",  "BASEPRI"));
        m.put("setBasePriority",             new Entry("sysreg_write", "BASEPRI"));
        m.put("getMainStackPointer",         new Entry("sysreg_read",  "MSP"));
        m.put("setMainStackPointer",         new Entry("sysreg_write", "MSP"));
        m.put("getProcessStackPointer",      new Entry("sysreg_read",  "PSP"));
        m.put("setProcessStackPointer",      new Entry("sysreg_write", "PSP"));
        m.put("getMainStackPointerLimit",    new Entry("sysreg_read",  "MSPLIM"));
        m.put("setMainStackPointerLimit",    new Entry("sysreg_write", "MSPLIM"));
        m.put("getProcessStackPointerLimit", new Entry("sysreg_read",  "PSPLIM"));
        m.put("setProcStackPointerLimit",    new Entry("sysreg_write", "PSPLIM"));
        m.put("getCurrentExceptionNumber",   new Entry("sysreg_read",  "IPSR"));

        // Mode/privilege queries: classified (mapped) but no clean accessor.
        m.put("isThreadMode",            new Entry("query", null));
        m.put("isThreadModePrivileged",  new Entry("query", null));
        m.put("isCurrentModePrivileged", new Entry("query", null));
        m.put("isUsingMainStack",        new Entry("query", null));
        m.put("isIRQinterruptsEnabled",  new Entry("query", null));
        m.put("isFIQinterruptsEnabled",  new Entry("query", null));
        m.put("setThreadModePrivileged", new Entry("mode_switch", null));

        // Hints -> ACLE nullary builtins.
        m.put("WaitForInterrupt", new Entry("hint_wfi",   null));
        m.put("WaitForEvent",     new Entry("hint_wfe",   null));
        m.put("SendEvent",        new Entry("hint_sev",   null));
        m.put("HintYield",        new Entry("hint_yield", null));

        // Barriers: classified, but C++ leaves them on the C11 atomic-fence path.
        m.put("DataMemoryBarrier",                 new Entry("barrier_dmb", null));
        m.put("DataSynchronizationBarrier",        new Entry("barrier_dsb", null));
        m.put("InstructionSynchronizationBarrier", new Entry("barrier_isb", null));

        // Single-register MCR/MRC/CDP -> ACLE __arm_mcr/__arm_mrc/__arm_cdp
        // (C++ emit_arm_coproc reorders the operands). The *2 forms are
        // MCRR/MRRC/CDP2 with a different operand shape; classified distinctly
        // so they stay on the registered-call path until handled.
        m.put("coprocessor_moveto",       new Entry("coproc_write",  null));
        m.put("coprocessor_moveto2",      new Entry("coproc_write2", null));
        m.put("coprocessor_movefromRt",   new Entry("coproc_read",   null));
        m.put("coprocessor_movefromRt2",  new Entry("coproc_read2",  null));
        m.put("coprocessor_movefrom2",    new Entry("coproc_read2",  null));
        m.put("coprocessor_function",     new Entry("coproc_cdp",    null));
        m.put("coprocessor_function2",    new Entry("coproc_cdp2",   null));

        // Coprocessor LDC/STC -> ACLE __arm_ldc/__arm_stc (verified operand order).
        m.put("coprocessor_load",      new Entry("coproc_load",   null));
        m.put("coprocessor_loadlong",  new Entry("coproc_loadl",  null));
        m.put("coprocessor_store",     new Entry("coproc_store",  null));
        m.put("coprocessor_storelong", new Entry("coproc_storel", null));

        // Supervisor / trap instructions.
        m.put("software_interrupt", new Entry("trap", null));
        m.put("software_bkpt",      new Entry("trap", null));
        m.put("software_hlt",       new Entry("trap", null));
        m.put("software_hvc",       new Entry("trap", null));
        m.put("software_smc",       new Entry("trap", null));
        m.put("software_udf",       new Entry("trap", null));
        m.put("secureMonitorCall",  new Entry("trap", null));

        // Mode switching (A/R).
        for (String mode : new String[] {
                "setIRQMode", "setFIQMode", "setSupervisorMode", "setAbortMode",
                "setUndefinedMode", "setSystemMode", "setMonitorMode", "setUserMode",
                "setStackMode", "setEndianState", "setISAMode" }) {
            m.put(mode, new Entry("mode_switch", null));
        }

        return m;
    }

    @Override
    public IntrinsicClassifier.Result classify(String rawName) {
        if (rawName == null || rawName.isEmpty()) {
            return IntrinsicClassifier.unknown();
        }

        Entry e = NAME_MAP.get(rawName);
        if (e != null) {
            return IntrinsicClassifier.result(e.klass, e.register);
        }

        // Named CP15 system-register accessors: the register is baked into the
        // name suffix. coproc_moveto_Control -> write "Control".
        if (rawName.startsWith("coproc_moveto_")) {
            return IntrinsicClassifier.result(
                "coproc_write", rawName.substring("coproc_moveto_".length()));
        }
        if (rawName.startsWith("coproc_movefrom_")) {
            return IntrinsicClassifier.result(
                "coproc_read", rawName.substring("coproc_movefrom_".length()));
        }

        return IntrinsicClassifier.unknown();
    }
}
