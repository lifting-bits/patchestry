/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */
package util.firmware;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import util.firmware.IntrinsicClassifier.Result;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class IntrinsicClassifierTest {

    private static Result arm(String name) {
        return IntrinsicClassifier.classify("ARM", name);
    }

    @Test
    public void interruptMasking() {
        Result disable = arm("disableIRQinterrupts");
        assertEquals("irq_mask_set", disable.klass);
        assertEquals("PRIMASK", disable.register);
        assertTrue(disable.mapped);

        Result enable = arm("enableIRQinterrupts");
        assertEquals("irq_mask_clear", enable.klass);
        assertEquals("PRIMASK", enable.register);

        Result fault = arm("disableFIQinterrupts");
        assertEquals("irq_mask_set", fault.klass);
        assertEquals("FAULTMASK", fault.register);
    }

    @Test
    public void specialRegisters() {
        Result ipsr = arm("getCurrentExceptionNumber");
        assertEquals("sysreg_read", ipsr.klass);
        assertEquals("IPSR", ipsr.register);
        assertTrue(ipsr.mapped);

        Result basepri = arm("setBasePriority");
        assertEquals("sysreg_write", basepri.klass);
        assertEquals("BASEPRI", basepri.register);
    }

    @Test
    public void hints() {
        assertEquals("hint_wfi", arm("WaitForInterrupt").klass);
        assertEquals("hint_wfe", arm("WaitForEvent").klass);
        assertEquals("hint_sev", arm("SendEvent").klass);
    }

    @Test
    public void coprocessorDirectAndNamed() {
        Result ldc = arm("coprocessor_load");
        assertEquals("coproc_load", ldc.klass);
        assertNull(ldc.register);
        assertTrue(ldc.mapped);

        // Named CP15 form resolves the register from the name suffix.
        Result named = arm("coproc_moveto_Control");
        assertEquals("coproc_write", named.klass);
        assertEquals("Control", named.register);
        assertTrue(named.mapped);
    }

    @Test
    public void barriersClassifiedButLeftToFencePath() {
        assertEquals("barrier_dmb", arm("DataMemoryBarrier").klass);
        assertEquals("barrier_dsb", arm("DataSynchronizationBarrier").klass);
    }

    @Test
    public void unknownAndNullAreUnmapped() {
        Result neon = arm("VectorAbs");
        assertEquals(IntrinsicClassifier.UNKNOWN, neon.klass);
        assertFalse(neon.mapped);

        Result nul = arm(null);
        assertEquals(IntrinsicClassifier.UNKNOWN, nul.klass);
        assertFalse(nul.mapped);
    }

    @Test
    public void archDispatchIsCaseInsensitiveAndScoped() {
        // Same name resolves under ARM but not under the AArch64 stub or an
        // unknown processor -- proving per-arch dispatch.
        assertTrue(IntrinsicClassifier.classify("arm", "disableIRQinterrupts").mapped);
        assertFalse(IntrinsicClassifier.classify("AARCH64", "disableIRQinterrupts").mapped);
        assertFalse(IntrinsicClassifier.classify("AARCH64", "SysOp_W").mapped);
        assertFalse(IntrinsicClassifier.classify("x86", "disableIRQinterrupts").mapped);
    }
}
