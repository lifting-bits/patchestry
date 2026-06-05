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

import util.firmware.UseropClassifier.Result;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class UseropClassifierTest {

    @Test
    public void interruptMasking() {
        Result disable = UseropClassifier.classify("disableIRQinterrupts");
        assertEquals("irq_mask_set", disable.klass);
        assertEquals("PRIMASK", disable.register);
        assertTrue(disable.mapped);

        Result enable = UseropClassifier.classify("enableIRQinterrupts");
        assertEquals("irq_mask_clear", enable.klass);
        assertEquals("PRIMASK", enable.register);

        Result fault = UseropClassifier.classify("disableFIQinterrupts");
        assertEquals("irq_mask_set", fault.klass);
        assertEquals("FAULTMASK", fault.register);
    }

    @Test
    public void specialRegisters() {
        Result ipsr = UseropClassifier.classify("getCurrentExceptionNumber");
        assertEquals("sysreg_read", ipsr.klass);
        assertEquals("IPSR", ipsr.register);
        assertTrue(ipsr.mapped);

        Result basepri = UseropClassifier.classify("setBasePriority");
        assertEquals("sysreg_write", basepri.klass);
        assertEquals("BASEPRI", basepri.register);
    }

    @Test
    public void hints() {
        assertEquals("hint_wfi", UseropClassifier.classify("WaitForInterrupt").klass);
        assertEquals("hint_wfe", UseropClassifier.classify("WaitForEvent").klass);
        assertEquals("hint_sev", UseropClassifier.classify("SendEvent").klass);
    }

    @Test
    public void coprocessorDirectAndNamed() {
        Result ldc = UseropClassifier.classify("coprocessor_load");
        assertEquals("coproc_load", ldc.klass);
        assertNull(ldc.register);
        assertTrue(ldc.mapped);

        // Named CP15 form resolves the register from the name suffix.
        Result named = UseropClassifier.classify("coproc_moveto_Control");
        assertEquals("coproc_write", named.klass);
        assertEquals("Control", named.register);
        assertTrue(named.mapped);
    }

    @Test
    public void barriersClassifiedButLeftToFencePath() {
        assertEquals("barrier_dmb", UseropClassifier.classify("DataMemoryBarrier").klass);
        assertEquals("barrier_dsb", UseropClassifier.classify("DataSynchronizationBarrier").klass);
    }

    @Test
    public void unknownAndNullAreUnmapped() {
        Result neon = UseropClassifier.classify("VectorAbs");
        assertEquals(UseropClassifier.UNKNOWN, neon.klass);
        assertFalse(neon.mapped);

        Result nul = UseropClassifier.classify(null);
        assertEquals(UseropClassifier.UNKNOWN, nul.klass);
        assertFalse(nul.mapped);
    }
}
