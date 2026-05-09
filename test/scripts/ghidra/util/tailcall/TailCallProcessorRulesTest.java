/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */
package util.tailcall;

import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.LanguageDescription;
import ghidra.program.model.lang.Processor;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class TailCallProcessorRulesTest {

    private static Language mockLanguage(String processorName, int sizeBits) {
        Language lang = mock(Language.class);
        Processor proc = Processor.findOrPossiblyCreateProcessor(processorName);
        when(lang.getProcessor()).thenReturn(proc);

        LanguageDescription desc = mock(LanguageDescription.class);
        when(desc.getSize()).thenReturn(sizeBits);
        when(lang.getLanguageDescription()).thenReturn(desc);

        return lang;
    }

    @Test
    public void lookupReturnsArmRulesForArmProcessor() {
        TailCallProcessorRules rules = TailCallProcessorRules.lookup(
            mockLanguage("ARM", 32));
        assertNotNull(rules);
        assertEquals(0, rules.redZoneSize());
    }

    @Test
    public void lookupReturnsAArch64RulesForAArch64Processor() {
        TailCallProcessorRules rules = TailCallProcessorRules.lookup(
            mockLanguage("AARCH64", 64));
        assertNotNull(rules);
        assertEquals(0, rules.redZoneSize());
    }

    @Test
    public void x86_64HasSysVRedZone() {
        TailCallProcessorRules rules = TailCallProcessorRules.lookup(
            mockLanguage("x86", 64));
        assertNotNull(rules);
        assertEquals(128, rules.redZoneSize());
    }

    @Test
    public void x86_32HasNoRedZone() {
        TailCallProcessorRules rules = TailCallProcessorRules.lookup(
            mockLanguage("x86", 32));
        assertNotNull(rules);
        assertEquals(0, rules.redZoneSize());
    }

    @Test
    public void mipsAvailable() {
        assertNotNull(TailCallProcessorRules.lookup(mockLanguage("MIPS", 32)));
    }

    @Test
    public void powerPcAndRiscvAvailable() {
        assertNotNull(TailCallProcessorRules.lookup(mockLanguage("PowerPC", 64)));
        assertNotNull(TailCallProcessorRules.lookup(mockLanguage("RISCV", 64)));
    }

    @Test
    public void unknownProcessorReturnsNull() {
        assertNull(TailCallProcessorRules.lookup(mockLanguage("XYZQQ", 32)));
    }

    @Test
    public void nullLanguageReturnsNull() {
        assertNull(TailCallProcessorRules.lookup(null));
    }
}
