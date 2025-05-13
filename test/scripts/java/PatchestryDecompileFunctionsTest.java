package scripts;
/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */


import org.junit.Test;
import static org.junit.Assert.*;

import java.util.function.Function;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Program;
import ghidra.test.AbstractGhidraHeadlessIntegrationTest;

public class PatchestryDecompileFunctionsTest extends AbstractGhidraHeadlessIntegrationTest {
    private Program pulseOxBinary;
    private Program bloodlightBinary;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        // NB: we set PULSEOX_FW_PATH in decompile-test.dockerfile, the test env.
        // The test environment is based on the main headless container, 
        // which must be built first before the test container!
        String pulseOxFirmwareLocation = System.getenv("PULSEOX_FW_PATH");
        pulseOxBinary = loadTestProgram(pulseOxFirmwareLocation);

        // the same is true of BLOODLIGHT_FW_PATH - it comes from 
        // decompile-test.dockerfile
        String bloodlightFirmwareLocation = System.getenv("BLOODLIGHT_FW_PATH");
        bloodlightBinary = loadTestProgram(bloodlightFirmwareLocation);
    }

    @Override
    public void tearDown() throws Exception {
        if (pulseOxBinary != null) {
            pulseOxBinary.release(this);
        }

        if (bloodlightBinary != null) {
            bloodlightBinary.release(this);
        }

        super.tearDown();
    }

    @Test
    public void testDecompileFunction() throws Exception {
        // PatchestryDecompileFunctions decompScript = new PatchestryDecompileFunctions();
        
        // decompScript.set(precompiledTestBinary, null, null, TaskMonitor.DUMMY);
        
        // String result = decompScript.decompileFunction(function);
        // assertNotNull("Decompilation result should not be null", result);
        assertTrue(false);
    }
}