package scripts;

/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import static org.junit.jupiter.api.Assertions.*;

import ghidra.program.model.listing.Program;
import ghidra.test.AbstractGenericTest;
import java.io.File;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class PatchestryDecompileFunctionsTest extends AbstractGenericTest {
    private Program pulseOxProgram;
    private Program bloodlightProgram;

    @BeforeAll
    public void setUp() throws Exception {
        // NB: we set PULSEOX_FW_PATH in decompile-test.dockerfile, the test env.
        // The test environment is based on the main headless container, 
        // which must be built first before the test container!
        File pulseOxFirmware = new File(System.getenv("PULSEOX_FW_PATH"));
        pulseOxProgram = importProgram(pulseOxFirmware);

        // the same is true of BLOODLIGHT_FW_PATH - it comes from 
        // decompile-test.dockerfile
        // File bloodlightFirmware = new File(System.getenv("BLOODLIGHT_FW_PATH"));
        // bloodlightProgram = importProgram(bloodlightFirmware);
    }

    @AfterAll
    public void tearDown() throws Exception {
        if (pulseOxProgram != null) {
            pulseOxProgram.release(this);
        }

        if (bloodlightProgram != null) {
            bloodlightProgram.release(this);
        }
        // super.tearDown();
    }

    @Test
    public void testDecompileFunction() throws Exception {
        assertTrue(false);
    }
}