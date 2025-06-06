/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import java.io.File;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import ghidra.app.util.importer.AutoImporter;
import ghidra.app.util.importer.MessageLog;
import ghidra.app.util.opinion.LoadResults;
import ghidra.framework.model.Project;
import ghidra.program.model.listing.Program;
import ghidra.test.AbstractGhidraHeadlessIntegrationTest;
import ghidra.test.TestEnv;
import ghidra.util.task.TaskMonitor;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class PatchestryDecompileFunctionsTest extends AbstractGhidraHeadlessIntegrationTest {
    private Program pulseOxProgram;
    private Program bloodlightProgram;
    private MessageLog log = new MessageLog();
    private TaskMonitor monitor = TaskMonitor.DUMMY;

    private Program load(File fw, Project project) throws Exception {
        LoadResults<Program> importResults = AutoImporter.importByUsingBestGuess(
            fw,
            project,
            project.getProjectData().getRootFolder().getPathname(),
            (Object) null,
            log,
            monitor
        );
        return (Program) importResults.getPrimaryDomainObject();
    }

    @BeforeAll
    public void setUp() throws Exception {
        TestEnv env = new TestEnv();
        Project project = env.getProject();
        // NB: we set PULSEOX_FW_PATH in decompile-test.dockerfile, the test env.
        // The test environment is based on the main headless container, 
        // which must be built first before the test container!
        File pulseOxFirmware = new File(System.getenv("PULSEOX_FW_PATH"));
        pulseOxProgram = load(pulseOxFirmware, project);

        // the same is true of BLOODLIGHT_FW_PATH - it comes from 
        // decompile-test.dockerfile
        File bloodlightFirmware = new File(System.getenv("BLOODLIGHT_FW_PATH"));
        bloodlightProgram = load(bloodlightFirmware, project);
    }

    @AfterAll
    public void tearDown() throws Exception {
        if (pulseOxProgram != null) {
            pulseOxProgram.release(this);
        }

        if (bloodlightProgram != null) {
            bloodlightProgram.release(this);
        }
    }

    @Test
    public void testDecompileFunction() throws Exception {
        assertTrue(false);
    }
}