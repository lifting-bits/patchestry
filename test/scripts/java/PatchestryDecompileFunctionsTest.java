/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.lang.reflect.Method;
import java.util.List;

import org.junit.jupiter.api.*;

import ghidra.app.util.importer.AutoImporter;
import ghidra.app.util.importer.MessageLog;
import ghidra.app.util.opinion.LoadResults;
import ghidra.framework.model.Project;
import ghidra.program.model.lang.CompilerSpec;
import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.LanguageID;
import ghidra.program.model.lang.LanguageService;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Program;
import ghidra.program.util.DefaultLanguageService;
import ghidra.test.AbstractGhidraHeadlessIntegrationTest;
import ghidra.test.TestEnv;
import ghidra.util.task.TaskMonitor;
import ghidra.app.script.GhidraScript;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class PatchestryDecompileFunctionsTest extends AbstractGhidraHeadlessIntegrationTest {
    private Program pulseOxProgram;
    private Program bloodlightProgram;
    private MessageLog log = new MessageLog();
    private TaskMonitor monitor = TaskMonitor.DUMMY;
    private PatchestryDecompileFunctions patchestryDecompileFunctions = new PatchestryDecompileFunctions();
    private String testFwArch = "ARM:LE:32:Cortex";

    private Program load(File fw, String arch, Project project) throws Exception {
        LanguageService languageService = DefaultLanguageService.getLanguageService();
        LanguageID langId = new LanguageID(arch);
        Language language = languageService.getLanguage(langId);
        CompilerSpec compilerSpec = language.getDefaultCompilerSpec();

        // lcs == language and compiler spec
        LoadResults<Program> importResults = AutoImporter.importByLookingForLcs(
            fw,
            project,
            project.getProjectData().getRootFolder().getPathname(),
            language,
            compilerSpec,
            this,
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
        pulseOxProgram = load(pulseOxFirmware, testFwArch, project);

        // the same is true of BLOODLIGHT_FW_PATH - it comes from 
        // decompile-test.dockerfile
        File bloodlightFirmware = new File(System.getenv("BLOODLIGHT_FW_PATH"));
        bloodlightProgram = load(bloodlightFirmware, testFwArch, project);
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
    public void testGetArch() throws Exception {
        String[] argsPulseOx = {
            "single", 
            "main", 
            "pulseox_main_testgetarch.txt"
        };
        patchestryDecompileFunctions.setScriptArgs(argsPulseOx);
        patchestryDecompileFunctions.setProgram(pulseOxProgram);

        assertTrue(patchestryDecompileFunctions.getArch() == "ARM");
        assertTrue(patchestryDecompileFunctions.getCurrentProgram().getLanguageID().toString() == testFwArch);
        assertTrue(patchestryDecompileFunctions.getCurrentProgram().getExecutablePath().contains("pulseox"));

         String[] argsBloodlight = {
            "single", 
            "main", 
            "bloodlight_main_testgetarch.txt"
        };
        patchestryDecompileFunctions.setScriptArgs(argsBloodlight);
        patchestryDecompileFunctions.setProgram(bloodlightProgram);

        assertTrue(patchestryDecompileFunctions.getArch() == "ARM");
        assertTrue(patchestryDecompileFunctions.getCurrentProgram().getLanguageID().toString() == testFwArch);
        assertTrue(patchestryDecompileFunctions.getCurrentProgram().getExecutablePath().contains("bloodlight"));
    }

    @Test 
    public void testGetLanguageID() throws Exception {
        String[] argsPulseOx = {
            "single", 
            "main", 
            "pulseox_main_testgetarch.txt"
        };
        patchestryDecompileFunctions.setScriptArgs(argsPulseOx);
        patchestryDecompileFunctions.setProgram(pulseOxProgram);
        assertNotNull(patchestryDecompileFunctions.getCurrentProgram().getLanguage());
        assertTrue(patchestryDecompileFunctions.getLanguageID().toString() == testFwArch);
        assertTrue(patchestryDecompileFunctions.getCurrentProgram().getExecutablePath().contains("pulseox"));

         String[] argsBloodlight = {
            "single", 
            "main", 
            "bloodlight_main_testgetarch.txt"
        };
        patchestryDecompileFunctions.setScriptArgs(argsBloodlight);
        patchestryDecompileFunctions.setProgram(bloodlightProgram);
        assertNotNull(patchestryDecompileFunctions.getCurrentProgram().getLanguage());
        assertTrue(patchestryDecompileFunctions.getLanguageID().toString() == testFwArch);
        assertTrue(patchestryDecompileFunctions.getCurrentProgram().getExecutablePath().contains("bloodlight"));
    }

    @Test 
    public void testGetDecompilerInterface() throws Exception {
        String[] argsPulseOx = {
            "single", 
            "main", 
            "pulseox_main_testdecompilerinterface.txt"
        };
        patchestryDecompileFunctions.setScriptArgs(argsPulseOx);
        patchestryDecompileFunctions.setProgram(pulseOxProgram);
        assertNotNull(patchestryDecompileFunctions.getCurrentProgram());

        Method getDecompilerInterface = PatchestryDecompileFunctions.class.getDeclaredMethod("getDecompilerInterface");
        getDecompilerInterface.setAccessible(true);
        getDecompilerInterface.invoke(patchestryDecompileFunctions);
    }

    @Test 
    public void testSerializeToFile() throws Exception {
         assertTrue(false);
    }

    @Test
    public void testGetAllFunctions() throws Exception {
        String[] args = {
            "all", 
            "pulseox_main_testgetallfunctions.txt"
        };
        patchestryDecompileFunctions.setScriptArgs(args);
        patchestryDecompileFunctions.setProgram(pulseOxProgram);
        assertNotNull(patchestryDecompileFunctions.getCurrentProgram());
        
        Method getAllFunctions = PatchestryDecompileFunctions.class.getDeclaredMethod("getAllFunctions");
        getAllFunctions.setAccessible(true);
        List<Function> output = (List<Function>) getAllFunctions.invoke(patchestryDecompileFunctions);
    }

    @Test
    public void testDecompileSingleFunction() throws Exception {
        String[] args = {
            "single", 
            "main", 
            "pulseox_main_testrunheadlessdecompilesingle.txt"
        };
        patchestryDecompileFunctions.setScriptArgs(args);
        patchestryDecompileFunctions.setProgram(pulseOxProgram);

        assertNotNull(patchestryDecompileFunctions.getCurrentProgram());

        Method runHeadless = PatchestryDecompileFunctions.class.getDeclaredMethod("runHeadless");
        runHeadless.setAccessible(true);
        runHeadless.invoke(patchestryDecompileFunctions);
    }

    @Test 
    public void testDecompileAllFunctions() throws Exception {
         assertTrue(false);
    }

    @Test 
    public void testRunAutoAnalysis() throws Exception {
         assertTrue(false);
    }
}