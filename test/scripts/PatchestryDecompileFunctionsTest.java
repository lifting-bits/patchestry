/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.lang.IllegalArgumentException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import ghidra.app.script.GhidraState;
import ghidra.framework.plugintool.PluginTool;
import java.lang.reflect.Field;

import org.junit.jupiter.api.*;

import com.google.gson.stream.JsonWriter;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;

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
    private TestEnv env = null;
    private Project project = null;
    private Program bloodlightProgram = null;
    private MessageLog log = new MessageLog();
    private TaskMonitor monitor = mock(TaskMonitor.class);
    private PatchestryDecompileFunctions decompileScript = new PatchestryDecompileFunctions();
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
        // stub out mocked class methods
        when(monitor.isCancelled()).thenReturn(false);

        env = new TestEnv();
        project = env.getProject();

        File bloodlightFirmware = new File(System.getenv("BLOODLIGHT_FW_PATH"));
        bloodlightProgram = load(bloodlightFirmware, testFwArch, project);

        String[] argsBloodlight = {
            "single", 
            "main", 
            "bloodlight_main_testgetarch.txt"
        };
        decompileScript.setScriptArgs(argsBloodlight);
        decompileScript.setProgram(bloodlightProgram);

        // if we don't mock this, because we aren't in GhidraScript context here, testing
        // decompiler-reliant stuff doesn't work
        GhidraState fakeState = new GhidraState(env.getTool(), project, bloodlightProgram, null, null, null);
        Field state = decompileScript.getClass().getSuperclass().getDeclaredField("state");
        state.setAccessible(true);
        state.set(decompileScript, fakeState);
    }

    @AfterAll
    public void cleanUp() throws Exception {
        if (bloodlightProgram != null) {
            bloodlightProgram.release(this);
        }

        if (env != null) {
            env.dispose();
        }
    }

    @Test 
    public void testGetArch() throws Exception {
        assertNotNull(decompileScript.getCurrentProgram());
        assertEquals(decompileScript.getArch(), "ARM");
        assertEquals(decompileScript.getCurrentProgram().getLanguageID().toString(), testFwArch);
        assertEquals(decompileScript.getCurrentProgram().getExecutablePath(), System.getenv("BLOODLIGHT_FW_PATH").toString());
    }

    @Test 
    public void testGetLanguageID() throws Exception {
        assertNotNull(decompileScript.getCurrentProgram());
        assertNotNull(decompileScript.getCurrentProgram().getLanguage());
        assertEquals(decompileScript.getLanguageID().toString(), testFwArch);
    }

    @Test
    public void testGetDecompilerInterface() throws Exception {
        assertNotNull(decompileScript.getCurrentProgram());
        assertNotNull(decompileScript.getDecompilerInterface());
    }

    @Test 
    public void testSerializeToFile() throws Exception {
        List<Function> empty = new ArrayList();
        assertThrows(IllegalArgumentException.class, () -> {decompileScript.serializeToFile(null, empty);});

        Path tempFile = Files.createTempFile("test-serialize-to-file", ".json");
        JsonWriter writer = new JsonWriter(Files.newBufferedWriter(tempFile));
        assertThrows(IllegalArgumentException.class, () -> {decompileScript.serializeToFile(writer, empty);});
        
        List<Function> usbFn = decompileScript.getGlobalFunctions("bl_usb__send_message");
        decompileScript.serializeToFile(writer, usbFn);
        String fileContents = Files.readString(tempFile);
        assertNotNull(fileContents);
    }

    @Disabled
    @Test
    public void testGetAllFunctions() throws Exception {
        String[] args = {
            "all", 
            "bloodlight_main_testgetallfunctions.txt"
        };
        decompileScript.setScriptArgs(args);
        decompileScript.setProgram(bloodlightProgram);
        assertNotNull(decompileScript.getCurrentProgram());
        
        Method getAllFunctions = PatchestryDecompileFunctions.class.getDeclaredMethod("getAllFunctions");
        getAllFunctions.setAccessible(true);
        List<Function> output = (List<Function>) getAllFunctions.invoke(decompileScript);
    }

    @Disabled
    @Test
    public void testDecompileSingleFunction() throws Exception {
        String[] args = {
            "single", 
            "main", 
            "bloodlight_main_testrunheadlessdecompilesingle.txt"
        };
        decompileScript.setScriptArgs(args);
        decompileScript.setProgram(bloodlightProgram);

        assertNotNull(decompileScript.getCurrentProgram());

        Method runHeadless = PatchestryDecompileFunctions.class.getDeclaredMethod("runHeadless");
        runHeadless.setAccessible(true);
        runHeadless.invoke(decompileScript);
    }

    @Disabled
    @Test 
    public void testDecompileAllFunctions() throws Exception {
    }

    @Disabled
    @Test 
    public void testRunAutoAnalysis() throws Exception {
    }
}