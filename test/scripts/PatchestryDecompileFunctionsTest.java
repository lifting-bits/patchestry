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
public class PatchestryDecompileFunctionsTest extends BaseTest {

    private PatchestryDecompileFunctions decompileScript = new PatchestryDecompileFunctions();

    @BeforeAll
    public void setUp() throws Exception {
        super.setUp();

        File bloodlightFirmware = new File(System.getenv("BLOODLIGHT_FW_PATH"));
        program = load(bloodlightFirmware, testFwArch, project);

        String[] argsBloodlight = {
            "single",
            "main",
            "bloodlight_main_testgetarch.txt"
        };
        decompileScript.setScriptArgs(argsBloodlight);
        decompileScript.setProgram(program);

        // if we don't fake this, because we aren't in GhidraScript context here, testing
        // decompiler-reliant stuff doesn't work
        GhidraState fakeState = new GhidraState(env.getTool(), project, program, null, null, null);
        Field state = decompileScript.getClass().getSuperclass().getDeclaredField("state");
        state.setAccessible(true);
        state.set(decompileScript, fakeState);

        Field monitorField = findFieldInHierarchy(decompileScript.getClass(), "monitor");
        monitorField.setAccessible(true);
        monitorField.set(decompileScript, fakeMonitor);
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
        assertThrows(IllegalArgumentException.class, () -> {
            decompileScript.serializeToFile(null, empty);
        });

        Path tempFile = Files.createTempFile("test-serialize-to-file", ".json");
        JsonWriter writer = new JsonWriter(Files.newBufferedWriter(tempFile));
        assertThrows(IllegalArgumentException.class, () -> {
            decompileScript.serializeToFile(writer, empty);
        });

        List<Function> usbFn = decompileScript.getGlobalFunctions("bl_usb__send_message");
        decompileScript.serializeToFile(writer, usbFn);
        String fileContents = Files.readString(tempFile);
        assertNotNull(fileContents);
        // todo (kaoudis) consider testing file contents here - but that is more about PcodeSerializer working correctly
    }

    @Test
    public void testGetAllFunctions() throws Exception {
        assertNotNull(decompileScript.getCurrentProgram());
        List<Function> functions = decompileScript.getAllFunctions();
        // todo (kaoudis) check this number somehow - is it EVERYTHING in bloodlight?
        assertEquals(functions.size(), 262);
    }

    @Disabled
    @Test
    public void testDecompileSingleFunction() throws Exception {
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
