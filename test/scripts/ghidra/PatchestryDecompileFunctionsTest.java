/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.*;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;

import ghidra.app.script.GhidraScript;
import ghidra.app.script.GhidraState;

import ghidra.app.util.importer.AutoImporter;
import ghidra.app.util.importer.MessageLog;
import ghidra.app.util.opinion.LoadResults;

import ghidra.framework.model.Project;

import ghidra.framework.plugintool.PluginTool;

import ghidra.program.model.lang.CompilerSpec;
import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.LanguageID;
import ghidra.program.model.lang.LanguageService;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Program;
import ghidra.program.model.listing.Variable;

import ghidra.program.util.DefaultLanguageService;

import ghidra.test.TestEnv;

import ghidra.util.task.TaskMonitor;

import com.google.gson.stream.JsonWriter;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.File;
import java.lang.IllegalArgumentException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.Map;
import java.util.Map.Entry;
import jdk.jfr.Timestamp;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class PatchestryDecompileFunctionsTest extends BaseTest {

    private PatchestryDecompileFunctions decompileScript = new PatchestryDecompileFunctions();
    private Path tempArgOutputFile = null;
    private String functionToDecompile = "spi_send_lsb_first";

    @BeforeAll
    public void setUp() throws Exception {
        super.setUp();

        // if we don't fake this, because we aren't in GhidraScript context here, testing
        // decompiler-reliant stuff doesn't work
        GhidraState fakeState = new GhidraState(env.getTool(), project, program, null, null, null);
        Field state = decompileScript.getClass().getSuperclass().getDeclaredField("state");
        state.setAccessible(true);
        state.set(decompileScript, fakeState);

        Field monitorField = findFieldInHierarchy(decompileScript.getClass(), "monitor");
        monitorField.setAccessible(true);
        monitorField.set(decompileScript, fakeMonitor);

        tempArgOutputFile = Files.createTempFile("test-decompile-single-fn", ".json");
    }

    private void setUpSingle() throws Exception {
        String[] argsBloodlight = {
            "single",
            functionToDecompile,
            tempArgOutputFile.toString()
        };
        decompileScript.setScriptArgs(argsBloodlight);
        decompileScript.setProgram(program);
        assertNotNull(decompileScript.getCurrentProgram());
        assertEquals(decompileScript.getScriptArgs(), argsBloodlight);
        assertFalse(decompileScript.getCurrentProgram().isClosed());
    }

    @Test
    public void testGetLanguageID() throws Exception {
        setUpSingle();
        assertNotNull(decompileScript.getCurrentProgram().getLanguage());
        assertEquals(decompileScript.getLanguageID().toString(), testFwArch);
    }

    @Test
    public void testGetDecompilerInterface() throws Exception {
        setUpSingle();
        assertNotNull(decompileScript.getDecompilerInterface());
    }

    @Test
    public void testSerializeToFile() throws Exception {
        setUpSingle();
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
        setUpSingle();
        List<Function> functions = decompileScript.getAllFunctions();
        // todo (kaoudis) check this number somehow - is it EVERYTHING in bloodlight?
        assertEquals(functions.size(), 262);
    }

    @Test
    public void testDecompileSingleFunction() throws Exception {
        setUpSingle();
        decompileScript.decompileSingleFunction();
        String fileContents = Files.readString(tempArgOutputFile);
        assertNotNull(fileContents);

        JsonObject topLevel = JsonParser.parseString(fileContents).getAsJsonObject();
        assertNotNull(topLevel);

        assertTrue(topLevel.has("architecture"));
        assertEquals(topLevel.get("architecture").getAsString(), "ARM");

        assertTrue(topLevel.has("id"));
        assertEquals(topLevel.get("id").getAsString(), testFwArch);

        assertTrue(topLevel.has("format"));
        assertEquals(topLevel.get("format").getAsString(), "Executable and Linking Format (ELF)");

        assertTrue(topLevel.has("functions"));
        // We can't "just" decomp one fn if it calls others, so this
        // set will TYPICALLY have more than one thing in it.
        // `spi_send_lsb_first` is pretty simple though and we're just using
        // it here for testing we get something back from decomp at the top level. 
        // We'll do more format checking in the actual pcode serialization test.
        Set<Entry<String, JsonElement>> functions = topLevel.get("functions").getAsJsonObject().entrySet();
        assertFalse(functions.isEmpty());

        JsonObject fn = functions.iterator().next().getValue().getAsJsonObject();
        assertTrue(fn.has("name"));
        assertEquals(fn.get("name").getAsString(), functionToDecompile);
        assertTrue(fn.has("is_intrinsic"));
        assertEquals(fn.get("is_intrinsic").getAsBoolean(), false);
        // further testing of this format and output occurs in test/scripts/ghidra/util/PcodeSerializerTest.java
    }

    @Test
    public void testDecompileAllFunctions() throws Exception {
        Path tempAllOutputFile = Files.createTempFile("test-decompile-all-fns", ".json");
        String[] argsAll = {
            "all",
            tempAllOutputFile.toString()
        };
        decompileScript.setScriptArgs(argsAll);
        assertEquals(decompileScript.getScriptArgs(), argsAll);

        assertNotNull(decompileScript.getCurrentProgram());
        decompileScript.decompileAllFunctions();

        String fileContents = Files.readString(tempAllOutputFile);
        assertNotNull(fileContents);

        JsonObject topLevel = JsonParser.parseString(fileContents).getAsJsonObject();
        assertNotNull(topLevel);

        assertTrue(topLevel.has("architecture"));
        assertEquals(topLevel.get("architecture").getAsString(), "ARM");

        assertTrue(topLevel.has("id"));
        assertEquals(topLevel.get("id").getAsString(), testFwArch);

        assertTrue(topLevel.has("format"));
        assertEquals(topLevel.get("format").getAsString(), "Executable and Linking Format (ELF)");

        assertTrue(topLevel.has("functions"));
        // further testing of this format and output occurs in test/scripts/ghidra/util/PcodeSerializerTest.java
    }
}
