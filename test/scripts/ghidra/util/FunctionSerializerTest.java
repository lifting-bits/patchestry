/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */
package util;

import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.*;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonWriter;

import ghidra.test.AbstractGhidraHeadlessIntegrationTest;
import ghidra.test.TestEnv;

import ghidra.app.decompiler.DecompInterface;

import ghidra.app.util.importer.AutoImporter;
import ghidra.app.util.importer.MessageLog;
import ghidra.app.util.opinion.LoadResults;

import ghidra.framework.model.Project;

import ghidra.program.model.lang.CompilerSpec;
import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.LanguageID;
import ghidra.program.model.lang.LanguageService;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Program;

import ghidra.program.util.DefaultLanguageService;

import ghidra.util.task.TaskMonitor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.StringWriter;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

// The FunctionSerializer is a utility class of the PatchestryListFunctions script.
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class FunctionSerializerTest extends AbstractGhidraHeadlessIntegrationTest {
    FunctionSerializer functionSerializer;
    TestEnv env;
    Project project;
    Program program;

    TaskMonitor fakeMonitor = mock(TaskMonitor.class);
    String testFwArch = "ARM:LE:32:Cortex";

    StringWriter stringWriter;
    BufferedWriter writer;
    DecompInterface decompInterface;
    List<Function> fns = new ArrayList<Function>();

    private Program load(File fw, String arch, Project project) throws Exception {
        LanguageService languageService = DefaultLanguageService.getLanguageService();
        LanguageID langId = new LanguageID(arch);
        Language language = languageService.getLanguage(langId);
        CompilerSpec compilerSpec = language.getDefaultCompilerSpec();
        MessageLog log = new MessageLog();

        // lcs == language and compiler spec
        LoadResults<Program> importResults = AutoImporter.importByLookingForLcs(
                fw,
                project,
                project.getProjectData().getRootFolder().getPathname(),
                language,
                compilerSpec,
                this,
                log,
                fakeMonitor
        );
        return (Program) importResults.getPrimaryDomainObject();
    }

    @BeforeAll
    public void setUp() throws Exception {
        env = new TestEnv();
        project = env.getProject();

        File bloodlightFirmware = new File(System.getenv("BLOODLIGHT_FW_PATH"));
        program = load(bloodlightFirmware, testFwArch, project);

        decompInterface = new DecompInterface();
        decompInterface.toggleCCode(false);
        decompInterface.toggleSyntaxTree(true);
        decompInterface.toggleJumpLoads(true);
        decompInterface.toggleParamMeasures(false);
        decompInterface.setSimplificationStyle("decompile");

        try {
            decompInterface.openProgram(program);
        } catch (Exception e) {
            fail("Unable to open program in DecompInterface");
        }

        FunctionIterator functionIterator = program.getFunctionManager().getFunctions(true);
        while (functionIterator.hasNext()) {
            fns.add(functionIterator.next());
        }
        assertFalse(fns.isEmpty());
    }

    @AfterAll
    public void cleanUp() throws Exception {
        if (program != null) {
            program.release(this);
        }
        if (env != null) {
            env.dispose();
        }

        if (!stringWriter.getBuffer().isEmpty()) {
            writer.close();
            stringWriter.close();
        }
    }

    @Test 
    public void testSerialize() throws Exception {
        stringWriter = new StringWriter();
        assertNotNull(stringWriter);

        writer = new BufferedWriter(stringWriter);
        assertNotNull(writer);
        
        functionSerializer = new FunctionSerializer(writer, program);
        assertNotNull(functionSerializer.currentProgram);
        assertFalse(functionSerializer.currentProgram.isClosed());

        JsonWriter jsonWriter = functionSerializer.serialize(fns);
        assertNotNull(jsonWriter);

        assertDoesNotThrow(() -> { stringWriter.toString(); });

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();

        assertEquals(jsonOutput.get("program").getAsString(), program.getName());
        Iterator<JsonElement> functions = jsonOutput.get("functions").getAsJsonArray().iterator();
        assertNotNull(functions);
        assertTrue(functions.hasNext());

        while (functions.hasNext()) {
            JsonObject element = functions.next().getAsJsonObject();
            assertTrue(element.has("name"));
            assertTrue(element.has("address"));
            assertTrue(element.has("is_thunk"));
        }
        assertFalse(functions.hasNext());
    }
}