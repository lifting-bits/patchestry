/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.*;

import ghidra.program.model.listing.Function;

import java.lang.reflect.Field;

import java.nio.file.Files;
import java.nio.file.Path;

import java.util.Collections;
import java.util.List;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class PatchestryListFunctionsTest extends BaseTest {
    private PatchestryListFunctions listingScript = new PatchestryListFunctions();
    private Path outputFile;

    @BeforeAll
    public void setUp() throws Exception {
        super.setUp();

        Field monitorField = findFieldInHierarchy(listingScript.getClass(), "monitor");
        monitorField.setAccessible(true);
        monitorField.set(listingScript, fakeMonitor);
    }

    @BeforeEach 
    protected void finalizeSetUp() throws Exception {
        outputFile = Files.createTempFile("test-list-and-serialize", ".json");
        String[] args = {
            outputFile.toString()
        };
        listingScript.setScriptArgs(args);
        listingScript.setProgram(program);
        assertEquals(listingScript.getScriptArgs(), args);
        assertNotNull(listingScript.getCurrentProgram());
        assertFalse(listingScript.getCurrentProgram().isClosed());
    }

    @Test 
    public void testGetAllFunctions() throws Exception {
        List<Function> functions = listingScript.getAllFunctions();
        assertNotNull(functions);
        assertFalse(functions.isEmpty());

        // should match with testGetAllFunctions in the PatchestryDecompileFunctionsTest.
        assertEquals(functions.size(), 262);
    }

    @Test 
    public void testSerializeToFile() throws Exception {
        List<Function> functions = listingScript.getAllFunctions();
        assertEquals(functions.size(), 262);

        assertThrows(IllegalArgumentException.class, () -> {
            listingScript.serializeToFile((Path) null, functions);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            listingScript.serializeToFile(outputFile, (List<Function>) null);
        });

        assertThrows(IllegalArgumentException.class, () -> {
            listingScript.serializeToFile(outputFile, Collections.emptyList());
        });

        listingScript.serializeToFile(outputFile, functions);
    }
}
