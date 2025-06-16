/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */
package scripts.ghidra.util;

import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.*;

import com.google.gson.stream.JsonWriter;

import ghidra.app.decompiler.DecompInterface;

import ghidra.util.task.TaskMonitor;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressFactory;
import ghidra.program.model.address.AddressSpace;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.FunctionManager;

import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import jdk.jfr.Timestamp;

import scripts.ghidra.BaseTest;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class PcodeSerializerTest extends BaseTest {
    JsonWriter fakeWriter = null;
    DecompInterface decompInterface = null;
    List<Function> fns = new ArrayList<Function>();
    PcodeSerializer serializer = null;

    @BeforeAll
    public void setUp() throws Exception {
        super.setUp();

        fakeWriter = mock(JsonWriter.class);
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

        serializer = new PcodeSerializer(
                fakeWriter,
                fns,
                "Cortex",
                fakeMonitor,
                program,
                decompInterface
        );

        assertNotNull(serializer.currentProgram);
        assertFalse(serializer.currentProgram.isClosed());
    }

    @Test
    public void testLabelFunction() throws Exception {
        for (Function func : fns) {
            if (!func.isThunk()) {
                assertEquals(serializer.label(func), serializer.label(func.getEntryPoint())); 
            } else {
                assertEquals(serializer.label(func), serializer.label(func.getThunkedFunction(true)));
            }
        }
    }

    @Test
    public void testLabelAddress() throws Exception {
        AddressFactory addressFactory = program.getAddressFactory();
        String ramSpaceLabel = serializer.label(addressFactory.getAddressSpace("ram").getMaxAddress());
        assertTrue(ramSpaceLabel.startsWith("ram"));
    }

    @Disabled
    @Test
    public void testLabelSequenceNumber() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test
    public void testLabelPcodeBlock() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test
    public void testLabelPcodeOp() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test
    public void testLabelDataType() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test
    public void testIntrinsicReturnType() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test
    public void testGetAddressFromPcodeOp() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test
    public void testIntrinsicLabelFromPcodeOp() throws Exception {
        assertTrue(false);
    }
    
    @Disabled
    @Test
    public void testIntrinsicLabelFromReturnDataType() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test
    public void testSerializePointerType() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test
    public void testSerializeTypedefType() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test
    public void testSerializeArrayType() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testSerializeBuiltInType() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testSerializeCompositeType() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testSerializeEnumType() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test
    public void testSerializeDataType() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testSerializeTypes() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testSerializePrototypeInt() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testSerializePrototypeFromFunctionSignature() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testSerializeHighVariable() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testRValueOfVarnode() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testIsOriginalRepresentative() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testResolvePcodeOp() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testIsOriginalRepresentativeHighVariable() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testAddressOfGlobal() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void makeGlobalFromData() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testClassifyVariable() throws Exception {
        assertTrue(false);
    }

    @Disabled
    @Test 
    public void testIsCharPointer() throws Exception {
        assertTrue(false);
    }
}
