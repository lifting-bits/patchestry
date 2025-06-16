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
import ghidra.app.decompiler.DecompileResults;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressFactory;
import ghidra.program.model.address.AddressSpace;

import ghidra.program.model.data.DataType;
import ghidra.program.model.data.DataTypeManager;
import ghidra.program.model.data.Undefined;
import ghidra.program.model.data.VoidDataType;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Parameter;
import ghidra.program.model.listing.Variable;

import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.HighVariable;
import ghidra.program.model.pcode.PcodeBlock;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.PcodeOpAST;
import ghidra.program.model.pcode.SequenceNumber;
import ghidra.program.model.pcode.Varnode;

import ghidra.util.UniversalID;
import ghidra.util.task.TaskMonitor;

import java.lang.reflect.Field;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

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
        assertFalse(fns.isEmpty());

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

    @Test
    public void testLabelSequenceNumber() throws Exception {
        DecompileResults result = decompInterface.decompileFunction(fns.get(0), 0, fakeMonitor);
        HighFunction high = result.getHighFunction();
        Iterator<PcodeOpAST> iterator = high.getPcodeOps();
        assertTrue(iterator.hasNext());
        while (iterator.hasNext()) {
            PcodeOp op = iterator.next();
            SequenceNumber sequenceNumber = op.getSeqnum();
            Address target = sequenceNumber.getTarget();
            assertTrue(serializer.label(sequenceNumber).startsWith(serializer.label(target)));
        }
    }

    @Test
    public void testLabelPcodeBlock() throws Exception {
        DecompileResults result = decompInterface.decompileFunction(fns.get(0), 1, fakeMonitor);
        HighFunction high = result.getHighFunction();
        Iterator<PcodeOpAST> iterator = high.getPcodeOps();
        
        assertTrue(iterator.hasNext());
        while (iterator.hasNext()) {
            PcodeOp op = iterator.next();
            PcodeBlock block = op.getParent();
            String label = serializer.label(block);
            assertTrue(label.startsWith(serializer.label(block.getStart())));
            assertTrue(label.endsWith(PcodeBlock.typeToName(block.getType())));
        }
    }

    @Test
    public void testLabelDataType() throws Exception {
         for (Function func : fns) {
            DataType returnType = func.getReturnType();
            String typeLabelInHex = serializer.label(returnType);

            UniversalID universalId = returnType.getUniversalID();
            assertNotNull(universalId);
            assertTrue(typeLabelInHex.endsWith(universalId.toString()));

            for (Parameter parameter : func.getParameters()) {
                DataType parameterType = parameter.getDataType();
                String paramTypeLabelInHex = serializer.label(parameterType);

                UniversalID paramUid = parameterType.getUniversalID();
                assertNotNull(paramUid);
                assertTrue(paramTypeLabelInHex.endsWith(paramUid.toString()));
            }

            for (Variable localVar : func.getAllVariables()) {
                DataType localType = localVar.getDataType();
                String variableTypeLabelInHex = serializer.label(localType);

                UniversalID localUid = localType.getUniversalID();
                assertNotNull(localUid);
                assertTrue(variableTypeLabelInHex.endsWith(localUid.toString()));
            }
        }
    }

    @Test 
    public void testLabelDataTypeNull() throws Exception {                
        DataType aNullDataType = null;
        String aNullLabel = serializer.label(aNullDataType);
        assertNotNull(aNullLabel);
        assertTrue(aNullLabel.length() > 0);
    }

    @Test
    public void testIntrinsicReturnType() throws Exception {
        DecompileResults result = decompInterface.decompileFunction(fns.get(0), 1, fakeMonitor);
        HighFunction high = result.getHighFunction();
        Iterator<PcodeOpAST> iterator = high.getPcodeOps();
        
        assertTrue(iterator.hasNext());
        while (iterator.hasNext()) {
            PcodeOp op = iterator.next();
            DataType calculatedReturnType = serializer.intrinsicReturnType(op);

            Varnode returnType = op.getOutput();
            if (returnType == null) {
                assertEquals(calculatedReturnType, VoidDataType.dataType);
            } else {
                HighVariable highVariable = returnType.getHigh();
                assertEquals(calculatedReturnType, highVariable.getDataType());
            }
        }
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
