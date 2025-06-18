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
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressFactory;
import ghidra.program.model.address.AddressSpace;

import ghidra.program.model.data.*;

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

import java.io.StringWriter;

import java.lang.reflect.Field;
import java.lang.Math;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import scripts.ghidra.BaseTest;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class PcodeSerializerTest extends BaseTest {
    StringWriter stringWriter = null;
    JsonWriter writer = null;
    DecompInterface decompInterface = null;
    List<Function> fns = new ArrayList<Function>();
    PcodeSerializer serializer = null;
    DataTypeManager dataTypeManager = null;

    @BeforeAll
    public void setUp() throws Exception {
        super.setUp();
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

        dataTypeManager = program.getDataTypeManager();
    }

    @BeforeEach 
    public void startTest() throws Exception {
        stringWriter = new StringWriter();
        assertNotNull(stringWriter);

        writer = new JsonWriter(stringWriter);
        assertNotNull(writer);

        serializer = new PcodeSerializer(
                writer,
                fns,
                "Cortex",
                fakeMonitor,
                program,
                decompInterface
        );
        assertNotNull(serializer.currentProgram);
        assertFalse(serializer.currentProgram.isClosed());
    }

    @AfterEach
    public void endTest() throws Exception {
        if (!stringWriter.getBuffer().isEmpty()) {
            writer.close();
            stringWriter.close();
        }
    }

    private int rand(int min, int max) {
        int range = max - min + 1;
        return (int)(Math.random() * range) + min;
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
        // get the HighFunction of one of the first fns in the test ELF
        // this still lacks the variety we might see in a real situation, but is more robust 
        // than just using a single function decomp for all testing
        // todo (kaoudis) save this result to the class and use it across tests
        DecompileResults result = decompInterface.decompileFunction(fns.get(rand(0, 10)), 0, fakeMonitor);
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
        DecompileResults result = decompInterface.decompileFunction(fns.get(rand(0, 10)), 0, fakeMonitor);
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
        DecompileResults result = decompInterface.decompileFunction(fns.get(rand(0, 10)), 0, fakeMonitor);
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

    @Test
    public void testIntrinsicLabelFromPcodeOp() throws Exception {
        DecompileResults result = decompInterface.decompileFunction(fns.get(rand(0, 10)), 0, fakeMonitor);
        HighFunction high = result.getHighFunction();
        Iterator<PcodeOpAST> iterator = high.getPcodeOps();
        
        assertTrue(iterator.hasNext());
        while (iterator.hasNext()) {
            PcodeOp op = iterator.next();

            if (op.getOpcode() != PcodeOp.CALLOTHER) {
                Throwable exception = assertThrows(UnsupportedOperationException.class, () -> {
                    serializer.intrinsicLabel(op);
                });

                assertTrue(exception.getMessage().startsWith("Can only label a CALLOTHER PcodeOp"));
            } else {
                String label = serializer.intrinsicLabel(op);
                Varnode returnType = op.getOutput();
                if (returnType == null) {
                    assertTrue(label.endsWith(serializer.label(VoidDataType.dataType)));
                } else {
                    assertTrue(label.endsWith(serializer.label(returnType.getHigh().getDataType())));
                }
            }
        }
    }

    @Test 
    public void testSerializeNull() throws Exception {
        assertTrue(writer.getSerializeNulls());
        DataType nullType = null;
        String nullName = serializer.label(nullType);
        writer.beginObject();
        writer.name(nullName);
        serializer.serializeType(nullType);
        writer.endObject();
        
        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();
        assertEquals(jsonOutput.get(nullName).toString(), "null");
    }

    @Test
    public void testSerializePointerType() throws Exception {
        DecompileResults result = decompInterface.decompileFunction(fns.get(rand(0, 10)), 0, fakeMonitor);
        Function function = result.getFunction();
        Pointer pointer = new PointerDataType(function.getReturnType());

        writer.beginObject();
        serializer.serializeType(pointer);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();
        String kind = jsonOutput.get("kind").getAsString();
        assertEquals(kind, "pointer");
        int size = jsonOutput.get("size").getAsInt();
        assertEquals(size, pointer.getLength());
        String elementType = jsonOutput.get("element_type").getAsString();
        assertEquals(elementType, serializer.label(pointer.getDataType()).toString());
    }

    @Test
    public void testSerializeTypedefTypePtr() throws Exception {
        // A typedef is a custom data type, and pulseox doesn't seem to have any, so let's make some.
        Pointer ptrType = new PointerDataType(IntegerDataType.dataType);
        String intPtrT = "int_ptr_t";
        TypeDef ptrTypeDef = new TypedefDataType(intPtrT, ptrType);

        writer.beginObject();
        serializer.serializeType(ptrTypeDef);
        writer.endObject();

        JsonObject jsonOutputPtr = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();
        String name = jsonOutputPtr.get("name").getAsString();
        assertEquals(name, intPtrT);
        String kind = jsonOutputPtr.get("kind").getAsString();
        assertEquals(kind, "typedef");
        int size = jsonOutputPtr.get("size").getAsInt();
        assertEquals(size, ptrTypeDef.getLength());
        String baseType = jsonOutputPtr.get("base_type").getAsString();
        assertEquals(baseType, serializer.label(ptrTypeDef.getBaseDataType()).toString());
    }

    @Disabled
    @Test
    public void testSerializeTypedefTypeStruct() throws Exception {
        // A typedef is a custom data type, and pulseox doesn't seem to have any, so let's make some.
        Structure structType = new StructureDataType("fancy_struct", 8);
        structType.add(IntegerDataType.dataType, "field1", null);
        String fancyStructT = "fancy_struct_t";
        TypeDef structTypeDef = new TypedefDataType(fancyStructT, structType);

        writer.beginObject();
        serializer.serializeType(structTypeDef);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();
        String name = jsonOutput.get("name").getAsString();
        assertEquals(name, fancyStructT);
        String kind = jsonOutput.get("kind").getAsString();
        assertEquals(kind, "typedef");
        int size = jsonOutput.get("size").getAsInt();
        assertEquals(size, structTypeDef.getLength());
        String baseType = jsonOutput.get("base_type").getAsString();
        assertEquals(baseType, structTypeDef.getBaseDataType().toString());
    }

    @Test
    public void testSerializeArrayType() throws Exception {
        DataType elementType = new IntegerDataType();
        int elements = 10;
        if (elementType == null) {
            fail("Unable to get element data type with which to test array data type serialization");
        }
        
        Array arrayType = new ArrayDataType(elementType, elements, elementType.getLength());

        writer.beginObject();
        serializer.serializeType(arrayType);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();
        String kind = jsonOutput.get("kind").getAsString();
        assertEquals(kind, "array");
        int size = jsonOutput.get("size").getAsInt();
        assertEquals(size, arrayType.getLength());
        int numElements = jsonOutput.get("num_elements").getAsInt();
        assertEquals(numElements, elements);
        String createdElementType = jsonOutput.get("element_type").getAsString();
        assertEquals(createdElementType, serializer.label(elementType));
    }

    @Test 
    public void testSerializeStructureComposite() throws Exception {
        StructureDataType structureType = new StructureDataType("ExtremelyFancyStruct", 0);

        DataType integer = new IntegerDataType();
        DataTypeComponent intComponent = structureType.add(integer, integer.getLength(), "foo", "");

        DataType string = new StringDataType();
        String stringValue = "bar";
        DataTypeComponent stringComponent = structureType.add(string, stringValue.getBytes().length, "bar", "");

        DataType lebool = new BooleanDataType();
        DataTypeComponent boolComponent = structureType.add(lebool, lebool.getLength(), "baz", "");

        writer.beginObject();
        serializer.serializeType(structureType);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();
        String name = jsonOutput.get("name").getAsString();
        assertEquals(name, "ExtremelyFancyStruct");
        String kind = jsonOutput.get("kind").getAsString();
        assertEquals(kind, "struct");
        int size = jsonOutput.get("size").getAsInt();
        assertEquals(size, structureType.getLength());
        
        Iterator<JsonElement> fields = jsonOutput.get("fields").getAsJsonArray().iterator();
        assertTrue(fields.hasNext());
        if (fields.hasNext()) {
            JsonObject field = fields.next().getAsJsonObject();

            String fieldName = field.get("name").getAsString();
            assertEquals(fieldName, "foo");

            int offset = field.get("offset").getAsInt();
            assertEquals(offset, intComponent.getOffset()); 

            String type = field.get("type").getAsString();
            assertEquals(type, serializer.label(integer).toString());
        }

        assertTrue(fields.hasNext());
        if (fields.hasNext()) {
            JsonObject field = fields.next().getAsJsonObject();

            String fieldName = field.get("name").getAsString();
            assertEquals(fieldName, stringValue);
            
            int offset = field.get("offset").getAsInt();
            assertEquals(offset, stringComponent.getOffset());

            String type = field.get("type").getAsString();
            assertEquals(type, serializer.label(string).toString());
        }

        assertTrue(fields.hasNext());
        if (fields.hasNext()) {
            JsonObject field = fields.next().getAsJsonObject();

            String fieldName = field.get("name").getAsString();
            assertEquals(fieldName, "baz");

            int offset = field.get("offset").getAsInt();
            assertEquals(offset, boolComponent.getOffset());

            String type = field.get("type").getAsString();
            assertEquals(type, serializer.label(lebool).toString());
        }

        assertFalse(fields.hasNext());
    }

    @Disabled 
    @Test 
    public void testSerializeUnionComposite() throws Exception {

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
