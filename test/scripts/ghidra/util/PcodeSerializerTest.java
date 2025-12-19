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

import com.google.gson.stream.JsonWriter;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;

import ghidra.app.util.importer.AutoImporter;
import ghidra.app.util.importer.MessageLog;
import ghidra.app.util.opinion.LoadResults;

import ghidra.docking.settings.Settings;
import ghidra.docking.settings.SettingsImpl;

import ghidra.framework.model.Project;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressFactory;

import ghidra.program.model.data.*;
import ghidra.program.model.listing.*;

import ghidra.program.model.mem.MemBuffer;
import ghidra.program.model.mem.MemoryBufferImpl;

import ghidra.program.model.pcode.*;

import ghidra.program.model.lang.CompilerSpec;
import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.LanguageID;
import ghidra.program.model.lang.LanguageService;

import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolIterator;

import ghidra.program.util.DefaultLanguageService;

import ghidra.util.UniversalID;
import ghidra.util.task.TaskMonitor;

import ghidra.test.AbstractGhidraHeadlessIntegrationTest;
import ghidra.test.TestEnv;

import java.io.File;
import java.io.StringWriter;

import java.lang.NullPointerException;
import java.lang.reflect.Field;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import domain.*;
// todo why can't we use BaseTest here??

// The PcodeSerializer is a utility class of the PatchestryDecompileFunctions script.
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class PcodeSerializerTest extends AbstractGhidraHeadlessIntegrationTest {
    TestEnv env;
    Project project;
    Program program;

    TaskMonitor fakeMonitor = mock(TaskMonitor.class);
    String testFwArch = "ARM:LE:32:Cortex";
    
    StringWriter stringWriter;
    JsonWriter writer;
    DecompInterface decompInterface;
    List<Function> fns = new ArrayList<Function>();
    PcodeSerializer serializer;
    DataTypeManager dataTypeManager;

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

    protected Field findFieldInHierarchy(Class<?> clazz, String fieldName) throws NoSuchFieldException {
        Class<?> current = clazz;
        while (current != null) {
            try {
                return current.getDeclaredField(fieldName);
            } catch (NoSuchFieldException e) {
                current = current.getSuperclass();
            }
        }
        throw new NoSuchFieldException("Field '" + fieldName + "' not found in class hierarchy");
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

    @AfterAll
    public void cleanUp() throws Exception {
        if (program != null) {
            program.release(this);
        }
        if (env != null) {
            env.dispose();
        }
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
    public void testGetAddress() throws Exception {
        // just using a contrived example to prove it does what it says on the tin
        AddressFactory addressFactory = program.getAddressFactory();
        Address address = addressFactory.getAddressSpace("ram").getAddress("0x100400");
        Varnode inputVarnode = new Varnode(address, 8);
        Varnode[] inputs = { inputVarnode };
        PcodeOp addOp = new PcodeOp(address, /*seqNum*/ 0, /*opcode*/ PcodeOp.INT_ADD, inputs);
        
        assertEquals(serializer.getAddress(addOp), address);
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
        assertEquals(baseType, serializer.label(structTypeDef.getBaseDataType()));
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
        JsonObject integerField = fields.next().getAsJsonObject();

        String integerFieldName = integerField.get("name").getAsString();
        assertEquals(integerFieldName, "foo");

        int integerOffset = integerField.get("offset").getAsInt();
        assertEquals(integerOffset, intComponent.getOffset()); 

        String integerType = integerField.get("type").getAsString();
        assertEquals(integerType, serializer.label(integer).toString());

        assertTrue(fields.hasNext());
        JsonObject strField = fields.next().getAsJsonObject();

        String strFieldName = strField.get("name").getAsString();
        assertEquals(strFieldName, stringValue);
        
        int strOffset = strField.get("offset").getAsInt();
        assertEquals(strOffset, stringComponent.getOffset());

        String strType = strField.get("type").getAsString();
        assertEquals(strType, serializer.label(string).toString());

        // the boolean
        assertTrue(fields.hasNext());
        JsonObject boolField = fields.next().getAsJsonObject();

        String boolFieldName = boolField.get("name").getAsString();
        assertEquals(boolFieldName, "baz");

        int boolOffset = boolField.get("offset").getAsInt();
        assertEquals(boolOffset, boolComponent.getOffset());

        String boolType = boolField.get("type").getAsString();
        assertEquals(boolType, serializer.label(lebool).toString());

        assertFalse(fields.hasNext());
    }

    @Test 
    public void testSerializeUnionComposite() throws Exception {
        UnionDataType unionDt = new UnionDataType("FancyUnion");
        
        DataType ptr = new Pointer64DataType(IntegerDataType.dataType);
        DataTypeComponent ptrComponent = unionDt.add(ptr, "intPtr", null);        

        DataType uni = new UnionDataType("Oignon");
        DataTypeComponent unionComponent = unionDt.add(uni, "legume", null);

        DataType dword = new DWordDataType();
        DataTypeComponent dwordComponent = unionDt.add(dword, "de woord", null);

        writer.beginObject();
        serializer.serializeType(unionDt);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();

        assertEquals(jsonOutput.get("name").getAsString(), "FancyUnion");
        assertEquals(jsonOutput.get("kind").getAsString(), "union");
        assertEquals(jsonOutput.get("size").getAsInt(), unionDt.getLength());

        Iterator<JsonElement> fields = jsonOutput.get("fields").getAsJsonArray().iterator();
        
        assertTrue(fields.hasNext());
        JsonObject ptrField = fields.next().getAsJsonObject();

        String ptrFieldName = ptrField.get("name").getAsString();
        assertEquals(ptrFieldName, "intPtr");

        int ptrOffset = ptrField.get("offset").getAsInt();
        assertEquals(ptrComponent.getOffset(), ptrOffset);

        String ptrType = ptrField.get("type").getAsString();
        assertEquals(ptrType, serializer.label(ptr).toString());

        assertTrue(fields.hasNext());
        JsonObject nestedUnionField = fields.next().getAsJsonObject();

        String unionFieldName = nestedUnionField.get("name").getAsString();
        assertEquals(unionFieldName, "legume");

        int unionOffset = nestedUnionField.get("offset").getAsInt();
        assertEquals(unionOffset, unionComponent.getOffset());

        String unionType = nestedUnionField.get("type").getAsString();
        assertEquals(unionType, serializer.label(uni).toString());

        assertTrue(fields.hasNext());
        JsonObject dwordField = fields.next().getAsJsonObject();

        String dwordFieldName = dwordField.get("name").getAsString();
        assertEquals(dwordFieldName, "de woord");

        int dwordOffset = dwordField.get("offset").getAsInt();
        assertEquals(dwordComponent.getOffset(), dwordOffset);

        String dwordType = dwordField.get("type").getAsString();
        assertEquals(dwordType, serializer.label(dword).toString());

        assertFalse(fields.hasNext());
    }

    @Test 
    public void testSerializeIntegerType() throws Exception {
        DataType integer = new IntegerDataType();

        writer.beginObject();
        serializer.serializeType(integer);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();

        assertEquals(jsonOutput.get("name").getAsString(), "int");
        assertEquals(jsonOutput.get("kind").getAsString(), "integer");
        assertEquals(jsonOutput.get("size").getAsInt(), integer.getLength());
    }

    @Test 
    public void testSerializeFloatType() throws Exception {
        DataType floaty = new FloatDataType();

        writer.beginObject();
        serializer.serializeType(floaty);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();

        assertEquals(jsonOutput.get("name").getAsString(), "float");
        assertEquals(jsonOutput.get("kind").getAsString(), "float");
        assertEquals(jsonOutput.get("size").getAsInt(), floaty.getLength());
    }

    @Test
    public void testSerializeBooleanType() throws Exception {
        DataType booly = new BooleanDataType();

        writer.beginObject();
        serializer.serializeType(booly);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();

        assertEquals(jsonOutput.get("name").getAsString(), "bool");
        assertEquals(jsonOutput.get("kind").getAsString(), "boolean");
        assertEquals(jsonOutput.get("size").getAsInt(), booly.getLength());
    }

    @Test 
    public void testSerializeEnumType() throws Exception {
        CategoryPath path = new CategoryPath("/TestEnums");
        EnumDataType enume = new EnumDataType(path, "hello", 4, dataTypeManager);
        enume.add("uno", 1);
        enume.add("dos", 2);
        enume.add("tres", 3);
        enume.add("yikes", -0);

        writer.beginObject();
        serializer.serializeType(enume);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();

        assertEquals(jsonOutput.get("name").getAsString(), "hello");
        assertEquals(jsonOutput.get("kind").getAsString(), "enum");
        assertEquals(jsonOutput.get("size").getAsInt(), enume.getLength());
    }

    @Test 
    public void testSerializeVoidType() throws Exception {
        DataType voidy = new VoidDataType();

        writer.beginObject();
        serializer.serializeType(voidy);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();

        assertEquals(jsonOutput.get("name").getAsString(), "void");
        assertEquals(jsonOutput.get("kind").getAsString(), "void");
        assertEquals(jsonOutput.get("size").getAsInt(), voidy.getLength());
    }

    @Test 
    public void testSerializeUndefined() throws Exception {
        // the argument is size - but we should be able to serialize any if we
        // can serialize at least one undefined DT
        DataType undef1 = Undefined.getUndefinedDataType(1);

        writer.beginObject();
        serializer.serializeType(undef1);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();
        
        assertEquals(jsonOutput.get("name").getAsString(), "undefined1");
        assertEquals(jsonOutput.get("kind").getAsString(), "undefined");
        assertEquals(jsonOutput.get("size").getAsInt(), undef1.getLength());
    }

    @Test 
    public void testSerializeDefaultType() throws Exception {
        DefaultDataType defaultType = DefaultDataType.dataType;

        writer.beginObject();
        serializer.serializeType(defaultType);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();
        
        assertEquals(jsonOutput.get("name").getAsString(), "undefined");
        assertEquals(jsonOutput.get("kind").getAsString(), "undefined");
        assertEquals(jsonOutput.get("size").getAsInt(), defaultType.getLength());
    }

    @Disabled
    @Test 
    public void testSerializeBitFieldType() throws Exception {
        // not implemented yet
    }

    @Test 
    public void testSerializeWideCharType() throws Exception {
        WideCharDataType wideChar = new WideCharDataType();
        AddressFactory addressFactory = program.getAddressFactory();
        Address address = addressFactory.getAddressSpace("ram").getMinAddress();

        Character value = 'Ā';
        MemBuffer buf = new MemoryBufferImpl(program.getMemory(), address);
        Settings settings = new SettingsImpl();
        int length = wideChar.getLength();
        // throw away the value since we are just working with TYPES
        wideChar.encodeValue(value, buf, settings, length);
        
        writer.beginObject();
        serializer.serializeType(wideChar);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();
        
        assertNull(jsonOutput.get("name"));
        assertEquals(jsonOutput.get("kind").getAsString(), "wchar");
        assertEquals(jsonOutput.get("size").getAsInt(), wideChar.getLength());
    }

    @Disabled
    @Test 
    public void testSerializeStringDataType() throws Exception {
        // not implemented yet
    }

    @Test 
    public void testNoArgSerializePrototypeNull() throws Exception {
        writer.beginObject();
        int serialized = serializer.serializePrototype();
        writer.endObject();
        assertEquals(serialized, 0);

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();

        assertEquals(jsonOutput.get("return_type").getAsString(), serializer.label((DataType) null));
        assertEquals(jsonOutput.get("is_variadic").getAsBoolean(), false);
        assertEquals(jsonOutput.get("is_noreturn").getAsBoolean(), false);
        assertEquals(jsonOutput.get("parameter_types").getAsJsonArray().isEmpty(), true);
    }

    @Test 
    public void testSerializePrototype() throws Exception {
        DecompileResults result = decompInterface.decompileFunction(fns.get(rand(0, 10)), 0, fakeMonitor);
        HighFunction high = result.getHighFunction();
        FunctionPrototype prototype = high.getFunctionPrototype();

        writer.beginObject();
        serializer.serializePrototype(prototype);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();

        assertEquals(jsonOutput.get("return_type").getAsString(), serializer.label(prototype.getReturnType()));
        assertEquals(jsonOutput.get("is_variadic").getAsBoolean(), prototype.isVarArg());
        assertEquals(jsonOutput.get("is_noreturn").getAsBoolean(), prototype.hasNoReturn());

        JsonArray parameterTypes = jsonOutput.get("parameter_types").getAsJsonArray();
        assertEquals(parameterTypes.size(), prototype.getNumParams());

        if (prototype.getParameterDefinitions() != null) {
            if (prototype.getParameterDefinitions().length > 0) {
                ArrayList<String> paramTypeStringsWritten = new ArrayList();
                for (JsonElement element : parameterTypes) {
                    paramTypeStringsWritten.add(element.getAsString());
                }

                ParameterDefinition[] paramDefs = prototype.getParameterDefinitions();
                for (ParameterDefinition pd : paramDefs) {
                    String dtLabel = serializer.label(pd.getDataType());
                    assertTrue(paramTypeStringsWritten.contains(dtLabel));
                }
            }
        }
    }

    @Test 
    public void testSerializeFunctionSignaturePrototype() throws Exception {
        DecompileResults result = decompInterface.decompileFunction(fns.get(rand(0, 10)), 0, fakeMonitor);
        HighFunction high = result.getHighFunction();
        FunctionSignature prototypeSignature = high.getFunction().getSignature();

        writer.beginObject();
        serializer.serializePrototype(prototypeSignature);
        writer.endObject();

        JsonObject jsonOutput = JsonParser
            .parseString(stringWriter.toString())
            .getAsJsonObject();

        assertEquals(jsonOutput.get("return_type").getAsString(), serializer.label(prototypeSignature.getReturnType()));
        assertEquals(jsonOutput.get("is_variadic").getAsBoolean(), prototypeSignature.hasVarArgs());
        assertEquals(jsonOutput.get("is_noreturn").getAsBoolean(), prototypeSignature.hasNoReturn());
        assertEquals(jsonOutput.get("calling_convention").getAsString(), prototypeSignature.getCallingConventionName());

        JsonArray parameterTypes = jsonOutput.get("parameter_types").getAsJsonArray();
        ParameterDefinition[] args = prototypeSignature.getArguments();
        if (args != null) {
            assertEquals(parameterTypes.size(), args.length);
            if (args.length > 0) {
                ArrayList<String> paramTypeStringsWritten = new ArrayList();
                for (JsonElement element : parameterTypes) {
                    paramTypeStringsWritten.add(element.getAsString());
                }

                for (ParameterDefinition pd : args) {
                    String dtLabel = serializer.label(pd.getDataType());
                    assertTrue(paramTypeStringsWritten.contains(dtLabel));
                }
            }
        }
    }
    
    @Test 
    public void testIsOriginalRepresentative() throws Exception {
        AddressFactory addressFactory = program.getAddressFactory();

        // just a random number really, no relation to real RAM address space.
        // todo (kaoudis) a representative address
        Address address = addressFactory.getAddressSpace("ram").getAddress("0x100400");
        Varnode inputVarnode = new Varnode(address, 8);
        Varnode[] inputs = { inputVarnode };
        PcodeOp addOp = new PcodeOp(address, /*seqNum*/ 0, /*opcode*/ PcodeOp.INT_ADD, inputs);
        assertTrue(serializer.isOriginalRepresentative(inputVarnode));

        Varnode nullOpVarnode = new Varnode(address, 4);
        assertTrue(serializer.isOriginalRepresentative(nullOpVarnode));

        Varnode notCallOtherVarnode = new Varnode(address, 16);
        PcodeOp loadOp = new PcodeOp(address, /*seqNum*/ 1, /*opcode*/ PcodeOp.LOAD, inputs, notCallOtherVarnode);
        assertTrue(serializer.isOriginalRepresentative(notCallOtherVarnode));

        Address lowAddress = addressFactory.getAddress("0x1");
        Varnode lowOffsetVarnode = new Varnode(lowAddress, 4);
        Varnode[] lowOffsetInput = { lowOffsetVarnode };
        PcodeOp addOp1 = new PcodeOp(lowAddress, /*seqNum*/ 2, /*opcode*/ PcodeOp.INT_ADD, lowOffsetInput);
        assertTrue(serializer.isOriginalRepresentative(lowOffsetVarnode));

        // todo (kaoudis) this doesn't trigger the false case! why?!
        // Address highAddress = addressFactory.getAddressSpace("ram").getAddress("0xffffff");
        // Varnode highOffsetVarnode = new Varnode(highAddress, 4);
        // Varnode[] emptyInputs = {};
        // PcodeOp callOtherOp = new PcodeOp(address, /*seqNum*/ 3, /*opcode*/ PcodeOp.CALLOTHER, emptyInputs, highOffsetVarnode);
        // Varnode output = callOtherOp.getOutput();
        // assertFalse(serializer.isOriginalRepresentative(output));
    }

    @Test 
    public void testResolvePcodeOp() throws Exception {
        assertNull(serializer.resolveOp((PcodeOp) null));

        AddressFactory addressFactory = program.getAddressFactory();
        Address address = addressFactory.getAddressSpace("ram").getAddress("0x100400");
        Varnode inputVarnode = new Varnode(address, 8);
        Varnode[] inputs = { inputVarnode };
        PcodeOp addOp = new PcodeOp(address, /*seqNum*/ 0, /*opcode*/ PcodeOp.INT_ADD, inputs);
        // no mapping set, since this is a contrived op!
        assertEquals(addOp, serializer.resolveOp(addOp));

        // todo (kaoudis) not quite sure how to test replacement since it's
        // context-dependent and moreover just "is thing in map or not"
    }

    @Test 
    public void testOriginalRepresentativeOf() throws Exception {
        assertNull(serializer.originalRepresentativeOf((HighVariable) null));
        
        DecompileResults result = decompInterface.decompileFunction(fns.get(6), 0, fakeMonitor);
        assertTrue(result.decompileCompleted());
        HighFunction high = result.getHighFunction();
        Iterator<PcodeOpAST> iterator = high.getPcodeOps();
        
        assertTrue(iterator.hasNext());
        while (iterator.hasNext()) {
            PcodeOp op = iterator.next();

            Varnode input = op.getInput(0);
            HighVariable inputHighVariable = input.getHigh();
            if (inputHighVariable != null) {
                // sort of the best we can do for now, as this has inconsistent results
                assertNotNull(serializer.originalRepresentativeOf(inputHighVariable));
            }

            Varnode returnType = op.getOutput();
            if (returnType != null) {
                HighVariable returnTypeHighVariable = returnType.getHigh();
                // sort of the best we can do for now, as this has inconsistent results
                assertNotNull(serializer.originalRepresentativeOf(returnTypeHighVariable));
            }
        }
    }

    @Test 
    public void testAddressOfGlobal() throws Exception {
        assertThrows(NullPointerException.class, 
            () -> {serializer.addressOfGlobal((HighVariable) null);});
       
        SymbolIterator symbols = program.getSymbolTable().getAllSymbols(true);
        if (!symbols.hasNext()) fail();

        for (Symbol symbol : symbols) {
            Address associatedAddress = symbol.getAddress();
            Function fn = program.getFunctionManager().getFunctionContaining(associatedAddress);
            if (fn != null) {
                DecompileResults results = decompInterface.decompileFunction(fn, 60, fakeMonitor);
                HighFunction highFunction = results.getHighFunction();
                GlobalSymbolMap globalMap = highFunction.getGlobalSymbolMap();
                Iterator<HighSymbol> highSymbols = globalMap.getSymbols();

                while (highSymbols.hasNext()) {
                    HighSymbol hs = highSymbols.next();
                    assertTrue(hs.isGlobal());

                    HighVariable highVariable = hs.getHighVariable();                    
                    Address result = serializer.addressOfGlobal(highVariable);
                    assertNotNull(result);
                }
            }
        }
    }

    @Test 
    public void testClassifyVariable() throws Exception {
        VariableClassification unknownOutput = serializer.classifyVariable((HighVariable) null);
        assertEquals(unknownOutput, VariableClassification.UNKNOWN);

        Function fn = fns.get(0);
        DecompileResults result = decompInterface.decompileFunction(fn, 60, fakeMonitor);
        assertTrue(result.decompileCompleted());

        HighFunction high = result.getHighFunction();
        LocalSymbolMap localSymbolMap = high.getLocalSymbolMap();
        Iterator<HighSymbol> symbols = localSymbolMap.getSymbols();
        if (!symbols.hasNext()) fail();

        while (symbols.hasNext()) {
            HighSymbol sym = symbols.next();
            HighVariable highVar = sym.getHighVariable();
            VariableClassification classification = serializer.classifyVariable(highVar);

            if (sym.isParameter()) {
                assertEquals(VariableClassification.PARAMETER, classification);
            } else {
                // it's a local if not a parameter
                assertEquals(VariableClassification.LOCAL, classification);
            }
        }

        Iterator<PcodeOpAST> pcodeOps = high.getPcodeOps();
        while (pcodeOps.hasNext()) {
            PcodeOp op = pcodeOps.next();
            for (Varnode input : op.getInputs()) {
                HighVariable hv = input.getHigh();
                if (hv != null) {
                    VariableClassification opClassification = serializer.classifyVariable(hv);
                    if (hv instanceof HighConstant) {
                        assertEquals(VariableClassification.CONSTANT, opClassification);
                    } else if (hv instanceof HighGlobal) {
                        assertEquals(VariableClassification.GLOBAL, opClassification);
                    } else if (hv instanceof HighTemporary) {
                        assertEquals(VariableClassification.TEMPORARY, opClassification);
                    }
                }
            }
        }
    }

    @Test 
    public void testIsCharPointer() throws Exception {
        assertThrows(NullPointerException.class, () -> {serializer.isCharPointer(null); });
        
        AddressFactory addressFactory = program.getAddressFactory();
        Address addr = addressFactory.getAddressSpace("ram").getMaxAddress();
        Varnode noHighVar = new Varnode(addr, 4);
        assertFalse(serializer.isCharPointer(noHighVar));

        DecompileResults result = decompInterface.decompileFunction(fns.get(rand(0, 10)), 0, fakeMonitor);
        HighFunction high = result.getHighFunction();
        
        Iterator<VarnodeAST> varnodes = high.getVarnodes(addressFactory.getAddressSpace("ram"));
        if (!varnodes.hasNext()) fail(); 
        while (varnodes.hasNext()) {
            Varnode node = varnodes.next();
            HighVariable hv = node.getHigh();
            
            boolean output = serializer.isCharPointer(node);
            assertNotNull(output);
            
            if (hv != null) {
                DataType highVarDataType = hv.getDataType();
                if (highVarDataType instanceof Pointer) {
                    DataType baseDataType = ((Pointer) highVarDataType).getDataType();
                    if (baseDataType instanceof CharDataType) {
                        assertTrue(output);
                    } else {
                        assertFalse(output);
                    }
                } else {
                    assertFalse(output);
                }
            } else {
                assertFalse(output);
            }
        }
    }

    /**
     * Test that char pointer constants at unmapped addresses serialize correctly.
     *
     * This test catches the bug fixed in commit e464be0: when a constant has a char pointer
     * type (e.g., value 0x3 inferred as char*) but doesn't point to valid mapped memory,
     * findNullTerminatedString() returns null. Previously, the code tried to call
     * listing.createData() at the unmapped address, causing a CodeUnitInsertionException.
     * Now it correctly treats such constants as plain constant values, outputting
     * {"kind": "constant", "value": <offset>}.
     */
    @Test
    public void testSerializeCharPointerConstantAtUnmappedAddress() throws Exception {
        // Search through decompiled functions for char pointer constants
        // at addresses that don't resolve to valid strings
        boolean foundCharPointerConstant = false;

        for (int i = 0; i < Math.min(fns.size(), 50); i++) {
            Function fn = fns.get(i);
            if (fn.isThunk()) continue;

            DecompileResults result = decompInterface.decompileFunction(fn, 60, fakeMonitor);
            if (!result.decompileCompleted()) continue;

            HighFunction high = result.getHighFunction();
            if (high == null) continue;

            // Look for constants in the function's pcode ops
            Iterator<PcodeOpAST> pcodeOps = high.getPcodeOps();
            while (pcodeOps.hasNext()) {
                PcodeOp op = pcodeOps.next();

                for (Varnode input : op.getInputs()) {
                    if (!input.isConstant()) continue;

                    HighVariable hv = input.getHigh();
                    if (hv == null) continue;

                    // Check if this is a char pointer type
                    if (!serializer.isCharPointer(input)) continue;

                    // Found a char pointer constant - verify serialization doesn't throw
                    foundCharPointerConstant = true;
                    stringWriter.getBuffer().setLength(0);

                    // This should NOT throw CodeUnitInsertionException after the fix
                    assertDoesNotThrow(() -> {
                        serializer.serializeInput(op, input);
                    }, "serializeInput should not throw for char pointer constant at unmapped address");

                    String json = stringWriter.toString();
                    JsonObject jsonOutput = JsonParser.parseString(json).getAsJsonObject();

                    // Verify it outputs either "string" (if valid string found) or "constant"
                    String kind = jsonOutput.get("kind").getAsString();
                    assertTrue(
                        kind.equals("string") || kind.equals("constant"),
                        "Char pointer constant should serialize as 'string' or 'constant', got: " + kind
                    );

                    if (kind.equals("constant")) {
                        assertTrue(
                            jsonOutput.has("value"),
                            "Constant kind should have 'value' field"
                        );
                    }
                }
            }
        }

        // If we didn't find any char pointer constants, the test is inconclusive
        // but not a failure - the firmware may not have any such cases
        if (!foundCharPointerConstant) {
            System.out.println("Warning: No char pointer constants found to test. " +
                "Test passed vacuously.");
        }
    }

    @Disabled
    @Test
    public void testSerializeInput() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testCreateStackPointerVarnodes() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testPrefixPtrSubcomponentWithAddressOf() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testCreateLocalForPtrSubcomponent() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testGetNextReferencedAddressOrMax() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testCreateGlobalForPtrSubcomponent() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testRewritePtrSubcomponent() throws Exception {
        // todo writeme
        fail();
    }  

    @Disabled 
    @Test 
    public void testNormalizeDataType() throws Exception {
        // todo writeme
        fail();
    } 

    @Disabled 
    @Test 
    public void testGetArgumentType() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testResolveCalledFunction() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testNeedsCastOperation() throws Exception {
        // todo writeme
        fail();
    }
    
    @Disabled 
    @Test 
    public void testNeedsAddressOfConversion() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testTypesAreCompatible() throws Exception {
        // todo writeme
        fail();
    } 

    @Disabled 
    @Test 
    public void testRewriteCallArgument() throws Exception {
        // todo writeme
        fail();
    } 

    @Disabled 
    @Test 
    public void testRewriteVoidReturnType() throws Exception {
        // todo writeme
        fail();
    }  

    @Disabled 
    @Test 
    public void testGetOrCreatePrefixOperations() throws Exception {
        // todo writeme
        fail();
    } 

    @Disabled 
    @Test 
    public void testTryFixupStackVarnode() throws Exception {
        // todo writeme
        fail();
    } 

    @Disabled 
    @Test 
    public void testMineForVarNodes() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testFixUpMissingLocalVariables() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testVariableOfHighVariable() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testVariableOfVarnode() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testVariableOfPcodeOp() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeOutput() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeLoadStoreAddress() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeLoadOp() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeStoreOp() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testNextUniqueAddress() throws Exception {
        // todo writeme
        fail();
    } 

    @Disabled 
    @Test 
    public void testCreateParamVarDecl() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testCreateLocalVariableDefinition() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testCreateNamedTemporaryRegisterVariableDeclaration() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testGetOrCreateLocalVariable() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeCallOp() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeBranchOp() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeCondBranchOp() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeGenericOp() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeDeclareParamVar() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeDeclareLocalVar() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeDeclareNamedTemporary() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeAddressOfOp() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeIntrinsicCallOp() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeCallOtherOp() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeReturnOp() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testMnemonic() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializePcodeOp() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testCanElideMultiEqual() throws Exception {
        // todo writeme
        fail();
    }
    
    @Disabled 
    @Test 
    public void testCanElideCopy() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testIsBranch() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializePcodeBasicBlock() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeEntryBlock() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeFunction() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeGlobals() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeIntrinsics() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerializeFunctions() throws Exception {
        // todo writeme
        fail();
    }

    @Disabled 
    @Test 
    public void testSerialize() throws Exception {
        // todo writeme
        fail();
    }
}
