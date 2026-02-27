/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util;

import ghidra.framework.options.Options;

import ghidra.app.cmd.function.CallDepthChangeInfo;

import ghidra.app.decompiler.component.DecompilerUtils;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;

import ghidra.app.plugin.processors.sleigh.SleighLanguage;

import ghidra.program.database.symbol.CodeSymbol;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressFactory;
import ghidra.program.model.address.AddressIterator;
import ghidra.program.model.address.AddressSet;
import ghidra.program.model.address.AddressSetView;
import ghidra.program.model.address.AddressSpace;
import ghidra.program.model.address.AddressOutOfBoundsException;

import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;

import ghidra.program.model.data.AbstractStringDataType;
import ghidra.program.model.data.BitFieldDataType;
import ghidra.program.model.data.DataType;
import ghidra.program.model.data.DataTypeManager;
import ghidra.program.model.data.StringDataType;
import ghidra.program.model.data.StringDataInstance;
import ghidra.program.model.data.CharDataType;

import ghidra.program.model.lang.CompilerSpec;
import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.Register;
import ghidra.program.model.lang.RegisterManager;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.FunctionSignature;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.listing.Parameter;
import ghidra.program.model.listing.Program;
import ghidra.program.model.listing.StackFrame;
import ghidra.program.model.listing.Variable;
import ghidra.program.model.listing.VariableStorage;
import ghidra.program.model.listing.Data;

import ghidra.program.model.mem.MemBuffer;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.mem.MemoryBufferImpl;

import ghidra.program.model.pcode.FunctionPrototype;
import ghidra.program.model.pcode.GlobalSymbolMap;
import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.HighCodeSymbol;
import ghidra.program.model.pcode.HighConstant;
import ghidra.program.model.pcode.HighGlobal;
import ghidra.program.model.pcode.HighLocal;
import ghidra.program.model.pcode.HighOther;
import ghidra.program.model.pcode.HighParam;
import ghidra.program.model.pcode.HighSymbol;
import ghidra.program.model.pcode.HighVariable;
import ghidra.program.model.pcode.LocalSymbolMap;
import ghidra.program.model.pcode.PartialUnion;
import ghidra.program.model.pcode.PcodeBlock;
import ghidra.program.model.pcode.PcodeBlockBasic;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.PcodeOpAST;
import ghidra.program.model.pcode.JumpTable;
import ghidra.program.model.pcode.SequenceNumber;
import ghidra.program.model.pcode.SymbolEntry;
import ghidra.program.model.pcode.Varnode;

import ghidra.program.model.data.AbstractFloatDataType;
import ghidra.program.model.data.AbstractIntegerDataType;
import ghidra.program.model.data.Array;
import ghidra.program.model.data.ArrayStringable;
import ghidra.program.model.data.BooleanDataType;
import ghidra.program.model.data.BuiltIn;
import ghidra.program.model.data.Composite;
import ghidra.program.model.data.CategoryPath;
import ghidra.program.model.data.DataType;
import ghidra.program.model.data.DataTypeComponent;
import ghidra.program.model.data.DefaultDataType;
import ghidra.program.model.data.Enum;
import ghidra.program.model.data.FunctionDefinition;
import ghidra.program.model.data.ParameterDefinition;
import ghidra.program.model.data.Pointer;
import ghidra.program.model.data.Structure;
import ghidra.program.model.data.TypeDef;
import ghidra.program.model.data.Undefined;
import ghidra.program.model.data.Union;
import ghidra.program.model.data.VoidDataType;
import ghidra.program.model.data.WideCharDataType;

import ghidra.program.model.symbol.Namespace;
import ghidra.program.model.symbol.Reference;
import ghidra.program.model.symbol.ReferenceManager;
import ghidra.program.model.symbol.SourceType;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolType;
import ghidra.program.model.symbol.SymbolTable;
import ghidra.program.model.symbol.SymbolUtilities;
import ghidra.program.model.symbol.FlowType;
import ghidra.program.model.symbol.Reference;

import ghidra.app.plugin.core.analysis.AutoAnalysisManager;

import ghidra.util.UniversalID;
import ghidra.util.task.TaskMonitor;

import com.google.gson.stream.JsonWriter;

import domain.*;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.OutputStreamWriter;
import java.io.File;

import java.nio.file.Files;
import java.nio.file.Path;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.TreeMap;

// The PcodeSerializer is a utility class of the PatchestryDecompileFunctions script.
public class PcodeSerializer {
	    public static final int MIN_CALLOTHER = 0x100000;
    	public static final int DECOMPILATION_TIMEOUT = 30;
    	public static final int DECLARE_PARAM_VAR = MIN_CALLOTHER + 0;
    	public static final int DECLARE_LOCAL_VAR = MIN_CALLOTHER + 1;
    	public static final int DECLARE_TEMP_VAR = MIN_CALLOTHER + 2;
    	public static final int ADDRESS_OF = MIN_CALLOTHER + 3;

		protected Program currentProgram;

		private String architecture;
		private String languageID;
		private AddressSpace externSpace;
		private AddressSpace ramSpace;
		private AddressSpace stackSpace;
		private AddressSpace constantSpace;
		private AddressSpace uniqueSpace;
		
		private JsonWriter writer;
		private TaskMonitor monitor;
		private ApiUtil apiUtil;
		private DecompInterface decompInterface;
		private BasicBlockModel basicBlockModel;
		
		// Tracks which functions to recover. The size of `functions` is
		// monotonically non-decreasing, with newly discovered functions
		// added to the end. The first `originalFunctionsSize` functions in
		// `functions` are meant to have their definitions (i.e. high p-code)
		// serialized to JSON.
		private List<Function> functions;
		private int originalFunctionsSize;
		private Set<String> seenFunctions;

		// The seen globals.
		private Map<Address, HighVariable> seenGlobalsMap;
		private Map<HighVariable, Address> addressOfGlobalMap;

		// The seen data
		private Map<Address, Data> seenDataMap;

		// The seen types. The size of `typesToSerialize` is monotonically
		// non-decreasing, so that as we add new things to `seenTypes`, we add
		// to the end of `typesToSerialize`. This lets us properly handle
		// tracking what recursive types need to be serialized.
		private Set<String> seenTypes;
		private List<DataType> typesToSerialize;
		
		// Current function being serialized, and current block within that
		// function being serialized.
		private HighFunction currentFunction;
		private PcodeBlockBasic currentBlock;

		// Jump table index for the current function, keyed by switch address.
		// Built once when currentFunction is set and cleared when it is reset.
		private Map<Address, JumpTable> jumpTableIndex;
		
		// We invent an entry block for each `HighFunction` to be serialized.
		// The operations within this entry block are custom `CALLOTHER`s, that
		// "declare" variables of various forms. The way to think about this is
		// with a visual analogy: when looking at a decompilation in Ghidra, the
		// first thing we see in the body of a function are the local variable
		// declarations. In our JSON output, we try to mimic this, and then
		// canonicalize accesses of things to target those variables, doing a
		// kind of de-SSAing.
		private List<PcodeOp> entryBlock;
		
		// When creating the `CALLOTHER`s for the `entryBlock`, we need to
		// synthesize addresses in the unique address space, and so we need to
		// keep track of what unique addresses we've already used/generated.
		private long nextUnique;
		private int nextSeqNum;
		
		private SleighLanguage language;
		
		// Stack pointer for this program's architecture. High p-code can have
		// two forms of stack references: `Varnode`s of whose `Address` is part
		// of the stack address space, and `Varnode`s representing registers,
		// where some of those are the stack pointer. In this latter case, we
		// need to be able to identify those and convert them into the former
		// case.
		private Register stackPointer;
		
		// Maps names of missing locals to invented `HighLocal`s used to
		// represent them. `Function`s often have many `Variable`s, not all of
		// which become `HighLocal`s or `HighParam`s. Sometimes when something
		// can't be precisely recognized, it is represented as a `HighOther`
		// connected to a `HighSymbol`. Confusingly, the `DataType` associated
		// with the `HighSymbol` is more representative of what the decompiler
		// actually shows, and the `HighOther` more representative of the
		// data type in the low `Variable` sourced from the `StackFrame`.
		private Map<String, HighLocal> missingLocalsMap;
		private Map<HighVariable, HighLocal> oldLocalsMap;
		
		// Maps `HighVariables` (really, `HighOther`s) that are attached to
		// register `Varnode`s to the `PcodeOp` containing those nodes. We
		// The same-named temporary/register may be associated with many such
		// independent `HighVariable`s, so to distinguish them to downstream
		// readers of the JSON, we want to 'version' the register variables by
		// their initial user.
		private Map<HighVariable, PcodeOp> temporaryAddressMap;
		
		// Replacement operations. Sometimes we have something that we actually
		// need to replace, and so this mapping allows us to do that without
		// having to aggressively rewrite things, especially output operands.
		private Map<PcodeOp, PcodeOp> replacementOperationsMap;
		
		// Sometimes we need to arrange for some operations to exist prior to
		// another one, e.g. if there is a `CALL foo, SP` that decompiles to
		// `foo(&local_x)`, then we really want to be able to represent `SP`,
		// the stack pointer, as a reference to the address of `local_x`, rather
		// than whatever it is.
		private Map<PcodeOp, List<PcodeOp>> prefixOperationsMap;
		
		// A mapping of `CALLOTHER` locations operating with named intrinsics to
		// the `PcodeOp`s representing those `CALLOTHER`s.
		private List<PcodeOp> callotherUsePcodeOps;

		public PcodeSerializer(
			JsonWriter writer,
			List<Function> functions,
			String languageId, 
			TaskMonitor monitor,
			Program currentProgram, 
			DecompInterface decompInterface
		) {
			this.writer = writer;

			this.functions = functions;
			this.originalFunctionsSize = functions.size();

			this.languageID = languageId;
			this.monitor = monitor;
			
			this.currentProgram = currentProgram;
			this.stackPointer = currentProgram.getCompilerSpec().getStackPointer();
			
			this.decompInterface = decompInterface;
			
			this.architecture = currentProgram.getLanguage().getProcessor().toString();
			this.language = (SleighLanguage) currentProgram.getLanguage();
			this.nextUnique = language.getUniqueBase();

			this.apiUtil = new ApiUtil(currentProgram);
			this.basicBlockModel = new BasicBlockModel(currentProgram);

			AddressFactory addressFactory = currentProgram.getAddressFactory();
			this.externSpace = addressFactory.getAddressSpace("extern");
			this.ramSpace = addressFactory.getAddressSpace("ram");
			this.stackSpace = addressFactory.getStackSpace();
			this.constantSpace = addressFactory.getConstantSpace();
			this.uniqueSpace = addressFactory.getUniqueSpace();

			this.seenFunctions = new TreeSet<>();
			this.seenTypes = new HashSet<>();
			this.seenGlobalsMap = new HashMap<>();
			this.typesToSerialize = new ArrayList<>();
			this.currentFunction = null;
			this.currentBlock = null;
			this.jumpTableIndex = null;
			this.nextSeqNum = 0;
			this.entryBlock = new ArrayList<>();
			this.missingLocalsMap = new HashMap<>();
			this.oldLocalsMap = new HashMap<>();
			this.temporaryAddressMap = new HashMap<>();
			this.replacementOperationsMap = new HashMap<>();
			this.prefixOperationsMap = new HashMap<>();
			this.addressOfGlobalMap = new HashMap<>();
			this.callotherUsePcodeOps = new ArrayList<>();
			this.seenDataMap = new HashMap<>();
		}

		String label(HighFunction function) throws Exception {
			return label(function.getFunction());
		}

		String label(Function function) throws Exception {
			
			// NOTE(pag): This full dethunking works in collaboration with
			//			  `seen_functions` checking in `serializeFunctions` to
			//			  deduplicate thunk functions.
			if (function.isThunk()) {
				function = function.getThunkedFunction(true);
			}

			return label(function.getEntryPoint());
		}

		static String label(Address address) throws Exception {
			return address.toString(true  /* show address space prefix */);
		}

		static String label(SequenceNumber sequenceNumber) throws Exception {
			return label(sequenceNumber.getTarget()) + Address.SEPARATOR +
				   Integer.toString(sequenceNumber.getTime()) + Address.SEPARATOR +
				   Integer.toString(sequenceNumber.getOrder());
		}

		static String label(PcodeBlock block) throws Exception {
			return label(block.getStart()) + Address.SEPARATOR +
				   Integer.toString(block.getIndex()) + Address.SEPARATOR +
				   PcodeBlock.typeToName(block.getType());
		}

		static String label(PcodeOp pcodeOp) throws Exception {
			return label(pcodeOp.getSeqnum());
		}

		String label(DataType type) throws Exception {
			// If type is null, assign VoidDataType in all cases.
			// We assume it as void type.
			if (type == null) {
				type = VoidDataType.dataType;
			}

			String name = type.getName();
			CategoryPath category = type.getCategoryPath();
			String concatenatedTypeCategoryAndLength = category.toString() + name + Integer.toString(type.getLength());
			String typeHexId = Integer.toHexString(concatenatedTypeCategoryAndLength.hashCode());

			UniversalID typeUid = type.getUniversalID();
			if (typeUid != null) {
				typeHexId += Address.SEPARATOR + typeUid.toString();
			}

			if (seenTypes.add(typeHexId)) {
				typesToSerialize.add(type);
			}
			return typeHexId;
		}
		
		// Figure out the return type of an intrinsic op.
		DataType intrinsicReturnType(PcodeOp pcodeOp) {
			Varnode returnValueVarnode = pcodeOp.getOutput();
			if (returnValueVarnode == null) {
				return VoidDataType.dataType;
			}

			HighVariable highVariable = returnValueVarnode.getHigh();
			if (highVariable != null) {
				return highVariable.getDataType();
			}
		
			assert false;
			return Undefined.getUndefinedDataType(returnValueVarnode.getSize());
		}

		Address getAddress(PcodeOp pcodeOp) throws Exception {
			SequenceNumber sequenceNumber = pcodeOp.getSeqnum();
			return sequenceNumber.getTarget();
		}
		
		// Return the label of an intrinsic with `CALLOTHER`. This is based
		// off of the return value.
		String intrinsicLabel(PcodeOp pcodeOp) throws Exception {
			if (pcodeOp.getOpcode() == PcodeOp.CALLOTHER) {
				int index = (int) pcodeOp.getInput(0).getOffset();
				String name = language.getUserDefinedOpName(index);
				return intrinsicLabel(name, intrinsicReturnType(pcodeOp));
			} else {
				throw new UnsupportedOperationException("Can only label a CALLOTHER PcodeOp, to which the first input should always be a constant representing the user-defined op index");
			}
		}
		
		String intrinsicLabel(
				String name, DataType returnDataType) throws Exception {
			return name + Address.SEPARATOR + label(returnDataType);
		}

		void serializeBuiltinType(
				DataType dataType, String kind) throws Exception {
			
			String displayName = null;
			if (dataType instanceof AbstractIntegerDataType) {
				AbstractIntegerDataType adt = (AbstractIntegerDataType) dataType;
				displayName = adt.getCDeclaration();
			}
			
			if (displayName == null) {
				displayName = dataType.getDisplayName();
			}

			writer.name("name").value(displayName);
			writer.name("size").value(dataType.getLength());
			writer.name("kind").value(kind);
		}

		void serializeCompositeType(
				Composite compositeDataType, String kind) throws Exception {
			writer.name("name").value(compositeDataType.getDisplayName());
			writer.name("kind").value(kind);
			writer.name("size").value(compositeDataType.getLength());
			writer.name("fields").beginArray();

			for (int i = 0; i < compositeDataType.getNumComponents(); i++) {
				DataTypeComponent dtc = compositeDataType.getComponent(i);
				writer.beginObject();
				writer.name("type").value(label(dtc.getDataType()));
				writer.name("offset").value(dtc.getOffset());

				if (dtc.getFieldName() != null) {
					writer.name("name").value(dtc.getFieldName());
				}
				writer.endObject();
			}
			writer.endArray();
		}

		void serializeEnumType(Enum enumDataType, String kind) throws Exception {
			writer.name("name").value(enumDataType.getDisplayName());
			writer.name("kind").value(kind);
			writer.name("size").value(enumDataType.getLength());	
			int enumCountOfEntries = enumDataType.getCount();
			writer.name("num_entries").value(enumCountOfEntries);
			writer.name("entries").beginArray();
			for (int i = 0; i < enumCountOfEntries; i++) {
				String enumDataTypeName = enumDataType.getName(i);
				if (enumDataTypeName != null) {
					writer.beginObject();
					writer.name("name").value(enumDataTypeName);
					writer.name("value").value(enumDataType.getValue(enumDataTypeName));
					writer.endObject();
				}
			}
			writer.endArray();
		}

		void serializeType(DataType dataType) throws Exception {
			if (dataType == null) {
				writer.nullValue();
				
			} else if (dataType instanceof Pointer) {
				Pointer ptr = (Pointer) dataType;
				writer.name("kind").value("pointer");
				writer.name("size").value(ptr.getLength());
				writer.name("element_type").value(label(ptr.getDataType()));

			} else if (dataType instanceof TypeDef) {
				TypeDef typeDef = (TypeDef) dataType;
				writer.name("name").value(typeDef.getDisplayName());
				writer.name("kind").value("typedef");
				writer.name("size").value(typeDef.getLength());
				writer.name("base_type").value(label(typeDef.getBaseDataType()));

			} else if (dataType instanceof Array) {
				Array array = (Array) dataType;
				writer.name("kind").value("array");
				writer.name("size").value(array.getLength());
				writer.name("num_elements").value(array.getNumElements());
				writer.name("element_type").value(label(array.getDataType()));

			} else if (dataType instanceof Structure) {
				serializeCompositeType((Composite) dataType, "struct");

			} else if (dataType instanceof Union) {
				serializeCompositeType((Composite) dataType, "union");

			} else if (dataType instanceof BooleanDataType) {
				// NB, BooleanDataType is technically a subclass of AbstractIntegerDataType
				// so MUST come first in the order here or all booleans will turn into ints
				serializeBuiltinType(dataType, "boolean");

			} else if (dataType instanceof AbstractIntegerDataType){
				serializeBuiltinType(dataType, "integer");

			} else if (dataType instanceof AbstractFloatDataType){
				serializeBuiltinType(dataType, "float");

			} else if (dataType instanceof Enum) {
				serializeEnumType((Enum) dataType, "enum");

			} else if (dataType instanceof VoidDataType) {
				serializeBuiltinType(dataType, "void");

			} else if (dataType instanceof Undefined || dataType instanceof DefaultDataType) {
				serializeBuiltinType(dataType, "undefined");

			} else if (dataType instanceof FunctionDefinition) {
				writer.name("kind").value("function");
				serializePrototype((FunctionSignature) dataType);

			} else if (dataType instanceof PartialUnion) {
				DataType parent = ((PartialUnion) dataType).getParent();
				if (parent != dataType) {
					serializeType(parent);
				} else {
					// PartialUnion stripped type is undefined type
					serializeType(((PartialUnion) dataType).getStrippedDataType());
				}
			} else if (dataType instanceof BitFieldDataType) {
				writer.name("kind").value("todo:BitFieldDataType");  // TODO(pag): Implement this
				writer.name("size").value(dataType.getLength());
				
			} else if (dataType instanceof WideCharDataType) {
				writer.name("kind").value("wchar");
				writer.name("size").value(dataType.getLength());
				
			} else if (dataType instanceof StringDataType) {
				writer.name("kind").value("todo:StringDataType");  // TODO(pag): Implement this
				writer.name("size").value(dataType.getLength());
				
			} else {
				throw new Exception("Unhandled type: " + dataType.getClass().getName());
			}
		}

		void serializeTypes() throws Exception {
			for (int i = 0; i < typesToSerialize.size(); i++) {
				DataType type = typesToSerialize.get(i);

				if (type != null) {
					writer.name(label(type)).beginObject();
				} else {
					// if the type is literally null, encapsulate it before naming it
					writer.beginObject();
					writer.name(label(type));
				}

				serializeType(type);
				writer.endObject();
			}

			System.out.println("Total serialized types: " + typesToSerialize.size());
		}
		
		int serializePrototype() throws Exception {
			writer.name("return_type").value(label((DataType) null));
			writer.name("is_variadic").value(false);
			writer.name("is_noreturn").value(false);
			writer.name("parameter_types").beginArray().endArray();
			return 0;
		}

		int serializePrototype(FunctionPrototype proto) throws Exception {
			if (proto == null) {
				return serializePrototype();
			}

			writer.name("return_type").value(label(proto.getReturnType()));
			writer.name("is_variadic").value(proto.isVarArg());
			writer.name("is_noreturn").value(proto.hasNoReturn());
			
			writer.name("parameter_types").beginArray();
			int numberOfParams = proto.getNumParams();
			for (int i = 0; i < numberOfParams; i++) {
				writer.value(label(proto.getParam(i).getDataType()));
			}
			writer.endArray();  // End of `parameter_types`.
			return numberOfParams;
		}
		
		int serializePrototype(FunctionSignature proto) throws Exception {
			if (proto == null) {
				return serializePrototype();
			}

			writer.name("return_type").value(label(proto.getReturnType()));
			writer.name("is_variadic").value(proto.hasVarArgs());
			writer.name("is_noreturn").value(proto.hasNoReturn());
			writer.name("calling_convention").value(proto.getCallingConventionName());
			
			ParameterDefinition[] arguments = proto.getArguments();
			writer.name("parameter_types").beginArray();
			int numberOfParameterTypes = (int) arguments.length;
			for (int i = 0; i < numberOfParameterTypes; i++) {
				writer.value(label(arguments[i].getDataType()));
			}
			writer.endArray();  // End of `parameter_types`.
			return numberOfParameterTypes;
		}
		
		// Returns `true` if a given representative is an original
		// representative.
		boolean isOriginalRepresentative(Varnode node) {
			if (node.isInput()) {
				return true;
			}
			
			// NOTE(pag): Don't use `resolveOp` here because that screws up the
			//			  variable creation logic.
			PcodeOp pcodeOp = node.getDef();
			if (pcodeOp == null) {
				return true;
			}
			
			if (pcodeOp.getOpcode() != PcodeOp.CALLOTHER) {
				return true;
			}
			
			if ((pcodeOp.getInputs().length > 0) && 
				(pcodeOp.getInput(0).getOffset() < MIN_CALLOTHER)) {
				return true;
			}
			
			// pcodeOp is CALLOTHER AND Varnode is an op output AND the offset is sufficiently high
			return false;
		}
		
		// Resolve an operation to a replacement operation, if any.
		PcodeOp resolveOp(PcodeOp pcodeOp) {
			if (pcodeOp == null) {
				return null;
			}

			PcodeOp replacementPcodeOp = replacementOperationsMap.get(pcodeOp);
			// if the map is empty or has no pcodeOp mapping, it rets null
			if (replacementPcodeOp != null) {
				return replacementPcodeOp;
			}
			return pcodeOp;
		}

		// Get the representative of a `HighVariable`, or if we've re-written
		// the representative with a `CALLOTHER`, then get the original
		// representative.
		Varnode originalRepresentativeOf(HighVariable highVariable) {
			if (highVariable == null) {
				return null;
			}
			
			Varnode rep = highVariable.getRepresentative();
			if (isOriginalRepresentative(rep)) {
				return rep; 
			}
			
			Varnode[] instances = highVariable.getInstances();
			if (instances.length <= 1) {
				return null;
			}
			
			return instances[1];
		}
		
		// Return the address of a high global variable.
		// todo(kaoudis) this needs to be consistent, right now it is not.
		// find out if the inconsistency is about decomp or something else.
		Address addressOfGlobal(HighVariable highGlobalVariable) throws Exception {
			HighSymbol highSymbol = highGlobalVariable.getSymbol();

			if (highSymbol != null && highSymbol.isGlobal()) {
				SymbolEntry entry = highSymbol.getFirstWholeMap();
				VariableStorage storage = entry.getStorage();
				if (storage != VariableStorage.BAD_STORAGE &&
					storage != VariableStorage.UNASSIGNED_STORAGE &&
					storage != VariableStorage.VOID_STORAGE) {
					return storage.getMinAddress();
				}
			}
			
			// todo (kaoudis) what is the rest of this even good for if all globals fall into the above???
			Varnode representativeVarnode = highGlobalVariable.getRepresentative();
			int typeId = AddressSpace.ID_TYPE_MASK & representativeVarnode.getSpace();
			if (typeId == AddressSpace.TYPE_RAM) {
				return ramSpace.getAddress(representativeVarnode.getOffset());

			} else if (typeId == AddressSpace.TYPE_EXTERNAL) {
				return externSpace.getAddress(representativeVarnode.getOffset());
			}
			
			Address fixedAddress = addressOfGlobalMap.get(highGlobalVariable);
			if (fixedAddress != null) {
				return fixedAddress;
			}
			
			System.out.println("Could not get address of variable " + highGlobalVariable.toString());
			return null;
		}

		Address makeGlobalFromData(Data data) throws Exception {
			if (data == null) {
				return null;
			}
			seenDataMap.put(data.getMinAddress(), data);
			return data.getMinAddress();
		}

		// Try to distinguish "local" variables from global ones. Roughly, we
		// want to make sure that the backing storage for a given variable
		// *isn't* RAM. Thus, UNIQUE, STACK, CONST, etc. are all in-scope for
		// locals.
		VariableClassification classifyVariable(HighVariable highVariable) throws Exception {
			if (highVariable == null) {
				return VariableClassification.UNKNOWN;
			}

			if (highVariable instanceof HighParam) {
				return VariableClassification.PARAMETER;

			} else if (highVariable instanceof HighLocal) {
				return VariableClassification.LOCAL;

			} else if (highVariable instanceof HighConstant) {
				return VariableClassification.CONSTANT;
			
			} else if (highVariable instanceof HighGlobal) {
				seenGlobalsMap.put(addressOfGlobal(highVariable), highVariable);
				return VariableClassification.GLOBAL;
			
			} else if (highVariable instanceof HighTemporary) {
				return VariableClassification.TEMPORARY;
			}

			HighSymbol highSymbol = highVariable.getSymbol();
			if (highSymbol != null) {
				if (highSymbol.isGlobal()) {
					seenGlobalsMap.put(addressOfGlobal(highVariable), highVariable);
					return VariableClassification.GLOBAL;

				} else if (highSymbol.isParameter() || highSymbol.isThisPointer()) {
					return VariableClassification.PARAMETER;
				}
			}

			Varnode highVariableRepresentativeVarnode = originalRepresentativeOf(highVariable);
			if (highVariableRepresentativeVarnode != null) {

				// TODO(pag): Consider checking if all uses of the unique
				//			  belong to the same block. We don't want to
				//			  introduce a kind of code motion risk into the
				//			  lifted representation.
				if (highVariableRepresentativeVarnode.isRegister() || highVariableRepresentativeVarnode.isUnique() || highVariable instanceof HighOther) {
					if (highVariableRepresentativeVarnode.getLoneDescend() != null) {
						return VariableClassification.TEMPORARY;
					} else {
						return VariableClassification.NAMED_TEMPORARY;
					}
				}
			}
			
			return VariableClassification.UNKNOWN;
		}

		boolean isCharPointer(Varnode node) throws Exception {
			HighVariable highVariable = variableOf(node.getHigh());
			if (highVariable == null) {
				return false;
			}

			DataType dataType = highVariable.getDataType();
			if (dataType instanceof Pointer) {
				DataType baseDataType = ((Pointer) dataType).getDataType();
				if (baseDataType instanceof TypeDef) {
					baseDataType = ((TypeDef )baseDataType).getBaseDataType();
				}
				if (baseDataType instanceof CharDataType) {
					return true;
				}
			}

			return false;
		}


		// Serialize an input or output varnode.
		void serializeInput(PcodeOp pcodeOp, Varnode node) throws Exception {
		//	assert !node.isFree();
		//	assert node.isInput();

			PcodeOp nodeDefPcodeOp = resolveOp(node.getDef());
			HighVariable highVariable = variableOf(node.getHigh());

			writer.beginObject();
			
			if (highVariable != null) {
				if (nodeDefPcodeOp == null) {
					nodeDefPcodeOp = highVariable.getRepresentative().getDef();
				}

				writer.name("type").value(label(highVariable.getDataType()));
			} else {
				writer.name("size").value(node.getSize());
			}

			switch (classifyVariable(highVariable)) {
				case UNKNOWN:
					if (nodeDefPcodeOp != null && !node.isInput() && nodeDefPcodeOp == pcodeOp) {
						if (node.isUnique()) {
							writer.name("kind").value("temporary");
						
						// TODO(pag): Figure this out.
						} else {
							assert false;
							writer.name("kind").value("unknown");
						}
	
					// NOTE(pag): Should be a `TEMPORARY` classification.
					} else if (node.isUnique()) {
						assert false;
						assert nodeDefPcodeOp != null;
						writer.name("kind").value("temporary");
						writer.name("operation").value(label(nodeDefPcodeOp));
					
					// NOTE(pag): Should be a `REGISTER` classification.
					} else if (node.isConstant()) {
						assert false;
						writer.name("kind").value("constant");
						writer.name("value").value(node.getOffset());

					} else {
						assert false;
						writer.name("kind").value("unknown");
					}
					break;
				case PARAMETER:
					writer.name("kind").value("parameter");
					writer.name("operation").value(label(getOrCreateLocalVariable(highVariable, pcodeOp)));
					break;
				case LOCAL:
					writer.name("kind").value("local");
					writer.name("operation").value(label(getOrCreateLocalVariable(highVariable, pcodeOp)));
					break;
				case NAMED_TEMPORARY:
					writer.name("kind").value("temporary");
					writer.name("operation").value(label(getOrCreateLocalVariable(highVariable, pcodeOp)));
					break;
				case TEMPORARY:
					assert nodeDefPcodeOp != null;
					writer.name("kind").value("temporary");
					writer.name("operation").value(label(nodeDefPcodeOp));
					break;
				case GLOBAL:
					writer.name("kind").value("global");
					writer.name("global").value(label(addressOfGlobal(highVariable)));
					break;
				case FUNCTION:
					writer.name("kind").value("function");
					writer.name("function").value(label(highVariable.getHighFunction()));
					break;
				case CONSTANT:
					if (node.isConstant()) {
						Data dataReferencedAsConstant = apiUtil.getDataReferencedAsConstant(node);
						if (dataReferencedAsConstant != null) {
							if (dataReferencedAsConstant.hasStringValue()) {
								writer.name("kind").value("string");
								writer.name("string_value").value(dataReferencedAsConstant.getValue().toString());
							} else {
								writer.name("kind").value("global");
								writer.name("global").value(label(makeGlobalFromData(dataReferencedAsConstant)));
							}
						} else if (isCharPointer(node) && highVariable != null
							&& !node.getAddress().equals(constantSpace.getAddress(0))) {
							String string = apiUtil.findNullTerminatedString(node.getAddress(), ((Pointer) highVariable.getDataType()));
							if (string != null) {
								writer.name("kind").value("string");
								writer.name("string_value").value(string);
							} else {
								// No valid string found at address - treat as constant value.
								// This happens when a small constant (e.g., 0x3) has a char pointer
								// type but doesn't point to valid mapped memory.
								writer.name("kind").value("constant");
								writer.name("value").value(node.getOffset());
							}
						} else {
							writer.name("kind").value("constant");
							writer.name("value").value(node.getOffset());
						}
					} else {
						assert false;
						writer.name("kind").value("unknown");
					}
					break;
			}

			writer.endObject();
		}
		
		// Returns the index of the first input `Varnode` referncing the stack
		// pointer, or `-1` if no direct references are found.
		int referencesStackPointer(PcodeOp op) throws Exception {
			int inputIndex = 0;
			for (Varnode node : op.getInputs()) {
				if (node.isRegister()) {
					Register register = language.getRegister(node.getAddress(), 0);
					if (register == null) {
						continue;
					}

					// TODO(pag): This doesn't seem to work? All `typeFlags` for
					// 			  all registers seem to be zero, at least for
					//			  x86.
					//
					// NOTE(pag): NCC group blog post on "earlyremoval" also
					//			  notes this curiosity.
					if ((register.getTypeFlags() & Register.TYPE_SP) != 0) {
						return inputIndex;
					}

					if (register == stackPointer) {
						return inputIndex;
					}

					// TODO(pag): Should we consider references to the frame
					//			  pointer, e.g. using the `CompilerSpec` or
					//			  `reg.isDefaultFramePointer()`?
				}
				
				++inputIndex;
			}
			
			return -1;
		}
		
		// Given a `PTRSUB SP, offset` that resolves to the base of a local
		// variable, or a `PTRSUB 0, addr` that resolves to the address of a
		// global variable, generate and `ADDRESS_OF var`.
		PcodeOp createAddressOf(
				Varnode defVarnode, SequenceNumber sequenceNumber, Varnode inputVarnode) {
			Varnode inputs[] = new Varnode[2];
			inputs[0] = new Varnode(constantSpace.getAddress(ADDRESS_OF), 4);
			inputs[1] = inputVarnode;
			return new PcodeOp(sequenceNumber, PcodeOp.CALLOTHER, inputs, defVarnode);
		}
		
		// Given an offset `var_offset` from the stack pointer in `op`, return
		// two `Varnode`s, the first referencing the relevant `HighVariable`
		// that contains the byte at that stack offset, and the second being
		// a constant byte displacement from the base of the stack variable.
		Varnode[] createStackPointerVarnodes(
				HighFunction highFunction, PcodeOp pcodeOp,
				int varOffset) throws Exception {
			
			Function function = highFunction.getFunction();
			StackFrame frame = function.getStackFrame();

			int stackFrameSize = frame.getFrameSize();
			int adjustOffset = 0;
			
			// Given the local symbol mapping for the high function, go find
			// a `HighSymbol` corresponding to `local_118`. This high symbol
			// will generally have a much better `DataType`, but initially
			// and confusingly won't have a corresponding `HighVariable`.
			LocalSymbolMap symbols = highFunction.getLocalSymbolMap();
			Address pcodeOpTargetAddress = pcodeOp.getSeqnum().getTarget();
			
			// Given a stack pointer offset, e.g. `-0x118`, go find the low
			// `Variable` representing `local_118`.
			Variable lowVariable = frame.getVariableContaining(varOffset);
			Address stackAddress = stackSpace.getAddress(varOffset);
			HighSymbol highSymbol = null;
			if (lowVariable != null) {
				highSymbol = symbols.findLocal(lowVariable.getVariableStorage(), pcodeOpTargetAddress);
				stackAddress = stackSpace.getAddress(lowVariable.getStackOffset());
				
			} else {
				highSymbol = symbols.findLocal(stackAddress, pcodeOpTargetAddress);
			}
			
			// Try to recover by locating the parameter containing the stack
			// address.
			if (highSymbol == null) {
				for (Variable param : frame.getParameters()) {
					VariableStorage storage = param.getVariableStorage();
					if (!storage.contains(stackAddress)) {
						continue;
					}
			
					int index = ((Parameter) param).getOrdinal();
					if (index >= symbols.getNumParams()) {
						break;
					}
					
					highSymbol = symbols.getParamSymbol(index);
					break;
				}
			}
			
			// This is usually for one of a few reasons:
			//		- Trying to lift `_start`
			//		- Trying to lift a variadic function using `va_list`.
			if (highSymbol == null) {
				return null;
			}

			Varnode pcodeOpInputVarnode = pcodeOp.getInput(0);
			UseVarnode newUseVarnode = new UseVarnode(
					stackAddress, highSymbol.getDataType().getLength());

			// We've already got a high variable for this missing local.
			HighVariable newHighVariable = highSymbol.getHighVariable();
			String symbolName = highSymbol.getName();
			if (newHighVariable != null && !newHighVariable.getName().equals("UNNAMED")) {

			// We need to invent a new `HighVariable` for this `HighSymbol`.
			// Unfortunately we can't use `HighSymbol.setHighVariable` for the
			// caching, so we need `missingLocalsMap`.
			} else {
				HighLocal highLocalVariable = oldLocalsMap.get(newHighVariable);
				if (highLocalVariable == null) {
					highLocalVariable = missingLocalsMap.get(symbolName);
				}

				if (highLocalVariable == null) {
					highLocalVariable = new HighLocal(
							highSymbol.getDataType(), newUseVarnode, null, pcodeOpTargetAddress, highSymbol);
					missingLocalsMap.put(symbolName, highLocalVariable);

					// println("Created " + local_var.getName() + " with type " + local_var.getDataType().toString());

					// Remap old-to-new.
					if (newHighVariable != null) {
						oldLocalsMap.put(newHighVariable, highLocalVariable);
					}
					
				}
				newHighVariable = highLocalVariable;
			}

			newUseVarnode.setHighVariable(newHighVariable);

			if (lowVariable != null) {
				adjustOffset = (varOffset - lowVariable.getStackOffset());
			}
			
			Varnode[] nodes = new Varnode[2];
			nodes[0] = newUseVarnode;
			nodes[1] = new Varnode(constantSpace.getAddress(adjustOffset),
								   ramSpace.getSize() / 8);
			return nodes;
		}
		
		// Update a `PTRSUB 0, addr` or a `PTRSUB SP, offset` to be prefixed
		// by an `ADDRESS_OF`, then operate on the `ADDRESS_OF` in the first
		// input, and use a modified offset in the second input.
		boolean prefixPtrSubcomponentWithAddressOf(
				HighFunction highFunction, PcodeOp pcodeOp,
				Varnode[] nodes) throws Exception {

			Address pcodeOpTargetAddress = pcodeOp.getSeqnum().getTarget();
			List<PcodeOp> pcodeOps = getOrCreatePrefixOperations(pcodeOp);
			
			// Figure out the tye of the pointer to the local variable being
			// referenced.
			DataTypeManager dataTypeManager = currentProgram.getDataTypeManager();
			DataType variableType = nodes[0].getHigh().getDataType();
			DataType nodeDataType = dataTypeManager.getPointer(variableType);
			
			// Create a unique address for this `Varnode`.
			Address address = nextUniqueAddress();
			SequenceNumber sequenceNumber = new SequenceNumber(address, nextSeqNum++);
			
			// Make the `Varnode` instances.
			DefinitionVarnode definitionVarnode = new DefinitionVarnode(
					address, nodeDataType.getLength());
			UseVarnode useVarnode = new UseVarnode(address, definitionVarnode.getSize());
			
			// Create a prefix `ADDRESS_OF` for the local variable.
			PcodeOp addressOf = this.createAddressOf(definitionVarnode, sequenceNumber, nodes[0]);
			pcodeOps.add(addressOf);
			
			// Track the logical value using a `HighOther`.
			Varnode[] instances = new Varnode[2];
			instances[0] = addressOf.getOutput();
			instances[1] = useVarnode;
			HighVariable tracker = new HighTemporary(
					nodeDataType, instances[0], instances, pcodeOpTargetAddress, highFunction);

			definitionVarnode.setDef(tracker, addressOf);
			useVarnode.setHighVariable(tracker);
			
			System.out.println(label(pcodeOp));
			System.out.println("  Rewriting " + pcodeOp.getSeqnum().toString() + ": " + pcodeOp.toString());

			// Rewrite the stack reference to point to the `HighVariable`.
			pcodeOp.setInput(useVarnode, 0);
			
			// Rewrite the offset.
			pcodeOp.setInput(nodes[1], 1);

			System.out.println("  to: " + pcodeOp.toString());
			
			return true;
		}
		
		// Given a `PTRSUB SP, offset`, try to invent a local variable at
		// `offset` in a similar way to how the decompiler would.
		boolean createLocalForPtrSubcomponent(
				HighFunction highFunction, PcodeOp pcodeOp,
				CallDepthChangeInfo cdci) throws Exception {
			
			Varnode offset = pcodeOp.getInput(1);
			if (!offset.isConstant()) {
				return false;
			}

			Varnode[] nodes = createStackPointerVarnodes(
					highFunction, pcodeOp, (int) offset.getOffset());
			if (nodes == null) {
				return false;
			}
			
			// We can replace the `PTRSUB SP, offset` with an
			// `ADDRESS_OF local`.
			if (nodes[1].getOffset() == 0) {
				PcodeOp newPcodeOp = createAddressOf(
						pcodeOp.getOutput(), pcodeOp.getSeqnum(), nodes[0]);
				replacementOperationsMap.put(pcodeOp, newPcodeOp);
				return true;
			}
			
			// We need to get the `ADDRESS_OF local`, then pass that to a
			// fixed-up `PTRSUB`.
			return prefixPtrSubcomponentWithAddressOf(highFunction, pcodeOp, nodes);
		}
		
		// Return the next referenced address after `start`, or the maximum
		// address in `start`'s address space.
		Address getNextReferencedAddressOrMax(Address start) {
			Address end = start.getAddressSpace().getMaxAddress();
			AddressSet range = new AddressSet(start, end);
			ReferenceManager references = currentProgram.getReferenceManager();
			AddressIterator addressIterator = references.getReferenceDestinationIterator(range, true);
			if (!addressIterator.hasNext()) {
				return end;
			}

			Address referencedAddress = addressIterator.next();
			if (!start.equals(referencedAddress)) {
				return referencedAddress;
			}
			
			if (addressIterator.hasNext()) {
				return addressIterator.next();
			}
			
			return end;
		}
		
		// Given a `PTRSUB const, const`, try to recognize it as a global variable
		// reference, or a field reference within a global variable.
		boolean createGlobalForPtrSubcomponent(
				HighFunction highFunction, PcodeOp pcodeOp) throws Exception {
			
			HighVariable zero = pcodeOp.getInput(0).getHigh();
			if (!(zero instanceof HighOther)) {
				return false;
			}
			
			if (!zero.getName().equals("UNNAMED")) {
				return false;
			}
			
			if (zero.getOffset() != -1) {
				return false;
			}
			
			Varnode offsetNode = pcodeOp.getInput(1);
			HighVariable offsetVariable = offsetNode.getHigh();
			if (!(offsetVariable instanceof HighConstant)) {
				return false;
			}
			
			HighSymbol highSymbol = offsetVariable.getSymbol();
			if (highSymbol == null) {
				return false;
			}

			SymbolEntry entry = highSymbol.getFirstWholeMap();
			VariableStorage storage = entry.getStorage();
			Address address = null;

			if (storage == VariableStorage.BAD_STORAGE ||
				storage == VariableStorage.UNASSIGNED_STORAGE ||
				storage == VariableStorage.VOID_STORAGE) {

				address = ramSpace.getAddress(offsetNode.getOffset());
			} else {
				address = storage.getMinAddress();
			}
			
			DataType type = highSymbol.getDataType();
			
			// Get the size in bytes. This might require calculating the length
			// of a string, for which we use the heuristic that the string
			// probably ends at the next referenced address.
			//
			// TODO(pag): This isn't a great heuristic because it's fairly
			//			  common for compilers to do suffix compression of
			//			  strings, i.e. given `"c"` can be a suffix of `"bc"`
			//			  which can be a suffix of `"abc"`, and so every string
			//			  in this case would show as having a maximum length of
			//			  `1` by this heuristic.
			int sizeInBytes = type.getLength();
			if (sizeInBytes == -1 && type instanceof AbstractStringDataType) {
				Listing listing = currentProgram.getListing();
				MemBuffer memory = listing.getCodeUnitAt(address);
				Address nextAddress = getNextReferencedAddressOrMax(address);
				sizeInBytes = ((AbstractStringDataType) type).getLength(
						memory, (int) nextAddress.subtract(address));
			}
			
			UseVarnode newUseVarnode = new UseVarnode(address, sizeInBytes);
			HighVariable globalVariable = seenGlobalsMap.get(address);
			if (globalVariable == null) {
				globalVariable = highSymbol.getHighVariable();
				if (globalVariable == null) {
					globalVariable = new HighGlobal(highSymbol, newUseVarnode, null);
				}

				seenGlobalsMap.put(address, globalVariable);
			}

			addressOfGlobalMap.put(globalVariable, address);

			// Rewrite the offset.
			Address offsetAsAddress = address.getAddressSpace().getAddress(offsetNode.getOffset());
			int subOffset = (int) offsetAsAddress.subtract(address);

			newUseVarnode.setHighVariable(globalVariable);
			if (subOffset == 0) {
				PcodeOp newPcodeOp = createAddressOf(
						pcodeOp.getOutput(), pcodeOp.getSeqnum(), newUseVarnode);
				replacementOperationsMap.put(pcodeOp, newPcodeOp);
				return true;
			}
			
			Varnode[] nodes = new Varnode[2];
			nodes[0] = newUseVarnode;
			nodes[1] = new Varnode(constantSpace.getAddress(subOffset),
								   offsetNode.getSize());
			
			// We need to get the `ADDRESS_OF global`, then pass that to a
			// fixed-up `PTRSUB`.
			return prefixPtrSubcomponentWithAddressOf(highFunction, pcodeOp, nodes);
		}
		
		// Try to rewrite/mutate a `PTRSUB`.
		boolean rewritePtrSubcomponent(
				HighFunction highFunction, PcodeOp pcodeOp,
				CallDepthChangeInfo cdci) throws Exception {

			// Look for `PTRSUB SP, offset` and convert into `PTRSUB local_N, M`.
			if (referencesStackPointer(pcodeOp) == 0) {
				return createLocalForPtrSubcomponent(highFunction, pcodeOp, cdci);
			}

			Varnode baseVarnode = pcodeOp.getInput(0);
			Varnode offsetVarnode = pcodeOp.getInput(1);
			if (baseVarnode.isConstant() && baseVarnode.getOffset() == 0 &&
				offsetVarnode.isConstant()) {
				return createGlobalForPtrSubcomponent(highFunction, pcodeOp);
			}

			return true;
		}

		DataType normalizeDataType(DataType dataType) throws Exception {
			if (dataType instanceof TypeDef) {
				return ((TypeDef) dataType).getBaseDataType();
			}
			return dataType;
		}

		DataType getArgumentType(Function callee, int paramIndex) {
			if (callee == null || paramIndex < 0) {
				return null;
			}

			FunctionSignature signature = callee.getSignature();
			if (signature == null) {
				return null;
			}

			ParameterDefinition[] params = signature.getArguments();
			if (params == null || paramIndex >= params.length) {
				return null;
			}

			ParameterDefinition param = params[paramIndex];
			return param != null ? param.getDataType() : null;
		}

		Function resolveCalledFunction(Varnode targetVarnode, Address caller) {
			if (!targetVarnode.isAddress()) {
				return null;
			}

			Address targetAddress = caller.getNewAddress(targetVarnode.getOffset());
			// Try to get function at the target address
			FunctionManager fm = currentProgram.getFunctionManager();
			Function callee = fm.getFunctionAt(targetAddress);
			if (callee == null) {
				callee = fm.getReferencedFunction(targetAddress);
			}

			return callee;
		}

		boolean needsCastOperation(DataType variableType, DataType argumentType) {
			if (variableType.getLength() <= 0 || argumentType.getLength() <= 0) {
				return false;
			}

			// TODO: Identify cases where we need explicit CAST operations to handle conversions.
 			//       Some cases like pointer conversion between different types or type conversion
			//       of builtin types are already handled during AST generation.

			// Handle float to int and int to float conversion
			if ((variableType instanceof AbstractFloatDataType && argumentType instanceof AbstractIntegerDataType) ||
				(variableType instanceof AbstractIntegerDataType && argumentType instanceof AbstractFloatDataType)) {
				return true;
			}

			if ((variableType instanceof Pointer && argumentType instanceof AbstractIntegerDataType) ||
				(variableType instanceof AbstractIntegerDataType && argumentType instanceof Pointer)) {
				return true;
			}

			if ((variableType instanceof Enum && argumentType instanceof AbstractIntegerDataType) ||
				(variableType instanceof AbstractIntegerDataType && argumentType instanceof Enum)) {
				return true;
			}

			// TODO: Handle conversion of undefined array to int or packed struct

			return false;
		}


		boolean needsAddressOfConversion(DataType variableType, DataType argumentType) {
			return ((variableType instanceof Composite) || (variableType instanceof Array)) && (argumentType instanceof Pointer);
		}

		boolean typesAreCompatible(DataType variableType, DataType argumentType) {
			return (variableType == argumentType) || ((variableType instanceof Pointer) && (argumentType instanceof Pointer));
		}


		boolean rewriteCallArgument(
				HighFunction highFunction, PcodeOp callOp) throws Exception {
			Address callerAddress = highFunction.getFunction().getEntryPoint();
			Varnode callTargetNode = callOp.getInput(0);

			Function callee = resolveCalledFunction(callTargetNode, callerAddress);
			if (callee == null) {
				// If the function cannot be resolved, we can't properly rewrite arguments
				return false;
			}

			DataTypeManager dataTypeManager = currentProgram.getDataTypeManager();
			Address callOpAddress = callOp.getSeqnum().getTarget();
			List<PcodeOp> prefixOperations = getOrCreatePrefixOperations(callOp);

			boolean modified = false;

			for (int i = 1; i < callOp.getInputs().length; i++) {
				Varnode input = callOp.getInput(i);
				HighVariable variable = input.getHigh();

				// Note: Check variable classification type and don't check for argument mismatch if
				//       it is of type UNKNOWN or CONSTANT
				VariableClassification variableClassification = classifyVariable(variableOf(variable));
				if ((variableClassification == VariableClassification.UNKNOWN) || (variableClassification == VariableClassification.CONSTANT)) {
					continue;
				}

				DataType variableType = normalizeDataType(variable.getDataType());
				DataType argumentType = normalizeDataType(getArgumentType(callee, i-1));

				// Skip if types match or both are compatible for implicit conversion
				if (variableType == null || argumentType == null
					|| typesAreCompatible(variableType, argumentType)) {
					continue;
				}

				if (needsAddressOfConversion(variableType, argumentType)) {
					Address address = nextUniqueAddress();
					SequenceNumber sequenceNumber = new SequenceNumber(address, nextSeqNum++);
					DefinitionVarnode definitionVarnode = new DefinitionVarnode(address, variableType.getLength());
					UseVarnode useVarnode = new UseVarnode(address, definitionVarnode.getSize());
					PcodeOp addressOf = createAddressOf(definitionVarnode, sequenceNumber, input);
					prefixOperations.add(addressOf);

					Varnode[] instances = new Varnode[2];
					instances[0] = addressOf.getOutput();
					instances[1] = useVarnode;
					DataType nodeDataType = dataTypeManager.getPointer(variableType);
					HighVariable tracker = new HighTemporary(
						nodeDataType, instances[0], instances, callOpAddress, highFunction);

					definitionVarnode.setDef(tracker, addressOf);
					useVarnode.setHighVariable(tracker);
					callOp.setInput(useVarnode, i);
					modified = true;

				} else if (needsCastOperation(variableType, argumentType)) {
					Address address = nextUniqueAddress();
					SequenceNumber sequenceNumber = new SequenceNumber(address, nextSeqNum++);
					DefinitionVarnode definitionVarnode = new DefinitionVarnode(address, variableType.getLength());

					Varnode castOutputVarnode = new DefinitionVarnode(address, argumentType.getLength());
					PcodeOp castOp = new PcodeOp(sequenceNumber, PcodeOp.CAST, new Varnode[] { input }, definitionVarnode);
					castOp.setOutput(castOutputVarnode);

					UseVarnode useVarnode = new UseVarnode(address, castOutputVarnode.getSize());
					prefixOperations.add(castOp);

					// Set up the high variables
					Varnode[] instances = new Varnode[2];
					instances[0] = castOutputVarnode;
					instances[1] = useVarnode;

					HighVariable tracker = new HighTemporary(
						argumentType, instances[0], instances, callOpAddress, highFunction);

					((DefinitionVarnode)castOutputVarnode).setDef(tracker, castOp);
					useVarnode.setHighVariable(tracker);

					// Update the call operation to use the cast result
					callOp.setInput(useVarnode, i);
					modified = true;
				}
			}
			return modified;
		}

		boolean rewriteVoidReturnType(
				HighFunction highFunction, PcodeOp callOp) throws Exception {
			Address callerAddress = highFunction.getFunction().getEntryPoint();
			Varnode callTargetNode = callOp.getInput(0);
			Function callee = resolveCalledFunction(callTargetNode, callerAddress);
			if (callee == null) {
				// If the function cannot be resolved, we can't properly rewrite return type
				return false;
			}

			// Only handle rewrite if return type is void but call_op output expects a non void return type
			Varnode callOutputVarnode = callOp.getOutput();
			if (callee.getReturnType() != VoidDataType.dataType || callOutputVarnode == null) {
				return false;
			}

			HighVariable highVariable = callOutputVarnode.getHigh();
			if (highVariable == null) {
				return false;
			}

			DataType fixedReturnType = highVariable.getDataType();
			if (fixedReturnType == null) {
				return false;
			}
			callee.setReturnType(fixedReturnType, SourceType.DEFAULT);
			return true;
		}
		
		// Get or create a prefix operations list. These are operations that
		// will precede `op` in our serialization, regardless of whether or
		// not `op` is elided.
		List<PcodeOp> getOrCreatePrefixOperations(PcodeOp pcodeOp) {
			List<PcodeOp> ops = prefixOperationsMap.get(pcodeOp);
			if (ops == null) {
				ops = new ArrayList<PcodeOp>();
				prefixOperationsMap.put(pcodeOp, ops);
			}
			return ops;
		}

		// Try to fixup direct stack pointer references in `op`.
		boolean tryFixupStackVarnode(
				HighFunction highFunction, PcodeOp pcodeOp,
				CallDepthChangeInfo cdci) throws Exception {

			int offset = referencesStackPointer(pcodeOp);
			if (offset == -1) {
				return true;
			}
			
			// Figure out what stack offset is pointed to by `SP`.
			Address operationTargetAddress = pcodeOp.getSeqnum().getTarget();
			int stackOffset = cdci.getDepth(operationTargetAddress);
			if (stackOffset == Function.UNKNOWN_STACK_DEPTH_CHANGE) {
				
				Function function = highFunction.getFunction();
				StackFrame frame = function.getStackFrame();
				Variable[] stackVariables = frame.getStackVariables();
				if (stackVariables == null || stackVariables.length == 0) {
					return false;
				}

				if (frame.growsNegative()) {
					stackOffset = stackVariables[0].getStackOffset();
				} else {
					stackOffset = stackVariables[stackVariables.length - 1].getStackOffset();
				}
			}

			Varnode stackPointerReference = pcodeOp.getInput(offset);
			Varnode[] nodes = createStackPointerVarnodes(
					highFunction, pcodeOp, stackOffset);
			if (nodes == null) {
				return false;
			}

			if (nodes[1].getOffset() != 0) {
				System.out.printf("??? " + Long.toHexString(nodes[1].getOffset()));
				return false;
			}

			List<PcodeOp> ops = getOrCreatePrefixOperations(pcodeOp);
			
			// Figure out the tye of the pointer to the local variable being
			// referenced.
			DataTypeManager dataTypeManager = currentProgram.getDataTypeManager();
			DataType variableType = nodes[0].getHigh().getDataType();
			DataType nodeDataType = dataTypeManager.getPointer(variableType);
			
			// Create a unique address for this `Varnode`.
			Address address = nextUniqueAddress();
			SequenceNumber sequenceNumber = new SequenceNumber(address, nextSeqNum++);
			
			// Make the `Varnode` instances.
			DefinitionVarnode definitionVarnode = new DefinitionVarnode(
					address, nodeDataType.getLength());
			UseVarnode useVarnode = new UseVarnode(address, definitionVarnode.getSize());
			
			// Create a prefix `ADDRESS_OF` for the local variable.
			PcodeOp addressOf = this.createAddressOf(definitionVarnode, sequenceNumber, nodes[0]);
			ops.add(addressOf);
			
			// Track the logical value using a `HighOther`.
			Varnode[] instances = new Varnode[2];
			instances[0] = addressOf.getOutput();
			instances[1] = useVarnode;
			HighVariable tracker = new HighTemporary(
					nodeDataType, instances[0], instances, operationTargetAddress, highFunction);

			definitionVarnode.setDef(tracker, addressOf);
			useVarnode.setHighVariable(tracker);
			
			pcodeOp.setInput(instances[1], offset);

			return tryFixupStackVarnode(highFunction, pcodeOp, cdci);
		}
		
		// The data model of high P-CODE is fundamentally value based. Lets
		// focus on the following example:
		//
		//		extern int do_with_int(int *);
		//
		//		int main() {
		//		  int x;
		//		  return do_with_int(&x);
		//		}
		//
		// Ignoring `INDIRECT`s, we might expect to see the following high
		// P-CODE for the above C code:
		//
		//		(unique res) CALL do_with_in (register RSP)
		//		--- RETURN 0 (unique res)
		//
		// Or:
		//
		//		(unique addr_of_x) PTRSUB (register RSP) (const NNN)
		//		(unique res) CALL do_with_in (unique addr_of_x)
		//		--- RETURN 0 (unique res)
		//
		// At first this is confusing: why not reference `x`? Why instead go
		// through the stack pointer register, `RSP`? The reason is that there
		// are no uses of the *value of x* in this code. Operations such as
		// `PTRSUB` operate on the address of things, and P-CODE doesn't
		// natively have an `ADDRESS_OF` operation (though we add one).
		//
		// The purpose of this method is to go and find the "real" `HighVariable`
		// if it exists by way of mining them from `MULTIEQUAL`, `COPY`, and
		// `INDIRECT`` operations, which exist to encode SSA form, as well as to
		// represent control-flow barriers in terms of data flow dependencies.
		void mineForVarNodes(PcodeOp pcodeOp) {
			for (Varnode inputVarnode : pcodeOp.getInputs()) {
				HighVariable highVariable = inputVarnode.getHigh();
				if (highVariable == null || !(highVariable instanceof HighLocal)) {
					continue;
				}

				HighSymbol highVariableSymbol = highVariable.getSymbol();
				String highVariableName = highVariable.getName();
				if (highVariableName == null || highVariableName.equals("UNNAMED")) {
					if (highVariableSymbol != null) {
						highVariableName = highVariableSymbol.getName();
					}
				}

				if (highVariableName == null || highVariableName.equals("UNNAMED")) {
					continue;
				}
				
				missingLocalsMap.put(highVariableName, (HighLocal) highVariable);
			}
		}

		// Create missing local variables. High p-code still includes things
		// like `PTRSUB SP, -offset` instead of treating the unrecognized data
		// as `local_<hex_offset>`. The decompiler, however, does these
		// automatic variable inventions.
		//
		// Returns `false` on failure.
		//
		// NOTE(pag): This function is very much inspired by the `MakeStackRefs`
		//		      script embedded in the Ghidra source.
		boolean fixUpMissingLocalVariables(
				HighFunction highFunction, int numberOfParameters) throws Exception {
			Function function = highFunction.getFunction();
			FunctionSignature signature = function.getSignature();
			FunctionPrototype proto = highFunction.getFunctionPrototype();
			LocalSymbolMap symbols = highFunction.getLocalSymbolMap();
			CallDepthChangeInfo cdci = new CallDepthChangeInfo(function);
			
			// Fill in the parameters first so that they are the first
			// things added to `entry_block`.
			for (int i = 0; i < numberOfParameters; ++i) {
				HighParam param = symbols.getParam(i);
				if (param == null) {
					HighSymbol paramHighSymbol = proto.getParam(i);
					param = new HighParam(paramHighSymbol.getDataType(), null, null, i, paramHighSymbol);
					missingLocalsMap.put(param.getName(), param);
				}

				createParamVarDecl(param);
			}

			// Now go look for operations directly referencing the stack pointer.
			for (PcodeBlockBasic block : highFunction.getBasicBlocks()) {
				Iterator<PcodeOp> pcodeOpIterator = block.getIterator();
				while (pcodeOpIterator.hasNext()) {
					PcodeOp pcodeOp = pcodeOpIterator.next();

					switch (pcodeOp.getOpcode()) {
						case PcodeOp.CALLOTHER:
							if (pcodeOp.getInput(0).getOffset() < MIN_CALLOTHER) {
								int index = (int) pcodeOp.getInput(0).getOffset();
								String userDefinedOpName = language.getUserDefinedOpName(index);
								if (userDefinedOpName != null) {
									callotherUsePcodeOps.add(pcodeOp);
								} else {
									System.out.println("Unsupported CALLOTHER at " + label(pcodeOp) + ": " + pcodeOp.toString());
									return false;
								}
							}
							break;
						case PcodeOp.CALL:
							// Rewrite call argument if there is type mismatch and can't be
							// handled during AST generation
							rewriteCallArgument(highFunction, pcodeOp);
							// Rewrite return type if the function retuns void but pcode op has 
							// valid output varnode.
							rewriteVoidReturnType(highFunction, pcodeOp);
							break;
						case PcodeOp.PTRSUB:
							if (!rewritePtrSubcomponent(highFunction, pcodeOp, cdci)) {
								System.out.println("Unsupported PTRSUB at " + label(pcodeOp) + ": " + pcodeOp.toString());
								return false;
							}
							break;
//						case PcodeOp.PTRADD:
//							if (!markPtrAddForElision(high_function, op)) {
//								println("Unsupported PTRADD at " + label(op) + ": " + op.toString());
//								return false;
//							}
//							break;
						case PcodeOp.MULTIEQUAL:
							if (!canElideMultiEqual(pcodeOp)) {
								System.out.println("Unsupported MULTIEQUAL at " + label(pcodeOp) + ": " + pcodeOp.toString());
								return false;
							}
							// Fall-through.
						case PcodeOp.INDIRECT:
							mineForVarNodes(pcodeOp);
							break;

						case PcodeOp.COPY:
							if (canElideCopy(pcodeOp)) {
								mineForVarNodes(pcodeOp);
								break;
							}
							// Fall-through.
						default:
							if (!tryFixupStackVarnode(highFunction, pcodeOp, cdci)) {
								System.out.println("Unsupported stack pointer reference at " + label(pcodeOp) + ": " + pcodeOp.toString());
								return false;
							}
							break;
					}
				}
			}

			return true;
		}
		
		HighVariable variableOf(HighVariable highVariable) {
			if (highVariable == null) {
				return null;
			}
			
			HighLocal fixedHighVariable = oldLocalsMap.get(highVariable);
			return fixedHighVariable != null ? fixedHighVariable : highVariable;
		}
		
		// Return the variable of a given `Varnode`. This applies local fixups.
		HighVariable variableOf(Varnode varnode) {
			return varnode == null ? null : variableOf(varnode.getHigh());
		}
		
		HighVariable variableOf(PcodeOp pcodeOp) {
			return variableOf(pcodeOp.getOutput());
		}

		// Handles serializing the output, if any, of `op`. We only actually
		// serialize the named outputs.
		void serializeOutput(PcodeOp pcodeOp) throws Exception {
			Varnode output = pcodeOp.getOutput();
			if (output == null) {
				return;
			}

			HighVariable outputHighVariable = variableOf(output);
			if (outputHighVariable != null) {
				writer.name("type").value(label(outputHighVariable.getDataType()));
			} else {
				writer.name("size").value(output.getSize());
			}
			
			// Only record an output node when the target is something named.
			// Otherwise, this p-code operation will be used as part of an
			// operand to something else.
			//
			// TODO(pag): Probably need some kind of verifier downstream to
			//			  ensure no code motion happens.
			VariableClassification klass = classifyVariable(outputHighVariable);
			switch (klass) {
				case PARAMETER:
				case LOCAL:
				case NAMED_TEMPORARY:
				case GLOBAL:
					break;
				default:
					return;
			}

			writer.name("output").beginObject();
			if (klass == VariableClassification.PARAMETER) {
				writer.name("kind").value("parameter");
				writer.name("operation").value(label(getOrCreateLocalVariable(outputHighVariable, pcodeOp)));
			} else if (klass == VariableClassification.LOCAL) {
				writer.name("kind").value("local");
				writer.name("operation").value(label(getOrCreateLocalVariable(outputHighVariable, pcodeOp)));				
			} else if (klass == VariableClassification.NAMED_TEMPORARY) {
				writer.name("kind").value("temporary");
				writer.name("operation").value(label(getOrCreateLocalVariable(outputHighVariable, pcodeOp)));
			} else if (klass == VariableClassification.GLOBAL) {
				writer.name("kind").value("global");
				writer.name("global").value(label(addressOfGlobal(outputHighVariable)));
			} else {
				assert false;
			}
 			writer.endObject();
		}
		
		// The address of a `LOAD` or `STORE` is spread across two operands:
		// the first being a constant representing the address space, and the
		// second being the actual address.
		void serializeLoadStoreAddress(PcodeOp pcodeOp) throws Exception {
			Varnode address = pcodeOp.getInput(1);
			if (!address.isConstant()) {
				serializeInput(pcodeOp, address);
				return;
			}
			
			Varnode addressSpaceVarnode = pcodeOp.getInput(0);
			assert addressSpaceVarnode.isConstant();

			writer.beginObject();
			writer.name("size").value(pcodeOp.getInput(1).getSize());
			System.out.println("!!! " + label(pcodeOp) + ": " + pcodeOp.toString());
			writer.endObject();
		}
		
		// Serialize a `LOAD space, address` op, eliding the address space.
		void serializeLoadOp(PcodeOp pcodeOp) throws Exception {
			serializeOutput(pcodeOp);
			writer.name("inputs").beginArray();
			serializeLoadStoreAddress(pcodeOp);
			writer.endArray();
		}
		
		// Serialize a `STORE space, address, value` op, eliditing the address
		// space.
		void serializeStoreOp(PcodeOp pcodeOp) throws Exception {
			serializeOutput(pcodeOp);
			writer.name("inputs").beginArray();
			serializeLoadStoreAddress(pcodeOp);
			serializeInput(pcodeOp, pcodeOp.getInput(2));
			writer.endArray();
		}
		
		// Product a new address in the `UNIQUE` address space.
		Address nextUniqueAddress() throws Exception {
			Address address = uniqueSpace.getAddress(nextUnique);
			nextUnique += uniqueSpace.getAddressableUnitSize();
			return address;
		}

		// Creates a pseudo p-code op using a `CALLOTHER` that logically
		// represents the definition of a parameter variable.
		//
		// NOTE(pag): The `HighParam` may have been invented and not have a
		//			  representative.
		//
		// TODO(pag): Do we want the `.getLength()` or `.getAlignedLength()`
		//			  for the parameter size in the absence of a representative?
		PcodeOp createParamVarDecl(HighVariable highVariable) throws Exception {
			HighParam param = (HighParam) highVariable;
			Address address = nextUniqueAddress();
			DefinitionVarnode definitionVarnode = new DefinitionVarnode(address, highVariable.getDataType().getAlignedLength());
			Varnode[] ins = new Varnode[2];
			SequenceNumber sequenceNumber = new SequenceNumber(address, nextSeqNum++);
			PcodeOp pcodeOp = new PcodeOp(sequenceNumber, PcodeOp.CALLOTHER, 2, definitionVarnode);
			pcodeOp.insertInput(new Varnode(constantSpace.getAddress(DECLARE_PARAM_VAR), 4), 0);
			pcodeOp.insertInput(new Varnode(constantSpace.getAddress(param.getSlot()), 4), 1);
			definitionVarnode.setDef(highVariable, pcodeOp);

			Varnode[] instances = highVariable.getInstances();
			Varnode[] newInstances = new Varnode[instances.length + 1];
			System.arraycopy(instances, 0, newInstances, 1, instances.length);
			newInstances[0] = definitionVarnode;

			highVariable.attachInstances(newInstances, definitionVarnode);

			entryBlock.add(pcodeOp);

			return pcodeOp;
		}

		// Creates a pseudo p-code op using a `CALLOTHER` that logically
		// represents the definition of a local variable.
		PcodeOp createLocalVariableDefinition(HighVariable highVariable) throws Exception {
			Address address = nextUniqueAddress();
			DefinitionVarnode definitionVarnode = new DefinitionVarnode(address, highVariable.getSize());
			Varnode[] ins = new Varnode[1];
			SequenceNumber sequenceNumber = new SequenceNumber(address, nextSeqNum++);
			PcodeOp pcodeOp = new PcodeOp(sequenceNumber, PcodeOp.CALLOTHER, 1, definitionVarnode);
			pcodeOp.insertInput(new Varnode(constantSpace.getAddress(DECLARE_LOCAL_VAR), 4), 0);
			definitionVarnode.setDef(highVariable, pcodeOp);

			Varnode[] instances = highVariable.getInstances();
			Varnode[] newInstances = new Varnode[instances.length + 1];
			System.arraycopy(instances, 0, newInstances, 1, instances.length);
			newInstances[0] = definitionVarnode;

			highVariable.attachInstances(newInstances, definitionVarnode);
			
			HighSymbol highSymbol = highVariable.getSymbol();
			entryBlock.add(pcodeOp);

			return pcodeOp;
		}

		// Creates a pseudo p-code op using a `CALLOTHER` that logically
		// represents the definition of a variable that stands in for a register.
		PcodeOp createNamedTemporaryRegisterVariableDeclaration(
				HighVariable highVariable, PcodeOp userPcodeOp) throws Exception {
			Varnode representativeVarnode = originalRepresentativeOf(highVariable);
			// assert representativeVarnode.isRegister();

			Address address = nextUniqueAddress();
			DefinitionVarnode definitionVarnode = new DefinitionVarnode(address, highVariable.getSize());
			Varnode[] ins = new Varnode[1];
			SequenceNumber sequenceNumber = new SequenceNumber(address, nextSeqNum++);
			PcodeOp pcodeOp = new PcodeOp(sequenceNumber, PcodeOp.CALLOTHER, 1, definitionVarnode);
			pcodeOp.insertInput(new Varnode(constantSpace.getAddress(DECLARE_TEMP_VAR), 4), 0);
			definitionVarnode.setDef(highVariable, pcodeOp);

			Varnode[] instances = highVariable.getInstances();
			Varnode[] newInstances = new Varnode[instances.length + 1];
			System.arraycopy(instances, 0, newInstances, 1, instances.length);
			newInstances[0] = definitionVarnode;

			highVariable.attachInstances(newInstances, definitionVarnode);

			entryBlock.add(pcodeOp);
			temporaryAddressMap.put(highVariable, userPcodeOp);

			return pcodeOp;
		}

		// Get or create a local variable pseudo definition op for the high
		// variable `var`.
		PcodeOp getOrCreateLocalVariable(
				HighVariable var, PcodeOp userPcodeOp) throws Exception {
			
			Varnode representative = var.getRepresentative();
			PcodeOp representativeDefOp = null;
			if (representative != null) {
				representativeDefOp = resolveOp(representative.getDef());			
				if (!isOriginalRepresentative(representative)) {
					return representativeDefOp;
				}
			}

			switch (classifyVariable(var)) {
				case PARAMETER:
					System.out.println("Creating late parameter for " + label(userPcodeOp) + ": " + userPcodeOp.toString());
					return createParamVarDecl(var);
				case LOCAL:
					return createLocalVariableDefinition(var);
				case NAMED_TEMPORARY:
					return createNamedTemporaryRegisterVariableDeclaration(var, userPcodeOp);
				default:
					break;
			}

			return representativeDefOp;
		}

		// Serialize a direct call. This enqueues the targeted for type lifting
		// `Function` if it can be resolved.
		void serializeCallOp(PcodeOp pcodeOp) throws Exception {
			Address callerAddress = currentFunction.getFunction().getEntryPoint();
			Varnode targetNode = pcodeOp.getInput(0);
			Function callee = null;

			if (targetNode.isAddress()) {
				Address targetAddress = callerAddress.getNewAddress(
						targetNode.getOffset());
				FunctionManager fm = currentProgram.getFunctionManager();
				callee = fm.getFunctionAt(targetAddress);
	
				// `target_address` may be a pointer to an external. Figure out
				// what we're calling.
				if (callee == null) {
					callee = fm.getReferencedFunction(targetAddress);	
				}
			}

			boolean hasReturnValue = pcodeOp.getOutput() != null || (callee != null && callee.getReturnType() != VoidDataType.dataType);
			writer.name("has_return_value").value(hasReturnValue);

			writer.name("target");
			if (callee != null) {

				functions.add(callee);

				writer.beginObject();
				writer.name("kind").value("function");
				writer.name("function").value(label(callee));
				writer.name("is_variadic").value(callee.hasVarArgs());
				writer.name("is_noreturn").value(callee.hasNoReturn());
				writer.endObject();

			} else {
				serializeInput(pcodeOp, targetNode);
			}
			
			writer.name("inputs").beginArray();			
			Varnode[] inputs = pcodeOp.getInputs();
			for (int i = 1; i < inputs.length; ++i) {
				serializeInput(pcodeOp, inputs[i]);
			}
			writer.endArray();
		}

		// Serialize an unconditional branch. This records the targeted block.
		void serializeBranchOp(PcodeOp pcodeOp) throws Exception {
			assert currentBlock.getOutSize() == 1;
			writer.name("target_block").value(label(currentBlock.getOut(0)));
		}

		// Serialize a conditional branch. This records the targeted blocks.
		//
		// TODO(pag): How does p-code handle delay slots? Are they separate
		//          blocks?
		//
		// XREF(pag): https://github.com/NationalSecurityAgency/ghidra/issues/2736
		//        describes how the `op` meaning of the branch, i.e. whether
		//        it branches on zero or not zero, can change over the course
		//        of simplification, and so the inputs representing the
		//        branch targets may not actually represent the `true` or
		//        `false` outputs in the traditional sense.
		void serializeCondBranchOp(PcodeOp pcodeOp) throws Exception {
			writer.name("taken_block").value(label(currentBlock.getTrueOut()));
			writer.name("not_taken_block").value(label(currentBlock.getFalseOut()));
			writer.name("condition");
			serializeInput(pcodeOp, pcodeOp.getInput(1));
		}

	// Walk backward through def-use chains, skipping mechanical transformations
	// (CAST, COPY, INT_ZEXT, INT_SEXT, INDIRECT), to reach the original integer
	// discriminant varnode (parameter, local, or call result).
	private Varnode traceSwitchDiscriminant(Varnode v) {
		int depth = 0;
		while (v != null && depth < 8) {
			PcodeOp def = v.getDef();
			if (def == null) {
				return v;  // param/local — done
			}
			int opc = def.getOpcode();
			if (opc == PcodeOp.CAST     || opc == PcodeOp.COPY   ||
				opc == PcodeOp.INT_ZEXT || opc == PcodeOp.INT_SEXT ||
				opc == PcodeOp.INDIRECT) {
				v = def.getInput(0);
				depth++;
			} else {
				return v;  // LOAD, INT_ADD, etc. — stop
			}
		}
		return v;
	}

	// Find the default block for a BRANCHIND.  The bounds-check block is the
	// BRANCHIND's predecessor that has exactly two exits: one into the BRANCHIND
	// block and one to the default/error block.
	private String findDefaultBlock() throws Exception {
		for (int i = 0; i < currentBlock.getInSize(); i++) {
			PcodeBlock pred = currentBlock.getIn(i);
			if (pred.getOutSize() == 2) {
				PcodeBlock out0 = pred.getOut(0);
				PcodeBlock out1 = pred.getOut(1);
				if (out0 == currentBlock) {
					return label(out1);
				}
				if (out1 == currentBlock) {
					return label(out0);
				}
			}
		}
		return null;
	}

	// Returns true if `v` traces through CAST/COPY/INT_ZEXT/INT_SEXT/INDIRECT
	// to the same varnode as `disc` (i.e. they represent the same discriminant).
	private boolean sameDiscriminant(Varnode v, Varnode disc) {
		int depth = 0;
		while (v != null && depth < 4) {
			if (v.equals(disc)) {
				return true;
			}
			PcodeOp def = v.getDef();
			if (def == null) {
				return false;
			}
			int opc = def.getOpcode();
			if (opc == PcodeOp.CAST     || opc == PcodeOp.COPY      ||
				opc == PcodeOp.INT_ZEXT || opc == PcodeOp.INT_SEXT  ||
				opc == PcodeOp.INDIRECT) {
				v = def.getInput(0);
				depth++;
			} else {
				return false;
			}
		}
		return false;
	}

	// Tier 1: scan the first few P-Code ops of `block` looking for a comparison
	// of the discriminant against a constant (INT_EQUAL, INT_NOTEQUAL, INT_LESS,
	// INT_LESSEQUAL, INT_SLESS, INT_SLESSEQUAL).  Returns the constant (case
	// value) or null if the pattern is not found within the first 4 ops.
	private Long recoverCaseValueFromBlockEntry(PcodeBlockBasic block, Varnode discriminant) {
		Iterator<PcodeOp> it = block.getIterator();
		int scanned = 0;
		while (it.hasNext() && scanned < 4) {
			PcodeOp op = it.next();
			int opc = op.getOpcode();
			if (opc == PcodeOp.INT_EQUAL     || opc == PcodeOp.INT_NOTEQUAL  ||
				opc == PcodeOp.INT_LESS      || opc == PcodeOp.INT_LESSEQUAL ||
				opc == PcodeOp.INT_SLESS     || opc == PcodeOp.INT_SLESSEQUAL) {
				Varnode lhs = op.getInput(0);
				Varnode rhs = op.getInput(1);
				if (rhs.isConstant() && sameDiscriminant(lhs, discriminant)) {
					return rhs.getOffset();
				}
				if (lhs.isConstant() && sameDiscriminant(rhs, discriminant)) {
					return lhs.getOffset();
				}
			}
			scanned++;
		}
		return null;
	}

	// Tier 0 (most reliable): Ghidra's switch analysis labels each case-target
	// block with a symbol like "caseD_HEX" (possibly in a namespace such as
	// "switchD_ADDR::caseD_HEX").  Read the primary symbol at the block start
	// address and parse the hex value after "caseD_".  Returns null when the
	// block has no such label (e.g. a user-renamed block or a non-switch block).
	private Long recoverCaseValueFromBlockLabel(PcodeBlock block) {
		Address addr = block.getStart();
		Symbol sym = currentProgram.getSymbolTable().getPrimarySymbol(addr);
		if (sym == null) {
			return null;
		}
		// getName() returns the local name without the namespace prefix.
		String name = sym.getName();
		int idx = name.indexOf("caseD_");
		if (idx < 0) {
			return null;
		}
		String hexPart = name.substring(idx + "caseD_".length());
		try {
			return Long.parseLong(hexPart, 16);
		} catch (NumberFormatException e) {
			return null;
		}
	}

	// Serialize an indirect branch (BRANCHIND). This records the computed
	// target expression and, when Ghidra has resolved the jump table, the
	// set of possible successor blocks, the default block, and the original
	// integer discriminant with per-case integer values.
	void serializeBranchIndOp(PcodeOp pcodeOp) throws Exception {
		writer.name("inputs").beginArray();
		serializeInput(pcodeOp, pcodeOp.getInput(0));
		writer.endArray();

		int n = currentBlock.getOutSize();
		if (n == 0) {
			return;
		}

		writer.name("successor_blocks").beginArray();
		for (int i = 0; i < n; i++) {
			writer.value(label(currentBlock.getOut(i)));
		}
		writer.endArray();

		// Emit the fallback block recovered from the bounds-check predecessor.
		// The C++ consumer jumps to this block when no switch case matches.
		String defaultBlock = findDefaultBlock();
		if (defaultBlock != null) {
			writer.name("fallback_block").value(defaultBlock);
		}

		JumpTable jt = jumpTableIndex.get(pcodeOp.getSeqnum().getTarget());
		Varnode discriminant = (jt != null) ? traceSwitchDiscriminant(pcodeOp.getInput(0)) : null;
		if (jt != null && discriminant != null) {
			serializeSwitchCases(pcodeOp, discriminant, jt, n);
		}
	}


	// Emits "switch_input" and "switch_cases" JSON fields for a resolved BRANCHIND.
	// Case values are recovered using a 3-tier strategy:
	//   Tier 0a – decompiler's own integer label values from JumpTable API (most authoritative).
	//   Tier 0  – Ghidra's own "caseD_HEX" block labels.
	//   Tier 1  – block-entry P-Code ops that compare the discriminant against a constant.
	// Both fields are omitted entirely when no tier produces a complete case map,
	// allowing the C++ side to fall back to successor_blocks-based dispatch.
	private void serializeSwitchCases(PcodeOp pcodeOp, Varnode discriminant,
			JumpTable jt, int n) throws Exception {

		Map<String,Long> caseMap = new HashMap<>();
		boolean caseMapOk = false;

		// Tier 0a: use the decompiler's own integer label values — most authoritative source.
		// getLabelValues()[i] and getCases()[i] are aligned: the i-th label routes to the
		// i-th case address.
		Integer[] labels = jt.getLabelValues();
		Address[] cases  = jt.getCases();
		if (labels != null && cases != null && labels.length > 0
				&& labels.length == cases.length) {
			for (int i = 0; i < labels.length; i++) {
				for (int j = 0; j < n; j++) {
					if (currentBlock.getOut(j).getStart().equals(cases[i])) {
						caseMap.put(label(currentBlock.getOut(j)), (long)(int) labels[i]);
						break;
					}
				}
			}
			caseMapOk = !caseMap.isEmpty();
		}

		// Tier 0: read Ghidra's own "caseD_HEX" block labels — reliable for any
		// compiled switch since Ghidra's analysis stores the correct case values
		// in the symbol name.
		if (!caseMapOk) {
			caseMap = new HashMap<>();
			caseMapOk = true;
			for (int i = 0; i < n; i++) {
				PcodeBlock blk = currentBlock.getOut(i);
				Long v = recoverCaseValueFromBlockLabel(blk);
				if (v == null) {
					caseMapOk = false;
					break;
				}
				caseMap.put(label(blk), v);
			}
		}

		// Tier 1: scan first P-Code ops at each case-target block for a
		// comparison of the discriminant against a constant.  Handles
		// computed-goto dispatch where blocks start with INT_EQUAL etc.
		if (!caseMapOk) {
			caseMap = new HashMap<>();
			caseMapOk = true;
			for (int i = 0; i < n; i++) {
				PcodeBlock blk = currentBlock.getOut(i);
				if (!(blk instanceof PcodeBlockBasic)) {
					caseMapOk = false;
					break;
				}
				Long v = recoverCaseValueFromBlockEntry((PcodeBlockBasic) blk, discriminant);
				if (v == null) {
					caseMapOk = false;
					break;
				}
				caseMap.put(label(blk), v);
			}
		}

		// If no tier recovered real case values, or the map is partial,
		// emit nothing — the C++ side will fall back to successor_blocks (Priority 2).
		if (!caseMapOk || caseMap.size() != n) {
			return;
		}

		writer.name("switch_input");
		serializeInput(pcodeOp, discriminant);
		writer.name("switch_cases").beginArray();
		for (int i = 0; i < n; i++) {
			String blockLabel = label(currentBlock.getOut(i));
			Long caseVal = caseMap.get(blockLabel);
			if (caseVal == null) { continue; }  // defensive: skip unmapped blocks
			writer.beginObject();
			writer.name("value").value(caseVal);
			writer.name("target_block").value(blockLabel);
			writer.endObject();
		}
		writer.endArray();
	}

		// Serialize a generic multi-input, single-output p-code operation.
		void serializeGenericOp(PcodeOp pcodeOp) throws Exception {
			writer.name("inputs").beginArray();
			for (Varnode inputVarnode : pcodeOp.getInputs()) {
				serializeInput(pcodeOp, inputVarnode);
			}
			writer.endArray();
		}

		// Serializes a pseudo-op `DECLARE_PARAM_VAR`, which is actually encoded
		// as a `CALLOTHER`.
		void serializeDeclareParamVar(PcodeOp pcodeOp) throws Exception {
			HighVariable highVariableOfPcodeOp = variableOf(pcodeOp);
			writer.name("name").value(highVariableOfPcodeOp.getName());
			writer.name("type").value(label(highVariableOfPcodeOp.getDataType()));
			writer.name("kind").value("parameter");  // So that it also looks like an input/output.
			if (highVariableOfPcodeOp instanceof HighParam) {
				writer.name("index").value(((HighParam) highVariableOfPcodeOp).getSlot());
			}
		}

		// Serializes a pseudo-op `DECLARE_LOCAL_VAR`, which is actually encoded
		// as a `CALLOTHER`.
		void serializeDeclareLocalVar(PcodeOp pcodeOp) throws Exception {
			HighVariable highVariableOfPcodeOp = variableOf(pcodeOp);
			HighSymbol highSymbol = highVariableOfPcodeOp.getSymbol();
			writer.name("kind").value("local");  // So that it also looks like an input/output.
			if (highSymbol != null && highVariableOfPcodeOp.getOffset() == -1 && highVariableOfPcodeOp.getName().equals("UNNAMED")) {
				writer.name("name").value(highSymbol.getName());
				writer.name("type").value(label(highSymbol.getDataType()));
			} else {
				writer.name("name").value(highVariableOfPcodeOp.getName());
				writer.name("type").value(label(highVariableOfPcodeOp.getDataType()));
			}
		}
		
		void serializeDeclareNamedTemporary(PcodeOp pcodeOp) throws Exception {
			HighVariable highVariableOfPcodeOp = variableOf(pcodeOp);
			Varnode representativeVarnode = originalRepresentativeOf(highVariableOfPcodeOp);

			// NOTE(pag): In practice, the `HighOther`s name associated with
			//			  this register is probably `UNNAMED`, which happens in
			//			  `HighOther.decode`; however, we'll be cautious and
			// 			  only canonicalize on the register name if the name
			//			  it is the default.
			if (highVariableOfPcodeOp.getName().equals("UNNAMED")) {
				if (representativeVarnode.isRegister()) {
					Register reg = language.getRegister(representativeVarnode.getAddress(), 0);
					if (reg != null) {
						writer.name("name").value(reg.getName());
					} else {
						writer.name("name").value("reg" + Address.SEPARATOR +
										   Long.toHexString(representativeVarnode.getOffset()));
					}
				} else {
					writer.name("name").value("temp");
				}
			} else {
				writer.name("name").value(highVariableOfPcodeOp.getName());
			}

			writer.name("kind").value("temporary");  // So that it also looks like an input/output.
			writer.name("type").value(label(highVariableOfPcodeOp.getDataType()));
			
			// NOTE(pag): The same register might appear multiple times, though
			//			  we can't guarantee that they will appear with the
			//			  same names. Thus, we want to record the address of
			//            the operation using the original register as a kind of
			//			  SSA-like version number downstream, e.g. in a Clang
			//			  AST.
			PcodeOp userPcodeOp = temporaryAddressMap.get(highVariableOfPcodeOp);
			if (userPcodeOp != null) {
				writer.name("address").value(label(userPcodeOp));
			}
		}
		
		// Serialize an `ADDRESS_OF`, used to the get the address of a local or
		// global variable. These are created from `PTRSUB` nodes.
		void serializeAddressOfOp(PcodeOp pcodeOp) throws Exception {
			serializeOutput(pcodeOp);
			writer.name("inputs").beginArray();
			serializeInput(pcodeOp, pcodeOp.getInputs()[1]);
			writer.endArray();
		}
		
		// Serialize a `CALLOTHER` as a call to an intrinsic.
		void serializeIntrinsicCallOp(PcodeOp pcodeOp) throws Exception {
			serializeOutput(pcodeOp);
			
			writer.name("target").beginObject();
			writer.name("kind").value("intrinsic");
			writer.name("function").value(intrinsicLabel(pcodeOp));
			writer.name("is_variadic").value(true);
			writer.name("is_noreturn").value(false);
			writer.endObject();  // End of `target`.
			
			writer.name("inputs").beginArray();			
			Varnode[] inputs = pcodeOp.getInputs();
			for (int i = 1; i < inputs.length; ++i) {
				serializeInput(pcodeOp, inputs[i]);
			}
			writer.endArray();
		}

		// Serialize a `CALLOTHER`. The first input operand is a constant
		// representing the user-defined opcode number. In our case, we have
		// our own user-defined opcodes for making things better mirror the
		// structure/needs of MLIR.
		void serializeCallOtherOp(PcodeOp pcodeOp) throws Exception {
			switch ((int) pcodeOp.getInput(0).getOffset()) {
			case DECLARE_PARAM_VAR:
				serializeDeclareParamVar(pcodeOp);
				break;
			case DECLARE_LOCAL_VAR:
				serializeDeclareLocalVar(pcodeOp);
				break;
			case DECLARE_TEMP_VAR:
				serializeDeclareNamedTemporary(pcodeOp);
				break;
			case ADDRESS_OF:
				serializeAddressOfOp(pcodeOp);
				break;
			default:
				serializeIntrinsicCallOp(pcodeOp);
				break;
			}
		}
		
		// Serialize a `RETURN N[, val]` as logically being `RETURN val`.
		void serializeReturnOp(PcodeOp pcodeOp) throws Exception {
			Varnode inputs[] = pcodeOp.getInputs();
			writer.name("inputs").beginArray();
			if (inputs.length == 2) {
				serializeInput(pcodeOp, inputs[1]);
			}
			writer.endArray();
		}

		// Get the mnemonic for a p-code operation. We have some custom
		// operations encoded as `CALLOTHER`s, so we get their names manually
		// here.
		//
		// TODO(pag): There is probably a way to register the name of a
		//        `CALLOTHER` via `Language.getSymbolTable()` using a
		//        `UseropSymbol`. It's not clear if there's really value in
		//        doing this, though.
		static String mnemonic(PcodeOp pcodeOp) {
			if (pcodeOp.getOpcode() == PcodeOp.CALLOTHER) {
				switch ((int) pcodeOp.getInput(0).getOffset()) {
				case DECLARE_PARAM_VAR:
					return "DECLARE_PARAMETER";
				case DECLARE_LOCAL_VAR:
					return "DECLARE_LOCAL";
				case DECLARE_TEMP_VAR:
					return "DECLARE_TEMPORARY";
				case ADDRESS_OF:
					return "ADDRESS_OF";
				default:
					break;
				}
			}
			return pcodeOp.getMnemonic();
		}

		void serializePcodeOp(PcodeOp pcodeOp) throws Exception {
			writer.beginObject();
			writer.name("mnemonic").value(mnemonic(pcodeOp));
			
			switch (pcodeOp.getOpcode()) {
				case PcodeOp.CALL:
				case PcodeOp.CALLIND:
					serializeOutput(pcodeOp);
					serializeCallOp(pcodeOp);
					break;
				case PcodeOp.CALLOTHER:
					serializeCallOtherOp(pcodeOp);
					break;
				case PcodeOp.CBRANCH:
					serializeCondBranchOp(pcodeOp);
					break;
				case PcodeOp.BRANCH:
					serializeBranchOp(pcodeOp);
					break;
				case PcodeOp.BRANCHIND:
					serializeBranchIndOp(pcodeOp);
					break;
				case PcodeOp.RETURN:
					serializeReturnOp(pcodeOp);
					break;
				case PcodeOp.LOAD:
					serializeLoadOp(pcodeOp);
					break;
				case PcodeOp.STORE:
					serializeStoreOp(pcodeOp);
					break;
//				case PcodeOp.COPY:
//				case PcodeOp.CAST:
				default:
					serializeOutput(pcodeOp);
					serializeGenericOp(pcodeOp);
					break;
			}

			writer.endObject();
		}
		
		// Returns `true` if we can elide a `MULTIEQUAL` operation. If all
		// inputs are of the identical `HighVariable`, then we can elide.
		boolean canElideMultiEqual(PcodeOp pcodeOp) throws Exception {
			HighVariable highVariableOfPcodeOp = variableOf(pcodeOp);
			if (highVariableOfPcodeOp == null) {
				return false;
			}

			for (Varnode inputVarnode : pcodeOp.getInputs()) {
				if (highVariableOfPcodeOp != variableOf(inputVarnode)) {
					return false;
				}
			}
			
			return true;
		}
		
		// Returns `true` if we can elide a copy operation. This only happens
		// when we copy a variable into itself.
		//
		// NOTE(pag): I think this comes about as a sort of "pre-PHI" operation.
		//
		// TODO(pag): This is toally unsafe if there's an intervening write to
		//			  relevant variable. Probably should investigate this case.
		boolean canElideCopy(PcodeOp pcodeOp) throws Exception {
			HighVariable highVariableOfPcodeOp = variableOf(pcodeOp);
			return highVariableOfPcodeOp != null && highVariableOfPcodeOp == variableOf(pcodeOp.getInput(0));
		}
		
		// Returns `true` if `op` is a branch operator.
		static boolean isBranch(PcodeOp pcodeOp) throws Exception {
			switch (pcodeOp.getOpcode()) {
				case PcodeOp.BRANCH:
				case PcodeOp.CBRANCH:
				case PcodeOp.BRANCHIND:
					return true;
				default:
					return false;
			}
		}
		
		// Serialize a high p-code basic block. This iterates over the p-code
		// operations within the block and serializes them individually.
		void serializePcodeBasicBlock(PcodeBlockBasic block) throws Exception {
			PcodeBlock parentBlock = block.getParent();
			if (parentBlock != null) {
				writer.name("parent_block").value(label(parentBlock));
			}
			
			boolean lastIsBranch = false;
			Iterator<PcodeOp> pcodeOpIterator = block.getIterator();
			ArrayList<PcodeOp> orderedPcodeOps = new ArrayList<>();
			
			while (pcodeOpIterator.hasNext()) {
				PcodeOp pcodeOp = resolveOp(pcodeOpIterator.next());
				
				// Inject the prefix operations into the ordered operations
				// list. These are to handle things like stack pointer
				// references flowing into `CALL` arguments.
				List<PcodeOp> prefixOps = prefixOperationsMap.get(pcodeOp);
				if (prefixOps != null) {
					for (PcodeOp prefixOp : prefixOps) {
						orderedPcodeOps.add(prefixOp);
					}
				}
				
				switch (pcodeOp.getOpcode()) {
					// NOTE(pag): INDIRECTs seem like a good way of modelling
					//            may- alias relations, as well as embedding
					//            control dependencies into the dataflow graph,
					//            e.g. to ensure code motion cannot happen from
					//            after a CALL to before a CALL, especially for
					//            stuff operating on stack slots. The idea at
					//            the time of this comment is that we will
					//            assume that eventual codegen also should not
					//            do any reordering, though enforcing that is
					//			  also tricky.
					case PcodeOp.INDIRECT:
						continue;
					
					// MULTIEQUALs are Ghidra's form of SSA-form PHI nodes.
					case PcodeOp.MULTIEQUAL:
						if (canElideMultiEqual(pcodeOp)) {
							continue;
						}
						break;
					
					// Some copies end up imlpementing the kind of forward edge
					// of a phi node (i.e. `MULTIEQUAL`) and can be elided.
					case PcodeOp.COPY:
						if (canElideCopy(pcodeOp)) {
							continue;
						}
						break;
	
					default:
						break;
				}

				orderedPcodeOps.add(pcodeOp);
			}
			
			// Serialize the operations.
			writer.name("operations").beginObject();
			for (PcodeOp pcodeOp : orderedPcodeOps) {
				writer.name(label(pcodeOp));
				currentBlock = block;
				serializePcodeOp(pcodeOp);
				lastIsBranch = isBranch(pcodeOp);
				currentBlock = null;
			}
			
			// Synthesize a fake `BRANCH` operation to the fall-through block.
			// We'll have a fall-through if we don't already end in a branch,
			// and if the last operation isn't a `RETURN` or a `CALL*` to a
			// `noreturn`-attributed function.
			String fallThroughLabel = "";
			if (!lastIsBranch && block.getOutSize() == 1) {
				fallThroughLabel = label(block) + ".exit";
				writer.name(fallThroughLabel).beginObject();
				writer.name("mnemonic").value("BRANCH");
				writer.name("target_block").value(label(block.getOut(0)));
				writer.endObject();  // End of BRANCH to `first_block`.	
			}
			
			writer.endObject();  // End of `operations`.

			// List out the operations in their order.
			writer.name("ordered_operations").beginArray();
			for (PcodeOp op : orderedPcodeOps) {
				writer.value(label(op));
			}
			if (!fallThroughLabel.equals("")) {
				writer.value(fallThroughLabel);
			}
			writer.endArray();  // End of `ordered_operations`.
		}

		// Emit a pseudo entry block to represent
		void serializeEntryBlock(
				String label, PcodeBlockBasic firstPcodeBasicBlock) throws Exception {
			writer.name(label).beginObject();
			writer.name("operations").beginObject();
			for (PcodeOp pseudoPcodeOp : entryBlock) {
				writer.name(label(pseudoPcodeOp));
				serializePcodeOp(pseudoPcodeOp);
			}

			// If there is a proper entry block, then invent a branch to it.
			if (firstPcodeBasicBlock != null) {
				writer.name("entry.exit").beginObject();
				writer.name("mnemonic").value("BRANCH");
				writer.name("target_block").value(label(firstPcodeBasicBlock));
				writer.endObject();  // End of BRANCH to `first_block`.
			}

			writer.endObject();  // End of operations.

			writer.name("ordered_operations").beginArray();
			for (PcodeOp pseudoPcodeOp : entryBlock) {
				writer.value(label(pseudoPcodeOp));
			}
			writer.value("entry.exit");
			writer.endArray();  // End of `ordered_operations`.
			writer.endObject();  // End of `entry` block.
		}
		
		// Serialize `function`. If we have `high_function` (the decompilation
		// of function) then we will serialize its type information. Otherwise,
		// we will serialize the type information of `functionToSerialize`. If
		// `shouldVisitPcode` is true, then this is a function for which we want to
		// fully lift, i.e. visit all the high p-code.
		void serializeFunction(
				HighFunction highFunction, Function functionToSerialize,
				boolean shouldVisitPcode) throws Exception {
			
			temporaryAddressMap.clear();
			oldLocalsMap.clear();
			missingLocalsMap.clear();
			entryBlock.clear();
			replacementOperationsMap.clear();
			prefixOperationsMap.clear();

			FunctionPrototype functionPrototype = null;
			writer.name("name").value(functionToSerialize.getName());
			writer.name("is_intrinsic").value(false);

			// If we have a high P-Code function, then serialize the blocks.
			if (highFunction != null) {
				functionPrototype = highFunction.getFunctionPrototype();

				writer.name("type").beginObject();
				
				int numberOfParams = 0;
				if (functionPrototype != null) {
					numberOfParams = serializePrototype(functionPrototype);
				} else {
					numberOfParams = serializePrototype(functionToSerialize.getSignature());
				}
				writer.endObject();  // End `type`.

				if (shouldVisitPcode && fixUpMissingLocalVariables(highFunction, numberOfParams)) {
					
					String entryLabel = null;
					PcodeBlockBasic firstPcodeBasicBlock = null;
					currentFunction = highFunction;
					jumpTableIndex = new HashMap<>();
					for (JumpTable jt : currentFunction.getJumpTables()) {
						jumpTableIndex.put(jt.getSwitchAddress(), jt);
					}

					writer.name("basic_blocks").beginObject();
					for (PcodeBlockBasic basicBlock : highFunction.getBasicBlocks()) {
						if (firstPcodeBasicBlock == null) {
							firstPcodeBasicBlock = basicBlock;
						}

						writer.name(label(basicBlock)).beginObject();
						serializePcodeBasicBlock(basicBlock);
						writer.endObject();
					}

					// If we created a fake entry block to represent variable
					// declarations then emit that here.
					if (!entryBlock.isEmpty()) {
						entryLabel = entryBlockLabel();
						serializeEntryBlock(entryLabel, firstPcodeBasicBlock);
					}
					
					writer.endObject();  // End of `basic_blocks`.
					currentFunction = null;
					jumpTableIndex = null;
					
					if (entryLabel != null) {
						writer.name("entry_block").value(entryLabel);

					} else if (firstPcodeBasicBlock != null) {
						writer.name("entry_block").value(label(firstPcodeBasicBlock));
					}
				}
			} else {
				writer.name("type").beginObject();
				serializePrototype(functionToSerialize.getSignature());
				writer.endObject();  // End `type`.
			}
		}
		
		String entryBlockLabel() throws Exception {
			return label(currentFunction) +  Address.SEPARATOR + "entry";
		}
		
		// Serialize the global variable declarations.
		void serializeGlobals() throws Exception {
			for (Map.Entry<Address, HighVariable> entry : seenGlobalsMap.entrySet()) {
				Address address = entry.getKey();
				HighVariable globalHighVariable = entry.getValue();
				
				// Try to get the global's name.
				String globalVariableName = globalHighVariable.getName();
				if (globalVariableName == null || (globalVariableName.equals("UNNAMED") && globalHighVariable.getOffset() == -1)) {
					HighSymbol globalVariableHighSymbol = globalHighVariable.getSymbol();
					if (globalVariableHighSymbol != null) {
						globalVariableName = globalVariableHighSymbol.getName();
					}
				}

				writer.name(label(address)).beginObject();
				writer.name("name").value(globalVariableName);
				writer.name("size").value(Integer.toString(globalHighVariable.getSize()));
				writer.name("type").value(label(globalHighVariable.getDataType()));
				writer.endObject();
			}

			for (Map.Entry<Address, Data> entry : seenDataMap.entrySet()) {
				Address address = entry.getKey();
				Data datavar = entry.getValue();
				String name = datavar.getDefaultLabelPrefix(null) + "_"
					+ SymbolUtilities.replaceInvalidChars(datavar.getDefaultValueRepresentation(), false)
					+ "_" + datavar.getMinAddress().toString();

				writer.name(label(address)).beginObject();
				writer.name("name").value(name);
				writer.name("size").value(Integer.toString(datavar.getDataType().getLength()));
				writer.name("type").value(label(datavar.getDataType()));
				writer.endObject();


			}

			System.out.println("Total serialized globals: " + Integer.toString(seenGlobalsMap.size()));
		}

		// Don't try to decompile some functions.
		private static final Set<String> IGNORED_NAMES = Set.of(
			"_start", "__libc_csu_fini", "__libc_csu_init", "__libc_start_main",
            "__data_start", "__dso_handle", "_IO_stdin_used",
            "_dl_relocate_static_pie", "__DTOR_END__", "__ashlsi3",
            "__ashldi3", "__ashlti3", "__ashrsi3", "__ashrdi3", "__ashrti3",
            "__divsi3", "__divdi3", "__divti3", "__lshrsi3", "__lshrdi3",
            "__lshrti3", "__modsi3", "__moddi3", "__modti3", "__mulsi3",
            "__muldi3", "__multi3", "__negdi2", "__negti2", "__udivsi3",
            "__udivdi3", "__udivti3", "__udivmoddi4", "__udivmodti4",
            "__umodsi3", "__umoddi3", "__umodti3", "__cmpdi2", "__cmpti2",
            "__ucmpdi2", "__ucmpti2", "__absvsi2", "__absvdi2", "__addvsi3",
            "__addvdi3", "__mulvsi3", "__mulvdi3", "__negvsi2", "__negvdi2",
            "__subvsi3", "__subvdi3", "__clzsi2", "__clzdi2", "__clzti2",
            "__ctzsi2", "__ctzdi2", "__ctzti2", "__ffsdi2", "__ffsti2",
            "__paritysi2", "__paritydi2", "__parityti2", "__popcountsi2",
            "__popcountdi2", "__popcountti2", "__bswapsi2", "__bswapdi2",
            "frame_dummy", "call_frame_dummy", "__do_global_dtors",
            "__do_global_dtors_aux", "call___do_global_dtors_aux",
            "__do_global_ctors", "__do_global_ctors_1", "__do_global_ctors_aux",
            "call___do_global_ctors_aux", "__gmon_start__", "_init_proc",
            ".init_proc", "_term_proc", ".term_proc", "__uClibc_main",
            "abort", "exit", "_Exit", "panic", "terminate",
            "_Jv_RegisterClasses",
            "__deregister_frame_info_bases", "__deregister_frame_info",
            "__register_frame_info", "__cxa_throw", "__cxa_finalize",
            "__cxa_allocate_exception", "__cxa_free_exception",
            "__cxa_begin_catch", "__cxa_end_catch", "__cxa_new_handler",
            "__cxa_get_globals", "__cxa_get_globals_fast",
            "__cxa_current_exception_type", "__cxa_rethrow", "__cxa_bad_cast",
            "__cxa_bad_typeid", "__allocate_exception", "__throw",
            "__free_exception",
            "__Unwind_RaiseException", "_Unwind_RaiseException", "_Unwind_Resume",
            "_Unwind_DeleteException", "_Unwind_GetGR", "_Unwind_SetGR",
            "_Unwind_GetIP", "_Unwind_SetIP", "_Unwind_GetRegionStart",
            "_Unwind_GetLanguageSpecificData", "_Unwind_ForcedUnwind",
            "__unw_getcontext", 
            "longjmp", "siglongjmp", "setjmp", "sigsetjmp",
            "__register_frame_info_bases", "__assert_fail",
            "_init", "_fini", "_ITM_registerTMCloneTable",
            "_ITM_deregisterTMCloneTable", "register_tm_clones",
            "deregister_tm_clones", "__sinit"
		);

		// Serialize all `CALLOTHER` intrinsics.
		void serializeIntrinsics() throws Exception {
			Set<String> seenIntrinsics = new HashSet<>();
			int numIntrinsics = 0;
			
			for (PcodeOp pcodeOp : callotherUsePcodeOps) {
				int index = (int) pcodeOp.getInput(0).getOffset();
				String name = language.getUserDefinedOpName(index);
				DataType returnType = intrinsicReturnType(pcodeOp);
				String label = intrinsicLabel(name, returnType);
				if (!seenIntrinsics.add(label)) {
					continue;
				}
				
				writer.name(label).beginObject();
				writer.name("name").value(name);
				writer.name("is_intrinsic").value(true);
				writer.name("type").beginObject();
				writer.name("return_type").value(label(returnType));
				writer.name("is_variadic").value(true);
				writer.name("is_noreturn").value(false);
				writer.name("parameter_types").beginArray().endArray();
				writer.endObject();  // End of `type`.
				writer.endObject();
				
				++numIntrinsics;
			}
			
			System.out.println("Total serialized intrinsics: " + Integer.toString(numIntrinsics));
		}
		
		// Serialize all functions.
		//
		// NOTE(pag): As we serialize functions, we might discover references
		//			  to other functions, causing `functions` will grow over
		// 			  time.
		void serializeFunctions() throws Exception {
			for (int i = 0; i < functions.size(); ++i) {
				Function function = functions.get(i);
				String functionLabel = label(function);
				if (!seenFunctions.add(functionLabel)) {
					continue;
				}
				
				boolean shouldVisitPcode = i < originalFunctionsSize &&
									  !function.isThunk() &&
									  !IGNORED_NAMES.contains(function.getName());

				DecompileResults functionDecompResults = decompInterface.decompileFunction(function, DECOMPILATION_TIMEOUT, this.monitor);
				HighFunction highFunction = functionDecompResults.getHighFunction();
				writer.name(functionLabel).beginObject();
				serializeFunction(highFunction, function, shouldVisitPcode);
				writer.endObject();
			}

			System.out.println("Total serialized functions: " + Integer.toString(functions.size()));
			
			if (!callotherUsePcodeOps.isEmpty()) {
				serializeIntrinsics();
			}
		}

		// Serialize the input function list to JSON. This function will also
		// serialize type information related to referenced functions and
		// variables.
		public void serialize() throws Exception {

			writer.beginObject();
			writer.name("architecture").value(this.architecture);
			writer.name("id").value(this.languageID);
			writer.name("format").value(currentProgram.getExecutableFormat());

			writer.name("functions").beginObject();
			serializeFunctions();
			writer.endObject();  // End of functions.
			
			writer.name("globals").beginObject();
			serializeGlobals();
			writer.endObject();  // End of globals.

			writer.name("types").beginObject();
			serializeTypes();
			writer.endObject();  // End of types.

			writer.endObject();

			// close ourselves so the caller doesn't need to
			writer.close();
		}
	}