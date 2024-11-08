/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import ghidra.app.script.GhidraScript;

import ghidra.app.cmd.function.CallDepthChangeInfo;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;

import ghidra.app.plugin.processors.sleigh.SleighLanguage;

import ghidra.program.database.symbol.CodeSymbol;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressFactory;
import ghidra.program.model.address.AddressSet;
import ghidra.program.model.address.AddressSetView;
import ghidra.program.model.address.AddressSpace;

import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;

import ghidra.program.model.data.BitFieldDataType;
import ghidra.program.model.data.DataType;

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
import ghidra.program.model.listing.Parameter;
import ghidra.program.model.listing.Program;
import ghidra.program.model.listing.StackFrame;
import ghidra.program.model.listing.Variable;
import ghidra.program.model.listing.VariableStorage;

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

import ghidra.program.model.symbol.ExternalManager;
import ghidra.program.model.symbol.Namespace;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolType;

import ghidra.util.UniversalID;

import com.google.gson.stream.JsonWriter;

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

public class PatchestryDecompileFunctions extends GhidraScript {

	protected static final int DECOMPILATION_TIMEOUT = 30;
	
	protected static final int MIN_CALLOTHER = 1000;
	protected static final int DECLARE_PARAM_VAR = MIN_CALLOTHER + 0;
	protected static final int DECLARE_LOCAL_VAR = MIN_CALLOTHER + 1;
	protected static final int DECLARE_TEMP_VAR = MIN_CALLOTHER + 2;
	
	// A custom `Varnode` used to represent the output of a `CALLOTHER` that
	// we have invented.
	protected class DefinitionVarnode extends Varnode {
		private PcodeOp def;
		private HighVariable high;

		public DefinitionVarnode(Address address, int size) {
			super(address, size);
		}

		public void setDef(HighVariable high, PcodeOp def) {
			this.def = def;
			this.high = high;
		}

		@Override
		public PcodeOp getDef() {
			return def;
		}

		@Override
		public HighVariable getHigh() {
			return high;
		}

		@Override
		public boolean isInput() {
			return false;
		}
	};
	
	// A custome `Varnode` used to represent a rewritten input of a `PTRSUB`
	// or other operation referencing a local variable that was not exactly
	// correctly understood in the high p-code.
	protected class UseVarnode extends Varnode {
		private HighVariable high;

		public UseVarnode(Address address, int size) {
			super(address, size);
		}

		public void setHigh(HighVariable high) {
			this.high = high;
		}

		@Override
		public HighVariable getHigh() {
			return high;
		}

		@Override
		public boolean isInput() {
			return true;
		}
	};

	private class PcodeSerializer extends JsonWriter {
		private Program program;
		private String arch;
		private AddressSpace ram_space;
		private AddressSpace stack_space;
		private AddressSpace constant_space;
		private AddressSpace unique_space;
		private FunctionManager fm;
		private ExternalManager em;
		private DecompInterface ifc;
		private BasicBlockModel bbm;
		private List<Function> functions;
		private int original_functions_size;
		private Set<Address> seen_functions;
		private Set<String> seen_types;
		private List<DataType> types_to_serialize;
		private HighFunction current_function;
		private PcodeBlockBasic current_block;
		
		// We invent an entry block for each `HighFunction` to be serialized.
		// The operations within this entry block are custom `CALLOTHER`s, that
		// "declare" variables of various forms. The way to think about this is
		// with a visual analogy: when looking at a decompilation in Ghidra, the
		// first thing we see in the body of a function are the local variable
		// declarations. In our JSON output, we try to mimic this, and then
		// canonicalize accesses of things to target those variables, doing a
		// kind of de-SSAing.
		private List<PcodeOp> entry_block;
		
		// When creating the `CALLOTHER`s for the `entry_block`, we need to
		// synthesize addresses in the unique address space, and so we need to
		// keep track of what unique addresses we've already used/generated.
		private long next_unique;
		private int next_seqnum;
		
		private SleighLanguage language;
		
		// Stack pointer for this program's architecture. High p-code can have
		// two forms of stack references: `Varnode`s of whose `Address` is part
		// of the stack address space, and `Varnode`s representing registers,
		// where some of those are the stack pointer. In this latter case, we
		// need to be able to identify those and convert them into the former
		// case.
		private Register stack_pointer;
		
		// Maps names of missing locals to invented `HighLocal`s used to
		// represent them. `Function`s often have many `Variable`s, not all of
		// which become `HighLocal`s or `HighParam`s. Sometimes when something
		// can't be precisely recognized, it is represented as a `HighOther`
		// connected to a `HighSymbol`. Confusingly, the `DataType` associated
		// with the `HighSymbol` is more representative of what the decompiler
		// actually shows, and the `HighOther` more representative of the
		// data type in the low `Variable` sourced from the `StackFrame`.
		private Map<String, HighLocal> missing_locals;
		private Map<HighVariable, HighLocal> old_locals;
		
		// Maps addresses to missing global variables.
		private Map<Address, HighGlobal> missing_globals;
		
		// Maps `HighVariables` (really, `HighOther`s) that are attached to
		// register `Varnode`s to the `PcodeOp` containing those nodes. We
		// The same-named temporary/register may be associated with many such
		// independent `HighVariable`s, so to distinguish them to downstream
		// readers of the JSON, we want to 'version' the register variables by
		// their initial user.
		private Map<HighVariable, PcodeOp> temporary_address;

		public PcodeSerializer(java.io.BufferedWriter writer,
				String arch_, FunctionManager fm_,
				ExternalManager em_, DecompInterface ifc_,
				BasicBlockModel bbm_,
				List<Function> functions_) {
			super(writer);

			this.program = fm_.getProgram();

			this.language = (SleighLanguage) program.getLanguage();
			AddressFactory address_factory = program.getAddressFactory();

			this.arch = arch_;
			this.ram_space = address_factory.getAddressSpace("ram");
			this.stack_space = address_factory.getStackSpace();
			this.constant_space = address_factory.getConstantSpace();
			this.unique_space = address_factory.getUniqueSpace();
			this.fm = fm_;
			this.em = em_;
			this.ifc = ifc_;
			this.bbm = bbm_;
			this.functions = functions_;
			this.original_functions_size = functions.size();
			this.seen_functions = new TreeSet<>();
			this.seen_types = new HashSet<>();
			this.types_to_serialize = new ArrayList<>();
			this.current_function = null;
			this.current_block = null;
			this.next_unique = language.getUniqueBase();
			this.next_seqnum = 0;
			this.entry_block = new ArrayList<>();
			this.stack_pointer = program.getCompilerSpec().getStackPointer();
			this.missing_globals = new HashMap<>();
			this.missing_locals = new HashMap<>();
			this.old_locals = new HashMap<>();
			this.temporary_address = new HashMap<>();
		}

		private static String label(HighFunction function) throws Exception {
			return label(function.getFunction());
		}

		private static String label(Function function) throws Exception {
			return label(function.getEntryPoint());
		}

		private static String label(Address address) throws Exception {
			return address.toString(true  /* show address space prefix */);
		}

		private static String label(SequenceNumber sn) throws Exception {
			return label(sn.getTarget()) + Address.SEPARATOR +
					Integer.toString(sn.getTime()) + Address.SEPARATOR +
					Integer.toString(sn.getOrder());
		}

		private static String label(PcodeBlock block) throws Exception {
			return label(block.getStart()) + Address.SEPARATOR +
					Integer.toString(block.getIndex()) + Address.SEPARATOR +
					PcodeBlock.typeToName(block.getType());
		}

		private static String label(PcodeOp op) throws Exception {
			return label(op.getSeqnum());
		}

		private String label(DataType type) throws Exception {
			// In type is null, assign VoidDataType in all cases.
			// We assume it as void type.
			if (type == null) {
				type = VoidDataType.dataType;
			}
			String name = type.getName();
			CategoryPath category = type.getCategoryPath();
			String concat_type = category.toString() + name + Integer.toString(type.getLength());
			String type_id = Integer.toHexString(concat_type.hashCode());

			UniversalID uid = type.getUniversalID();
			if (uid != null) {
				type_id += Address.SEPARATOR + uid.toString();
			}

			if (seen_types.add(type_id)) {
				types_to_serialize.add(type);
			}
			return type_id;
		}

		private void serializePointerType(Pointer ptr) throws Exception {
			name("kind").value("pointer");
			name("size").value(ptr.getLength());
			name("element_type").value(label(ptr.getDataType()));
		}

		private void serializeTypedefType(TypeDef typedef) throws Exception {
			name("name").value(typedef.getDisplayName());
			name("kind").value("typedef");
			name("size").value(typedef.getLength());
			name("base_type").value(label(typedef.getBaseDataType()));
		}

		private void serializeArrayType(Array arr) throws Exception {
			name("kind").value("array");
			name("size").value(arr.getLength());
			name("num_elements").value(arr.getNumElements());
			name("element_type").value(label(arr.getDataType()));
		}

		private void serializeBuiltinType(DataType data_type, String kind) throws Exception {
			
			String display_name = null;
			if (data_type instanceof AbstractIntegerDataType) {
				display_name = ((AbstractIntegerDataType) data_type).getCDeclaration();
			}
			
			if (display_name == null) {
				display_name = data_type.getDisplayName();
			}

			name("name").value(display_name);
			name("size").value(data_type.getLength());
			name("kind").value(kind);
		}

		private void serializeCompositeType(Composite data_type, String kind) throws Exception {
			name("name").value(data_type.getDisplayName());
			name("kind").value(kind);
			name("size").value(data_type.getLength());
			name("fields").beginArray();

			for (int i = 0; i < data_type.getNumComponents(); i++) {
				DataTypeComponent dtc = data_type.getComponent(i);
				beginObject();
				name("type").value(label(dtc.getDataType()));
				name("offset").value(dtc.getOffset());

				if (dtc.getFieldName() != null) {
					name("name").value(dtc.getFieldName());
				}
				endObject();
			}
			endArray();
		}

		private void serialize(DataType data_type) throws Exception {
			if (data_type == null) {
				nullValue();
				return;
			}

			if (data_type instanceof Pointer) {
				serializePointerType((Pointer) data_type);

			} else if (data_type instanceof TypeDef) {
				serializeTypedefType((TypeDef) data_type);

			} else if (data_type instanceof Array) {
				serializeArrayType((Array) data_type);

			} else if (data_type instanceof Structure) {
				serializeCompositeType((Composite) data_type, "struct");

			} else if (data_type instanceof Union) {
				serializeCompositeType((Composite) data_type, "union");

			} else if (data_type instanceof AbstractIntegerDataType){
				serializeBuiltinType(data_type, "integer");

			} else if (data_type instanceof AbstractFloatDataType){
				serializeBuiltinType(data_type, "float");

			} else if (data_type instanceof BooleanDataType){
				serializeBuiltinType(data_type, "boolean");

			} else if (data_type instanceof Enum) {
				serializeBuiltinType(data_type, "enum");

			} else if (data_type instanceof VoidDataType) {
				serializeBuiltinType(data_type, "void");

			} else if (data_type instanceof Undefined || data_type.toString().contains("undefined")) {
				serializeBuiltinType(data_type, "undefined");

			} else if (data_type instanceof FunctionDefinition) {
				name("kind").value("function");
				serializePrototype((FunctionSignature) data_type);

			} else if (data_type instanceof PartialUnion) {
				name("kind").value("todo");  // TODO(pag): Implement this
				name("size").value(data_type.getLength());

			} else if (data_type instanceof BitFieldDataType) {
				name("kind").value("todo");  // TODO(pag): Implement this
				name("size").value(data_type.getLength());
				
			} else if (data_type instanceof WideCharDataType) {
				name("kind").value("todo");  // TODO(pag): Implement this
				name("size").value(data_type.getLength());
				
			} else {
				throw new Exception("Unhandled type: " + data_type.getClass().getName());
			}
		}

		private void serializeTypes() throws Exception {
			for (int i = 0; i < types_to_serialize.size(); i++) {
				DataType type = types_to_serialize.get(i);
				name(label(type)).beginObject();
				serialize(type);
				endObject();
			}

			println("Total serialized types: " + types_to_serialize.size());
		}
		
		private int serializePrototype() throws Exception {
			name("return_type").value(label((DataType) null));
			name("is_variadic").value(false);
			name("is_variadic").value(false);
			name("parameter_types").beginArray().endArray();
			return 0;
		}

		private int serializePrototype(FunctionPrototype proto) throws Exception {
			if (proto == null) {
				return serializePrototype();
			}

			name("return_type").value(label(proto.getReturnType()));
			name("is_variadic").value(proto.isVarArg());
			name("is_noreturn").value(proto.hasNoReturn());
			
			name("parameter_types").beginArray();
			int num_params = proto.getNumParams();
			for (int i = 0; i < num_params; i++) {
				value(label(proto.getParam(i).getDataType()));
			}
			endArray();  // End of `parameter_types`.
			return num_params;
		}
		
		private int serializePrototype(FunctionSignature proto) throws Exception {
			if (proto == null) {
				return serializePrototype();
			}

			name("return_type").value(label(proto.getReturnType()));
			name("is_variadic").value(proto.hasVarArgs());
			name("is_noreturn").value(proto.hasNoReturn());
			name("calling_convention").value(proto.getCallingConventionName());
			
			ParameterDefinition[] arguments = proto.getArguments();
			name("parameter_types").beginArray();
			int num_params = (int) arguments.length;
			for (int i = 0; i < num_params; i++) {
				value(label(arguments[i].getDataType()));
			}
			endArray();  // End of `parameter_types`.
			return num_params;
		}

		private void serialize(HighVariable high_var) throws Exception {
			if (high_var == null) {
				nullValue();
				return;
			}

			beginObject();
			name("name").value(high_var.getName());
			name("type").value(label(high_var.getDataType()));
			endObject();
		}

		// Return the r-value of a varnode.
		private Varnode rValueOf(Varnode node) throws Exception {
			return node;
//			HighVariable high = node.getHigh();
//			if (high == null) {
//				return node;
//			}
//
//			Varnode rep = high.getRepresentative();
//			return rep == null ? node : rep;
		}
		
		private enum VariableClassification {
			UNKNOWN,
			PARAMETER,
			LOCAL,
			NAMED_TEMPORARY,
			TEMPORARY,
			GLOBAL,
			FUNCTION,
			CONSTANT
		};
		
		// Returns `true` if a given representative is an original
		// representative.
		private static boolean isOriginalRepresentative(Varnode node) {
			if (node.isInput()) {
				return true;
			}

			PcodeOp op = node.getDef();
			if (op == null) {
				return true;
			}
			
			if (op.getOpcode() != PcodeOp.CALLOTHER) {
				return true;
			}
			
			if (op.getInput(0).getOffset() < MIN_CALLOTHER) {
				return true;
			}
			
			return false;
		}

		// Get the representative of a `HighVariable`, or if we've re-written
		// the representative with a `CALLOTHER`, then get the original
		// representative.
		private static Varnode originalRepresentativeOf(HighVariable var) {
			if (var == null) {
				return null;
			}
			
			Varnode rep = var.getRepresentative();
			if (isOriginalRepresentative(rep)) {
				return rep; 
			}
			
			Varnode[] instances = var.getInstances();
			if (instances.length <= 1) {
				return null;
			}
			
			return instances[1];
		}

		// Try to distinguish "local" variables from global ones. Roughly, we
		// want to make sure that the backing storage for a given variable
		// *isn't* RAM. Thus, UNIQUE, STACK, CONST, etc. are all in-scope for
		// locals.
		private VariableClassification classifyVariable(HighVariable var) throws Exception {
			if (var == null) {
				return VariableClassification.UNKNOWN;
			}

			if (var instanceof HighParam) {
				return VariableClassification.PARAMETER;

			} else if (var instanceof HighLocal) {
				return VariableClassification.LOCAL;

			} else if (var instanceof HighConstant) {
				return VariableClassification.CONSTANT;
			
			} else if (var instanceof HighGlobal) {
				return VariableClassification.GLOBAL;
			}

			HighSymbol symbol = var.getSymbol();
			if (symbol != null) {
				if (symbol.isGlobal()) {
					return VariableClassification.GLOBAL;
				} else if (symbol.isParameter() || symbol.isThisPointer()) {
					return VariableClassification.PARAMETER;
				}
			}

			Varnode rep = originalRepresentativeOf(var);
			if (rep != null) {

				// TODO(pag): Consider checking if all uses of the unique
				//			  belong to the same block. We don't want to
				//			  introduce a kind of code motion risk into the
				//			  lifted representation.
				if (rep.isRegister() || rep.isUnique() || var instanceof HighOther) {
					if (rep.getLoneDescend() != null) {
						return VariableClassification.TEMPORARY;
					} else {
						return VariableClassification.NAMED_TEMPORARY;
					}
				}
			}
			
			return VariableClassification.UNKNOWN;
		}
		
		// Serialize an input or output varnode.
		private void serializeInput(PcodeOp op, Varnode node) throws Exception {
			assert !node.isFree();
			assert node.isInput();

			PcodeOp def = node.getDef();
			HighVariable var = variableOf(node.getHigh());

			beginObject();
			
			if (var != null) {
				name("type").value(label(var.getDataType()));
			} else {
				name("size").value(node.getSize());
			}

			switch (classifyVariable(var)) {
				case VariableClassification.UNKNOWN:
					if (def != null && !node.isInput() && def == op) {
						if (node.isUnique()) {
							name("kind").value("temporary");
						
						// TODO(pag): Figure this out.
						} else {
							assert false;
							name("kind").value("unknown");
						}
	
					// NOTE(pag): Should be a `TEMPORARY` classification.
					} else if (node.isUnique()) {
						assert false;
						assert def != null;
						name("kind").value("temporary");
						name("operation").value(label(def));
					
					// NOTE(pag): Should be a `REGISTER` classification.
					} else if (node.isConstant()) {
						assert false;
						name("kind").value("constant");
						name("value").value(node.getOffset());

					} else {
						assert false;
						name("kind").value("unknown");
					}
					break;
				case VariableClassification.PARAMETER:
					name("kind").value("parameter");
					name("operation").value(label(getOrCreateLocalVariable(var, op)));
					break;
				case VariableClassification.LOCAL:
					name("kind").value("local");
					name("operation").value(label(getOrCreateLocalVariable(var, op)));
					break;
				case VariableClassification.NAMED_TEMPORARY:
					name("kind").value("temporary");
					name("operation").value(label(getOrCreateLocalVariable(var, op)));
					break;
				case VariableClassification.TEMPORARY:
					assert def != null;
					if (def == null) {
						println("Null def: " + label(op));
						println("      Op: " + op.toString());
						println("    Node: " + node.toString());
						
						if (var != null) {
							println("     Var: " + var.getName() + " @ " + Integer.toString(var.getOffset()) + ": " + var.toString());
							println("    Type: " + var.getDataType().toString());
							HighSymbol sym = var.getSymbol();
							if (sym != null) {
								println("     Sym: " + sym.getName() + ": " + sym.toString());
							}
						}
					}
					name("kind").value("temporary");
					name("operation").value(label(def));
					break;
				case VariableClassification.GLOBAL:
					name("kind").value("global");
					// TODO(pag): Fill this in.
					break;
				case VariableClassification.FUNCTION:
					name("kind").value("function");
					name("function").value(label(var.getHighFunction()));
					break;
				case VariableClassification.CONSTANT:
					if (node.isConstant()) {
						name("kind").value("constant");
						name("value").value(node.getOffset());
					} else {
						assert false;
						name("kind").value("unknown");
					}
					break;
			}

			endObject();
		}
		
		// Returns the index of the first input `Varnode` referncing the stack
		// pointer, or `-1` if no direct references are found.
		private int referencesStackPointer(PcodeOp op) {
			int input_index = 0;
			for (Varnode node : op.getInputs()) {
				if (node.isRegister()) {
					Register reg = language.getRegister(node.getAddress(), 0);
					
					// TODO(pag): This doesn't seem to work? All `typeFlags` for
					// 			  all registers seem to be zero, at least for
					//			  x86.
					if ((reg.getTypeFlags() & Register.TYPE_SP) != 0) {
						return input_index;
					}

					if (reg == stack_pointer) {
						return input_index;
					}

					// TODO(pag): Should we consider references to the frame
					//			  pointer, e.g. using the `CompilerSpec` or
					//			  `reg.isDefaultFramePointer()`?
				}
				
				++input_index;
			}
			
			return -1;
		}
		
		// Given a `PTRSUB SP, offset`, try to invent a local variable at
		// `offset` in a similar way to how the decompiler would.
		private boolean createLocalForPtrSubcomponent(
				HighFunction high_function, PcodeOp op) {
			Function function = high_function.getFunction();
			StackFrame frame = function.getStackFrame();
			Varnode offset_node = op.getInput(1);
			if (!offset_node.isConstant()) {
				return false;
			}

			int frame_size = frame.getFrameSize();
			int var_offset = (int) offset_node.getOffset();
			int adjust_offset = 0;
			
			// Given the local symbol mapping for the high function, go find
			// a `HighSymbol` corresponding to `local_118`. This high symbol
			// will generally have a much better `DataType`, but initially
			// and confusingly won't have a corresponding `HighVariable`.
			LocalSymbolMap symbols = high_function.getLocalSymbolMap();
			Address pc = op.getSeqnum().getTarget();
			
			// Given a stack pointer offset, e.g. `-0x118`, go find the low
			// `Variable` representing `local_118`.
			Variable var = frame.getVariableContaining(var_offset);
			Address stack_address = stack_space.getAddress(var_offset);
			HighSymbol sym = null;
			if (var != null) {
				sym = symbols.findLocal(var.getVariableStorage(), pc);
				stack_address = stack_space.getAddress(var.getStackOffset());
				
			} else {
				sym = symbols.findLocal(stack_address, pc);
			}
			
			// Try to recover by locating the parameter containing the stack
			// address.
			if (sym == null) {
				for (Variable param : frame.getParameters()) {
					VariableStorage storage = param.getVariableStorage();
					if (!storage.contains(stack_address)) {
						continue;
					}
			
					int index = ((Parameter) param).getOrdinal();
					if (index >= symbols.getNumParams()) {
						break;
					}
					
					sym = symbols.getParamSymbol(index);
					break;
				}
			}
			
			// This is usually for one of a few reasons:
			//		- Trying to lift `_start`
			//		- Trying to lift a variadic function using `va_list`.
			if (sym == null) {
				return false;
			}

			Varnode var_node = op.getInput(0);
			UseVarnode new_var_node = new UseVarnode(
					stack_address, sym.getDataType().getLength());

			// We've already got a high variable for this missing local.
			HighVariable new_var = sym.getHighVariable();
			String sym_name = sym.getName();
			if (new_var != null && !new_var.getName().equals("UNNAMED")) {
				// println("Using existing high sym " + sym_name + " with var named " + new_var.getName() + " and type " + new_var.getDataType().toString());

			// We need to invent a new `HighVariable` for this `HighSymbol`.
			// Unfortunately we can't use `HighSymbol.setHighVariable` for the
			// caching, so we need `missing_locals`.
			} else {
				HighLocal local_var = old_locals.get(new_var);
				if (local_var == null) {
					local_var = missing_locals.get(sym_name);
				}

				if (local_var == null) {
					local_var = new HighLocal(
							sym.getDataType(), new_var_node, null, pc, sym);
					missing_locals.put(sym_name, local_var);

					// println("Created " + local_var.getName() + " with type " + local_var.getDataType().toString());

					// Remap old-to-new.
					if (new_var != null) {
						old_locals.put(new_var, local_var);
					}
					
				}
				new_var = local_var;
			}

			int new_var_offset = new_var.getOffset() == -1 ? 0 : new_var.getOffset();
			if (var != null) {
				adjust_offset = (var_offset - var.getStackOffset()) + new_var_offset;
			}
			
			// println("  Rewriting " + op.getSeqnum().toString() + ": " + op.toString());

			// Rewrite the stack reference to point to the `HighVariable`.
			new_var_node.setHigh(new_var);
			op.setInput(new_var_node, 0);
			
			// Rewrite the offset.
			op.setInput(new Varnode(constant_space.getAddress(adjust_offset), offset_node.getSize()), 1);

			// println("  to: " + op.toString());
			
			return true;
		}
		
		// Given a `PTRSUB const, const`, try to recognize it as a global variable
		// reference, or a field reference within a global variable.
		private boolean createGlobalForPtrSubcomponent(
				HighFunction high_function, PcodeOp op) {
			
			HighVariable zero = op.getInput(0).getHigh();
			if (!(zero instanceof HighOther)) {
				return false;
			}
			
			if (!zero.getName().equals("UNNAMED")) {
				return false;
			}
			
			if (zero.getOffset() != -1) {
				return false;
			}
			
			Varnode offset_node = op.getInput(1);
			HighVariable offset_var = offset_node.getHigh();
			if (!(offset_var instanceof HighConstant)) {
				return false;
			}
			
			HighSymbol high_sym = offset_var.getSymbol();
			if (high_sym == null) {
				return false;
			}

			// println("Found variable use " + high_sym.getName());

			SymbolEntry entry = high_sym.getFirstWholeMap();
			Address address = entry.getStorage().getMinAddress();

			UseVarnode new_var_node = new UseVarnode(
					address, high_sym.getDataType().getLength());

			HighGlobal global_var = missing_globals.get(address);
			if (global_var == null) {
				HighVariable maybe_global_var = high_sym.getHighVariable();
				if (maybe_global_var instanceof HighGlobal) {
					global_var = (HighGlobal) maybe_global_var;
				} else {
					global_var = new HighGlobal(high_sym, new_var_node, null);
				}
				
				missing_globals.put(address, global_var);
			}

			// println("Rewriting " + op.getSeqnum().toString() + ": " + op.toString());

			// Rewrite the global reference to point to the `HighVariable`.
			new_var_node.setHigh(global_var);
			op.setInput(new_var_node, 0);
			
			// Rewrite the offset.
			Address offset_as_address = address.getAddressSpace().getAddress(offset_node.getOffset());
			int sub_offset = (int) offset_as_address.subtract(address);

			op.setInput(new Varnode(constant_space.getAddress(sub_offset), offset_node.getSize()), 1);
			
			// println("  to: " + op.toString());
			
			return true;
		}
		
		// Try to rewrite/mutate a `PTRSUB`.
		private boolean rewritePtrSubcomponent(
				HighFunction high_function, PcodeOp op) {

			// Look for `PTRSUB SP, offset` and convert into `PTRSUB local_N, M`.
			if (referencesStackPointer(op) == 0) {
				return createLocalForPtrSubcomponent(high_function, op);
			}

			Varnode base_node = op.getInput(0);
			Varnode offset_node = op.getInput(1);
			if (base_node.isConstant() && base_node.getOffset() == 0 &&
				offset_node.isConstant()) {
				return createGlobalForPtrSubcomponent(high_function, op);
			}

			return true;
		}
		
//		// Look for `PTRADD SP, -1, 1` or something like it that is used to
//		// calculate the return address location.
//		private boolean markPtrAddForElision(
//				HighFunction high_function, PcodeOp op) {
//			int sp_index = referencesStackPointer(op);
//			if (sp_index != 0) {
//				return true;
//			}
//			
//			if (op.getInput(2).getOffset() != 1) {
//				return false;
//			}
//			
//			Function function = high_function.getFunction();
//			StackFrame frame = function.getStackFrame();
//
//		}

		// Create missing local variables. High p-code still includes things
		// like `PTRSUB SP, -offset` instead of treating the unrecognized data
		// as `local_<hex_offset>`. The decompiler, however, does these
		// automatic variable inventions.
		//
		// Returns `false` on failure.
		//
		// NOTE(pag): This function is very much inspired by the `MakeStackRefs`
		//		      script embedded in the Ghidra source.
		private boolean fixupOperations(
				HighFunction high_function, int num_params) throws Exception {
			Function function = high_function.getFunction();
			FunctionSignature signature = function.getSignature();
			FunctionPrototype proto = high_function.getFunctionPrototype();
			LocalSymbolMap symbols = high_function.getLocalSymbolMap();
			
			// Fill in the parameters first so that they are the first
			// things added to `entry_block`.
			for (int i = 0; i < num_params; ++i) {
				HighParam param = symbols.getParam(i);
				if (param == null) {
					HighSymbol param_sym = proto.getParam(i);
					// println("Inventing HighParam for " + param_sym.getName() + " in " + function.getName());
					param = new HighParam(param_sym.getDataType(), null, null, i, param_sym);
					missing_locals.put(param.getName(), param);
				}

				createParamVarDecl(param);
			}
			
//			Iterator<HighSymbol> iter = symbols.getSymbols();
//			while (iter.hasNext()) {
//				HighSymbol sym = iter.next();
//				println("High symbol: " + sym.getName());
//				HighVariable var = sym.getHighVariable();
//				if (var != null) {
//					println("  " + var.toString());
//				}
//				println("  " + sym.getDataType().toString());
//			}
			
			//CallDepthChangeInfo cdci = new CallDepthChangeInfo(function);
//			StackFrame frame = function.getStackFrame();
//			Set<Variable> seen_vars = new TreeSet<>();
//			Variable[] vars = frame.getStackVariables();
//			for (Variable var : vars) {
//				if (!var.isStackVariable() || !var.hasStackStorage()) {
//					assert false;
//					continue;
//				}
//
//				println("stack variable: " + var.toString());
//			}
			
			
			// Now go look for operations directly referencing the stack pointer.
			for (PcodeBlockBasic block : high_function.getBasicBlocks()) {
				Iterator<PcodeOp> op_iterator = block.getIterator();
				while (op_iterator.hasNext()) {
					PcodeOp op = op_iterator.next();

					switch (op.getOpcode()) {
						case PcodeOp.PTRSUB:
							if (!rewritePtrSubcomponent(high_function, op)) {
								println("Unsupported PTRSUB at " + label(op) + ": " + op.toString());
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
							if (canElideMultiEqual(op)) {
								continue;
							}
							// Fall-through.
						default:
							if (referencesStackPointer(op) != -1) {
								println("Unsupported stack pointer reference at " + label(op) + ": " + op.toString());
								return false;
							}
							break;
					}
				}
			}
			
			// Need to go and invent these.
//			Variable[] missing = (Variable[]) seen_vars.toArray();
			
			return true;
		}
		
		private HighVariable variableOf(HighVariable var) {
			if (var == null) {
				return null;
			}
			
			HighLocal fixed_var = old_locals.get(var);
			return fixed_var != null ? fixed_var : var;
		}
		
		// Return the variable of a given `Varnode`. This applies local fixups.
		private HighVariable variableOf(Varnode node) {
			return node == null ? null : variableOf(node.getHigh());
		}
		
		private HighVariable variableOf(PcodeOp op) {
			return variableOf(op.getOutput());
		}

		// Handles serializing the output, if any, of `op`. We only actually
		// serialize the named outputs.
		private void serializeOutput(PcodeOp op) throws Exception {
			Varnode output = op.getOutput();
			if (output == null) {
				return;
			}

			HighVariable var = variableOf(output);
			if (var != null) {
				name("type").value(label(var.getDataType()));
			} else {
				name("size").value(output.getSize());
			}
			
			// Only record an output node when the target is something named.
			// Otherwise, this p-code operation will be used as part of an
			// operand to something else.
			//
			// TODO(pag): Probably need some kind of verifier downstream to
			//			  ensure no code motion happens.
			VariableClassification klass = classifyVariable(var);
			switch (klass) {
				case VariableClassification.PARAMETER:
				case VariableClassification.LOCAL:
				case VariableClassification.NAMED_TEMPORARY:
				case VariableClassification.GLOBAL:
					break;
				default:
					return;
			}
			
			name("output").beginObject();
			if (klass == VariableClassification.PARAMETER) {
				name("kind").value("parameter");
				name("operation").value(label(getOrCreateLocalVariable(var, op)));
			} else if (klass == VariableClassification.LOCAL) {
				name("kind").value("local");
				name("operation").value(label(getOrCreateLocalVariable(var, op)));				
			} else if (klass == VariableClassification.NAMED_TEMPORARY) {
				name("kind").value("temporary");
				name("operation").value(label(getOrCreateLocalVariable(var, op)));
			} else if (klass == VariableClassification.GLOBAL) {
				name("kind").value("global");
				// TODO(pag): Global refs.
			} else {
				assert false;
			}
 			endObject();
		}
		
		// The address of a `LOAD` or `STORE` is spread across two operands:
		// the first being a constant representing the address space, and the
		// second being the actual address.
		private void serializeLoadStoreAddress(PcodeOp op) throws Exception {
			Varnode aspace = op.getInput(0);
			assert aspace.isConstant();

			Varnode address = rValueOf(op.getInput(1));

			beginObject();
			name("size").value(op.getInput(1).getSize());
			println(op.toString());
			//          if (address.isConstant()) {
			//            AddressSpace address_space = null;
			//            
			//            Address address = new Address();
			//          }

			endObject();
		}
		
		// Product a new address in the `UNIQUE` address space.
		private Address nextUniqueAddress() throws Exception {
			Address address = unique_space.getAddress(next_unique);
			next_unique += unique_space.getAddressableUnitSize();
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
		private PcodeOp createParamVarDecl(HighVariable var) throws Exception {
			HighParam param = (HighParam) var;
			Address address = nextUniqueAddress();
			DefinitionVarnode def = new DefinitionVarnode(address, var.getDataType().getAlignedLength());
			Varnode[] ins = new Varnode[2];
			SequenceNumber loc = new SequenceNumber(address, next_seqnum++);
			PcodeOp op = new PcodeOp(loc, PcodeOp.CALLOTHER, 2, def);
			op.insertInput(new Varnode(constant_space.getAddress(DECLARE_PARAM_VAR), 4), 0);
			op.insertInput(new Varnode(constant_space.getAddress(param.getSlot()), 4), 1);
			def.setDef(var, op);

			Varnode[] instances = var.getInstances();
			Varnode[] new_instances = new Varnode[instances.length + 1];
			System.arraycopy(instances, 0, new_instances, 1, instances.length);
			new_instances[0] = def;

			var.attachInstances(new_instances, def);

			entry_block.add(op);

			return op;
		}

		// Creates a pseudo p-code op using a `CALLOTHER` that logically
		// represents the definition of a local variable.
		private PcodeOp createLocalVarDecl(HighVariable var) throws Exception {
			Address address = nextUniqueAddress();
			DefinitionVarnode def = new DefinitionVarnode(address, var.getSize());
			Varnode[] ins = new Varnode[1];
			SequenceNumber loc = new SequenceNumber(address, next_seqnum++);
			PcodeOp op = new PcodeOp(loc, PcodeOp.CALLOTHER, 1, def);
			op.insertInput(new Varnode(constant_space.getAddress(DECLARE_LOCAL_VAR), 4), 0);
			def.setDef(var, op);

			Varnode[] instances = var.getInstances();
			Varnode[] new_instances = new Varnode[instances.length + 1];
			System.arraycopy(instances, 0, new_instances, 1, instances.length);
			new_instances[0] = def;

			var.attachInstances(new_instances, def);
			
			HighSymbol sym = var.getSymbol();
			entry_block.add(op);

			return op;
		}

		// Creates a pseudo p-code op using a `CALLOTHER` that logically
		// represents the definition of a variable that stands in for a register.
		private PcodeOp createNamedTemporaryDecl(
				HighVariable var, PcodeOp user_op) throws Exception {
			Varnode rep = originalRepresentativeOf(var);
			assert rep.isRegister();

			Address address = nextUniqueAddress();
			DefinitionVarnode def = new DefinitionVarnode(address, var.getSize());
			Varnode[] ins = new Varnode[1];
			SequenceNumber loc = new SequenceNumber(address, next_seqnum++);
			PcodeOp op = new PcodeOp(loc, PcodeOp.CALLOTHER, 1, def);
			op.insertInput(new Varnode(constant_space.getAddress(DECLARE_TEMP_VAR), 4), 0);
			def.setDef(var, op);

			Varnode[] instances = var.getInstances();
			Varnode[] new_instances = new Varnode[instances.length + 1];
			System.arraycopy(instances, 0, new_instances, 1, instances.length);
			new_instances[0] = def;

			var.attachInstances(new_instances, def);

			entry_block.add(op);
			temporary_address.put(var, user_op);

			return op;
		}

		// Get or create a local variable pseudo definition op for the high
		// variable `var`.
		private PcodeOp getOrCreateLocalVariable(
				HighVariable var, PcodeOp user_op) throws Exception {
			Varnode representative = var.getRepresentative();
			PcodeOp def = null;
			if (representative != null) {
				def = representative.getDef();			
				if (!isOriginalRepresentative(representative)) {
					return def;
				}
			}

			switch (classifyVariable(var)) {
				case VariableClassification.PARAMETER:
					println("Creating late parameter for " + label(user_op) + ": " + user_op.toString());
					return createParamVarDecl(var);
				case VariableClassification.LOCAL:
					return createLocalVarDecl(var);
				case VariableClassification.NAMED_TEMPORARY:
					return createNamedTemporaryDecl(var, user_op);
				default:
					break;
			}

			return def;
		}

		// Serialize a direct call. This enqueues the targeted for type lifting
		// `Function` if it can be resolved.
		private void serializeCallOp(PcodeOp op) throws Exception {
			Address caller_address = current_function.getFunction().getEntryPoint();
			Varnode target_node = op.getInput(0);
			Function callee = null;

			if (target_node.isAddress()) {
				Address target_address = caller_address.getNewAddress(target_node.getOffset());
				String target_label = label(target_address);
				callee = fm.getFunctionAt(target_address);
	
				// `target_address` may be a pointer to an external. Figure out
				// what we're calling.
	
				if (callee == null) {
					callee = fm.getReferencedFunction(target_address);
					if (callee != null) {
						target_address = callee.getEntryPoint();
						target_label = label(target_address);
					}	
				}
			}

			name("target");
			if (callee != null) {
				functions.add(callee);

				beginObject();
				name("kind").value("function");
				name("function").value(label(callee));
				name("is_noreturn").value(callee.hasNoReturn());
				endObject();

			} else {
				serializeInput(op, rValueOf(target_node));
			}
			
			name("inputs").beginArray();			
			Varnode[] inputs = op.getInputs();
			for (int i = 1; i < inputs.length; ++i) {
				serializeInput(op, rValueOf(inputs[i]));
			}
			endArray();
		}

		// Serialize an unconditional branch. This records the targeted block.
		private void serializeBranchOp(PcodeOp op) throws Exception {
			assert current_block.getOutSize() == 1;
			name("target_block").value(label(current_block.getOut(0)));
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
		private void serializeCondBranchOp(PcodeOp op) throws Exception {
			name("taken_block").value(label(current_block.getTrueOut()));
			name("not_taken_block").value(label(current_block.getFalseOut()));
			name("condition");
			serializeInput(op, rValueOf(op.getInput(1)));
		}

		// Serialize a generic multi-input, single-output p-code operation.
		private void serializeGenericOp(PcodeOp op) throws Exception {
			name("inputs").beginArray();
			for (Varnode input : op.getInputs()) {
				serializeInput(op, rValueOf(input));
			}
			endArray();
		}

		// Serializes a pseudo-op `DECLARE_PARAM_VAR`, which is actually encoded
		// as a `CALLOTHER`.
		private void serializeDeclareParamVar(PcodeOp op) throws Exception {
			HighVariable var = variableOf(op);
			name("name").value(var.getName());
			name("type").value(label(var.getDataType()));
			name("kind").value("parameter");  // So that it also looks like an input/output.
			if (var instanceof HighParam) {
				name("index").value(((HighParam) var).getSlot());
			}
		}

		// Serializes a pseudo-op `DECLARE_LOCAL_VAR`, which is actually encoded
		// as a `CALLOTHER`.
		private void serializeDeclareLocalVar(PcodeOp op) throws Exception {
			HighVariable var = variableOf(op);
			HighSymbol sym = var.getSymbol();
			name("kind").value("local");  // So that it also looks like an input/output.
			if (sym != null && var.getOffset() == -1 && var.getName().equals("UNNAMED")) {
				name("name").value(sym.getName());
				name("type").value(label(sym.getDataType()));
			} else {
				name("name").value(var.getName());
				name("type").value(label(var.getDataType()));
			}
		}
		
		private void serializeDeclareNamedTemporary(PcodeOp op) throws Exception {
			HighVariable var = variableOf(op);
			Varnode rep = originalRepresentativeOf(var);

			// NOTE(pag): In practice, the `HighOther`s name associated with
			//			  this register is probably `UNNAMED`, which happens in
			//			  `HighOther.decode`; however, we'll be cautious and
			// 			  only canonicalize on the register name if the name
			//			  it is the default.
			if (var.getName().equals("UNNAMED")) {
				if (rep.isRegister()) {
					Register reg = language.getRegister(rep.getAddress(), 0);
					name("name").value(reg.getName());
				} else {
					name("name").value("temp");
				}
			} else {
				name("name").value(var.getName());
			}

			name("kind").value("temporary");  // So that it also looks like an input/output.
			name("type").value(label(var.getDataType()));
			
			// NOTE(pag): The same register might appear multiple times, though
			//			  we can't guarantee that they will appear with the
			//			  same names. Thus, we want to record the address of
			//            the operation using the original register as a kind of
			//			  SSA-like version number downstream, e.g. in a Clang
			//			  AST.
			PcodeOp user_op = temporary_address.get(var);
			if (user_op != null) {
				name("address").value(label(user_op));
			}
		}

		// Serialize a `CALLOTHER`. The first input operand is a constant
		// representing the user-defined opcode number. In our case, we have
		// our own user-defined opcodes for making things better mirror the
		// structure/needs of MLIR.
		private void serializeCallOtherOp(PcodeOp op) throws Exception {
			switch ((int) op.getInput(0).getOffset()) {
			case DECLARE_PARAM_VAR:
				serializeDeclareParamVar(op);
				break;
			case DECLARE_LOCAL_VAR:
				serializeDeclareLocalVar(op);
				break;
			case DECLARE_TEMP_VAR:
				serializeDeclareNamedTemporary(op);
				break;
			default:
				serializeOutput(op);
				serializeGenericOp(op);
				break;
			}
		}

		// Get the mnemonic for a p-code operation. We have some custom
		// operations encoded as `CALLOTHER`s, so we get their names manually
		// here.
		//
		// TODO(pag): There is probably a way to register the name of a
		//        `CALLOTHER` via `Language.getSymbolTable()` using a
		//        `UseropSymbol`. It's not clear if there's really value in
		//        doing this, though.
		private static String mnemonic(PcodeOp op) {
			if (op.getOpcode() == PcodeOp.CALLOTHER) {
				switch ((int) op.getInput(0).getOffset()) {
				case DECLARE_PARAM_VAR:
					return "DECLARE_PARAMETER";
				case DECLARE_LOCAL_VAR:
					return "DECLARE_LOCAL";
				case DECLARE_TEMP_VAR:
					return "DECLARE_TEMPORARY";
				default:
					break;
				}
			}
			return op.getMnemonic();
		}

		private void serialize(PcodeOp op) throws Exception {

			beginObject();
			name("mnemonic").value(mnemonic(op));
			
			switch (op.getOpcode()) {
				case PcodeOp.CALL:
				case PcodeOp.CALLIND:
					serializeOutput(op);
					serializeCallOp(op);
					break;
				case PcodeOp.CALLOTHER:
					serializeCallOtherOp(op);
					break;
				case PcodeOp.CBRANCH:
					serializeCondBranchOp(op);
					break;
				case PcodeOp.BRANCH:
					serializeBranchOp(op);
					break;
				case PcodeOp.LOAD:
				case PcodeOp.COPY:
				case PcodeOp.CAST:
				default:
					serializeOutput(op);
					serializeGenericOp(op);
					break;
			}

			endObject();
		}
		
		// Returns `true` if we can elide a `MULTIEQUAL` operation. If all
		// inputs are of the identical `HighVariable`, then we can elide.
		private boolean canElideMultiEqual(PcodeOp op) throws Exception {
			HighVariable high = variableOf(op);
			if (high == null) {
				return false;
			}

			for (Varnode node : op.getInputs()) {
				if (high != variableOf(node)) {
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
		private boolean canElideCopy(PcodeOp op) throws Exception {
			HighVariable high = variableOf(op);
			return high != null && high == variableOf(op.getInput(0));
		}
		
		// Returns `true` if `op` is a branch operator.
		private static boolean isBranch(PcodeOp op) throws Exception {
			switch (op.getOpcode()) {
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
		private void serialize(PcodeBlockBasic block) throws Exception {
			PcodeBlock parent_block = block.getParent();
			if (parent_block != null) {
				name("parent_block").value(label(parent_block));
			}
			
			boolean last_is_branch = false;
			Iterator<PcodeOp> op_iterator = block.getIterator();
			ArrayList<PcodeOp> ordered_operations = new ArrayList<>();
			name("operations").beginObject();
			while (op_iterator.hasNext()) {
				PcodeOp op = op_iterator.next();
				
				switch (op.getOpcode()) {
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
						if (canElideMultiEqual(op)) {
							continue;
						}
						break;
					
					// Some copies end up imlpementing the kind of forward edge
					// of a phi node (i.e. `MULTIEQUAL`) and can be elided.
					case PcodeOp.COPY:
						if (canElideCopy(op)) {
							continue;
						}
						break;

					default:
						break;
				}
				
				ordered_operations.add(op);
				name(label(op));
				current_block = block;
				serialize(op);
				last_is_branch = isBranch(op);
				current_block = null;
			}
			
			// Synthesize a fake `BRANCH` operation to the fall-through block.
			// We'll have a fall-through if we don't already end in a branch,
			// and if the last operation isn't a `RETURN` or a `CALL*` to a
			// `noreturn`-attributed function.
			String fall_through_label = "";
			if (!last_is_branch && block.getOutSize() == 1) {
				fall_through_label = label(block) + ".exit";
				name(fall_through_label).beginObject();
				name("mnemonic").value("BRANCH");
				name("target_block").value(label(block.getOut(0)));
				endObject();  // End of BRANCH to `first_block`.	
			}
			
			endObject();

			// List out the operations in their order.
			name("ordered_operations").beginArray();
			for (PcodeOp op : ordered_operations) {
//				beginArray();
//				value(mnemonic(op));
				value(label(op));
//				endArray();
			}
			if (!fall_through_label.equals("")) {
				value(fall_through_label);
			}
			endArray();
		}

		// Emit a pseudo entry block to represent
		private void serializeEntryBlock(PcodeBlockBasic first_block) throws Exception {
			name("entry").beginObject();
			name("operations").beginObject();
			for (PcodeOp pseudo_op : entry_block) {
				name(label(pseudo_op));
				serialize(pseudo_op);
			}

			// If there is a proper entry block, then invent a branch to it.
			if (first_block != null) {
				name("entry.exit").beginObject();
				name("mnemonic").value("BRANCH");
				name("target_block").value(label(first_block));
				endObject();  // End of BRANCH to `first_block`.
			}

			endObject();  // End of operations.

			name("ordered_operations").beginArray();
			for (PcodeOp pseudo_op : entry_block) {
				value(label(pseudo_op));
			}
			value("entry.exit");
			endArray();  // End of `ordered_operations`.
			endObject();  // End of `entry` block.
		}
		
		// Serialize `function`. If we have `high_function` (the decompilation
		// of function) then we will serialize its type information. Otherwise,
		// we will serialize the type information of `function`. If
		// `visit_pcode` is true, then this is a function for which we want to
		// fully lift, i.e. visit all the high p-code.
		private void serialize(
				HighFunction high_function, Function function,
				boolean visit_pcode) throws Exception {
			
			temporary_address.clear();
			old_locals.clear();
			missing_locals.clear();
			entry_block.clear();

			FunctionPrototype proto = null;
			name("name").value(function.getName());

			// If we have a high P-Code function, then serialize the blocks.
			if (high_function != null) {
				proto = high_function.getFunctionPrototype();

				name("type").beginObject();
				
				int num_params = 0;
				if (proto != null) {
					num_params = serializePrototype(proto);
				} else {
					num_params = serializePrototype(function.getSignature());
				}
				endObject();  // End `type`.

				if (visit_pcode && fixupOperations(high_function, num_params)) {
					
					PcodeBlockBasic first_block = null;
					current_function = high_function;

					name("basic_blocks").beginObject();
					for (PcodeBlockBasic block : high_function.getBasicBlocks()) {
						if (first_block == null) {
							first_block = block;
						}

						name(label(block)).beginObject();
						serialize(block);
						endObject();
					}

					// If we created a fake entry block to represent variable
					// declarations then emit that here.
					if (!entry_block.isEmpty()) {
						serializeEntryBlock(first_block);
					}
					
					endObject();  // End of `basic_blocks`.
					current_function = null;
					
					if (!entry_block.isEmpty()) {
						name("entry_block").value("entry");

					} else if (first_block != null) {
						name("entry_block").value(label(first_block));
					}
				}
			} else {
				name("type").beginObject();
				serializePrototype(function.getSignature());
				endObject();  // End `type`.
			}
		}
		
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
            "__register_frame_info_bases", "__assert_fail"
		);

		// Don't try to recover the definitions of some functions.
		private static boolean ignoreFunction(String name) {
			return IGNORED_NAMES.contains(name);
		}

		// Serialize the input function list to JSON. This function will also
		// serialize type information related to referenced functions and
		// variables.
		public void serialize() throws Exception {

			beginObject();
			name("arch").value(getArch());
			name("format").value(currentProgram.getExecutableFormat());
			name("functions").beginObject();

			for (int i = 0; i < functions.size(); ++i) {
				Function function = functions.get(i);
				Address function_address = function.getEntryPoint();
				if (!seen_functions.add(function_address)) {
					continue;
				}

				DecompileResults res = ifc.decompileFunction(function, DECOMPILATION_TIMEOUT, null);
				HighFunction high_function = res.getHighFunction();
				name(label(function)).beginObject();
				serialize(
						high_function,
						function,
						(i < original_functions_size &&
								!ignoreFunction(function.getName())));
				endObject();
			}

			endObject();  // End of functions.

			name("types").beginObject();
			serializeTypes();
			endObject();  // End of types.

			endObject();
		}
	}

	private String getArch() throws Exception {
		if (currentProgram.getLanguage() == null ||
				currentProgram.getLanguage().getProcessor() == null) {
			return "unknown";
		}
		return currentProgram.getLanguage().getProcessor().toString();
	}

	private DecompInterface getDecompilerInterface() throws Exception {
		if (currentProgram == null) {
			throw new Exception("Unable to initialize decompiler: invalid current program.");
		}
		DecompInterface decompiler = new DecompInterface();
		decompiler.setOptions(new DecompileOptions());
		if (!decompiler.openProgram(currentProgram)) {
			throw new Exception("Unable to initialize decompiler: " + decompiler.getLastMessage());
		}
		return decompiler;
	}

	private void serializeToFile(Path file, List<Function> functions) throws Exception {
		if (file == null || functions == null || functions.isEmpty()) {
			throw new IllegalArgumentException("Invalid file path or empty function list");
		}

		final var serializer = new PcodeSerializer(
				Files.newBufferedWriter(file), getArch(),
				currentProgram.getFunctionManager(), currentProgram.getExternalManager(),
				getDecompilerInterface(), new BasicBlockModel(currentProgram), functions);
		serializer.serialize();
		serializer.close();
	}

	private List<Function> getAllFunctions() {
		if (currentProgram == null || currentProgram.getFunctionManager() == null) {
			return Collections.emptyList();
		}
		FunctionIterator functionIter = currentProgram.getFunctionManager().getFunctions(true);
		List<Function> functions = new ArrayList<>();
		while (functionIter.hasNext() && !monitor.isCancelled()) {
			functions.add(functionIter.next());
		}
		return functions;
	}

	private void decompileSingleFunction() throws Exception {
		if (getScriptArgs().length < 3) {
			throw new IllegalArgumentException("Insufficient arguments. Expected: <function_name> <output_file> as argument");
		}
		serializeToFile(Path.of(getScriptArgs()[2]), getGlobalFunctions(getScriptArgs()[1]));
	}

	private void decompileAllFunctions() throws Exception {
		if (getScriptArgs().length < 2) {
			throw new IllegalArgumentException("Insufficient arguments. Expected: <output_file> as argument");
		}
		serializeToFile(Path.of(getScriptArgs()[1]), getAllFunctions());
	}

	private void runHeadless() throws Exception {
		if (getScriptArgs().length < 1) {
			throw new IllegalArgumentException("mode is not specified for headless execution");
		}

		// Execution mode
		String mode = getScriptArgs()[0];
		println("Running in mode: " + mode);
		switch (mode.toLowerCase()) {
		case "single":
			decompileSingleFunction();
			break;
		case "all":
			decompileAllFunctions();
			break;
		default:
			throw new IllegalArgumentException("Invalid mode: " + mode);
		}
	}

	private void decompileSingleFunctionInGUI() throws Exception {
		List<Function> functions = null;
		if (currentProgram != null) {
			FunctionManager manager = currentProgram.getFunctionManager();
			if (manager != null) {
				Function function = manager.getFunctionContaining(currentAddress);
				if (function != null) {
					functions = new ArrayList<>();
					functions.add(function);
				}
			}
		}

		if (functions == null) {
			String functionNameArg = askString("functionNameArg", "Function name to decompile: ");
			functions = getGlobalFunctions(functionNameArg);
		}

		File outputDirectory = askDirectory("outputFilePath", "Select output directory");
		File outputFilePath = new File(outputDirectory, "patchestry.json");
		serializeToFile(outputFilePath.toPath(), functions);
	}

	private void decompileAllFunctionsInGUI() throws Exception {
		File outputDirectory = askDirectory("outputFilePath", "Select output directory");
		File outputFilePath = new File(outputDirectory, "patchestry.json");
		serializeToFile(outputFilePath.toPath(), getAllFunctions());
	}

	// GUI mode execution
	private void runGUI() throws Exception {
		String mode = askString("mode", "Please enter mode:");
		println("Running in mode: " + mode);
		switch (mode.toLowerCase()) {
		case "single":
			decompileSingleFunctionInGUI();
			break;
		case "all":
			decompileAllFunctionsInGUI();
			break;
		default:
			throw new IllegalArgumentException("Invalid mode: " + mode);
		}
	}

	// Script entry point
	@Override
	public void run() throws Exception {
		try {
			if (isRunningHeadless()) {
				runHeadless();
			} else {
				runGUI();
			}
		} catch (Exception e) {
			println("Error: " + e.getMessage());
			e.printStackTrace(new PrintWriter(new OutputStreamWriter(System.err)));
			throw e;
		}
	}
}
