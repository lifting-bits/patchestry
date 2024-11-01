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

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressFactory;
import ghidra.program.model.address.AddressSpace;

import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;

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
import ghidra.program.model.listing.Program;
import ghidra.program.model.listing.StackFrame;
import ghidra.program.model.listing.Variable;
import ghidra.program.model.listing.VariableStorage;

import ghidra.program.model.pcode.FunctionPrototype;
import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.HighConstant;
import ghidra.program.model.pcode.HighLocal;
import ghidra.program.model.pcode.HighOther;
import ghidra.program.model.pcode.HighParam;
import ghidra.program.model.pcode.HighSymbol;
import ghidra.program.model.pcode.HighVariable;
import ghidra.program.model.pcode.LocalSymbolMap;
import ghidra.program.model.pcode.PcodeBlock;
import ghidra.program.model.pcode.PcodeBlockBasic;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.SequenceNumber;
import ghidra.program.model.pcode.SymbolEntry;
import ghidra.program.model.pcode.Varnode;

import ghidra.program.model.data.AbstractFloatDataType;
import ghidra.program.model.data.AbstractIntegerDataType;
import ghidra.program.model.data.Array;
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
	protected static final int DECLARE_REGISTER = MIN_CALLOTHER + 2;
	
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
		private String arch;
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
		
		// Maps `HighVariables` (really, `HighOther`s) that are attached to
		// register `Varnode`s to the `PcodeOp` containing those nodes. We
		// The same-named register may be associated with many such independent
		// `HighVariable`s, so to distinguish them to downstream readers of the
		// JSON, we want to 'version' the register variables by their initial
		// user.
		private Map<HighVariable, PcodeOp> register_address;

		public PcodeSerializer(java.io.BufferedWriter writer,
				String arch_, FunctionManager fm_,
				ExternalManager em_, DecompInterface ifc_,
				BasicBlockModel bbm_,
				List<Function> functions_) {
			super(writer);

			Program program = fm_.getProgram();

			this.language = (SleighLanguage) program.getLanguage();
			AddressFactory address_factory = program.getAddressFactory();

			this.arch = arch_;
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
			this.missing_locals = new HashMap<>();
			this.old_locals = new HashMap<>();
			this.register_address = new HashMap<>();
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
			} else {
				throw new Exception("Unhandled type: " + data_type.toString());
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
		
		private void serializePrototype() throws Exception {
			name("return_type").value(label((DataType) null));
			name("is_variadic").value(false);
			name("is_variadic").value(false);
			name("parameter_types").beginArray().endArray();
		}

		private void serializePrototype(FunctionPrototype proto) throws Exception {
			if (proto == null) {
				serializePrototype();
				return;
			}

			name("return_type").value(label(proto.getReturnType()));
			name("is_variadic").value(proto.isVarArg());
			name("is_noreturn").value(proto.hasNoReturn());
			
			name("parameter_types").beginArray();
			for (int i = 0; i < proto.getNumParams(); i++) {
				value(label(proto.getParam(i).getDataType()));
			}
			endArray();  // End of `parameter_types`.
		}
		
		private void serializePrototype(FunctionSignature proto) throws Exception {
			if (proto == null) {
				serializePrototype();
				return;
			}

			name("return_type").value(label(proto.getReturnType()));
			name("is_variadic").value(proto.hasVarArgs());
			name("is_noreturn").value(proto.hasNoReturn());
			name("calling_convention").value(proto.getCallingConventionName());
			
			ParameterDefinition[] arguments = proto.getArguments();
			name("parameter_types").beginArray();
			for (int i = 0; i < arguments.length; i++) {
				value(label(arguments[i].getDataType()));
			}
			endArray();  // End of `parameter_types`.
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
			REGISTER,
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
			}

			HighSymbol symbol = var.getSymbol();
			if (symbol != null) {
				if (symbol.isGlobal()) {
					return VariableClassification.GLOBAL;
				} else if (symbol.isParameter() || symbol.isThisPointer()) {
					return VariableClassification.PARAMETER;
				}
			}
			
			if (var instanceof HighOther) {
				Varnode rep = originalRepresentativeOf(var);

//				println("OTHER var: " + var.toString());
//				for (Varnode inst : var.getInstances()) {
//					if (inst.isInput()) {
//						println("  -- input: " + inst.toString());
//					} else {
//						println("  -- output: " + inst.toString());
//						PcodeOp op = inst.getDef();
//						if (op != null) {
//							println("     -> " + op.toString() + " at " + label(op));
//						}
//					}
//				}

				if (rep != null) {
					if (rep.isRegister()) {
						return VariableClassification.REGISTER;

					// TODO(pag): Consider checking if all uses of the unique
					//			  belong to the same block. We don't want to
					//			  introduce a kind of code motion risk into the
					//			  lifted representation.
					} else if (rep.isUnique()) {
						
						PcodeOp def = rep.getDef();
						if (def != null && def.getOpcode() == PcodeOp.PTRSUB &&
								def.getInput(0).isRegister()) {
							println("Unique-def: " + label(def));
							println(def.toString());
							
							Function func = current_function.getFunction();
							StackFrame frame = func.getStackFrame();
							HighVariable v = variableOf(def);
							if (v != null) {
								println("  output high: " + v.toString() + " " + v.getName());
							}
							
							v = def.getInput(0).getHigh();
							if (v != null) {
								println("  reg high: " + v.toString() + " " + v.getName());
							}

							v = def.getInput(1).getHigh();
							if (v != null) {
								println("  offset high: " + v.toString() + " " + v.getName());
								println("  offset value: " + Long.toString(v.getRepresentative().getOffset()));
								HighConstant x;
								Variable ov = frame.getVariableContaining(v.getOffset());
								if (ov != null) {
									println("    offset var: " + ov.toString());
								}
							}
						}
						
						return VariableClassification.TEMPORARY;
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
				case VariableClassification.REGISTER:
					name("kind").value("register");
					name("operation").value(label(getOrCreateLocalVariable(var, op)));
					break;
				case VariableClassification.TEMPORARY:
					assert def != null;
					assert def != null;
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
			
			// Given a stack pointer offset, e.g. `-0x118`, go find the low
			// `Variable` representing `local_118`.
			Variable var = frame.getVariableContaining(var_offset);
			if (var == null) {
				return false;
			}
			
			println("Found variable use " + var.toString() + " at offset " + var_offset + " rel to " + Integer.toString(var.getStackOffset()));
			
			// Given the local symbol mapping for the high function, go find
			// a `HighSymbol` corresponding to `local_118`. This high symbol
			// will generally have a much better `DataType`, but initially
			// and confusingly won't have a corresponding `HighVariable`.
			LocalSymbolMap symbols = high_function.getLocalSymbolMap();
			Address pc = op.getSeqnum().getTarget();
			HighSymbol sym = symbols.findLocal(var.getVariableStorage(), pc);
			
			if (sym == null) {
				return false;
			}
			
			Varnode var_node = op.getInput(0);
			UseVarnode new_var_node = new UseVarnode(
					stack_space.getAddress(var.getStackOffset()), sym.getSize());

			// We've already got a high variable for this missing local.
			HighVariable new_var = sym.getHighVariable();
			String sym_name = sym.getName();
			if (new_var != null && !new_var.getName().equals("UNNAMED")) {
				println("Using existing high sym " + sym_name + " with var named " + new_var.getName() + " and type " + new_var.getDataType().toString());

			// We need to invent a new `HighVariable` for this `HighSymbol`.
			// Unfortunately we can't use `HighSymbol.setHighVariable` for the
			// caching, so we need `missing_locals`.
			} else {
				HighLocal local = old_locals.get(new_var);
				if (local == null) {
					local = missing_locals.get(sym_name);
				} else {
					if (missing_locals.get(sym_name) != local) {
						println("!!! WHAT??");
					}
				}

				if (local == null) {
					local = new HighLocal(
							sym.getDataType(), new_var_node, null, pc, sym);
					missing_locals.put(sym_name, local);
					
					println("Created " + local.getName() + " with type " + local.getDataType().toString());
						
					// Remap old-to-new.
					if (new_var != null) {
						old_locals.put(new_var, local);
					}
					
				} else {
					println("  Mapped to previously missing local with name " + local.getName());
				}
				new_var = local;
			}

			
			println("  Rewriting: " + op.toString());

			// Rewrite the stack reference to point to the `HighVariable`.
			new_var_node.setHigh(new_var);
			op.setInput(new_var_node, 0);
			
			// Rewrite the offset.
			int new_var_offset = new_var.getOffset() == -1 ? 0 : new_var.getOffset();
			int adjust_offset = (var_offset - var.getStackOffset()) + new_var_offset;
			op.setInput(new Varnode(constant_space.getAddress(adjust_offset), 4), 1);
			
			println("  to: " + op.toString());
			
			return true;
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
		private boolean createMissingLocals(HighFunction high_function) throws Exception {
			Function function = high_function.getFunction();

			// Fill in the parameters first so that they are the first
			// things added to `entry_block`.
			LocalSymbolMap symbols = high_function.getLocalSymbolMap();
			for (int i = 0; i < symbols.getNumParams(); ++i) {
				HighParam param = symbols.getParam(i);
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
					int input_index = referencesStackPointer(op);
					if (input_index == -1) {
						continue;
					}

					switch (op.getOpcode()) {
						case PcodeOp.PTRSUB:
							if (input_index != 0 || !createLocalForPtrSubcomponent(high_function, op)) {
								println("Unsupported stack pointer reference at " + label(op) + ": " + op.toString());
								return false;
							}
							break;
							
						default:
							println("Unsupported stack pointer reference at " + label(op) + ": " + op.toString());
							return false;
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
				case VariableClassification.REGISTER:
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
			} else if (klass == VariableClassification.REGISTER) {
				name("kind").value("register");
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
		private PcodeOp createParamVarDecl(HighVariable var) throws Exception {
			HighParam param = (HighParam) var;
			Address address = nextUniqueAddress();
			DefinitionVarnode def = new DefinitionVarnode(address, var.getSize());
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
		private PcodeOp createRegisterDecl(
				HighVariable var, PcodeOp user_op) throws Exception {
			Varnode rep = originalRepresentativeOf(var);
			assert rep.isRegister();

			Address address = nextUniqueAddress();
			DefinitionVarnode def = new DefinitionVarnode(address, var.getSize());
			Varnode[] ins = new Varnode[1];
			SequenceNumber loc = new SequenceNumber(address, next_seqnum++);
			PcodeOp op = new PcodeOp(loc, PcodeOp.CALLOTHER, 1, def);
			op.insertInput(new Varnode(constant_space.getAddress(DECLARE_REGISTER), 4), 0);
			def.setDef(var, op);

			Varnode[] instances = var.getInstances();
			Varnode[] new_instances = new Varnode[instances.length + 1];
			System.arraycopy(instances, 0, new_instances, 1, instances.length);
			new_instances[0] = def;

			var.attachInstances(new_instances, def);

			entry_block.add(op);
			register_address.put(var, user_op);

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
					return createParamVarDecl(var);
				case VariableClassification.LOCAL:
					return createLocalVarDecl(var);
				case VariableClassification.REGISTER:
					return createRegisterDecl(var, user_op);
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
				endObject();

			} else {
				serializeInput(op, rValueOf(target_node));
			}
			
			name("arguments").beginArray();			
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
			if (var instanceof HighParam) {
				name("index").value(((HighParam) var).getSlot());
			}
		}

		// Serializes a pseudo-op `DECLARE_LOCAL_VAR`, which is actually encoded
		// as a `CALLOTHER`.
		private void serializeDeclareLocalVar(PcodeOp op) throws Exception {
			HighVariable var = variableOf(op);
			HighSymbol sym = var.getSymbol();
			if (sym != null && var.getOffset() == -1 && var.getName().equals("UNNAMED")) {
				name("name").value(sym.getName());
				name("type").value(label(sym.getDataType()));
			} else {
				name("name").value(var.getName());
				name("type").value(label(var.getDataType()));
			}
		}
		
		private void serializeDeclareRegister(PcodeOp op) throws Exception {
			HighVariable var = variableOf(op);
			Varnode rep = originalRepresentativeOf(var);
			assert rep.isRegister();
			
			Register reg = language.getRegister(rep.getAddress(), 0);

			// NOTE(pag): In practice, the `HighOther`s name associated with
			//			  this register is probably `UNNAMED`, which happens in
			//			  `HighOther.decode`; however, we'll be cautious and
			// 			  only canonicalize on the register name if the name
			//			  it is the default.
			if (var.getName().equals("UNNAMED")) {
				name("name").value(reg.getName());
			} else {
				name("name").value(var.getName());
			}
			
			name("type").value(label(var.getDataType()));
			
			// NOTE(pag): The same register might appear multiple times, though
			//			  we can't guarantee that they will appear with the
			//			  same names. Thus, we want to record the address of
			//            the operation using the original register as a kind of
			//			  SSA-like version number downstream, e.g. in a Clang
			//			  AST.
			PcodeOp user_op = register_address.get(var);
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
			case DECLARE_REGISTER:
				serializeDeclareRegister(op);
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
					return "DECLARE_PARAM_VAR";
				case DECLARE_LOCAL_VAR:
					return "DECLARE_LOCAL_VAR";
				case DECLARE_REGISTER:
					return "DECLARE_REGISTER";
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

				// TODO(pag): What about `isAddress()`? How do `MULTIEQUAL`s
				//			  interact with global variables in RAM?
				if (node.isConstant() || node.isHash()) {
					return false;
				}
				
				if (high != variableOf(node)) {
					return false;
				}
			}
			
			return true;
		}
		
		// Serialize a high p-code basic block. This iterates over the p-code
		// operations within the block and serializes them individually.
		private void serialize(PcodeBlockBasic block) throws Exception {
			PcodeBlock parent_block = block.getParent();
			if (parent_block != null) {
				name("parent_block").value(label(parent_block));
			}

			PcodeOp op = null;
			Iterator<PcodeOp> op_iterator = block.getIterator();
			name("operations").beginObject();
			while (op_iterator.hasNext()) {
				op = op_iterator.next();
				
				switch (op.getOpcode()) {
					// NOTE(pag): INDIRECTs seem like a good way of modelling may-
					//            alias relations, as well as embedding control
					//            dependencies into the dataflow graph, e.g. to
					//            ensure code motion cannot happen from after a CALL
					//            to before a CALL, especially for stuff operating
					//            on stack slots. The idea at the time of this
					//            comment is that we will assume that eventual
					//            codegen also should not do any reordering, though
					//            enforcing that is also tricky.
					case PcodeOp.INDIRECT:
						break;
					
					// MULTIEQUALs are Ghidra's form of SSA-form PHI nodes.
					case PcodeOp.MULTIEQUAL:
						if (canElideMultiEqual(op)) {
							break;
						}
						
						// Fall-through.

					default:
						name(label(op));
						current_block = block;
						serialize(op);
						current_block = null;
						break;
				}
			}
			endObject();

			// List out the operations in their order.
			op_iterator = block.getIterator();
			name("ordered_operations").beginArray();
			while (op_iterator.hasNext()) {
				beginArray();
				value(mnemonic(op));
				value(label(op_iterator.next()));
				endArray();
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
			
			register_address.clear();
			old_locals.clear();
			missing_locals.clear();

			FunctionPrototype proto = null;
			name("name").value(function.getName());

			// If we have a high P-Code function, then serialize the blocks.
			if (high_function != null) {
				proto = high_function.getFunctionPrototype();

				name("type").beginObject();
				if (proto != null) {
					serializePrototype(proto);
				} else {
					FunctionSignature signature = function.getSignature();
					serializePrototype(signature);
				}
				endObject();  // End `prototype`.

				if (visit_pcode && createMissingLocals(high_function)) {
					
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
						entry_block.clear();

					} else if (first_block != null) {
						name("entry_block").value(label(first_block));
					}
				}
			}
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
				serialize(high_function, function, i < original_functions_size);
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
