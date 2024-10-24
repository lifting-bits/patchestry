/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import ghidra.app.script.GhidraScript;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;

import ghidra.program.model.address.Address;

import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.FunctionManager;

import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;

import ghidra.program.model.listing.Program;

import ghidra.program.model.pcode.FunctionPrototype;
import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.HighParam;
import ghidra.program.model.pcode.HighVariable;
import ghidra.program.model.pcode.PcodeBlock;
import ghidra.program.model.pcode.PcodeBlockBasic;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.SequenceNumber;
import ghidra.program.model.pcode.Varnode;

import ghidra.program.model.data.AbstractFloatDataType;
import ghidra.program.model.data.AbstractIntegerDataType;
import ghidra.program.model.data.Array;
import ghidra.program.model.data.BooleanDataType;
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
import java.util.Set;
import java.util.TreeSet;

public class PatchestryDecompileFunctions extends GhidraScript {

    static final int decompilation_timeout = 30;
	
    private class PcodeSerializer extends JsonWriter {
    	private String arch;
    	private FunctionManager fm;
    	private ExternalManager em;
    	private DecompInterface ifc;
    	private BasicBlockModel bbm;
    	private List<Function> functions;
    	private int original_functions_size;
    	private Set<Address> seen_functions;
        private Set<String> seen_types;
        private List<DataType> types_to_serialize;
    	
        public PcodeSerializer(java.io.BufferedWriter writer,
        					   String arch_, FunctionManager fm_,
        					   ExternalManager em_, DecompInterface ifc_,
        					   BasicBlockModel bbm_,
        					   List<Function> functions_) {
            super(writer);
            this.arch = arch_;
            this.fm = fm_;
            this.em = em_;
            this.ifc = ifc_;
            this.bbm = bbm_;
            this.functions = functions_;
            this.original_functions_size = functions.size();
            this.seen_functions = new TreeSet<>();
            this.seen_types = new HashSet<>();
            this.types_to_serialize = new ArrayList<>();
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
        
        // Return the r-value of a varnode.
        private Varnode rValueOf(Varnode node) throws Exception {
        	while (true) {
        		PcodeOp def = node.getDef();
        		if (def == null) {
        			break;
        		}
        		
        		if (def.getOpcode() != PcodeOp.INDIRECT) {
        			break;
        		}
        		
        		Varnode i0 = def.getInput(0);
        		if (!i0.isConstant()) {
        			break;
        		}
        		
        		assert i0.getOffset() == 0;
        		node = def.getInput(1);
        	}
        	
        	return node;
        }
        
        // Return the l-value of a varnode.
        private HighVariable lValueOf(Varnode node) throws Exception {
        	HighVariable var = node.getHigh();
        	while (var == null) {
        		PcodeOp def = node.getDef();
        		if (def == null) {
        			break;
        		}
        		
        		if (def.getOpcode() != PcodeOp.INDIRECT) {
        			break;
        		}
        		
        		Varnode i0 = def.getInput(0);
        		
        		// A constant varnode for input 0 in an INDIRECT op means that
        		// the referenced operation producing input 1 is the producer
        		// of the value.
        		if (i0.isConstant()) {
        			assert i0.getOffset() == 0;
        			break;
        		}
        		
        		node = i0;
        		var = node.getHigh();
        	}
        	
        	return var;
        }

        private void serializePointerType(Pointer ptr) throws Exception {
            name("name").value(ptr.getDisplayName());
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
            name("name").value(arr.getDisplayName());
            name("kind").value("array");
            name("size").value(arr.getLength());
            name("num_elements").value(arr.getNumElements());
            name("element_type").value(label(arr.getDataType()));
        }
            
        private void serializeBuiltinType(DataType data_type, String kind) throws Exception {
            name("name").value(data_type.getDisplayName());
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

        private void serializeFunctionDefinition(FunctionDefinition fd) throws Exception {
            name("name").value(fd.getDisplayName());
            name("kind").value("function");
            name("return_type").value(label(fd.getReturnType()));
            name("is_variadic").value(fd.hasVarArgs());
            ParameterDefinition[] arguments = fd.getArguments();
            name("parameters").beginArray();
            for (int i = 0; i < arguments.length; i++) {
                beginObject();
                String name = arguments[i].getName();
                if (name != null && !name.isEmpty()) {
                    name("name").value(arguments[i].getName());
                }
                name("size").value(arguments[i].getLength());
                name("type").value(label(arguments[i].getDataType()));
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
                serializeFunctionDefinition((FunctionDefinition) data_type);
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

        private void serialize(FunctionPrototype proto) throws Exception {
            if (proto == null) {
                nullValue();
                return;
            }

            name("parameters").beginArray();
            for (int i = 0; i < proto.getNumParams(); i++) {
                HighVariable hv = proto.getParam(i).getHighVariable();
                // Assert if hv is not an instance of HighParam
                assert hv instanceof HighParam;
                if (hv != null) {
                    beginObject();
                    String hv_name = hv.getName();
                    if (hv_name != null && !hv_name.isEmpty()) {
                        name("name").value(hv_name);
                    }
                    name("type").value(label(hv.getDataType()));
                    endObject();
                }
            }
            endArray();
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

        private void serialize(Varnode node) throws Exception {
            if (node == null) {
            	assert false;
                nullValue();
                return;
            }
            
            // Make sure INDIRECTs don't leak back into our output. We won't
            // have the ability to reference them.
            PcodeOp def = node.getDef();
            if (def != null) {
            	assert def.getOpcode() != PcodeOp.INDIRECT;
            }

            beginObject();

            Address address = node.getAddress();
            name("space").value(address.getAddressSpace().getName());
            name("offset").value(node.getOffset());
            name("size").value(node.getSize());
            name("address").value(label(node.getAddress()));
            HighVariable high_var = node.getHigh();
            if (high_var != null) {
                name("variable").beginObject();
                name("name").value(high_var.getName());
                name("type").value(label(high_var.getDataType()));
                endObject();
            }
            endObject();
        }

        // Serialize a direct call. This enqueues the targeted for type lifting
        // `Function` if it can be resolved.
        private void serializeDirectCallOp(Address caller_address, PcodeOp op) throws Exception {
        	Varnode target_address_node = op.getInput(0);
    		if (!target_address_node.isAddress()) {
    			throw new Exception("Unexpected non-address input to CALL");
    		}

    		Address target_address = caller_address.getNewAddress(target_address_node.getOffset());
    		String target_label = label(target_address);
    		Function callee = fm.getFunctionAt(target_address);
    		
    		// `target_address` may be a pointer to an external. Figure out
    		// what we're calling.
    		if (callee == null) {
    			callee = fm.getReferencedFunction(target_address);
    			if (callee != null) {
    				target_address = callee.getEntryPoint();
    				println("Call through " + target_label +
    						" targets " + callee.getName() +
    						" at " + label(target_address));
    				target_label = label(target_address);
    			}
    		}
    		
    		name("target_address").value(target_label);

    		if (callee != null) {
    			functions.add(callee);
    		} else {
    			println("Could not find function at address " + target_label +
    					" called by " + caller_address.toString());
    		}
        }
        
        // Serialize an unconditional branch. This records the targeted block.
        private void serializeBranchOp(PcodeBlockBasic block, PcodeOp op) throws Exception {
        	assert block.getOutSize() == 1;
    		name("target_block").value(label(block.getOut(0)));
        }
        
        // Serialize a conditional branch. This records the targeted blocks.
        //
        // TODO(pag): How does p-code handle delay slots? Are they separate
        //		      blocks?
        //
        // XREF(pag): https://github.com/NationalSecurityAgency/ghidra/issues/2736
        //			  describes how the `op` meaning of the branch, i.e. whether
        //			  it branches on zero or not zero, can change over the course
        //			  of simplification, and so the inputs representing the
        //			  branch targets may not actually represent the `true` or
        //			  `false` outputs in the traditional sense.
        private void serializeCondBranchOp(PcodeBlockBasic block, PcodeOp op) throws Exception {
        	name("taken_block").value(label(block.getTrueOut()));
        	name("not_taken_block").value(label(block.getFalseOut()));
        	name("condition");
        	serialize(rValueOf(op.getInput(1)));
        }
        
        // Serialize a generic multi-input, single-output p-code operation.
        private void serializeGenericOp(PcodeOp op) throws Exception {
        	name("output");
            serialize(op.getOutput());
            name("inputs").beginArray();
            for (var input : op.getInputs()) {
                serialize(rValueOf(input));
            }
            endArray();
        }

        private void serialize(HighFunction function, PcodeBlockBasic block, PcodeOp op) throws Exception {
        	Address function_address = function.getFunction().getEntryPoint();
            beginObject();
            name("mnemonic").value(op.getMnemonic());
            name("name").value(label(op.getSeqnum()));
            switch (op.getOpcode()) {
            	case PcodeOp.CALL:
            		serializeDirectCallOp(function_address, op);
            		break;
            	case PcodeOp.CBRANCH:
            		serializeCondBranchOp(block, op);
                	break;
            	case PcodeOp.BRANCH:
            		serializeBranchOp(block, op);
                	break;
            	default:
            		serializeGenericOp(op);
            		break;
            }
            endObject();
        }
        
        // Serialize a high p-code basic block. This iterates over the p-code
        // operations within the block and serializes them individually.
        private void serialize(HighFunction function, PcodeBlockBasic block) throws Exception {
        	PcodeBlock parent_block = block.getParent();
        	if (parent_block != null) {
        		name("parent_block").value(label(parent_block));
        	}
        	
            PcodeOp op = null;
            Iterator<PcodeOp> op_iterator = block.getIterator();
            name("operations").beginObject();
            while (op_iterator.hasNext()) {
            	op = op_iterator.next();
            	
            	// NOTE(pag): INDIRECTs seem like a good way of modelling may-
            	//		      alias relations, as well as embedding control
            	//			  dependencies into the dataflow graph, e.g. to
            	//			  ensure code motion cannot happen from after a CALL
            	//			  to before a CALL, especially for stuff operating
            	//			  on stack slots. The idea at the time of this
            	//			  comment is that we will assume that eventual
            	//			  codegen also should not do any reordering, though
            	//			  enforcing that is also tricky.
            	if (op.getOpcode() != PcodeOp.INDIRECT) {
            		name(label(op));
            		serialize(function, block, op);
            	}
            }
            endObject();
            
            // List out the operations in their order.
            op_iterator = block.getIterator();
            name("ordered_operations").beginArray();
            while (op_iterator.hasNext()) {
            	value(label(op_iterator.next()));
            }
            endArray();
        }

        // Serialize `function`. If we have `high_function` (the decompilation
        // of function) then we will serialize its type information. Otherwise,
        // we will serialize the type information of `function`. If
        // `visit_pcode` is true, then this is a function for which we want to
        // fully lift, i.e. visit all the high p-code.
        private void serialize(HighFunction high_function, Function function, boolean visit_pcode) throws Exception {
            
            name("name").value(function.getName());
            FunctionPrototype proto = high_function.getFunctionPrototype();
            name("prototype").beginObject();
            if (proto != null) {
                serialize(proto);
            }
            endObject();
            
            // If we have a high P-Code function, then serialize the blocks.
            if (high_function != null) {
            	if (visit_pcode) {
	                name("basic_blocks").beginObject();
	                for (PcodeBlockBasic block : high_function.getBasicBlocks()) {
	                	name(label(block)).beginObject();
	                    serialize(high_function, block);
	                    endObject();
	                }
	                endObject();
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

        		DecompileResults res = ifc.decompileFunction(function, decompilation_timeout, null);
        		HighFunction high_function = res.getHighFunction();
                        if (high_function == null) {
                            continue;
                        }
                
        		name(label(function_address)).beginObject();
        		serialize(high_function, function, i < original_functions_size);
        		endObject();
            }
            
            // Serialize Types
            name("types").beginObject();
            serializeTypes();
            endObject();

            endObject().endObject();
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
