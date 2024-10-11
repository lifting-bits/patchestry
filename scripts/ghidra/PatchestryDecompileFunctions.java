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

import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.PcodeBlock;
import ghidra.program.model.pcode.PcodeBlockBasic;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.SequenceNumber;
import ghidra.program.model.pcode.Varnode;

import ghidra.program.model.symbol.ExternalManager;

import com.google.gson.stream.JsonWriter;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.OutputStreamWriter;
import java.io.File;

import java.nio.file.Files;
import java.nio.file.Path;

import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;

public class PatchestryDecompileFunctions extends GhidraScript {
	
    private class PcodeSerializer extends JsonWriter {
    	private String arch;
    	private FunctionManager fm;
    	private ExternalManager em;
    	private DecompInterface ifc;
    	private BasicBlockModel bbm;
    	private List<Function> functions;
    	private int original_functions_size;
    	private Set<Address> seen_functions;
    	
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
        }
        
        private static String label(Address address) throws Exception {
    		return address.toString(true  /* show address space prefix */);
        }
        
        private static String label(PcodeBlock block) throws Exception {
        	return Integer.toString(block.getIndex());
        }

        private static String label(SequenceNumber sn) throws Exception {
        	return label(sn.getTarget()) + Address.SEPARATOR +
        		   Integer.toString(sn.getTime()) + Address.SEPARATOR +
        		   Integer.toString(sn.getOrder());
        }

        private void serialize(Varnode node) throws Exception {
            if (node == null) {
                nullValue();
                return;
            }

            beginObject();

            if (node.isConstant()) {
                name("type").value("const");
            } else if (node.isUnique()) {
                name("type").value("unique");
            } else if (node.isRegister()) {
                name("type").value("register");
            } else if (node.isAddress()) {
                name("type").value("ram");
            } else if (node.getAddress().isStackAddress()) {
                name("type").value("stack");
            } else {
                throw new Exception("Unknown Varnode kind: " + node.toString());
            }

            name("offset").value(node.getOffset());
            name("size").value(node.getSize());

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
    		String target_address_string = label(target_address);
    		Function callee = fm.getFunctionAt(target_address);
    		
    		// `target_address` may be a pointer to an external. Figure out
    		// what we're calling.
    		if (callee == null) {
    			callee = fm.getReferencedFunction(target_address);
    			if (callee != null) {
    				target_address = callee.getEntryPoint();
    				println("Call through " + target_address_string +
    						" targets " + callee.getName() +
    						" at " + label(target_address));
    				target_address_string = label(target_address);
    			}
    		}
    		
    		name("target_address").value(target_address_string);

    		if (callee != null) {
    			functions.add(callee);
    		} else {
    			println("Could not find function at address " + target_address_string +
    					" called by " + caller_address.toString());
    		}
        }
        
        // Serialize a conditional branch. This records the targeted blocks.
        //
        // TODO(pag): How does p-code handle delay slots? Are they separate
        //		      blocks?
        //
        // TODO(pag): Ian as previously mentioned how the true/false targets
        //			  can be reversed. Investigate this.
        private void serializeCondBranchOp(PcodeBlockBasic block, PcodeOp op) throws Exception {
        	name("taken_block").value(label(block.getTrueOut()));
        	name("not_taken_block").value(label(block.getFalseOut()));
        }
        
        // Serialize a generic multi-input, single-output p-code operation.
        private void serializeGenericOp(PcodeOp op) throws Exception {
        	name("output");
            serialize(op.getOutput());
            name("inputs").beginArray();
            for (var input : op.getInputs()) {
                serialize(input);
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
            name("pcode").beginArray();
            PcodeOp last_op = null;
            Iterator<PcodeOp> op_iterator = block.getIterator();
            while (op_iterator.hasNext()) {
            	last_op = op_iterator.next();
                serialize(function, block, last_op);
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

        		DecompileResults res = ifc.decompileFunction(function, 30, null);
        		HighFunction high_function = res.getHighFunction();

        		name(label(function_address)).beginObject();
        		serialize(high_function, function, i < original_functions_size);
        		endObject();
            }
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
