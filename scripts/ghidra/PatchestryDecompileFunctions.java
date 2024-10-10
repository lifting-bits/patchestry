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
import ghidra.program.model.pcode.PcodeBlockBasic;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.Varnode;

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
	
    public class PcodeSerializer extends JsonWriter {
    	private String arch;
    	private DecompInterface ifc;
    	private BasicBlockModel bbm;
    	private List<Function> functions;
    	private Set<Address> seen_functions;
    	
        public PcodeSerializer(java.io.BufferedWriter writer,
        					   String arch_, DecompInterface ifc_,
        					   BasicBlockModel bbm_, List<Function> functions_) {
            super(writer);
            this.arch = arch_;
            this.ifc = ifc_;
            this.bbm = bbm_;
            this.functions = functions_;
            this.seen_functions = new TreeSet<>();
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
            return;
        }

        private void serialize(PcodeOp op) throws Exception {
            beginObject();
            name("mnemonic").value(op.getMnemonic());
            name("address").value(op.getSeqnum().getTarget().toString());
            name("output");
            serialize(op.getOutput());
            name("inputs").beginArray();
            for (var input : op.getInputs()) {
                serialize(input);
            }
            endArray().endObject();
        }

        private void serialize(PcodeBlockBasic block) throws Exception {
            name("pcode").beginArray();
            PcodeOp last_op = null;
            Iterator<PcodeOp> op_iterator = block.getIterator();
            while (op_iterator.hasNext()) {
            	last_op = op_iterator.next();
                serialize(last_op);
            }
            endArray();
        
            // TODO(pag): How does P-Code handle delay slots? Are they separate
            //		      blocks?
            if (last_op.getOpcode() == PcodeOp.CBRANCH) {
            	name("taken_block").value(Integer.toString(block.getTrueOut().getIndex()));
            	name("not_taken_block").value(Integer.toString(block.getFalseOut().getIndex()));
            }
        }

        private void serialize(
    		HighFunction high_function, Function function) throws Exception {
            
            name("name").value(function.getName());
            
            // If we have a high P-Code function, then serialize the blocks.
            if (high_function != null) {
                name("basic_blocks").beginObject();
                for (PcodeBlockBasic block : high_function.getBasicBlocks()) {
                	name(Integer.toString(block.getIndex())).beginObject();
                    serialize(block);
                    endObject();
                }
                endObject();
            }
        }

        public JsonWriter serialize() throws Exception {

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

        		name(function_address.toString()).beginObject();
        		serialize(high_function, function);
        		endObject();
            }
            return endObject().endObject();
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
    		Files.newBufferedWriter(file), getArch(), getDecompilerInterface(),
    		new BasicBlockModel(currentProgram), functions);
        serializer.serialize().close();
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
