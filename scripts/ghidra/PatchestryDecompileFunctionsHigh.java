/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import ghidra.app.script.GhidraScript;

import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;

import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;

import ghidra.program.model.listing.Program;

import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.PcodeBlockBasic;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.Varnode;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.decompiler.DecompileException;
import ghidra.app.decompiler.DecompileOptions;

import com.google.gson.stream.JsonWriter;

import java.util.Iterator;
import java.util.Map;
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


public class PatchestryDecompileFunctionsHigh extends GhidraScript {

    int decompile_time = 30;

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

    private String getArch() throws Exception {
        if (currentProgram.getLanguage() == null || currentProgram.getLanguage().getProcessor() == null) {
            return "unknown";
        }
        return currentProgram.getLanguage().getProcessor().toString();
    }

    private String inferOSType(String executableFormat) throws Exception {
        if (executableFormat == null) {
            return "unknown";
        }

        executableFormat = executableFormat.toLowerCase();
        if (executableFormat.contains("pe")) {
            return "windows";
        } else if (executableFormat.contains("elf")) {
            return "linux";
        } else if (executableFormat.contains("mach-o")) {
            return "macos";
        } else {
            return "unknown";
        }
    }

    private String getOS() throws Exception {
        String executableFormat = currentProgram.getExecutableFormat();
        println("executableFormat " + executableFormat);
        return inferOSType(executableFormat);
    }

    public class PcodeSerializer extends JsonWriter {
        public PcodeSerializer(java.io.BufferedWriter writer) {
            super(writer);
        }

        public JsonWriter serialize(Varnode node) throws Exception {
            if (node == null) {
                return nullValue();
            }

            beginArray();

            if (node.isConstant()) {
                value("const"); 
            } else if (node.isUnique()) {
                value("unique"); 
            } else if (node.isRegister()) {
                value("register"); 
            } else if (node.isAddress()) {
                value("ram"); 
            } else {
                throw new Exception("Unknown Varnode kind.");
            }

            value(node.getOffset());
            value(node.getSize());

            return endArray();
        }

        public JsonWriter serialize(PcodeOp op) throws Exception {
            beginObject();
            name("mnemonic").value(op.getMnemonic());
            name("address").value(op.getSeqnum().getTarget().toString());
            name("output");
            serialize(op.getOutput());
            name("inputs").beginArray();
            for (var input : op.getInputs()) {
                serialize(input);
            }
            return endArray().endObject();
        }

        public JsonWriter serialize(PcodeBlockBasic block) throws Exception {
            beginObject();
            name("label").value("bb_" + block.getStart().toString());
            name("pcode").beginArray();
            Iterator<PcodeOp> opIterator = block.getIterator();
            while (opIterator.hasNext()) {
                serialize(opIterator.next());
            }
            return endArray().endObject();
        }

        public JsonWriter serialize(HighFunction function) throws Exception {
            beginObject();
            name("name").value(function.getFunction().getName());
            name("basic_blocks").beginArray();
            for (var curBlock : function.getBasicBlocks()) {
                serialize(curBlock);
            }
            return endArray().endObject();
        }

        public JsonWriter serialize(List<HighFunction> functions) throws Exception {
            beginObject();
            name("arch").value(getArch());
            name("os").value(getOS());
            name("functions").beginArray();
            for (HighFunction function : functions) {
                serialize(function);
            }
            return endArray().endObject();
        }
    }

    private void serializeToFile(Path file, HighFunction function) throws Exception {
        if (file == null || function == null) {
            throw new IllegalArgumentException("Invalid file path or empty function list");
        }
        try (BufferedWriter writer = Files.newBufferedWriter(file)) {
            final var serializer = new PcodeSerializer(writer);
            serializer.serialize(function).close();
        }
    }

    private void serializeToFile(Path file, List<HighFunction> functions) throws Exception {
        if (file == null || functions == null || functions.isEmpty()) {
            throw new IllegalArgumentException("Invalid file path or empty function list");
        }
        final var serializer = new PcodeSerializer(Files.newBufferedWriter(file));
        serializer.serialize(functions).close();
    }

    private List<HighFunction> getAllFunctions(DecompInterface ifc) {
        if (currentProgram == null || currentProgram.getFunctionManager() == null) {
            return Collections.emptyList();
        }
        FunctionIterator functionIter = currentProgram.getFunctionManager().getFunctions(true);
        List<HighFunction> functions = new ArrayList<>();
        while (functionIter.hasNext() && !monitor.isCancelled()) {
            DecompileResults res = ifc.decompileFunction(functionIter.next(), decompile_time, null);
            functions.add(res.getHighFunction());
        }
        return functions;
    }

    private void decompileSingleFunction(String[] args) throws Exception {
        if (args.length < 3) {
            throw new IllegalArgumentException("Insufficient arguments. Expected: <function_name> <output_file> as argument");
        }
        String functionNameArg = args[1];
        String outputFilePath = args[2];
        final var functions = getGlobalFunctions(functionNameArg);
        if (functions.isEmpty()) {
            println("Function not found: " + functionNameArg);
            return;
        }

        if (functions.size() > 1) {
            println("Warning: Found more than one function named: " + functionNameArg);
        }

        println("Serializing function: " + functions.get(0).getName() + " @ " + functions.get(0).getEntryPoint());
        DecompInterface ifc = getDecompilerInterface();
        List<HighFunction> high_functions = new ArrayList<>();

        for (Function func : functions) {
            DecompileResults res = ifc.decompileFunction(func, 30, null);
            high_functions.add(res.getHighFunction());
        }


        // Serialize to the file
        serializeToFile(Path.of(outputFilePath), high_functions);
    }

    private void decompileAllFunctions(String[] args) throws Exception {
        if (args.length < 2) {
            throw new IllegalArgumentException("Insufficient arguments. Expected: <output_file> as argument");
        }
        String outputFilePath = args[1];
        DecompInterface ifc = getDecompilerInterface();
        List<HighFunction> functions = getAllFunctions(ifc);
        if (functions.isEmpty()) {
            println("No functions found in the current program");
            return;
        }

        // Serialize to the file
        serializeToFile(Path.of(outputFilePath), functions);
    }

    private void runHeadless() throws Exception {
        if (getScriptArgs().length < 1) {
            throw new IllegalArgumentException("mode is not specified for headless execution");
        }
        String mode = getScriptArgs()[0];
        println("Running in mode: " + mode);
        switch (mode.toLowerCase()) {
            case "single":
                decompileSingleFunction(getScriptArgs());
                break;
            case "all":
                decompileAllFunctions(getScriptArgs());
                break;
            default:
                throw new IllegalArgumentException("Invalid mode: " + mode);
        }
    }

    private String[] getArguments() throws Exception {
        String mode = askString("mode", "Please enter mode:");
        if (mode.toLowerCase() == "single") {
            String functionNameArg = askString("functionNameArg", "Function name to decompile:");
            File outputDirectory = askDirectory("outputFilePath", "Select output directory");
            String outputFilePath = new File(outputDirectory, "patchestry.json").getAbsolutePath();
            return new String[] { mode, functionNameArg, outputFilePath };
        } else if (mode.toLowerCase() == "all") {
            File outputDirectory = askDirectory("outputFilePath", "Select output directory");
            String outputFilePath = new File(outputDirectory, "patchestry.json").getAbsolutePath();
            return new String[] { mode, outputFilePath};
        } else {
            throw new IllegalArgumentException("Invalid mode: " + mode);
        }
    }

    // GUI mode execution
    private void runGUI() throws Exception {
        String mode = askString("mode", "Please enter mode:");
        println("Running in mode: " + mode);
        switch (mode.toLowerCase()) {
            case "single":
                decompileSingleFunction(getArguments());
                break;
            case "all":
                decompileAllFunctions(getArguments());
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