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


public class PatchestryDecompileFunctions extends GhidraScript {

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

    private InstructionIterator getInstructions(CodeBlock block) throws Exception {
        if (block == null || currentProgram.getListing() == null) {
            throw new IllegalArgumentException("Invalid block");
        }
        return currentProgram.getListing().getInstructions(block, true);
    }

    private CodeBlockIterator getBasicBlocks(Function function) throws Exception {
        if (function == null) {
            throw new IllegalArgumentException("Invalid function");
        }
        final var model = new BasicBlockModel(currentProgram);
        return model.getCodeBlocksContaining(function.getBody(), monitor);
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
            name("output");
            serialize(op.getOutput());
            name("inputs").beginArray();
            for (var input : op.getInputs()) {
                serialize(input);
            }
            return endArray().endObject();
        }

        public JsonWriter serialize(Instruction instruction) throws Exception {
            beginObject();
            name("mnemonic").value(instruction.getMnemonicString());
            name("address").value(instruction.getAddressString(false, false));
            name("pcode").beginArray();
            for (var curOp : instruction.getPcode()) {
                serialize(curOp);
            }
            return endArray().endObject();
        }

        public JsonWriter serialize(CodeBlock block) throws Exception {
            beginObject();
            name("label").value(block.getName());
            name("instructions").beginArray();
            for (var curInst : getInstructions(block)) {
                serialize(curInst);
            }
            return endArray().endObject();
        }

        public JsonWriter serialize(Function function) throws Exception {
            beginObject();
            name("name").value(function.getName());
            name("basic_blocks").beginArray();
            for (var curBlock : getBasicBlocks(function)) {
                serialize(curBlock);
            }
            return endArray().endObject();
        }

        public JsonWriter serialize(List<Function> functions) throws Exception {
            beginObject();
            name("arch").value(getArch());
            name("os").value(getOS());
            name("functions").beginArray();
            for (Function function : functions) {
                serialize(function);
            }
            return endArray().endObject();
        }
    }

    private void serializeToFile(Path file, Function function) throws Exception {
        if (file == null || function == null) {
            throw new IllegalArgumentException("Invalid file path or empty function list");
        }
        try (BufferedWriter writer = Files.newBufferedWriter(file)) {
            final var serializer = new PcodeSerializer(writer);
            serializer.serialize(function).close();
        }
    }

    private void serializeToFile(Path file, List<Function> functions) throws Exception {
        if (file == null || functions == null || functions.isEmpty()) {
            throw new IllegalArgumentException("Invalid file path or empty function list");
        }
        final var serializer = new PcodeSerializer(Files.newBufferedWriter(file));
        serializer.serialize(functions).close();
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
        String functionNameArg = getScriptArgs()[1];
        String outputFilePath = getScriptArgs()[2];
        final var functions = getGlobalFunctions(functionNameArg);
        if (functions.isEmpty()) {
            println("Function not found: " + functionNameArg);
            return;
        }

        if (functions.size() > 1) {
            println("Warning: Found more than one function named: " + functionNameArg);
        }

        println("Serializing function: " + functions.get(0).getName() + " @ " + functions.get(0).getEntryPoint());

        // Serialize to the file
        serializeToFile(Path.of(outputFilePath), functions);
    }

    private void decompileAllFunctions() throws Exception {
        if (getScriptArgs().length < 2) {
            throw new IllegalArgumentException("Insufficient arguments. Expected: <output_file> as argument");
        }
        String outputFilePath = getScriptArgs()[1];
        List<Function> functions = getAllFunctions();
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
        String functionNameArg = askString("functionNameArg", "Function name to decompile: ");
        File outputDirectory = askDirectory("outputFilePath", "Select output directory");
        File outputFilePath = new File(outputDirectory, "patchestry.json");

        final var functions = getGlobalFunctions(functionNameArg);
        if (functions.isEmpty()) {
            println("Function not found: " + functionNameArg);
            return;
        }

        if (functions.size() > 1) {
            println("Warning: Found more than one function named: " + functionNameArg);
        }

        // Serialize to the file
        serializeToFile(outputFilePath.toPath(), functions);
    }

    private void decompileAllFunctionsInGUI() throws Exception {
        File outputDirectory = askDirectory("outputFilePath", "Select output directory");
        File outputFilePath = new File(outputDirectory, "patchestry.json");
        List<Function> functions = getAllFunctions();
        if (functions.isEmpty()) {
            println("No functions found in the current program");
            return;
        }

        // Serialize to the file
        serializeToFile(outputFilePath.toPath(), functions);
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
