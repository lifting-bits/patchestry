/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import ghidra.app.script.GhidraScript;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;

import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;

import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.Varnode;

import com.google.gson.stream.JsonWriter;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.lang.ProcessBuilder;

public class PatchestryScript extends GhidraScript {

    private InstructionIterator getInstructions(CodeBlock block) throws Exception {
        return currentProgram.getListing().getInstructions(block, true);
    }

    private CodeBlockIterator getBasicBlocks(Function function) throws Exception {
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

            if (node.isConstant())
                value("const");
            else if (node.isUnique())
                value("unique");
            else if (node.isRegister())
                value("register");
            else if (node.isAddress())
                value("ram");
            else
                throw new Exception("Unknown Varnode kind.");

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
    }

    private void serializeToFile(Path file, Function function) throws Exception {
        final var serializer = new PcodeSerializer(Files.newBufferedWriter(file));
        println("Serializing function: " + function.getName() + " @ " + function.getEntryPoint());
        serializer.serialize(function).close();
    }
    
    private void runHeadless() throws Exception {
        final var args = getScriptArgs();
        if (args.length != 2) {
            println("Usage:");
            println("\tFUNCTION_NAME");
            println("\tOUTPUT_FILE");
            return;
        }
        
        final var functionName = args[0];
        final var outputPath = args[1];
        
        final var functions = getGlobalFunctions(functionName);
        if (functions.isEmpty()) {
            println("Function not found: " + functionName);
            return;
        }

        if (functions.size() > 1) {
            println("Found more than one function named: " + functionName);
        }

        serializeToFile(Path.of(outputPath), functions.get(0));
    }

    private void runGUI() throws Exception {
        final var curFunction = getFunctionContaining(currentAddress);
        final var functionName = curFunction.getName();
        final var jsonPath = Files.createTempFile(functionName + '.', ".patchestry.json");
        
        serializeToFile(jsonPath, curFunction);

        final var mlirPath = Files.createTempFile(functionName + '.', ".patchestry.out");
        final var binaryPath = "patchestry";
        
        final var cmd = new ArrayList<String>();
        cmd.add(binaryPath);
        cmd.add(jsonPath.toString());
        cmd.add(mlirPath.toString());
        println("Calling: " + cmd.toString());
        new ProcessBuilder(cmd).inheritIO().start().waitFor();

        println(Files.readString(mlirPath));
    }

    public void run() throws Exception {
        if (isRunningHeadless()) {
            runHeadless();
        } else {
            runGUI();
        }
    }
}
