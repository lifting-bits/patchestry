/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import java.nio.file.Files;
import java.util.ArrayList;
import java.lang.ProcessBuilder;

import ghidra.app.script.GhidraScript;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.listing.Program;

import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;

import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.Varnode;

import com.google.gson.stream.JsonWriter;

public class DecompileHeadless extends GhidraScript {

    @Override
    public void run() throws Exception {
        // Ensure we have a valid program
        final var program = state.getCurrentProgram();
        if (program == null) {
            println("No active program.");
            return;
        }

        // Get the function name from the script arguments
        String functionName = getScriptArgs()[0];

        // Find the function by name
        final var function = findFunctionByName(program, functionName);
        if (function == null) {
            println("Function not found: " + functionName);
            return;
        }

        final var json = Files.createFile("patchestry.json");
        final var serializer = new PcodeSerializer(Files.newBufferedWriter(json));
        serializer.serialize(function).close();
    }

    private Function findFunctionByName(Program program, String functionName) {
        final var iter = program.getFunctionManager().getFunctions(true);
        for (Function function : iter) {
            if (function.getName().equals(functionName)) {
                return function;
            }
        }

        return null;
    }

    //
    // This is copy-pasted from the GUI ghidra script
    // It should be refactored to be a separate library
    //
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

        public JsonWriter serialize(Function function) throws Exception {
            println("Serializing function: " + function.getName());
            beginObject();
            name("name").value(function.getName());
            name("basic_blocks").beginArray();
            for (var block : getBasicBlocks(function)) {
                serialize(block);
            }
            return endArray().endObject();
        }

        public JsonWriter serialize(CodeBlock block) throws Exception {
            beginObject();
            name("label").value(block.getName());
            name("instructions").beginArray();
            for (var instruction : getInstructions(block)) {
                serialize(instruction);
            }
            return endArray().endObject();
        }

        public JsonWriter serialize(Instruction instruction) throws Exception {
            beginObject();
            name("mnemonic").value(instruction.getMnemonicString());
            name("pcode").beginArray();
            for (var op : instruction.getPcode()) {
                serialize(op);
            }
            return endArray().endObject();
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

    }
}