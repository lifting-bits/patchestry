/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import ghidra.app.script.GhidraScript;

import ghidra.program.model.listing.Program;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;

import ghidra.program.model.listing.Listing;

import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;

import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;

import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.Varnode;

import ghidra.util.exception.CancelledException;

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

public class PatchestryListFunctions extends GhidraScript {

    public class FunctionSerializer extends JsonWriter {
        public FunctionSerializer(java.io.BufferedWriter writer) {
            super(writer);
        }

        public JsonWriter serialize(Function function) throws Exception {
            beginObject();
            name("name").value(function.getName());
            name("address").value(function.getEntryPoint().toString());
            name("is_thunk").value(function.isThunk());
            return endObject();
        }

        public JsonWriter serialize(List<Function> functions) throws Exception {
            beginObject();
            name("program").value(currentProgram.getName());
            name("functions").beginArray();
            for (Function function : functions) {
                serialize(function);
            }
            return endArray().endObject();
        }
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

    private void serializeToFile(Path file, List<Function> functions) throws Exception { 
        if (file == null || functions == null || functions.isEmpty()) {
            throw new IllegalArgumentException("Invalid file path or empty function list");
        }

        final var serializer = new FunctionSerializer(Files.newBufferedWriter(file));
        serializer.serialize(functions).close();
    }

    private void runHeadless() throws Exception {
        if (getScriptArgs().length != 1) {
            throw new IllegalArgumentException("Output file is not specified");
        }

        if (currentProgram == null) {
            println("Error: No program is currently loaded.");
            return;
        }
        
        final var outputFilePath = getScriptArgs()[0];
        final var functions = getAllFunctions();
        serializeToFile(Path.of(outputFilePath), functions);
    }

    private void runGUI() throws Exception {
        if (currentProgram == null) {
            println("Error: No program is currently loaded.");
            return;
        }

        File outputDirectory = askDirectory("outputDirectory", "Select Output Directory");
        File outputFilePath = new File(outputDirectory, "functions.json");
        final var functions = getAllFunctions();
        serializeToFile(outputFilePath.toPath(), functions);
    }

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