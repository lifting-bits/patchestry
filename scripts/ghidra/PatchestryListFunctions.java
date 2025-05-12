/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import util.FunctionSerializer;

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
    /* For test setup purposes, we don't want to control this script from the command line. */
    protected void setProgram(Program program) throws RuntimeException {
        if (program != null && getCurrentProgram() == null) {
            currentProgram = program;
        } else {
            currentProgram = getCurrentProgram();
        }
    }

    protected List<Function> getAllFunctions() {
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

    protected void serializeToFile(Path file, List<Function> functions) throws Exception { 
        if (file == null) {
            throw new IllegalArgumentException("Invalid file writer");
        }

        if (functions == null || functions.isEmpty()) {
            throw new IllegalArgumentException("Empty function list");
        }

        final var serializer = new FunctionSerializer(Files.newBufferedWriter(file), currentProgram);
        serializer.serialize(functions);
        serializer.close();
    }

    protected void runHeadless() throws Exception {
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