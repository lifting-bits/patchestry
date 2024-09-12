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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.lang.ProcessBuilder;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class PatchestryListFunctions extends GhidraScript {

    private void runHeadless() throws Exception {
        final var args = getScriptArgs();
        if (args.length != 1) {
            println("Usage:\n\tOUTPUT_FILE");
            return;
        }

        final var outputPath = args[0];

        Program program = getCurrentProgram();
        if (program == null) {
            println("Error: No program is currently loaded.");
            return;
        }

        try(PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
            writer.println("Functions in the binary:");
            writer.println("========================");

            FunctionIterator functions = program.getFunctionManager().getFunctions(true);
            for (Function function : functions) {
                if (monitor.isCancelled()) {
                    throw new CancelledException();
                }
                writer.printf("%s at 0x%s%n", function.getName(), function.getEntryPoint().toString());
            }

            println("Function list has been written to: " + outputPath);

        } catch(CancelledException e) {
            println("Operation was cancelled by the user.");
        } catch(IOException e) {
            println("Error writing to file: " + e.getMessage());
        } catch(Exception e) {
            println("An unexpected error occurred: " + e.getMessage());
        }
    }

    private void runGUI() throws Exception {
        if (currentProgram == null) {
            popup("Error: No program is currently loaded.");
            return;
        }

        Listing listing = currentProgram.getListing();
        FunctionIterator functions = currentProgram.getFunctionManager().getFunctions(true);
        StringBuilder functionInfo = new StringBuilder();

        for (Function function : functions) {
            String functionName = function.getName();
            String functionAddress = function.getEntryPoint().toString();
            functionInfo.append(String.format("%s at 0x%s%ns\n", functionName, functionAddress));
        }

        if (functionInfo.length() > 0) {
            popup(functionInfo.toString());
        } else {
            popup("No functions found!");
        }
    }

    public void run() throws Exception {
        if (isRunningHeadless()) {
            runHeadless();
        } else {
            runGUI();
        }
    }
}
