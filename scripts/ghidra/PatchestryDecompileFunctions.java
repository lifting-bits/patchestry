/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

import domain.*;
import util.*;

import ghidra.app.script.GhidraScript;

import ghidra.framework.options.Options;

import ghidra.app.cmd.function.CallDepthChangeInfo;

import ghidra.app.decompiler.component.DecompilerUtils;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;

import ghidra.app.plugin.processors.sleigh.SleighLanguage;

import ghidra.program.database.symbol.CodeSymbol;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressFactory;
import ghidra.program.model.address.AddressIterator;
import ghidra.program.model.address.AddressSet;
import ghidra.program.model.address.AddressSetView;
import ghidra.program.model.address.AddressSpace;
import ghidra.program.model.address.AddressOutOfBoundsException;

import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;

import ghidra.program.model.data.AbstractStringDataType;
import ghidra.program.model.data.BitFieldDataType;
import ghidra.program.model.data.DataType;
import ghidra.program.model.data.DataTypeManager;
import ghidra.program.model.data.StringDataType;
import ghidra.program.model.data.StringDataInstance;
import ghidra.program.model.data.CharDataType;

import ghidra.program.model.lang.CompilerSpec;
import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.Register;
import ghidra.program.model.lang.RegisterManager;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.FunctionSignature;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.listing.Parameter;
import ghidra.program.model.listing.Program;
import ghidra.program.model.listing.StackFrame;
import ghidra.program.model.listing.Variable;
import ghidra.program.model.listing.VariableStorage;
import ghidra.program.model.listing.Data;

import ghidra.program.model.mem.MemBuffer;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.mem.MemoryBufferImpl;

import ghidra.program.model.pcode.FunctionPrototype;
import ghidra.program.model.pcode.GlobalSymbolMap;
import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.HighCodeSymbol;
import ghidra.program.model.pcode.HighConstant;
import ghidra.program.model.pcode.HighGlobal;
import ghidra.program.model.pcode.HighLocal;
import ghidra.program.model.pcode.HighOther;
import ghidra.program.model.pcode.HighParam;
import ghidra.program.model.pcode.HighSymbol;
import ghidra.program.model.pcode.HighVariable;
import ghidra.program.model.pcode.LocalSymbolMap;
import ghidra.program.model.pcode.PartialUnion;
import ghidra.program.model.pcode.PcodeBlock;
import ghidra.program.model.pcode.PcodeBlockBasic;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.SequenceNumber;
import ghidra.program.model.pcode.SymbolEntry;
import ghidra.program.model.pcode.Varnode;

import ghidra.program.model.data.AbstractFloatDataType;
import ghidra.program.model.data.AbstractIntegerDataType;
import ghidra.program.model.data.Array;
import ghidra.program.model.data.ArrayStringable;
import ghidra.program.model.data.BooleanDataType;
import ghidra.program.model.data.BuiltIn;
import ghidra.program.model.data.Composite;
import ghidra.program.model.data.CategoryPath;
import ghidra.program.model.data.DataType;
import ghidra.program.model.data.DataTypeComponent;
import ghidra.program.model.data.DefaultDataType;
import ghidra.program.model.data.Enum;
import ghidra.program.model.data.FunctionDefinition;
import ghidra.program.model.data.ParameterDefinition;
import ghidra.program.model.data.Pointer;
import ghidra.program.model.data.Structure;
import ghidra.program.model.data.TypeDef;
import ghidra.program.model.data.Undefined;
import ghidra.program.model.data.Union;
import ghidra.program.model.data.VoidDataType;
import ghidra.program.model.data.WideCharDataType;

import ghidra.program.model.symbol.ExternalManager;
import ghidra.program.model.symbol.Namespace;
import ghidra.program.model.symbol.Reference;
import ghidra.program.model.symbol.ReferenceManager;
import ghidra.program.model.symbol.SourceType;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolType;
import ghidra.program.model.symbol.SymbolTable;
import ghidra.program.model.symbol.SymbolUtilities;
import ghidra.program.model.symbol.FlowType;
import ghidra.program.model.symbol.Reference;

import ghidra.app.plugin.core.analysis.AutoAnalysisManager;

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
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.TreeMap;

public class PatchestryDecompileFunctions extends GhidraScript {
    /* For test setup purposes, we don't want to control this script from the command line. */
    protected void setProgram(Program program) throws RuntimeException {
        if (program != null && getCurrentProgram() == null) {
            currentProgram = program;
        } else {
            currentProgram = getCurrentProgram();
        }
    }

    String getLanguageID() throws Exception {
        if (currentProgram.getLanguage() == null
                || currentProgram.getLanguage().getLanguageDescription() == null) {
            return "unknown";
        }
        return currentProgram.getLanguage().getLanguageDescription().getLanguageID().toString();
    }

    DecompInterface getDecompilerInterface() throws Exception {
        if (currentProgram == null) {
            throw new Exception("Unable to initialize decompiler: invalid current program.");
        }

        DecompileOptions options = DecompilerUtils.getDecompileOptions(state.getTool(), currentProgram);
        DecompInterface decompiler = new DecompInterface();

        decompiler.setOptions(options);
        decompiler.toggleCCode(false);
        decompiler.toggleSyntaxTree(true);
        decompiler.toggleJumpLoads(true);
        decompiler.toggleParamMeasures(false);
        decompiler.setSimplificationStyle("decompile");

        if (!decompiler.openProgram(currentProgram)) {
            throw new Exception("Unable to initialize decompiler: " + decompiler.getLastMessage());
        }
        return decompiler;
    }

    void serializeToFile(JsonWriter writer, List<Function> functions) throws Exception {
        if (writer == null) {
            throw new IllegalArgumentException("Invalid file writer");
        }

        if (functions == null || functions.isEmpty()) {
            throw new IllegalArgumentException("Empty function list");
        }

        final var serializer = new PcodeSerializer(
            writer, 
            functions, 
            getLanguageID(),
            monitor,
            currentProgram,
            getDecompilerInterface()
        );
        serializer.serialize();
    }

    List<Function> getAllFunctions() {
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

    void decompileSingleFunction() throws Exception {
        if (getScriptArgs().length < 3) {
            throw new IllegalArgumentException("Insufficient arguments. Expected: <function_name> <output_file> as argument");
        }
        JsonWriter writer = new JsonWriter(Files.newBufferedWriter(Path.of(getScriptArgs()[2])));
        serializeToFile(writer, getGlobalFunctions(getScriptArgs()[1]));
    }

    void decompileAllFunctions() throws Exception {
        if (getScriptArgs().length < 2) {
            throw new IllegalArgumentException("Insufficient arguments. Expected: <output_file> as argument");
        }
        JsonWriter writer = new JsonWriter(Files.newBufferedWriter(Path.of(getScriptArgs()[1])));
        serializeToFile(writer, getAllFunctions());
    }

    // Update and re-run auto analysis.
    void runAutoAnalysis() throws Exception {
        Program program = getCurrentProgram();

        if (program == null) {
            throw new IllegalStateException("Program cannot be null");
        }

        if (program.isClosed()) {
            throw new UnsupportedOperationException("Program must be open for analysis to occur");
        }

        Options options = program.getOptions(Program.ANALYSIS_PROPERTIES);
        options.setBoolean("Aggressive Instruction Finder", true);
        options.setBoolean("Decompiler Parameter ID", true);
        for (String option : options.getOptionNames()) {
            println(option + " = " + options.getValueAsString(option));
        }

        AutoAnalysisManager mgr = AutoAnalysisManager.getAnalysisManager(program);
        mgr.scheduleOneTimeAnalysis(
                mgr.getAnalyzer("Aggressive Instruction Finder"), program.getMemory());
        mgr.scheduleOneTimeAnalysis(
                mgr.getAnalyzer("Decompiler Parameter ID"), program.getMemory());

        mgr.startAnalysis(monitor);
        mgr.waitForAnalysis(null, monitor);
    }

    void runHeadless() throws Exception {
        if (getScriptArgs().length < 1) {
            throw new IllegalArgumentException("mode is not specified for headless execution");
        }

        // Update and run auto analysis before accessing decompiler interface.
        runAutoAnalysis();

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

    void decompileSingleFunctionInGUI() throws Exception {
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
        JsonWriter writer = new JsonWriter(Files.newBufferedWriter(outputFilePath.toPath()));
        serializeToFile(writer, functions);
    }

    void decompileAllFunctionsInGUI() throws Exception {
        File outputDirectory = askDirectory("outputFilePath", "Select output directory");
        File outputFilePath = new File(outputDirectory, "patchestry.json");
        JsonWriter writer = new JsonWriter(Files.newBufferedWriter(outputFilePath.toPath()));
        serializeToFile(writer, getAllFunctions());
    }

    // GUI mode execution
    void runGUI() throws Exception {
        String mode = askString("mode", "Please enter mode:");
        println("Running in mode: " + mode);

        // Update and run auto analysis before accessing decompiler interface.
        runAutoAnalysis();
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
