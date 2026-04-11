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
    // --- Extraout sanitizer flags ---
    //
    // --sanitize-extraout (default true; pass --no-sanitize-extraout to disable)
    //     Master switch for the extraout/unaff/in_ register-alias sanitizer
    //     pass in PcodeSerializer. When off, the serializer emits its
    //     legacy JSON shape unchanged. Tier 1 (declarative via
    //     PrototypeModel.getUnaffectedList) is active on every architecture
    //     when the flag is on; Tier 2 (analytical callee walk) runs only
    //     when the current program's processor is in the allowlist
    //     util.PcodeSerializer.TIER2_DEFAULT_ARCHITECTURES.
    //
    // --sanitize-extraout-analytical={auto|on|off} (default auto)
    //     Controls Tier 2 (analytical callee-walk) preservation analysis.
    //     `auto` defers to the architecture allowlist
    //     (util.PcodeSerializer.TIER2_DEFAULT_ARCHITECTURES). `on` forces
    //     Tier 2 regardless of architecture; `off` disables Tier 2 even on
    //     allowlisted architectures.
    private boolean sanitizeExtraout = true;
    private util.PcodeSerializer.AnalyticalTierMode analyticalMode =
        util.PcodeSerializer.AnalyticalTierMode.AUTO;

    // Script args with all recognized flags stripped out. The positional
    // command processors (decompileSingleFunction / decompileAllFunctions)
    // index into this rather than getScriptArgs() so flags and positional
    // arguments can be intermixed on the command line.
    private String[] positionalArgs = new String[0];

    /* For test setup purposes, we don't want to control this script from the command line. */
    protected void setProgram(Program program) throws RuntimeException {
        if (program != null && getCurrentProgram() == null) {
            currentProgram = program;
        } else {
            currentProgram = getCurrentProgram();
        }
    }

    // Parse --sanitize-extraout and --sanitize-extraout-analytical from
    // the raw getScriptArgs() list, storing the result on instance fields
    // and producing a positional-args-only view in `positionalArgs`.
    //
    // Both `--flag value` and `--flag=value` forms are accepted so the
    // Ghidra headless driver (which passes script args as a plain list)
    // and wrapper shell scripts can use whichever is convenient.
    //
    // Idempotent: callers (including ensureFlagsParsed) may invoke this
    // repeatedly. Flag-controlled fields reset to their defaults at the
    // top so that successive calls reflect only the current raw args.
    private void parseScriptFlags() {
        // Reset to defaults so that repeat calls do not carry state from
        // a previous argument list. This matters for tests that reuse a
        // PatchestryDecompileFunctions instance across methods under
        // @TestInstance(PER_CLASS). Defaults match the field initializers
        // at the top of the class.
        sanitizeExtraout = true;
        analyticalMode = util.PcodeSerializer.AnalyticalTierMode.AUTO;
        positionalArgs = new String[0];

        String[] raw = getScriptArgs();
        List<String> positional = new ArrayList<>(raw.length);
        for (int i = 0; i < raw.length; ++i) {
            String arg = raw[i];
            if (arg == null) {
                continue;
            }
            if (arg.equals("--sanitize-extraout")) {
                sanitizeExtraout = true;
                continue;
            }
            if (arg.equals("--no-sanitize-extraout")) {
                sanitizeExtraout = false;
                continue;
            }
            if (arg.startsWith("--sanitize-extraout=")) {
                String value = arg.substring("--sanitize-extraout=".length());
                sanitizeExtraout = parseBoolFlag(arg, value);
                continue;
            }
            if (arg.equals("--sanitize-extraout-analytical")) {
                if (i + 1 >= raw.length) {
                    throw new IllegalArgumentException(
                        "--sanitize-extraout-analytical requires a value (auto|on|off)");
                }
                analyticalMode = parseAnalyticalMode(raw[++i]);
                continue;
            }
            if (arg.startsWith("--sanitize-extraout-analytical=")) {
                String value = arg.substring(
                    "--sanitize-extraout-analytical=".length());
                analyticalMode = parseAnalyticalMode(value);
                continue;
            }
            positional.add(arg);
        }
        positionalArgs = positional.toArray(new String[0]);
    }

    private static boolean parseBoolFlag(String flagName, String value) {
        if (value == null) {
            throw new IllegalArgumentException(flagName + " requires a value");
        }
        String v = value.toLowerCase();
        if (v.equals("true") || v.equals("on") || v.equals("1") || v.equals("yes")) {
            return true;
        }
        if (v.equals("false") || v.equals("off") || v.equals("0") || v.equals("no")) {
            return false;
        }
        throw new IllegalArgumentException(
            flagName + ": unrecognized boolean value '" + value + "'");
    }

    private static util.PcodeSerializer.AnalyticalTierMode parseAnalyticalMode(String value) {
        if (value == null) {
            throw new IllegalArgumentException(
                "--sanitize-extraout-analytical requires a value (auto|on|off)");
        }
        switch (value.toLowerCase()) {
            case "auto":
                return util.PcodeSerializer.AnalyticalTierMode.AUTO;
            case "on":
            case "true":
            case "1":
            case "yes":
                return util.PcodeSerializer.AnalyticalTierMode.ON;
            case "off":
            case "false":
            case "0":
            case "no":
                return util.PcodeSerializer.AnalyticalTierMode.OFF;
            default:
                throw new IllegalArgumentException(
                    "--sanitize-extraout-analytical: unrecognized value '"
                        + value + "' (expected auto, on, or off)");
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
            getDecompilerInterface(),
            sanitizeExtraout,
            analyticalMode
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

    // Resolve a function by name using a multi-strategy fallback chain.
    // Supports: global name, hex address, mangled C++ symbols, namespace-
    // qualified names (A::B::fn), and unqualified local names.
    List<Function> resolveFunction(String name) throws Exception {
        // Step 1: Global namespace lookup (existing Ghidra API).
        List<Function> functions = getGlobalFunctions(name);
        if (functions != null && !functions.isEmpty()) {
            return functions;
        }

        FunctionManager fm = currentProgram.getFunctionManager();
        SymbolTable symTable = currentProgram.getSymbolTable();

        // Step 2: Address-based lookup (e.g. "0x08001234" or hex digits).
        if (name.startsWith("0x") || name.startsWith("0X")) {
            try {
                Address addr = currentProgram.getAddressFactory()
                    .getDefaultAddressSpace().getAddress(name);
                Function fn = fm.getFunctionAt(addr);
                if (fn != null) {
                    println("Resolved '" + name + "' by address to: " + fn.getName(true));
                    return new ArrayList<>(List.of(fn));
                }
            } catch (Exception e) {
                // Not a valid address, continue to next fallback.
            }
        }

        // Step 3: Symbol/label lookup — handles mangled C++ names stored as
        // labels at the function's entry point by Ghidra's ELF importer.
        for (Symbol sym : symTable.getSymbols(name)) {
            Function fn = fm.getFunctionAt(sym.getAddress());
            if (fn != null) {
                println("Resolved '" + name + "' via symbol to: " + fn.getName(true));
                return new ArrayList<>(List.of(fn));
            }
        }

        // Step 4: Namespace-qualified name (e.g. "Debug::Command::VarHandler::GetVarInfo").
        if (name.contains("::")) {
            String[] parts = name.split("::");
            String localName = parts[parts.length - 1];
            for (Symbol sym : symTable.getSymbols(localName)) {
                if (sym.getSymbolType() == SymbolType.FUNCTION) {
                    String fullPath = sym.getName(true);
                    if (fullPath.equals(name)) {
                        Function fn = fm.getFunctionAt(sym.getAddress());
                        if (fn != null) {
                            println("Resolved '" + name + "' via namespace path.");
                            return new ArrayList<>(List.of(fn));
                        }
                    }
                }
            }
        }

        // Step 5: Local name match via symbol table (last resort).
        // Uses the symbol table index instead of iterating all functions.
        String searchName = name.contains("::") ? name.substring(name.lastIndexOf("::") + 2) : name;
        List<Function> matches = new ArrayList<>();
        for (Symbol sym : symTable.getSymbols(searchName)) {
            if (sym.getSymbolType() == SymbolType.FUNCTION) {
                Function fn = fm.getFunctionAt(sym.getAddress());
                if (fn != null) {
                    matches.add(fn);
                }
            }
        }

        if (!matches.isEmpty()) {
            if (matches.size() > 1) {
                println("WARNING: Multiple functions match '" + searchName + "':");
                for (Function fn : matches) {
                    println("  " + fn.getName(true) + " @ " + fn.getEntryPoint());
                }
                println("Using first match: " + matches.get(0).getName(true));
            } else {
                println("Resolved '" + name + "' via local name to: "
                    + matches.get(0).getName(true));
            }
            return new ArrayList<>(List.of(matches.get(0)));
        }

        throw new IllegalArgumentException(
            "Function not found: '" + name + "'. "
            + "Searched by: global name, address, symbol/label, namespace path, and local name. "
            + "Use 'all' mode to decompile everything, or verify the function name in Ghidra."
        );
    }

    // Ensure positionalArgs reflects the current getScriptArgs(). Always
    // re-parses so that test instances which reuse a PatchestryDecompileFunctions
    // across multiple method invocations (@TestInstance(PER_CLASS)) see the
    // freshly-set script args. Safe to call multiple times; parseScriptFlags()
    // is pure w.r.t. the current raw args array.
    private void ensureFlagsParsed() {
        parseScriptFlags();
    }

    void decompileSingleFunction() throws Exception {
        ensureFlagsParsed();
        if (positionalArgs.length < 3) {
            throw new IllegalArgumentException("Insufficient arguments. Expected: <function_name> <output_file> as argument");
        }
        JsonWriter writer = new JsonWriter(Files.newBufferedWriter(Path.of(positionalArgs[2])));
        serializeToFile(writer, resolveFunction(positionalArgs[1]));
    }

    void decompileAllFunctions() throws Exception {
        ensureFlagsParsed();
        if (positionalArgs.length < 2) {
            throw new IllegalArgumentException("Insufficient arguments. Expected: <output_file> as argument");
        }
        JsonWriter writer = new JsonWriter(Files.newBufferedWriter(Path.of(positionalArgs[1])));
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
        options.setBoolean("Decompiler Switch Analysis", true);
        for (String option : options.getOptionNames()) {
            println(option + " = " + options.getValueAsString(option));
        }

        AutoAnalysisManager mgr = AutoAnalysisManager.getAnalysisManager(program);
        mgr.scheduleOneTimeAnalysis(
                mgr.getAnalyzer("Aggressive Instruction Finder"), program.getMemory());
        mgr.scheduleOneTimeAnalysis(
                mgr.getAnalyzer("Decompiler Parameter ID"), program.getMemory());
        mgr.scheduleOneTimeAnalysis(
                mgr.getAnalyzer("Decompiler Switch Analysis"), program.getMemory());

        mgr.startAnalysis(monitor);
        mgr.waitForAnalysis(null, monitor);
    }

    void runHeadless() throws Exception {
        parseScriptFlags();
        if (positionalArgs.length < 1) {
            throw new IllegalArgumentException("mode is not specified for headless execution");
        }

        // Update and run auto analysis before accessing decompiler interface.
        runAutoAnalysis();

        // Execution mode
        String mode = positionalArgs[0];
        println("Running in mode: " + mode);
        if (sanitizeExtraout) {
            println("[extraout] sanitizer flag enabled (analytical=" + analyticalMode + ")");
        }
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
            functions = resolveFunction(functionNameArg);
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
