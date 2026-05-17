/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 *
 * PatchestryDecompileCFunction
 * ----------------------------
 * Headless Ghidra postscript that decompiles a single function to plain C
 * source and writes it to the supplied output path. Designed as a debugging
 * companion to PatchestryDecompileFunctions (which emits P-Code JSON):
 * when patchir-decomp fails to lift a function, this script lets you see
 * what Ghidra's own decompiler produced for the same body so you can
 * compare expectations.
 *
 * The output is preceded by C declarations for every type the function
 * references (return type, parameters, locals, and their transitive
 * dependencies), so the emitted .c is self-contained for inspection.
 *
 * Usage (headless): single positional arg form, mirrors
 *   PatchestryDecompileFunctions single mode.
 *
 *   postScript PatchestryDecompileCFunction <function-or-address> <output-file>
 *
 * `<function-or-address>` accepts: a global function name, a hex address
 * (e.g. 0x0000f69c), or any address inside a function body — the latter
 * resolves via getFunctionContaining so op keys from failure logs work
 * directly.
 */

import ghidra.app.script.GhidraScript;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.decompiler.DecompiledFunction;
import ghidra.app.decompiler.component.DecompilerUtils;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressFactory;
import ghidra.program.model.data.DataType;
import ghidra.program.model.data.DataTypeWriter;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Parameter;
import ghidra.program.model.listing.Program;
import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.HighSymbol;
import ghidra.program.model.pcode.LocalSymbolMap;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolTable;
import ghidra.program.model.symbol.SymbolType;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class PatchestryDecompileCFunction extends GhidraScript {

    // Matches the timeout used by PcodeSerializer for parity with the
    // P-Code JSON path. 60 s is plenty for any single function.
    private static final int DECOMPILATION_TIMEOUT_SECONDS = 60;

    // Test setup hook (parity with sibling scripts) — keep package-private.
    protected void setProgram(Program program) throws RuntimeException {
        if (program != null && getCurrentProgram() == null) {
            currentProgram = program;
        } else {
            currentProgram = getCurrentProgram();
        }
    }

    private String getLanguageID() {
        if (currentProgram.getLanguage() == null
                || currentProgram.getLanguage().getLanguageDescription() == null) {
            return "unknown";
        }
        return currentProgram.getLanguage().getLanguageDescription().getLanguageID().toString();
    }

    private DecompInterface openDecompiler() throws Exception {
        if (currentProgram == null) {
            throw new IllegalStateException(
                "Unable to initialize decompiler: invalid current program.");
        }
        DecompileOptions options =
            DecompilerUtils.getDecompileOptions(state.getTool(), currentProgram);
        DecompInterface decompiler = new DecompInterface();
        decompiler.setOptions(options);
        // Enable C-code emission — this is the whole point of this script
        // and is the only material difference from
        // PatchestryDecompileFunctions.getDecompilerInterface().
        decompiler.toggleCCode(true);
        decompiler.toggleSyntaxTree(true);
        decompiler.toggleJumpLoads(true);
        decompiler.toggleParamMeasures(false);
        decompiler.setSimplificationStyle("decompile");
        if (!decompiler.openProgram(currentProgram)) {
            throw new IllegalStateException(
                "Unable to initialize decompiler: " + decompiler.getLastMessage());
        }
        return decompiler;
    }

    // Multi-strategy lookup: global name → hex address → enclosing-function
    // resolution → symbol/label → namespace-qualified name. Mirrors the
    // chain in PatchestryDecompileFunctions.resolveFunction but additionally
    // honours getFunctionContaining so any address inside the body (e.g.
    // an op key like ram:0000f728) resolves to the enclosing function.
    Function resolveFunction(String name) throws Exception {
        FunctionManager fm = currentProgram.getFunctionManager();
        SymbolTable symTable = currentProgram.getSymbolTable();
        AddressFactory addrFactory = currentProgram.getAddressFactory();

        // 1. Global name lookup.
        for (Function fn : currentProgram.getListing().getGlobalFunctions(name)) {
            return fn;
        }

        // 2. Address lookup. Accepts 0x-prefixed hex or bare hex.
        Address addr = null;
        try {
            String addrStr = name;
            if (addrStr.startsWith("0x") || addrStr.startsWith("0X")) {
                addrStr = addrStr.substring(2);
            }
            addr = addrFactory.getDefaultAddressSpace().getAddress(addrStr);
        } catch (Exception ignored) {
            // Not an address; fall through.
        }
        if (addr != null) {
            Function fn = fm.getFunctionAt(addr);
            if (fn != null) {
                return fn;
            }
            fn = fm.getFunctionContaining(addr);
            if (fn != null) {
                println("Resolved '" + name + "' to enclosing function: "
                    + fn.getName(true) + " @ " + fn.getEntryPoint());
                return fn;
            }
        }

        // 3. Symbol/label lookup (mangled C++ names attached as labels).
        for (Symbol sym : symTable.getSymbols(name)) {
            Function fn = fm.getFunctionAt(sym.getAddress());
            if (fn != null) {
                return fn;
            }
        }

        // 4. Namespace-qualified name (e.g. Foo::Bar::method).
        if (name.contains("::")) {
            String[] parts = name.split("::");
            String localName = parts[parts.length - 1];
            for (Symbol sym : symTable.getSymbols(localName)) {
                if (sym.getSymbolType() == SymbolType.FUNCTION
                        && sym.getName(true).equals(name)) {
                    Function fn = fm.getFunctionAt(sym.getAddress());
                    if (fn != null) {
                        return fn;
                    }
                }
            }
        }

        throw new IllegalArgumentException(
            "Function not found: '" + name + "'. Tried: global name, address, "
            + "enclosing-function, symbol/label, namespace path.");
    }

    // Collect the data types the function references at the surface: its
    // return type, parameters, and decompiler-recovered locals. DataTypeWriter
    // expands these transitively (struct fields, pointee types, typedef
    // targets), so seeding it with the top-level types is sufficient.
    private List<DataType> collectReferencedTypes(Function fn, DecompileResults results) {
        // LinkedHashSet: dedup while keeping a stable, reproducible order.
        Set<DataType> seeds = new LinkedHashSet<>();

        DataType returnType = fn.getReturnType();
        if (returnType != null) {
            seeds.add(returnType);
        }
        for (Parameter param : fn.getParameters()) {
            DataType paramType = param.getDataType();
            if (paramType != null) {
                seeds.add(paramType);
            }
        }

        // Locals come from the decompiler's HighFunction, which is available
        // because openDecompiler() enables toggleSyntaxTree(true). It may
        // still be null if the syntax tree failed to build — emit the body
        // anyway in that case.
        HighFunction highFn = results.getHighFunction();
        if (highFn != null) {
            LocalSymbolMap localSymbols = highFn.getLocalSymbolMap();
            Iterator<HighSymbol> it = localSymbols.getSymbols();
            while (it.hasNext()) {
                DataType localType = it.next().getDataType();
                if (localType != null) {
                    seeds.add(localType);
                }
            }
        }

        return new ArrayList<>(seeds);
    }

    // Emit C declarations for `types` (and their transitive dependencies) as a
    // preamble ahead of the function body. DataTypeWriter orders dependencies
    // before dependents and flushes the writer itself; it never closes the
    // underlying Writer, so the caller's try-with-resources keeps ownership.
    // Type emission is best-effort: a failure here must not suppress the C
    // body, which is the primary debugging artifact.
    private void writeTypeDefinitions(BufferedWriter writer, List<DataType> types) {
        if (types.isEmpty()) {
            return;
        }
        try {
            // Write the marker before constructing DataTypeWriter: its
            // constructor immediately emits the standard builtin typedefs,
            // so the marker must precede it to head the whole type block.
            writer.write("// --- referenced type definitions ---\n");
            DataTypeWriter typeWriter =
                new DataTypeWriter(currentProgram.getDataTypeManager(), writer);
            // throwExceptionOnInvalidType=false: skip a malformed type rather
            // than aborting the whole preamble.
            typeWriter.write(types, monitor, false);
            writer.write('\n');
        } catch (Exception e) {
            println("Warning: failed to emit type definitions for "
                + "decompile-c output: " + e.getMessage());
        }
    }

    private String renderHeader(Function fn) {
        StringBuilder sb = new StringBuilder();
        sb.append("// function:  ").append(fn.getName(true)).append('\n');
        sb.append("// entry:     ").append(fn.getEntryPoint()).append('\n');
        sb.append("// signature: ").append(fn.getSignature()).append('\n');
        sb.append("// arch:      ").append(getLanguageID()).append('\n');
        sb.append("// (decompiled by Ghidra DecompInterface for debugging;\n");
        sb.append("//  referenced type definitions are emitted below)\n");
        return sb.toString();
    }

    void runHeadless() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            throw new IllegalArgumentException(
                "Insufficient arguments. Expected: <function-or-address> <output-file>");
        }
        String target = args[0];
        Path outputPath = Path.of(args[1]);

        Function fn = resolveFunction(target);
        println("Decompiling " + fn.getName(true) + " @ " + fn.getEntryPoint()
            + " -> " + outputPath);

        DecompInterface decompiler = openDecompiler();
        try {
            DecompileResults results = decompiler.decompileFunction(
                fn, DECOMPILATION_TIMEOUT_SECONDS, monitor);
            if (results == null || !results.decompileCompleted()) {
                String reason = results == null
                    ? "<null results>" : results.getErrorMessage();
                throw new RuntimeException(
                    "Decompilation did not complete for "
                    + fn.getName(true) + ": " + reason);
            }
            DecompiledFunction decompiled = results.getDecompiledFunction();
            String cSource = decompiled == null ? null : decompiled.getC();
            if (cSource == null) {
                throw new RuntimeException(
                    "Ghidra produced no C output for " + fn.getName(true));
            }
            List<DataType> referencedTypes = collectReferencedTypes(fn, results);
            try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
                writer.write(renderHeader(fn));
                writer.write('\n');
                writeTypeDefinitions(writer, referencedTypes);
                writer.write(cSource);
                if (!cSource.endsWith("\n")) {
                    writer.write('\n');
                }
            }
        } finally {
            decompiler.dispose();
        }
    }

    @Override
    public void run() throws Exception {
        // Headless-only. The script is meant to be invoked via the
        // decompile-headless.sh wrapper; GUI mode would just need an
        // askString prompt, which we intentionally skip to keep the
        // entrypoint minimal.
        runHeadless();
    }
}
