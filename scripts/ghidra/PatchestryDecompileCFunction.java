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
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Program;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolTable;
import ghidra.program.model.symbol.SymbolType;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;

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

    private String renderHeader(Function fn) {
        StringBuilder sb = new StringBuilder();
        sb.append("// function:  ").append(fn.getName(true)).append('\n');
        sb.append("// entry:     ").append(fn.getEntryPoint()).append('\n');
        sb.append("// signature: ").append(fn.getSignature()).append('\n');
        sb.append("// arch:      ").append(getLanguageID()).append('\n');
        sb.append("// (decompiled by Ghidra DecompInterface for debugging)\n");
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
            try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
                writer.write(renderHeader(fn));
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
