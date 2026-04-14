/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util;

import ghidra.app.decompiler.DecompileResults;
import ghidra.app.decompiler.DecompInterface;

import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.Register;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Program;

import ghidra.program.model.pcode.HighFunction;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.program.model.pcode.PcodeOpAST;
import ghidra.program.model.pcode.Varnode;

import ghidra.util.task.TaskMonitor;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * Per-callee written-register cache used by the extraout sanitizer in
 * {@code PcodeSerializer}.
 *
 * <p>Ghidra's decompiler invents {@code extraout_rN} local variables when it
 * cannot prove that a callee preserves register {@code rN} across a call.
 * For well-behaved helpers that actually leave the register untouched, this
 * produces noise in the JSON export (declared-but-undefined locals) that
 * downstream tools then stumble on.</p>
 *
 * <p>This class answers the question "did callee {@code F} write to register
 * {@code R} anywhere in its body?" by decompiling {@code F} once, walking its
 * high pcode, and collecting the register outputs. The result is cached per
 * {@link Function} for the duration of a serialization run so each callee is
 * analyzed at most once.</p>
 *
 * <p>Conservatism rules:
 * <ul>
 *   <li>If the callee cannot be decompiled, the analysis marks the callee
 *       as <em>unreliable</em> and {@link #isPreservedAcross} returns
 *       {@code false} for every register.</li>
 *   <li>If the callee's body contains any {@code CALL}, {@code CALLIND}, or
 *       {@code CALLOTHER} (a nested call, indirect call, or unmodeled
 *       userop), the analysis likewise marks the callee as unreliable.
 *       Proving preservation transitively across a nested call is
 *       out of scope for this first cut.</li>
 *   <li>Register writes canonicalize through {@link Register#getParentRegister}
 *       so writing a sub-register counts as writing the parent. The
 *       {@link #isPreservedAcross} query walks the target register's parent
 *       chain to catch overlap in the opposite direction.</li>
 * </ul>
 * </p>
 */
public final class RegisterKillAnalysis {
    private final DecompInterface decomp;
    private final TaskMonitor monitor;
    private final int timeoutSeconds;
    private final Map<Function, Set<Register>> writtenRegistersCache;
    private final Set<Function> unreliable;

    public RegisterKillAnalysis(DecompInterface decomp, TaskMonitor monitor, int timeoutSeconds) {
        this.decomp = decomp;
        this.monitor = monitor;
        this.timeoutSeconds = timeoutSeconds;
        this.writtenRegistersCache = new HashMap<>();
        this.unreliable = new HashSet<>();
    }

    /**
     * Returns the set of registers that {@code callee} writes anywhere in
     * its body. A register that is NOT in this set is provably preserved
     * across any direct call to {@code callee} — but only if
     * {@link #isUnreliable(Function)} also returns {@code false} for the
     * callee.
     *
     * <p>Results are cached per {@link Function}.</p>
     */
    public Set<Register> getWrittenRegisters(Function callee) {
        if (callee == null) {
            return Collections.emptySet();
        }
        Set<Register> cached = writtenRegistersCache.get(callee);
        if (cached != null) {
            return cached;
        }
        Set<Register> written = analyze(callee);
        writtenRegistersCache.put(callee, written);
        return written;
    }

    /**
     * Returns {@code true} when the analysis cannot prove anything about
     * {@code callee} (decompilation failed, or the body contains nested
     * calls / indirect calls / unmodeled userops).
     */
    public boolean isUnreliable(Function callee) {
        if (callee == null) {
            return true;
        }
        // Force the cache entry so the unreliable flag is populated.
        getWrittenRegisters(callee);
        return unreliable.contains(callee);
    }

    /**
     * Returns {@code true} if {@code callee} is known to leave {@code reg}
     * unchanged across a direct call.
     *
     * <p>The check considers register aliasing: if a sub-register of
     * {@code reg} (or any ancestor) was written by the callee, {@code reg}
     * is not considered preserved.</p>
     */
    public boolean isPreservedAcross(Function callee, Register reg) {
        if (callee == null || reg == null) {
            return false;
        }
        if (isUnreliable(callee)) {
            return false;
        }
        Set<Register> written = getWrittenRegisters(callee);
        // Exact match or any ancestor of `reg` written: not preserved.
        Register cursor = reg;
        while (cursor != null) {
            if (written.contains(cursor)) {
                return false;
            }
            cursor = cursor.getParentRegister();
        }
        return true;
    }

    /**
     * Drops all cached analysis results. Call between serialization runs if
     * the underlying program state has changed.
     */
    public void clear() {
        writtenRegistersCache.clear();
        unreliable.clear();
    }

    // --- internal ---

    private Set<Register> analyze(Function callee) {
        Set<Register> written = new HashSet<>();
        if (decomp == null) {
            unreliable.add(callee);
            return written;
        }
        // Thunks: follow to the thunked function. If the thunk chain is
        // visible and short, analyze the target; otherwise give up.
        Function target = callee;
        int thunkGuard = 8;
        while (target != null && target.isThunk() && thunkGuard-- > 0) {
            Function next = target.getThunkedFunction(false);
            if (next == null || next == target) {
                break;
            }
            target = next;
        }
        if (target == null) {
            unreliable.add(callee);
            return written;
        }

        DecompileResults results;
        try {
            results = decomp.decompileFunction(target, timeoutSeconds, monitor);
        } catch (Exception e) {
            unreliable.add(callee);
            return written;
        }
        if (results == null) {
            unreliable.add(callee);
            return written;
        }
        HighFunction hf = results.getHighFunction();
        if (hf == null) {
            unreliable.add(callee);
            return written;
        }

        Program program = target.getProgram();
        Language language = program != null ? program.getLanguage() : null;
        if (language == null) {
            unreliable.add(callee);
            return written;
        }

        Iterator<PcodeOpAST> it = hf.getPcodeOps();
        while (it.hasNext()) {
            PcodeOp op = it.next();
            int opcode = op.getOpcode();
            // Nested calls / indirect calls / userops break our locality
            // assumption. Mark the callee unreliable so any later
            // isPreservedAcross query returns false.
            if (opcode == PcodeOp.CALL
                    || opcode == PcodeOp.CALLIND
                    || opcode == PcodeOp.CALLOTHER) {
                unreliable.add(callee);
                // Keep collecting writes we can see up to this point —
                // they are still informative for future features, but
                // isPreservedAcross will short-circuit on the unreliable
                // flag so correctness is preserved.
                continue;
            }
            Varnode out = op.getOutput();
            if (out == null) {
                continue;
            }
            if (!out.isRegister()) {
                continue;
            }
            Register reg = language.getRegister(out.getAddress(), out.getSize());
            if (reg == null) {
                reg = language.getRegister(out.getAddress(), 0);
            }
            if (reg == null) {
                continue;
            }
            written.add(reg);
            Register parent = reg.getParentRegister();
            while (parent != null) {
                written.add(parent);
                parent = parent.getParentRegister();
            }
        }
        return written;
    }
}
