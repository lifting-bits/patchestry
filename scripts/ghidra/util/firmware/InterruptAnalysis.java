/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.firmware;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressRange;
import ghidra.program.model.address.AddressSpace;

import ghidra.program.model.lang.Language;
import ghidra.program.model.lang.Register;

import ghidra.program.model.listing.Bookmark;
import ghidra.program.model.listing.BookmarkManager;
import ghidra.program.model.listing.BookmarkType;
import ghidra.program.model.listing.FlowOverride;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.listing.Program;

import ghidra.program.model.mem.Memory;
import ghidra.program.model.mem.MemoryAccessException;

import ghidra.program.model.scalar.Scalar;

import ghidra.program.model.util.PropertyMapManager;
import ghidra.program.model.util.StringPropertyMap;

import ghidra.util.exception.CancelledException;
import ghidra.util.task.TaskMonitor;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Detects interrupt/exception handlers in bare-metal ARM firmware and records
 * the metadata the lifter needs to handle them faithfully:
 *
 * <ul>
 *   <li>Function tag {@code ISR} (+ companion {@code ISR_AUTO} for auto-applied
 *       findings) on each vector-table entry handler. The serializer turns this
 *       into {@code is_interrupt}, which the C++ side renders as a void(void)
 *       prototype + interrupt attribute.</li>
 *   <li>Function tag {@code ISR_REACHABLE} on functions transitively called from
 *       a handler (for downstream concurrency/shared-state analysis; these are
 *       NOT interrupt entry points and keep their normal prototype).</li>
 *   <li>{@code KIND_PROPMAP}: handler entry address -&gt; Clang ARMInterruptAttr
 *       kind string ("" Generic / M-profile, or IRQ/FIQ/SWI/ABORT/UNDEF).</li>
 *   <li>{@code EXC_RETURN_PROPMAP}: exception-return site address -&gt; encoded
 *       {@code form|value} for non-standard returns (M-profile explicit
 *       EXC_RETURN magic, or A/R CPSR-restoring returns). The serializer injects
 *       a synthetic {@code exception_return} CALLOTHER from these.</li>
 *   <li>{@link FlowOverride#RETURN} on each exception-return terminator so the
 *       decompiler terminates the CFG cleanly instead of branching to the
 *       unmapped EXC_RETURN address.</li>
 * </ul>
 *
 * <p>Architecture-neutral: detection is behind {@link InterruptModel}; v1 wires
 * Cortex-M (M-profile) and classic ARMv7-A/R models. Idempotent: re-running
 * clears only auto-applied ({@code ISR_AUTO}) tags and this pass's bookmarks/
 * propmaps, so analyst-added bare {@code ISR} tags survive (human override).
 *
 * <p>Modeled on {@link util.tailcall.TailCallAnalysis}.
 */
public final class InterruptAnalysis {

    public static final String ISR_TAG            = "ISR";
    public static final String ISR_AUTO_TAG       = "ISR_AUTO";
    public static final String ISR_REACHABLE_TAG  = "ISR_REACHABLE";
    public static final String BOOKMARK_CATEGORY  = "InterruptAnalysis";
    // entry address -> ARMInterruptAttr kind ("", "IRQ", "FIQ", "SWI",
    // "ABORT", "UNDEF").
    public static final String KIND_PROPMAP       = "Interrupt.Kind";
    // return-site address -> "form|value" (value omitted for classic returns).
    public static final String EXC_RETURN_PROPMAP = "Interrupt.ExcReturn";

    // ARM EXC_RETURN magic values occupy 0xFFFFFFE0..0xFFFFFFFF.
    private static final long EXC_RETURN_LOW = 0xFFFFFFE0L;

    private InterruptAnalysis() {}

    // ----- public entry point ------------------------------------------------

    public static void run(Program program, TaskMonitor monitor) throws CancelledException {
        if (program == null) {
            return;
        }
        InterruptModel model = selectModel(program);
        if (model == null) {
            return;
        }

        int clearTx = program.startTransaction("InterruptAnalysis clear");
        try {
            clearPriorAutoState(program);
        } finally {
            program.endTransaction(clearTx, true);
        }

        List<Handler> handlers = model.enumerateHandlers(program, monitor);
        if (handlers.isEmpty()) {
            return;
        }

        FunctionManager fm = program.getFunctionManager();

        // ISR entry functions (with their kind) and their transitive callees.
        Map<Function, String> entryKind = new HashMap<>();
        for (Handler h : handlers) {
            Function f = fm.getFunctionContaining(h.entry);
            if (f != null) {
                // First slot wins if the same function backs multiple vectors.
                entryKind.putIfAbsent(f, h.kind);
            }
        }
        if (entryKind.isEmpty()) {
            return;
        }
        Set<Function> reachable = computeReachable(entryKind.keySet(), monitor);

        int txId = program.startTransaction("InterruptAnalysis");
        boolean commit = true;
        try {
            BookmarkManager bm = program.getBookmarkManager();
            StringPropertyMap kindMap = ensureMap(program, KIND_PROPMAP);
            StringPropertyMap excMap  = ensureMap(program, EXC_RETURN_PROPMAP);

            // Tag entry handlers + record kind.
            for (Map.Entry<Function, String> e : entryKind.entrySet()) {
                if (monitor.isCancelled()) {
                    throw new CancelledException();
                }
                Function f = e.getKey();
                f.addTag(ISR_TAG);
                f.addTag(ISR_AUTO_TAG);
                if (kindMap != null) {
                    kindMap.add(f.getEntryPoint(), e.getValue());
                }
                if (bm != null) {
                    bm.setBookmark(f.getEntryPoint(), BookmarkType.ANALYSIS,
                        BOOKMARK_CATEGORY,
                        "ISR entry kind=" + (e.getValue().isEmpty() ? "Generic" : e.getValue()));
                }
            }

            // Tag reachable callees (not themselves entries).
            for (Function f : reachable) {
                if (!entryKind.containsKey(f)) {
                    f.addTag(ISR_REACHABLE_TAG);
                    f.addTag(ISR_AUTO_TAG);
                }
            }

            // Reclassify exception-return terminators across all ISR-context
            // functions (entries + reachable). Set RETURN only where no override
            // exists, so analyst-set overrides are preserved.
            Set<Function> isrContext = new HashSet<>(entryKind.keySet());
            isrContext.addAll(reachable);
            Listing listing = program.getListing();
            for (Function f : isrContext) {
                if (monitor.isCancelled()) {
                    throw new CancelledException();
                }
                for (Instruction instr : listing.getInstructions(f.getBody(), true)) {
                    if (!model.isExceptionReturn(instr)) {
                        continue;
                    }
                    if (instr.getFlowOverride() == FlowOverride.NONE) {
                        instr.setFlowOverride(FlowOverride.RETURN);
                    }
                    ReturnSite rs = model.decodeReturnSite(instr);
                    if (rs != null && rs.explicit && excMap != null) {
                        // form|value ; value omitted (-1) for classic returns.
                        String encoded = rs.form + "|"
                            + (rs.value == null ? "" : Long.toString(rs.value));
                        excMap.add(instr.getAddress(), encoded);
                        if (bm != null) {
                            bm.setBookmark(instr.getAddress(), BookmarkType.ANALYSIS,
                                BOOKMARK_CATEGORY, "exception_return " + encoded);
                        }
                    }
                }
            }
        } catch (CancelledException e) {
            commit = false;
            throw e;
        } catch (RuntimeException e) {
            commit = false;
            throw e;
        } finally {
            program.endTransaction(txId, commit);
        }
    }

    // ----- model selection ---------------------------------------------------

    private static InterruptModel selectModel(Program program) {
        Language lang = program.getLanguage();
        if (lang == null || lang.getProcessor() == null) {
            return null;
        }
        if (!"ARM".equalsIgnoreCase(lang.getProcessor().toString())) {
            return null; // AArch64 / other architectures: out of scope for v1.
        }
        String id = lang.getLanguageID().getIdAsString();
        // M-profile: Ghidra uses "ARM:LE:32:Cortex" (and v7m/v8m variants).
        if (id.contains("Cortex") || id.contains("v7m") || id.contains("v8m")
            || id.contains("v6m")) {
            return new CortexMInterruptModel();
        }
        // Everything else ARM32 (v4/v5/v6/v7/v8 A/R) uses the classic model.
        return new ClassicArmInterruptModel();
    }

    // ----- shared reachability ----------------------------------------------

    private static Set<Function> computeReachable(Set<Function> roots, TaskMonitor monitor)
            throws CancelledException {
        Set<Function> seen = new HashSet<>();
        Deque<Function> work = new ArrayDeque<>(roots);
        while (!work.isEmpty()) {
            if (monitor.isCancelled()) {
                throw new CancelledException();
            }
            Function caller = work.poll();
            for (Function callee : caller.getCalledFunctions(monitor)) {
                if (seen.add(callee)) {
                    work.add(callee);
                }
            }
        }
        return seen;
    }

    // ----- idempotent clear --------------------------------------------------

    private static void clearPriorAutoState(Program program) {
        FunctionManager fm = program.getFunctionManager();
        // Remove auto-applied tags. A function with ISR_AUTO had its ISR /
        // ISR_REACHABLE tags applied by us; strip them. Analyst-added bare ISR
        // tags (no ISR_AUTO companion) are preserved.
        for (Function f : fm.getFunctions(true)) {
            boolean wasAuto = false;
            for (ghidra.program.model.listing.FunctionTag t : f.getTags()) {
                if (ISR_AUTO_TAG.equals(t.getName())) {
                    wasAuto = true;
                    break;
                }
            }
            if (wasAuto) {
                f.removeTag(ISR_TAG);
                f.removeTag(ISR_REACHABLE_TAG);
                f.removeTag(ISR_AUTO_TAG);
            }
        }

        PropertyMapManager pmm = program.getUsrPropertyManager();
        if (pmm != null) {
            clearMap(program, pmm, KIND_PROPMAP);
            clearMap(program, pmm, EXC_RETURN_PROPMAP);
        }

        BookmarkManager bm = program.getBookmarkManager();
        if (bm != null) {
            Iterator<Bookmark> it = bm.getBookmarksIterator(BookmarkType.ANALYSIS);
            List<Bookmark> toRemove = new ArrayList<>();
            while (it.hasNext()) {
                Bookmark b = it.next();
                if (BOOKMARK_CATEGORY.equals(b.getCategory())) {
                    toRemove.add(b);
                }
            }
            for (Bookmark b : toRemove) {
                bm.removeBookmark(b);
            }
        }
        // NOTE: flow overrides set by a prior run are not reset here; re-setting
        // RETURN is idempotent and we only set it where no override existed.
    }

    private static void clearMap(Program program, PropertyMapManager pmm, String name) {
        StringPropertyMap existing = pmm.getStringPropertyMap(name);
        if (existing == null) {
            return;
        }
        for (AddressRange r : program.getMemory().getAddressRanges()) {
            try {
                existing.removeRange(r.getMinAddress(), r.getMaxAddress());
            } catch (Exception e) {
                // non-fatal: new entries overwrite at matching addresses
            }
        }
    }

    private static StringPropertyMap ensureMap(Program program, String name) {
        PropertyMapManager pmm = program.getUsrPropertyManager();
        if (pmm == null) {
            return null;
        }
        StringPropertyMap existing = pmm.getStringPropertyMap(name);
        if (existing != null) {
            return existing;
        }
        try {
            return pmm.createStringPropertyMap(name);
        } catch (Exception e) {
            return pmm.getStringPropertyMap(name);
        }
    }

    // ----- shared helpers for models ----------------------------------------

    static boolean isExcReturnValue(long v) {
        long u = v & 0xFFFFFFFFL;
        return u >= EXC_RETURN_LOW;
    }

    // True if any operand of the instruction is the named register.
    static boolean usesRegister(Instruction instr, String regName) {
        for (int i = 0; i < instr.getNumOperands(); i++) {
            for (Object o : instr.getOpObjects(i)) {
                if (o instanceof Register && ((Register) o).getName().equalsIgnoreCase(regName)) {
                    return true;
                }
            }
        }
        return false;
    }

    // ----- model interface + value types ------------------------------------

    interface InterruptModel {
        // Vector-table-derived handler entries (entry address + attribute kind).
        List<Handler> enumerateHandlers(Program program, TaskMonitor monitor)
            throws CancelledException;

        // True if this instruction is an exception-return terminator.
        boolean isExceptionReturn(Instruction instr);

        // Describe the return site (explicit/non-standard vs plain), or null.
        ReturnSite decodeReturnSite(Instruction instr);
    }

    static final class Handler {
        final Address entry;
        final String kind; // "" | IRQ | FIQ | SWI | ABORT | UNDEF
        Handler(Address entry, String kind) {
            this.entry = entry;
            this.kind = kind;
        }
    }

    static final class ReturnSite {
        final boolean explicit; // non-standard / side-effecting return
        final String form;      // serializer userop name when explicit
        final Long value;       // M-profile EXC_RETURN magic, else null
        ReturnSite(boolean explicit, String form, Long value) {
            this.explicit = explicit;
            this.form = form;
            this.value = value;
        }
        static ReturnSite plain() {
            return new ReturnSite(false, null, null);
        }
    }

    // ----- Cortex-M (M-profile) model ---------------------------------------

    static final class CortexMInterruptModel implements InterruptModel {
        // Slots beyond the initial SP/reset to scan. 16 system exceptions +
        // device IRQs; scan generously and stop at the first unreadable slot.
        private static final int MAX_SLOTS = 256;

        @Override
        public List<Handler> enumerateHandlers(Program program, TaskMonitor monitor) {
            List<Handler> out = new ArrayList<>();
            Memory mem = program.getMemory();
            AddressSpace space = program.getAddressFactory().getDefaultAddressSpace();
            // v1: vector table at the default address (VTOR = 0x0). A future
            // option/symbol can override this base.
            long base = 0L;
            // Slot 0 is the initial SP; slot 1 is the reset vector, which is the
            // program entry running in thread mode -- NOT an exception handler,
            // so it must not receive the interrupt attribute. Start at slot 2
            // (NMI) so reset is never tagged.
            for (int i = 2; i < MAX_SLOTS; i++) {
                Address slot = space.getAddress(base + (long) i * 4);
                int raw;
                try {
                    raw = mem.getInt(slot);
                } catch (MemoryAccessException e) {
                    break; // off the end of mapped memory
                }
                if (raw == 0) {
                    continue;
                }
                Address handler = space.getAddress((raw & 0xFFFFFFFEL)); // clear Thumb bit
                if (!mem.contains(handler)) {
                    continue;
                }
                out.add(new Handler(handler, "")); // M-profile: Generic kind
            }
            return out;
        }

        @Override
        public boolean isExceptionReturn(Instruction instr) {
            String m = instr.getMnemonicString().toLowerCase();
            if (m.startsWith("bx")) {
                Register r = instr.getRegister(0);
                if (r != null && "lr".equalsIgnoreCase(r.getName())) {
                    return true;
                }
                // bx rN where rN holds an EXC_RETURN literal (stack switch).
                return holdsExcReturn(instr, r);
            }
            if (m.startsWith("pop")) {
                return usesRegister(instr, "pc");
            }
            if (m.startsWith("ldr")) {
                Register d = instr.getRegister(0);
                return d != null && "pc".equalsIgnoreCase(d.getName());
            }
            return false;
        }

        @Override
        public ReturnSite decodeReturnSite(Instruction instr) {
            String m = instr.getMnemonicString().toLowerCase();
            if (m.startsWith("bx")) {
                Register r = instr.getRegister(0);
                if (r != null && !"lr".equalsIgnoreCase(r.getName())) {
                    Long lit = excReturnLiteral(instr, r);
                    if (lit != null) {
                        return new ReturnSite(true, "exception_return", lit);
                    }
                }
            }
            return ReturnSite.plain();
        }

        // Backward def-use scan for a constant EXC_RETURN load into `reg`.
        private boolean holdsExcReturn(Instruction instr, Register reg) {
            return excReturnLiteral(instr, reg) != null;
        }

        private Long excReturnLiteral(Instruction bx, Register reg) {
            if (reg == null) {
                return null;
            }
            // Bound the backward walk to the function containing `bx`:
            // getPrevious() follows linear address order, not the CFG, so an
            // unbounded scan could read into a physically-preceding but
            // control-flow-unrelated function and pick up an unrelated def.
            Program program = bx.getProgram();
            Function fn = program == null ? null
                : program.getFunctionManager().getFunctionContaining(bx.getAddress());
            Instruction cur = bx.getPrevious();
            for (int steps = 0; cur != null && steps < 16; steps++, cur = cur.getPrevious()) {
                if (fn != null && !fn.getBody().contains(cur.getAddress())) {
                    break; // crossed the function boundary — stop scanning
                }
                Register dst = cur.getRegister(0);
                if (dst == null || !dst.getName().equalsIgnoreCase(reg.getName())) {
                    continue;
                }
                // mov/ldr reg, #imm — read the scalar operand if present.
                for (int op = 1; op < cur.getNumOperands(); op++) {
                    Scalar s = cur.getScalar(op);
                    if (s != null && isExcReturnValue(s.getUnsignedValue())) {
                        return s.getUnsignedValue() & 0xFFFFFFFFL;
                    }
                }
                // First definition of reg that isn't an EXC_RETURN immediate:
                // stop — the value didn't come from a magic literal.
                return null;
            }
            return null;
        }
    }

    // ----- classic ARMv7-A/R model ------------------------------------------

    static final class ClassicArmInterruptModel implements InterruptModel {
        // Slot index -> ARMInterruptAttr kind. 0=reset, 5=reserved skipped.
        private static String slotKind(int i) {
            switch (i) {
                case 1: return "UNDEF";
                case 2: return "SWI";
                case 3: return "ABORT"; // prefetch abort
                case 4: return "ABORT"; // data abort
                case 6: return "IRQ";
                case 7: return "FIQ";
                default: return null;
            }
        }

        @Override
        public List<Handler> enumerateHandlers(Program program, TaskMonitor monitor) {
            List<Handler> out = new ArrayList<>();
            Memory mem = program.getMemory();
            AddressSpace space = program.getAddressFactory().getDefaultAddressSpace();
            Listing listing = program.getListing();
            // Try low vectors (0x0) then high vectors (0xFFFF0000).
            for (long base : new long[] { 0x0L, 0xFFFF0000L }) {
                for (int i = 0; i <= 7; i++) {
                    String kind = slotKind(i);
                    if (kind == null) {
                        continue;
                    }
                    Address slot = space.getAddress(base + (long) i * 4);
                    if (!mem.contains(slot)) {
                        continue;
                    }
                    Address handler = decodeSlotTarget(program, listing, mem, space, slot);
                    if (handler != null && mem.contains(handler)) {
                        out.add(new Handler(handler, kind));
                    }
                }
            }
            return out;
        }

        // A classic vector slot holds an instruction: `B <handler>` (PC-relative)
        // or `LDR PC,[PC,#off]` (pointer in a literal pool).
        private Address decodeSlotTarget(Program program, Listing listing, Memory mem,
                                         AddressSpace space, Address slot) {
            Instruction instr = listing.getInstructionAt(slot);
            if (instr == null) {
                return null;
            }
            String m = instr.getMnemonicString().toLowerCase();
            // Unconditional branch to the handler (`B <handler>`). Gate on the
            // flow type rather than a mnemonic prefix: startsWith("b") would also
            // match data-processing ops (BIC/BFI) and the BL call, which are not
            // vector-slot branches.
            if (instr.getFlowType().isJump()) {
                Address[] flows = instr.getFlows();
                if (flows != null && flows.length > 0) {
                    return flows[0];
                }
            }
            if (m.startsWith("ldr")) {
                // ldr pc,[pc,#off]: the referenced literal holds the handler ptr.
                ghidra.program.model.symbol.Reference[] refs = instr.getReferencesFrom();
                for (ghidra.program.model.symbol.Reference ref : refs) {
                    Address pool = ref.getToAddress();
                    try {
                        int ptr = mem.getInt(pool);
                        return space.getAddress(ptr & 0xFFFFFFFEL);
                    } catch (MemoryAccessException e) {
                        // try next reference
                    }
                }
            }
            return null;
        }

        @Override
        public boolean isExceptionReturn(Instruction instr) {
            String m = instr.getMnemonicString().toLowerCase();
            // subs pc,lr,#N / movs pc,lr — dest pc, restores CPSR via S suffix.
            if (m.startsWith("subs") || m.startsWith("movs")) {
                Register d = instr.getRegister(0);
                return d != null && "pc".equalsIgnoreCase(d.getName());
            }
            // rfe — return from exception.
            if (m.startsWith("rfe")) {
                return true;
            }
            // ldm{..}^ that loads PC. The user-mode/CPSR-restore '^' is not
            // reliably exposed in the mnemonic; approximate with ldm-loads-pc.
            if (m.startsWith("ldm")) {
                return usesRegister(instr, "pc");
            }
            return false;
        }

        @Override
        public ReturnSite decodeReturnSite(Instruction instr) {
            // Every classic exception return performs a CPSR/SPSR restore that a
            // plain `return` would drop, so mark all as explicit (no magic
            // value; the form carries the meaning).
            return new ReturnSite(true, "exception_return_cpsr", null);
        }
    }
}
