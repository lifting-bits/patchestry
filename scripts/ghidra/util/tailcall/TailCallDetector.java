/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.tailcall;

import ghidra.app.cmd.function.CallDepthChangeInfo;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressSetView;

import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;
import ghidra.program.model.block.CodeBlockReference;
import ghidra.program.model.block.CodeBlockReferenceIterator;

import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.listing.Program;

import ghidra.program.model.symbol.FlowType;
import ghidra.program.model.symbol.Reference;
import ghidra.program.model.symbol.ReferenceManager;

import ghidra.util.exception.CancelledException;
import ghidra.util.task.TaskMonitor;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

// Walks every function and reports basic blocks that end in an unconditional
// branch whose target is structurally consistent with a separate function
// entry (SP-delta balanced, target outside body or analyzer blind-spot,
// at least one corroborator).
public final class TailCallDetector {

    public enum Corroborator { PROLOGUE, NORETURN, THUNK }

    public static final class Candidate {
        public final Address from;
        public final Address target;
        public final Function inFunction;
        public final EnumSet<Corroborator> corroborators;

        Candidate(Address from, Address target, Function inFunction,
                  EnumSet<Corroborator> corroborators) {
            this.from = from;
            this.target = target;
            this.inFunction = inFunction;
            this.corroborators = corroborators;
        }
    }

    private static final Set<String> EXTRA_NORETURN = Set.of(
        "abort", "exit", "_exit", "_Exit",
        "__assert_fail", "__stack_chk_fail",
        "panic", "longjmp", "siglongjmp",
        "__cxa_throw", "_Unwind_Resume"
    );

    private final Program program;
    private final TailCallProcessorRules rules;
    private final TaskMonitor monitor;

    private final FunctionManager fm;
    private final ReferenceManager rm;
    private final Listing listing;
    private final BasicBlockModel bbm;

    public TailCallDetector(Program program, TailCallProcessorRules rules,
                            TaskMonitor monitor) {
        this.program = program;
        this.rules = rules;
        this.monitor = monitor;
        this.fm = program.getFunctionManager();
        this.rm = program.getReferenceManager();
        this.listing = program.getListing();
        this.bbm = new BasicBlockModel(program);
    }

    public List<Candidate> detect() throws CancelledException {
        List<Candidate> out = new ArrayList<>();
        for (Function f : fm.getFunctions(true)) {
            if (monitor.isCancelled()) throw new CancelledException();
            if (f.isExternal() || f.isThunk()) continue;
            try {
                detectInFunction(f, out);
            } catch (CancelledException e) {
                throw e;
            } catch (Exception e) {
                // Per-function failure is non-fatal.
            }
        }
        return out;
    }

    private void detectInFunction(Function f, List<Candidate> out) throws Exception {
        AddressSetView body = f.getBody();
        if (body == null || body.isEmpty()) return;

        CallDepthChangeInfo cdci;
        try {
            cdci = new CallDepthChangeInfo(f, monitor);
        } catch (Exception e) {
            return;
        }

        int redZone = rules.redZoneSize();

        CodeBlockIterator blocks = bbm.getCodeBlocksContaining(body, monitor);
        while (blocks.hasNext()) {
            if (monitor.isCancelled()) throw new CancelledException();
            CodeBlock block = blocks.next();
            Instruction last = listing.getInstructionContaining(block.getMaxAddress());
            if (last == null) continue;

            FlowType ft = last.getFlowType();
            if (ft == null) continue;
            if (!ft.isJump() || ft.isConditional() || ft.isComputed() || ft.isCall()) {
                continue;
            }

            Address target = primaryFlow(last);
            if (target == null) continue;

            int spAtBranch;
            try {
                spAtBranch = cdci.getDepth(last.getAddress());
            } catch (Exception e) {
                continue;
            }
            if (Math.abs(spAtBranch) > redZone) continue;

            boolean targetOutside = !body.contains(target);
            boolean blindSpot = !targetOutside && targetLooksLikeEntry(target);
            if (!targetOutside && !blindSpot) continue;

            EnumSet<Corroborator> hits = EnumSet.noneOf(Corroborator.class);
            Instruction targetInsn = listing.getInstructionAt(target);
            if (targetInsn != null && rules.isFunctionPrologue(targetInsn)) {
                hits.add(Corroborator.PROLOGUE);
            }
            if (predecessorEndsWithNoreturn(block)) {
                hits.add(Corroborator.NORETURN);
            }
            if (isSingleInstructionThunk(f, last)) {
                hits.add(Corroborator.THUNK);
            }
            if (hits.isEmpty()) continue;

            out.add(new Candidate(last.getAddress(), target, f, hits));
        }
    }

    private Address primaryFlow(Instruction ins) {
        Address[] flows = ins.getFlows();
        return (flows == null || flows.length == 0) ? null : flows[0];
    }

    // Inside-body address that looks like a function entry Ghidra missed:
    // not already a function, no incoming call xref, starts a basic block.
    private boolean targetLooksLikeEntry(Address target) {
        if (fm.getFunctionAt(target) != null) return false;
        for (Reference ref : rm.getReferencesTo(target)) {
            if (ref.getReferenceType() != null && ref.getReferenceType().isCall()) {
                return false;
            }
        }
        try {
            CodeBlock[] containing = bbm.getCodeBlocksContaining(target, monitor);
            for (CodeBlock cb : containing) {
                if (cb.getFirstStartAddress().equals(target)) return true;
            }
        } catch (CancelledException e) {
            return false;
        }
        return false;
    }

    private boolean predecessorEndsWithNoreturn(CodeBlock block) {
        try {
            CodeBlockReferenceIterator preds = block.getSources(monitor);
            while (preds.hasNext()) {
                CodeBlockReference ref = preds.next();
                CodeBlock src = ref.getSourceBlock();
                if (src == null) continue;
                Instruction term = listing.getInstructionContaining(src.getMaxAddress());
                if (term == null) continue;
                FlowType ft = term.getFlowType();
                if (ft == null || !ft.isCall()) continue;
                Address callTarget = primaryFlow(term);
                if (callTarget == null) continue;
                Function callee = fm.getFunctionAt(callTarget);
                if (callee == null) continue;
                if (callee.hasNoReturn()) return true;
                if (EXTRA_NORETURN.contains(callee.getName())) return true;
            }
        } catch (CancelledException e) {
            // fall through
        }
        return false;
    }

    private boolean isSingleInstructionThunk(Function f, Instruction last) {
        long bytes = f.getBody().getNumAddresses();
        if (bytes <= 0) return false;
        int minLen = program.getLanguage().getInstructionAlignment();
        if (minLen <= 0) minLen = 2;
        long approxInsnCount = (bytes + minLen - 1) / minLen;
        return approxInsnCount <= rules.thunkInstructionLimit()
            && f.getEntryPoint().equals(blockStart(last));
    }

    private Address blockStart(Instruction last) {
        try {
            CodeBlock[] blocks = bbm.getCodeBlocksContaining(last.getAddress(), monitor);
            if (blocks != null && blocks.length > 0) {
                return blocks[0].getFirstStartAddress();
            }
        } catch (CancelledException e) {
            // ignore
        }
        return last.getAddress();
    }
}
