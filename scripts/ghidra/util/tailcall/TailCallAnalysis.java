/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.tailcall;

import ghidra.app.cmd.function.CreateFunctionCmd;
import ghidra.app.plugin.core.analysis.AutoAnalysisManager;

import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressRange;
import ghidra.program.model.address.AddressSet;

import ghidra.program.model.listing.Bookmark;
import ghidra.program.model.listing.BookmarkManager;
import ghidra.program.model.listing.BookmarkType;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Program;

import ghidra.program.model.util.PropertyMapManager;
import ghidra.program.model.util.StringPropertyMap;

import ghidra.util.exception.CancelledException;
import ghidra.util.task.TaskMonitor;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;

// Detect tail-call sites and split merged functions via CreateFunctionCmd.
// Findings: Bookmark category "TailCallAnalysis" + StringPropertyMap
// "TailCall.SiteAddr" (branch address -> callee Address.toString(true)).
public final class TailCallAnalysis {

    public static final String BOOKMARK_CATEGORY = "TailCallAnalysis";
    public static final String SITE_PROPMAP = "TailCall.SiteAddr";
    // Legacy LongPropertyMap name; cleared on migration.
    private static final String LEGACY_SITE_PROPMAP = "TailCall.Site";

    public static void run(Program program, TaskMonitor monitor) throws CancelledException {
        if (program == null) return;
        TailCallProcessorRules rules = TailCallProcessorRules.lookup(program.getLanguage());
        if (rules == null) return;

        // Drop stale state from prior runs (siteMap/bookmarks are
        // append-only otherwise).
        int clearTx = program.startTransaction("TailCallAnalysis clear");
        try {
            clearPriorState(program);
        } finally {
            program.endTransaction(clearTx, true);
        }

        TailCallDetector detector = new TailCallDetector(program, rules, monitor);
        List<TailCallDetector.Candidate> candidates = detector.detect();
        if (candidates.isEmpty()) return;

        FunctionManager fm = program.getFunctionManager();
        AddressSet splitTargets = new AddressSet();
        AddressSet affectedContainers = new AddressSet();

        int txId = program.startTransaction("TailCallAnalysis");
        boolean commit = true;
        try {
            BookmarkManager bm = program.getBookmarkManager();
            StringPropertyMap siteMap = ensureSiteMap(program);

            for (TailCallDetector.Candidate c : candidates) {
                if (monitor.isCancelled()) throw new CancelledException();

                String action;
                if (fm.getFunctionAt(c.target) != null) {
                    action = "detected";
                } else {
                    CreateFunctionCmd cmd = new CreateFunctionCmd(c.target);
                    if (cmd.applyTo(program, monitor)) {
                        action = "split";
                        splitTargets.add(c.target);
                        if (c.inFunction != null) {
                            affectedContainers.add(c.inFunction.getEntryPoint());
                        }
                    } else {
                        action = "skipped";
                    }
                }

                recordBookmark(bm, c, action);
                // Skip "skipped" sites. toString(true) keeps the space
                // prefix so cross-space callees re-resolve correctly.
                if (siteMap != null && !"skipped".equals(action)) {
                    siteMap.add(c.from, c.target.toString(true));
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

        // Re-decompile the new entries and the wrappers whose body shrank,
        // otherwise the serializer would still emit the pre-split shape.
        if (commit && !splitTargets.isEmpty()) {
            rerunAnalysis(program, splitTargets, affectedContainers, monitor);
        }
    }

    private static void rerunAnalysis(Program program, AddressSet splitTargets,
                                      AddressSet affectedContainers, TaskMonitor monitor) {
        try {
            AutoAnalysisManager mgr = AutoAnalysisManager.getAnalysisManager(program);
            AddressSet rerun = new AddressSet();
            rerun.add(splitTargets);
            rerun.add(affectedContainers);
            int rerunTx = program.startTransaction("TailCallAnalysis rerun");
            try {
                if (mgr.getAnalyzer("Decompiler Parameter ID") != null) {
                    mgr.scheduleOneTimeAnalysis(
                        mgr.getAnalyzer("Decompiler Parameter ID"), rerun);
                }
                mgr.startAnalysis(monitor);
                mgr.waitForAnalysis(null, monitor);
            } finally {
                program.endTransaction(rerunTx, true);
            }
        } catch (Exception e) {
            // Re-analysis failure is non-fatal.
        }
    }

    // Drain siteMap entries + remove TailCallAnalysis bookmarks.
    private static void clearPriorState(Program program) {
        PropertyMapManager pmm = program.getUsrPropertyManager();
        if (pmm != null) {
            // Drop legacy LongPropertyMap so StringPropertyMap creates clean.
            try {
                pmm.removePropertyMap(LEGACY_SITE_PROPMAP);
            } catch (Exception e) {
                // legacy map absent / removal unsupported — non-fatal
            }
            StringPropertyMap existing = pmm.getStringPropertyMap(SITE_PROPMAP);
            if (existing != null) {
                for (AddressRange r : program.getMemory().getAddressRanges()) {
                    try {
                        existing.removeRange(r.getMinAddress(), r.getMaxAddress());
                    } catch (Exception e) {
                        // non-fatal: new entries overwrite at matching addresses
                    }
                }
            }
        }

        BookmarkManager bm = program.getBookmarkManager();
        if (bm != null) {
            Iterator<Bookmark> it = bm.getBookmarksIterator(BookmarkType.ANALYSIS);
            List<Bookmark> toRemove = new ArrayList<>();
            while (it.hasNext()) {
                Bookmark bookmark = it.next();
                if (BOOKMARK_CATEGORY.equals(bookmark.getCategory())) {
                    toRemove.add(bookmark);
                }
            }
            for (Bookmark b : toRemove) {
                bm.removeBookmark(b);
            }
        }
    }

    private static StringPropertyMap ensureSiteMap(Program program) {
        PropertyMapManager pmm = program.getUsrPropertyManager();
        if (pmm == null) return null;
        StringPropertyMap existing = pmm.getStringPropertyMap(SITE_PROPMAP);
        if (existing != null) return existing;
        try {
            return pmm.createStringPropertyMap(SITE_PROPMAP);
        } catch (Exception e) {
            return pmm.getStringPropertyMap(SITE_PROPMAP);
        }
    }

    private static void recordBookmark(BookmarkManager bm,
                                       TailCallDetector.Candidate c, String action) {
        if (bm == null) return;
        StringBuilder sb = new StringBuilder();
        sb.append("from=").append(c.from.toString(true));
        sb.append(" target=").append(c.target.toString(true));
        sb.append(" corroborators=").append(joinCorroborators(c.corroborators));
        sb.append(" action=").append(action);
        bm.setBookmark(c.from, BookmarkType.ANALYSIS, BOOKMARK_CATEGORY, sb.toString());
    }

    private static String joinCorroborators(EnumSet<TailCallDetector.Corroborator> set) {
        if (set == null || set.isEmpty()) return "none";
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (TailCallDetector.Corroborator c : set) {
            if (!first) sb.append(',');
            sb.append(c.name().toLowerCase());
            first = false;
        }
        return sb.toString();
    }
}
