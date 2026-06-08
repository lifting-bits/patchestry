/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.firmware;

import java.util.HashMap;
import java.util.Map;

/**
 * Architecture-neutral entry point for classifying CALLOTHER userops into a
 * shared taxonomy so the C++ AST layer can spell each as a compiler builtin
 * (ACLE) / library intrinsic (CMSIS) without re-deriving the vocabulary.
 *
 * <p>The taxonomy ({@code irq_mask_set}, {@code sysreg_read}, {@code coproc_*},
 * {@code barrier_*}, {@code hint_*}, {@code trap}, {@code mode_switch},
 * {@code query}, {@code builtin}, {@link #UNKNOWN}) is the stable contract. The
 * per-architecture *vocabulary* (name -> class) lives in an
 * {@link ArchIntrinsicClassifier}, selected here by the program's processor
 * string (case-insensitive). To add an architecture, implement
 * {@link ArchIntrinsicClassifier} and register it in {@link #buildRegistry}.
 *
 * <p>Pure and deterministic.
 */
public final class IntrinsicClassifier {

    /** Taxonomy tag for an unrecognized / unmapped userop. */
    public static final String UNKNOWN = "unknown";

    /** Classification result for a single userop. */
    public static final class Result {
        public final String klass;     // taxonomy tag, never null ("unknown" when unmapped)
        public final String register;  // decoded system register, or null
        public final boolean mapped;   // false when klass == UNKNOWN

        Result(String klass, String register) {
            this.klass    = klass;
            this.register = register;
            this.mapped   = !UNKNOWN.equals(klass);
        }
    }

    /** Build a mapped result. For use by {@link ArchIntrinsicClassifier} impls. */
    static Result result(String klass, String register) {
        return new Result(klass, register);
    }

    /** The shared "not classified" result. */
    static Result unknown() {
        return new Result(UNKNOWN, null);
    }

    // An architecture with no registered classifier maps nothing.
    private static final ArchIntrinsicClassifier EMPTY = name -> unknown();

    // processor string (lower-case) -> classifier.
    private static final Map<String, ArchIntrinsicClassifier> REGISTRY = buildRegistry();

    private static Map<String, ArchIntrinsicClassifier> buildRegistry() {
        Map<String, ArchIntrinsicClassifier> m = new HashMap<>();
        m.put("arm", new Arm32IntrinsicClassifier());
        m.put("aarch64", new AArch64IntrinsicClassifier());
        return m;
    }

    private IntrinsicClassifier() {}

    /**
     * Classify {@code name} using the vocabulary for {@code processor} (the
     * Ghidra processor string, e.g. "ARM"/"AARCH64", matched case-insensitively).
     * An unknown processor or unrecognized name yields {@link #UNKNOWN} with
     * {@code mapped == false}.
     */
    public static Result classify(String processor, String name) {
        String key = (processor == null) ? "" : processor.toLowerCase();
        return REGISTRY.getOrDefault(key, EMPTY).classify(name);
    }
}
