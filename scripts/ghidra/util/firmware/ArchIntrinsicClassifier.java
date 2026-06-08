/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.firmware;

/**
 * Per-architecture userop vocabulary: maps an architecture's SLEIGH userop
 * names onto the shared {@link IntrinsicClassifier} taxonomy. Implementations
 * are registered by processor string in {@link IntrinsicClassifier}.
 *
 * <p>Implementations must be pure and deterministic, and must return
 * {@link IntrinsicClassifier#unknown()} (never null) for names they do not
 * recognize so the missing-intrinsic inventory can surface them.
 */
@FunctionalInterface
public interface ArchIntrinsicClassifier {
    IntrinsicClassifier.Result classify(String rawName);
}
