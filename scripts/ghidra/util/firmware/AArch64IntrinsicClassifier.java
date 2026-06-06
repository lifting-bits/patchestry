/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

package util.firmware;

/**
 * AArch64 userop vocabulary -- extension-point stub. Returns
 * {@link IntrinsicClassifier#unknown()} for every name today, so AArch64
 * userops appear in the missing-intrinsic inventory (origin/manifest) but are
 * not yet spelled.
 *
 * <p>TODO (when implemented): map the AArch64 SLEIGH userops
 * (Processors/AARCH64/data/languages/*.sinc) onto the shared taxonomy --
 * system-register read/write via {@code SysOp_R}/{@code SysOp_W} /
 * {@code UnkSytemRegRead}/{@code UnkSytemRegWrite}, the cache/TLB maintenance
 * families {@code DC_*}/{@code IC_*}/{@code TLBI_*}/{@code AT_*} (likely new
 * generic classes {@code cache_op}/{@code tlb_op}), {@code DataMemoryBarrier}/
 * {@code DataSynchronizationBarrier}/{@code InstructionSynchronizationBarrier}
 * (barriers), and {@code WaitForInterrupt}/{@code WaitForEvent}/
 * {@code SendEvent}/{@code Yield} (hints). The C++ side then needs an AArch64
 * speller registered by the "AARCH64" processor string.
 */
public final class AArch64IntrinsicClassifier implements ArchIntrinsicClassifier {

    @Override
    public IntrinsicClassifier.Result classify(String rawName) {
        return IntrinsicClassifier.unknown();
    }
}
