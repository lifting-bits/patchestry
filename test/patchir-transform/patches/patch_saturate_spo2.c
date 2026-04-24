// RUN: true
typedef float SFtype;

// Single-return replacement for `spo2_lookup`. Scales the incoming float
// by 2.0 and narrows to u8 — trivially wrong semantically, intentionally
// so. The point is a patch body with no control flow so every statement
// lowers to straight-line CIR terminated by a single top-level cir.return.
// Used by inline_patches_replace_positive.yaml to exercise the
// `replaceCallWithPatch` + `inline-patches` code path
// (PatchOperationImpl.cpp:289), which the multi-return
// patch__replace__spo2_lookup cannot reach (blocked by Bug F / nested
// cir.return inside cir.scope).
unsigned char patch__replace__saturate_spo2(SFtype param_0) {
    return (unsigned char)(param_0 * (SFtype)2.0f);
}
