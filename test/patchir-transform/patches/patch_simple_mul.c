// RUN: true
typedef unsigned int uint32_t;

// Single-return u32(u32, u32) replacement used by
// inline_patches_replace_binop.yaml to exercise
// `replaceOperationWithPatch` + `inline-patches`
// (PatchOperationImpl.cpp:379). `patch__replace__int_mul` has an early-
// return inside `cir.if` and is blocked by Bug F; this one has no
// control flow and inlines cleanly.
uint32_t patch__replace__passthrough_mul(uint32_t a, uint32_t b) {
    return a * b;
}
