// RUN: true
typedef unsigned int uint32_t;

// Void probe function used as a regression witness for
// `replaceOperationWithPatch`'s arity pre-check: attempting to replace
// a value-producing `cir.binop` with this function (which returns
// nothing) used to leave a half-applied state (patch call + original
// op both present in CIR). See
// test/patchir-transform/replace_binop_void_reject.json.
void patch__probe__void_mul(uint32_t a, uint32_t b) {
    (void) a;
    (void) b;
}
