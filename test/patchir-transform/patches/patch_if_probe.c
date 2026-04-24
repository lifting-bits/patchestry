// RUN: true

// Minimal probe patch used to verify that name-based matching of a
// region-carrying op (cir.if) works. apply_before inserts this call
// immediately ahead of the matched cir.if; captures cannot reach into
// the op's regions, so the probe just takes the condition operand.

void patch__before__if_probe(_Bool cond) {
    (void) cond;
}
