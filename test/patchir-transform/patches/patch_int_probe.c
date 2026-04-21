// RUN: true

// Minimal apply_after probe used to exercise capture-by-result binding.
// Receives the result of a matched binop via `captures: - result: 0`
// + `source: capture`. Parameter is unsigned so it matches the binop's
// !u32i result type — otherwise the pass would insert a signed cast
// and the test wouldn't be able to pin the probe's arg back to the
// binop's raw SSA result.

typedef unsigned int uint32_t;

void patch__after__int_probe(uint32_t result) {
    (void) result;
}
