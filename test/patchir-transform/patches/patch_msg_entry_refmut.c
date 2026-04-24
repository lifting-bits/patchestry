// RUN: true

// Entry-point patch that writes through the parameter reference slot.
// Regression cover for commit 500016e: with `apply_at_entrypoint` +
// `source: operand` + `is_reference: true`, the patch must be handed
// the caller's parameter alloca directly (the slot the function itself
// reads from) rather than a fresh `arg_ref` temporary. The body here
// is intentionally minimal — the LIT check is structural (that the
// parameter alloca SSA value is what flows into the patch call).
void patch__entrypoint__message_entry_refmut(void **msg_ref) {
    if (*msg_ref == (void *)0) {
        *msg_ref = (void *)1;
    }
}
