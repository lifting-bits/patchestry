// RUN: true

// CWE-415: Null-after-free guard for double free prevention.
// This patch is applied after a free() call with is_reference: true.
// The is_reference mechanism creates an alloca for the pointer operand
// and passes its address (void**). By setting *ptr_ref = NULL, we null
// the pointer after free. The update_state_after_patch mechanism then
// propagates this NULL to all subsequent dominated uses of the pointer,
// making the second free(ptr) become free(NULL) — a safe no-op.
//
// Parameter type is void** to match the alloca type produced by
// is_reference for a void* operand.
void patch__after__free(void **ptr_ref) {
    *ptr_ref = (void *)0;
}
