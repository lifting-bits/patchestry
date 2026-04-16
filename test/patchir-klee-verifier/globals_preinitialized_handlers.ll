; RUN: %patchir-klee-verifier %s --target-function dispatch -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Regression test: preinitialized function-pointer table must not be
; clobbered. Before this fix, emitPointerField unconditionally stored
; null into every FunctionPointer-inferred field, turning
; `@ops = global { ptr null, ptr @real_handler }` into a guaranteed
; null-function-pointer crash before the target could ever invoke the
; real handler. KLEE never explored the real handler path.
;
; The fix walks each global's initializer alongside its type. For
; pointer fields whose initializer is a concrete constant (Function,
; GlobalVariable, ConstantExpr), the per-global init is a no-op —
; the initializer is already in place at load time. For null/undef
; pointer fields, the existing symbolization/null-store path runs.

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

%struct.ops_t = type { ptr, ptr }

define i32 @real_handler(i32 %x) {
entry:
  %r = add i32 %x, 1
  ret i32 %r
}

; field 0: null (should be symbolized/stored null as before)
; field 1: @real_handler (MUST be preserved)
@ops = global %struct.ops_t { ptr null, ptr @real_handler }

define i32 @dispatch(i32 %x) {
entry:
  %fn_p = getelementptr inbounds %struct.ops_t, ptr @ops, i32 0, i32 1
  %fn   = load ptr, ptr %fn_p
  ; Indirect call — inference sees the loaded pointer as a CallBase
  ; callee and marks (ops_t, 1) -> FunctionPointer.
  %ret  = call i32 %fn(i32 %x)
  ret i32 %ret
}

; --- Harness main() dispatches through __klee_init_globals ---
; CHECK:       define i32 @main()
; CHECK:       call void @__klee_init_globals()

; --- Per-global wrapper for @ops: non-trivial initializer triggers the
; initializer-aware walk. Field 0 has a null initializer → symbolized
; (stored null for FunctionPointer kind). Field 1 has @real_handler as
; its initializer → preserved (no store emitted). ---
; CHECK:       define internal void @__klee_init_g_ops()

; Field 0 (null init) is symbolized — either a flat symbolic buffer
; via malloc (Unknown/ScalarBytes inference) or a direct null store
; (FunctionPointer inference). Either shape indicates field 0 is
; being initialized by the pass.
; CHECK:         store ptr {{.*}}, ptr @ops

; Field 1 must NOT receive any store — the initializer @real_handler
; is authoritative. No GEP into (ops_t, 0, 1) should be the target of
; a store inside this wrapper.
; CHECK-NOT:     store ptr {{.*}}getelementptr{{.*}}i32 0, i32 1

; CHECK:         ret void
