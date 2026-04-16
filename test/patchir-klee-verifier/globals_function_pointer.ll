; RUN: %patchir-klee-verifier %s --target-function dispatch -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: function-pointer field init. A struct whose pointer field is
; only ever used as the callee of an indirect call. The inference pass
; classifies it as FunctionPointer via the CallBase use-site check, and
; the per-type init function emits `store ptr null` for the field
; instead of allocating a symbolic byte buffer.
;
; At runtime, a symbolic execution that actually invokes the field will
; reach a clean null-function-pointer error from KLEE — much better
; than the pre-change behavior where the 8 bytes of the field were
; made symbolic and became an arbitrary jump target.

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

%struct.Ops = type { i32, ptr }

@ops = global %struct.Ops zeroinitializer

define i32 @dispatch(i32 %x) {
entry:
  %fn_p = getelementptr inbounds %struct.Ops, ptr @ops, i32 0, i32 1
  %fn   = load ptr, ptr %fn_p
  ; Indirect call — the inference pass sees the loaded pointer as a
  ; CallBase callee and marks (Ops, 1) -> FunctionPointer.
  %ret  = call i32 %fn(i32 %x)
  ret i32 %ret
}

; --- Harness main() dispatches through __klee_init_globals ---
; CHECK:       define i32 @main()
; CHECK:       call void @__klee_init_globals()

; --- Per-global wrapper for @ops: zero initializer routes to the
; trivial-init fast path in buildTypeInitBody. The walker symbolizes
; field 0 (i32) inline and stores null into field 1 (fn pointer inferred
; as FunctionPointer). No malloc, no flat symbolic over field 1. ---
; CHECK:       define internal void @__klee_init_g_ops()
; CHECK:       call void @klee_make_symbolic(
; CHECK:       store ptr null,
; CHECK:       ret void

; --- No per-type init should be emitted for %struct.Ops — the walk is
; inlined into the per-global wrapper and the only pointer field is
; FunctionPointer (stored as null, no recursive pointee allocation). ---
; CHECK-NOT:   define internal void @__klee_init_type_struct_Ops
