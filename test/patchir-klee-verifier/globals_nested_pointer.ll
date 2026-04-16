; RUN: %patchir-klee-verifier %s --target-function read_buffer -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: nested-pointer init. A non-const global of a struct type whose
; second field is a `ptr` field that the target function dereferences.
;
; The type-inference pass in patchir-klee-verifier sees the loaded
; pointer used as the base of a typed `load i32, ptr %buf` and records
; `(Device, 1) -> PointeeType(i32)`. The per-type init function for
; %struct.Device then mallocs sizeof(i32) bytes for the buf field,
; symbolizes them, stores the address, and the runtime can dereference
; the field without hitting an unconstrained symbolic pointer.
;
; The fast-path fields (i32 id, i32 len) still get one flat
; klee_make_symbolic call each.

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

%struct.Device = type { i32, ptr, i32 }

@dev = global %struct.Device zeroinitializer

define i32 @read_buffer() {
entry:
  %id_p  = getelementptr inbounds %struct.Device, ptr @dev, i32 0, i32 0
  %id    = load i32, ptr %id_p
  %buf_p = getelementptr inbounds %struct.Device, ptr @dev, i32 0, i32 1
  %buf   = load ptr, ptr %buf_p
  ; Typed load through the pointer field — this is the use site that
  ; the inference pass consumes to learn `(Device, 1) -> i32*`.
  %val   = load i32, ptr %buf
  %sum   = add i32 %id, %val
  ret i32 %sum
}

; --- Harness main() dispatches through __klee_init_globals ---
; CHECK:       define i32 @main()
; CHECK:       call void @__klee_init_globals()

; --- Per-global wrapper for @dev: zero initializer routes to the
; trivial-init fast path in buildTypeInitBody. The walker emits
; klee_make_symbolic for fields 0 and 2 (i32), and for field 1 (ptr)
; emits the runtime depth gate + malloc + call to the cached per-type
; init for the pointee (i32). Because depth is a compile-time 0 at the
; top level, the depth-check constant-folds to `br i1 false`, but the
; CFG (init.recurse/init.null/init.cont) is still emitted. ---
; CHECK:       define internal void @__klee_init_g_dev()
; CHECK:       call void @klee_make_symbolic(
; CHECK:       init.recurse:
; CHECK:       call ptr @malloc(
; CHECK:       call void @__klee_init_type_i32(
; CHECK:       init.null:
; CHECK:       store ptr null,
; CHECK:       init.cont:

; --- Per-type init for i32 (used by the pointer field's pointee) ---
; CHECK:       define internal void @__klee_init_type_i32(ptr %p, i32 %depth)
; CHECK:       call void @klee_make_symbolic(ptr %p, i64 4,

; --- No inline per-global klee_make_symbolic on @dev in main() ---
; CHECK-NOT:   call void @klee_make_symbolic(ptr @dev,
