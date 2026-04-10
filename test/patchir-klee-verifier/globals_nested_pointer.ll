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

; --- Per-global wrapper for @dev ---
; CHECK:       define internal void @__klee_init_g_dev()
; CHECK:       call void @__klee_init_type_struct_Device(ptr @dev, i32 0)

; --- Per-type init for %struct.Device ---
; Expected body: klee_make_symbolic on field 0 (i32), the runtime depth
; gate + malloc + nested init call for field 1 (ptr -> i32), and
; klee_make_symbolic on field 2 (i32). We check the structural
; ingredients rather than exact SSA numbering.
;
; The IR-builder creates recurse_bb before null_bb, so the `init.recurse`
; block appears first in the function text even though the CFG branches
; to `init.null` on the too-deep path.
; CHECK:       define internal void @__klee_init_type_struct_Device(ptr %p, i32 %depth)
; CHECK:       call void @klee_make_symbolic(
; CHECK:       icmp sge i32
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
