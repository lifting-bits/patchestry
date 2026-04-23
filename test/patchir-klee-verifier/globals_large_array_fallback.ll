; RUN: %patchir-klee-verifier %s --target-function probe --klee-init-array-expand-limit=8 -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: large-array fallback. An array above the expand limit collapses
; to a single flat klee_make_symbolic over the whole storage, matching
; the pre-change behavior for globals of this shape. We set the limit
; to 8 in the RUN line so the test fixture stays small (16 elements >
; 8), rather than hard-coding a fixture with 65+ elements.
;
; The element type here is a scalar struct (no pointers), so even with
; a normal limit we'd just take the pointer-free fast path. The cap
; matters when the element type contains pointers — in that case the
; fall-back documents a known limitation (nested pointer fields inside
; array elements stay as unconstrained symbolic bytes).

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

%struct.Entry = type { i32, i32 }

@table = global [16 x %struct.Entry] zeroinitializer

define i32 @probe(i32 %i) {
entry:
  %ep  = getelementptr inbounds [16 x %struct.Entry], ptr @table, i32 0, i32 %i
  %v_p = getelementptr inbounds %struct.Entry, ptr %ep, i32 0, i32 0
  %v   = load i32, ptr %v_p
  ret i32 %v
}

; --- Harness main() dispatches through __klee_init_globals ---
; CHECK:       define i32 @main()
; CHECK:       call void @__klee_init_globals()

; --- Per-global wrapper for @table: zero initializer routes to the
; trivial-init fast path in buildTypeInitBody. Since %struct.Entry
; contains no pointers, the fast path emits a single flat
; klee_make_symbolic over the entire [16 x Entry] storage (128 bytes).
; No 16 element-wise inits, no per-element walk. ---
; CHECK:       define internal void @__klee_init_g_table()
; CHECK:       call void @klee_make_symbolic(ptr @table, i64 128,
; CHECK:       ret void

; --- No per-type init functions for arr16 or Entry — both are pointer-
; free so the walk is inlined/flat, not dispatched through a cached
; per-type init. ---
; CHECK-NOT:   define internal void @__klee_init_type_arr16_struct_Entry
; CHECK-NOT:   define internal void @__klee_init_type_struct_Entry
