; RUN: %patchir-klee-verifier %s --target-function walk_list -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: self-referential type init. A linked-list node where the `next`
; field points to the same struct type. The inference pass discovers
; `(Node, 1) -> PointeeType(%struct.Node)` via the GEP at `cur->next->v`.
;
; The critical property being verified is that the per-type init
; function for %struct.Node closes the type cycle at codegen time: its
; body contains exactly one call to itself (`@__klee_init_type_struct_Node`).
; Without the cache-before-body-build trick in getOrCreatePerTypeInit,
; the recursive build would spin forever trying to inline Node's body.
;
; At runtime, the depth argument bounds KLEE's evaluation: at depth 2
; the `next` field is set to null via the init.null branch. The target
; can walk two links symbolically; a third hop hits a null pointer.

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

%struct.Node = type { i32, ptr }

@head = global %struct.Node zeroinitializer

define i32 @walk_list() {
entry:
  %v1p    = getelementptr inbounds %struct.Node, ptr @head, i32 0, i32 0
  %v1     = load i32, ptr %v1p
  %next_p = getelementptr inbounds %struct.Node, ptr @head, i32 0, i32 1
  %next   = load ptr, ptr %next_p
  ; Typed GEP through the next pointer — this is the use site that
  ; lets the inference pass recognize `(Node, 1) -> Node*`.
  %v2p    = getelementptr inbounds %struct.Node, ptr %next, i32 0, i32 0
  %v2     = load i32, ptr %v2p
  %sum    = add i32 %v1, %v2
  ret i32 %sum
}

; --- Harness main() dispatches through __klee_init_globals ---
; CHECK:       define i32 @main()
; CHECK:       call void @__klee_init_globals()

; --- Per-global wrapper for @head: the zero initializer makes this a
; trivial-init case, so the walk is delegated to buildTypeInitBody.
; The walker symbolizes field 0 (i32) inline and emits the recurse/null
; branching for field 1 (pointer) directly in the per-global body. For
; the recursive pointee allocated via malloc, it calls the cached
; per-type init @__klee_init_type_struct_Node. ---
; CHECK:       define internal void @__klee_init_g_head()
; CHECK:       call void @klee_make_symbolic(ptr @head,
; CHECK:       call ptr @malloc(
; CHECK:       call void @__klee_init_type_struct_Node(

; --- Per-type init for %struct.Node: must recursively call itself ---
; The presence of `call void @__klee_init_type_struct_Node(...)` inside
; @__klee_init_type_struct_Node's own body is the observable outcome of
; the codegen-time cycle breaker: the cache returns the in-progress
; Function* instead of re-entering body construction.
;
; The IR-builder creates recurse_bb before null_bb, so the `init.recurse`
; block appears first in function text even though the CFG branches to
; `init.null` on the too-deep path.
; CHECK:       define internal void @__klee_init_type_struct_Node(ptr %p, i32 %depth)
; CHECK:       call void @klee_make_symbolic(
; CHECK:       icmp sge i32 %depth
; CHECK:       init.recurse:
; CHECK:       call ptr @malloc(
; CHECK:       call void @__klee_init_type_struct_Node(
; CHECK:       init.null:
; CHECK:       store ptr null,
; CHECK:       init.cont:
; CHECK:       ret void

; --- No inline per-global klee_make_symbolic on @head in main() ---
; CHECK-NOT:   call void @klee_make_symbolic(ptr @head,
