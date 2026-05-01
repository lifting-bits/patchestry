; RUN: %patchir-klee-verifier %s --target-function bl_device__process_entry -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: All predicate kinds — nonnull, alignment, relation (ne), range.
; Modeled after patchir-transform all_predicates.yaml: contract attached to a
; @snprintf call inside bl_device__process_entry. Verifies that all predicate
; kinds emit the expected ICmp patterns at the contract site.

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

declare i32 @snprintf(ptr, i32, ptr)

define i32 @bl_device__process_entry(ptr %dest, i32 %dest_size, ptr %format) {
entry:
  %r = call i32 @snprintf(ptr %dest, i32 %dest_size, ptr %format), !static_contract !0
  ret i32 %r
}

!0 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}, {kind=alignment, target=Arg(0), align=4}, {kind=relation, target=Arg(1), relation=neq, value=0}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=32]}]"}

; --- External stub for @snprintf (emitted first; stubExternalFunctions
; gives the declaration a body before harness codegen) ---
; CHECK:       define i32 @snprintf(ptr %0, i32 %1, ptr %2)
; CHECK:       call void @klee_make_symbolic(

; --- Target body: precondition assumes before the @snprintf call ---
; CHECK:       define i32 @bl_device__process_entry(ptr %dest, i32 %dest_size, ptr %format)

; Precondition 1: nonnull on Arg(0) (icmp ne ptr)
; CHECK:       icmp ne ptr
; CHECK:       call void @klee_assume(

; Precondition 2: alignment on Arg(0) (ptrtoint + urem + icmp eq)
; CHECK:       ptrtoint ptr
; CHECK:       urem
; CHECK:       icmp eq
; CHECK:       call void @klee_assume(

; Precondition 3: relation neq on Arg(1) (icmp ne i32)
; CHECK:       icmp ne i32
; CHECK:       call void @klee_assume(

; The contracted call
; CHECK:       call i32 @snprintf(

; Postcondition: range on ReturnValue [0, 32].
; Block layout: entry -> after.contract -> assert.fail -> assert.cont.
; CHECK:       icmp sge i64
; CHECK:       icmp sle i64 %{{[0-9]+}}, 32
; CHECK:       br i1 %{{[0-9]+}}, label %assert.cont, label %assert.fail
; CHECK:       after.contract:
; CHECK:       ret i32
; CHECK:       assert.fail:
; CHECK:       call void @klee_abort()
; CHECK:       unreachable
; CHECK:       assert.cont:
; CHECK:       br label %after.contract

; --- Harness main() ---
; CHECK:       define i32 @main()
; CHECK:       call void @klee_make_symbolic(
; CHECK:       call i32 @bl_device__process_entry(
; CHECK:       ret i32 0
