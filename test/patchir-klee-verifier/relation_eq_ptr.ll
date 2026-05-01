; RUN: %patchir-klee-verifier %s --target-function check_eq_ptr -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: PK_RelEqArgConst on pointer — relation eq, value=0 (null check)

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

declare void @inner(ptr)

define void @check_eq_ptr(ptr %p) {
entry:
  call void @inner(ptr %p), !static_contract !0
  ret void
}

!0 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=eq, value=0}], postconditions=[]"}

; CHECK-LABEL: define void @check_eq_ptr(ptr %p)
; CHECK:       icmp eq ptr
; CHECK:       call void @klee_assume(
; CHECK:       call void @inner(
; CHECK-NOT:   call void @klee_abort(
