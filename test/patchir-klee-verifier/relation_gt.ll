; RUN: %patchir-klee-verifier %s --target-function check_gt -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: PK_RelGtArgConst — relation gt (arg0 > 0)

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

declare void @inner(i32)

define void @check_gt(i32 %val) {
entry:
  call void @inner(i32 %val), !static_contract !0
  ret void
}

!0 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=gt, value=0}], postconditions=[]"}

; CHECK-LABEL: define void @check_gt(i32 %val)
; CHECK:       icmp sgt i32 %{{[a-zA-Z0-9_]+}}, 0
; CHECK:       call void @klee_assume(
; CHECK:       call void @inner(
; CHECK-NOT:   call void @klee_abort(
