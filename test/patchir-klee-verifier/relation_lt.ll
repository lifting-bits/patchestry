; RUN: %patchir-klee-verifier %s --target-function check_lt -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: PK_RelLtArgConst — relation lt (arg0 < 100)

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

define void @check_lt(i32 %val) {
entry:
  ret void
}

define void @caller() {
entry:
  call void @check_lt(i32 0), !static_contract !0
  ret void
}

!0 = !{!"check_lt", !"preconditions=[{kind=relation, target=Arg(0), relation=lt, value=100}], postconditions=[]"}

; CHECK:       define i32 @main()
; CHECK:       call void @klee_make_symbolic(
; CHECK:       icmp slt i32 %{{[0-9]+}}, 100
; CHECK:       call void @klee_assume(
; CHECK:       call void @check_lt(
; CHECK-NOT:   call void @klee_abort(
; CHECK:       ret i32 0
