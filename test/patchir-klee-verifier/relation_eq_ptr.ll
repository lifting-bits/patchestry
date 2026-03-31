; RUN: %patchir-klee-verifier %s --target-function check_eq_ptr -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: PK_RelEqArgConst on pointer — relation eq, value=0 (null check)

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

define void @check_eq_ptr(ptr %p) {
entry:
  ret void
}

define void @caller() {
entry:
  call void @check_eq_ptr(ptr null), !static_contract !0
  ret void
}

!0 = !{!"check_eq_ptr", !"preconditions=[{kind=relation, target=Arg(0), relation=eq, value=0}], postconditions=[]"}

; CHECK:       define i32 @main()
; CHECK:       call void @klee_make_symbolic(
; CHECK:       icmp eq ptr
; CHECK:       call void @klee_assume(
; CHECK:       call void @check_eq_ptr(
; CHECK-NOT:   call void @klee_assert(
; CHECK:       ret i32 0
