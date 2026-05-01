; RUN: %patchir-klee-verifier %s --target-function check_lte -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: PK_RelLeArgConst — relation lte (arg0 <= 50)

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

declare void @inner(i32)

define void @check_lte(i32 %val) {
entry:
  call void @inner(i32 %val), !static_contract !0
  ret void
}

!0 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=lte, value=50}], postconditions=[]"}

; --- Target body: sle assume before the inner call ---
; CHECK:       define void @check_lte(i32 %val)
; CHECK:       icmp sle i32 %{{[a-zA-Z0-9_]+}}, 50
; CHECK:       call void @klee_assume(
; CHECK:       call void @inner(

; --- No postconditions ---
; CHECK-NOT:   call void @klee_abort(

; --- Harness main() ---
; CHECK:       define i32 @main()
; CHECK:       call void @klee_make_symbolic(
; CHECK:       call void @check_lte(
; CHECK:       ret i32 0
