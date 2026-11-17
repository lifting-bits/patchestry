; RUN: %patchir-klee-verifier %s --target-function add_one -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: a function no input can make fail.  Under KLEE (CI) every path must
; end without an error.
; KLEE-EXPECT: no-errors

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

define i32 @add_one(i32 %x) {
entry:
  %small = icmp ult i32 %x, 100
  br i1 %small, label %low, label %high
low:
  %r = add i32 %x, 1
  ret i32 %r
high:
  ret i32 100
}

; CHECK: define i32 @main()
; CHECK: call i32 @add_one(
; CHECK-NOT: call void @klee_abort(
