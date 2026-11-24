; RUN: %patchir-klee-verifier %s --target-function checked -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: a seeded harm for the KLEE gate.  One input reaches abort(); under
; KLEE (CI) the run must find it, and the gate fails if it does not or if the
; expectation below is changed to no-errors.
; KLEE-EXPECT: errors: abort

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

declare void @abort() noreturn

define i32 @checked(i32 %x) {
entry:
  %bad = icmp eq i32 %x, 7
  br i1 %bad, label %fail, label %ok
fail:
  call void @abort()
  unreachable
ok:
  ret i32 %x
}

; abort gets a body that ends the path in an error.
; CHECK: define void @abort()
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @klee_abort()
; CHECK: define i32 @checked(i32 %x)
; CHECK: call void @abort()
; CHECK: define i32 @main()
