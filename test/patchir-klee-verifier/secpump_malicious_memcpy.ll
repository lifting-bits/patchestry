; RUN: %patchir-klee-verifier %s --target-function memcpy_caller -S -o %t.ll
; RUN: %file-check -check-prefix=HARNESS %s --input-file %t.ll

; Test: KLEE harness generation for the SecPump-QEMU MaliciousMemCpy
; replacement. Mirrors the structure produced end-to-end by patchir-decomp +
; patchir-transform with secpump_klee_spec.yaml, but distilled by hand so the
; verifier unit-test does not depend on the lifter or the transform pass.
;
; Verifies:
;   1. External callee (MaliciousMemCpy) is stubbed with klee_make_symbolic.
;   2. The contracted call to patch__replace__MaliciousMemCpy is wrapped with
;      klee_assume on each precondition (dest non-null, src non-null).
;   3. main() symbolizes the caller's stack target before invoking the target.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@AttackBuffer = global [256 x i8] zeroinitializer

declare void @MaliciousMemCpy(ptr, ptr, i32)

define internal void @secpump_assert_fail() noreturn {
entry:
  br label %loop
loop:
  br label %loop
}

define internal void @patch__replace__MaliciousMemCpy(ptr %dest, ptr %src, i32 %n, i32 %dest_cap) {
entry:
  %ovf = icmp ugt i32 %n, %dest_cap
  br i1 %ovf, label %abort, label %copy
abort:
  call void @secpump_assert_fail()
  unreachable
copy:
  call void @MaliciousMemCpy(ptr %dest, ptr %src, i32 %n)
  ret void
}

define void @memcpy_caller(ptr %dest, ptr %src, i32 %n) {
entry:
  call void @patch__replace__MaliciousMemCpy(ptr %dest, ptr %src, i32 %n, i32 4), !static_contract !0
  ret void
}

!0 = !{!"static_contract", !"preconditions=[{id=\22dest_nonnull\22, kind=relation, target=Arg(0), relation=neq, value=0}; {id=\22src_nonnull\22, kind=relation, target=Arg(1), relation=neq, value=0}]"}

; HARNESS-LABEL: define void @memcpy_caller(ptr %dest, ptr %src, i32 %n)
; HARNESS:       icmp ne ptr %dest, null
; HARNESS:       call void @klee_assume(
; HARNESS:       icmp ne ptr %src, null
; HARNESS:       call void @klee_assume(
; HARNESS:       call void @patch__replace__MaliciousMemCpy(

; The verifier registers the symbolic-input helpers needed by the harness.
; HARNESS:       declare void @klee_make_symbolic(
; HARNESS:       declare void @klee_assume(

; main entry: KLEE driver that symbolizes the target's three pointer/size
; arguments and invokes memcpy_caller.
; HARNESS-LABEL: define i32 @main()
; HARNESS:       call void @klee_make_symbolic(
; HARNESS:       call void @memcpy_caller(
