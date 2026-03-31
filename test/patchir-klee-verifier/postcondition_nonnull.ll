; RUN: %patchir-klee-verifier %s --target-function get_buffer -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: PK_Nonnull as postcondition on return value (ptr)

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

@buf = global [16 x i8] zeroinitializer

define ptr @get_buffer() {
entry:
  ret ptr @buf
}

define void @caller() {
entry:
  %r = call ptr @get_buffer(), !static_contract !0
  ret void
}

!0 = !{!"get_buffer", !"preconditions=[], postconditions=[{kind=nonnull, target=ReturnValue}]"}

; CHECK:       define i32 @main()

; No args — no precondition assumes
; CHECK-NOT:   call void @klee_assume(

; Call target
; CHECK:       call ptr @get_buffer()

; Postcondition: return value nonnull
; CHECK:       icmp ne ptr
; CHECK:       call void @klee_assert(

; CHECK:       ret i32 0
