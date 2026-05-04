; RUN: %patchir-klee-verifier %s --target-function get_buffer -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: PK_Nonnull as postcondition on the return value (ptr) of an inner
; call. The contract instrumentation lives at the inner call site inside the
; target body, not in main().

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

@buf = global [16 x i8] zeroinitializer

declare ptr @inner_get()

define ptr @get_buffer() {
entry:
  %r = call ptr @inner_get(), !static_contract !0
  ret ptr %r
}

!0 = !{!"static_contract", !"preconditions=[], postconditions=[{kind=nonnull, target=ReturnValue}]"}

; Block layout after splitBasicBlock + emitKleePredicate:
;   entry -> after.contract -> assert.fail -> assert.cont -> (br after.contract)
; CHECK-LABEL: define ptr @get_buffer()
; CHECK:       call ptr @inner_get()
; CHECK:       icmp ne ptr
; CHECK:       br i1 %{{[0-9]+}}, label %assert.cont, label %assert.fail
; CHECK:       after.contract:
; CHECK:       assert.fail:
; CHECK:       call void @klee_abort()
; CHECK:       unreachable
; CHECK:       assert.cont:
; CHECK:       br label %after.contract
