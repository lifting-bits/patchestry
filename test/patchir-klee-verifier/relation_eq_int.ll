; RUN: %patchir-klee-verifier %s --target-function contract__entrypoint__message_entry_check -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: Relation eq predicate on integer argument.
; Modeled after patchir-transform usb_security_patches.yaml
; test_entrypoint_insertion_contract_static: precondition arg0 == 1.
;
; Verifies:
;   1. PK_RelEqArgConst emits icmp eq on integer arg
;   2. klee_assume is generated for the eq precondition
;   3. No postconditions (expr kind is unsupported, skipped)

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

define void @contract__entrypoint__message_entry_check(i32 %flag) {
entry:
  ret void
}

define void @test_caller() {
entry:
  call void @contract__entrypoint__message_entry_check(i32 0), !static_contract !0
  ret void
}

!0 = !{!"contract__entrypoint__message_entry_check", !"preconditions=[{kind=relation, target=Arg(0), relation=eq, value=1}], postconditions=[]"}

; --- Target preserved ---
; CHECK:       define void @contract__entrypoint__message_entry_check(i32 %flag)

; --- Harness main() ---
; CHECK:       define i32 @main()

; Symbolic arg0 (integer)
; CHECK:       call void @klee_make_symbolic(

; Precondition: arg0 == 1
; CHECK:       icmp eq i32 %{{[0-9]+}}, 1
; CHECK:       call void @klee_assume(

; Void call to target
; CHECK:       call void @contract__entrypoint__message_entry_check(

; No postconditions
; CHECK-NOT:   call void @klee_abort(

; CHECK:       ret i32 0
