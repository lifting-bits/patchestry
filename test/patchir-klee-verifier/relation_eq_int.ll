; RUN: %patchir-klee-verifier %s --target-function contract__entrypoint__message_entry_check -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: Relation eq predicate on integer argument.
; Modeled after patchir-transform usb_security_patches.yaml
; test_entrypoint_insertion_contract_static: precondition arg0 == 1.
;
; Verifies:
;   1. PK_RelEqArgConst emits icmp eq on integer arg
;   2. klee_assume is generated for the eq precondition before the inner call
;   3. No postconditions => no assertion chain emitted

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

declare void @inner(i32)

define void @contract__entrypoint__message_entry_check(i32 %flag) {
entry:
  call void @inner(i32 %flag), !static_contract !0
  ret void
}

!0 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=eq, value=1}], postconditions=[]"}

; CHECK-LABEL: define void @contract__entrypoint__message_entry_check(i32 %flag)
; CHECK:       icmp eq i32 %{{[a-zA-Z0-9_]+}}, 1
; CHECK:       call void @klee_assume(
; CHECK:       call void @inner(
; CHECK-NOT:   call void @klee_abort(
