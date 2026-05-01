; RUN: %patchir-klee-verifier %s --target-function usbd_ep_write_packet -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: Multiple relation preconditions + range on arg + range on return.
; Modeled after patchir-transform usb_security_patches.yaml
; usb_endpoint_write_validation_contract_static.
;
; Verifies:
;   1. Two relation-ne preconditions (arg0 != 0, arg1 != 0)
;   2. Range precondition on arg2 (0..512) — exercises PK_RangeArg
;   3. Range postcondition on return (0..512)
;   4. Mixed arg types: two pointers + one integer
;
; Static contracts are an operation-level property: the metadata sits on the
; matched call op (here `@hal_usb_write` inside the target's body), and the
; verifier instruments klee_assume/klee_assert pairs at that call site,
; resolving Arg(N) against the call's operand and ReturnValue against its SSA
; result.

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

declare i32 @hal_usb_write(ptr, ptr, i32)

define i32 @usbd_ep_write_packet(ptr %usb_device, ptr %buffer, i32 %buffer_length) {
entry:
  %r = call i32 @hal_usb_write(ptr %usb_device, ptr %buffer, i32 %buffer_length), !static_contract !0
  ret i32 %r
}

!0 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=neq, value=0}, {kind=relation, target=Arg(1), relation=neq, value=0}, {kind=range, target=Arg(2), range=[min=0, max=512]}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=512]}]"}

; CHECK-LABEL: define i32 @usbd_ep_write_packet(ptr %usb_device, ptr %buffer, i32 %buffer_length)
; Precondition 1 — Arg(0) (ptr) != 0
; CHECK:       icmp ne ptr
; CHECK:       call void @klee_assume(
; Precondition 2 — Arg(1) (ptr) != 0
; CHECK:       icmp ne ptr
; CHECK:       call void @klee_assume(
; Precondition 3 — range on Arg(2) [0, 512] (PK_RangeArg)
; CHECK:       icmp sge i64
; CHECK:       icmp sle i64 %{{[0-9]+}}, 512
; CHECK:       call void @klee_assume(
; The contracted call
; CHECK:       call i32 @hal_usb_write(
; Postcondition — range on ReturnValue [0, 512]; layout
; entry -> after.contract -> assert.fail -> assert.cont
; CHECK:       icmp sge i64
; CHECK:       icmp sle i64 %{{[0-9]+}}, 512
; CHECK:       br i1 %{{[0-9]+}}, label %assert.cont, label %assert.fail
; CHECK:       after.contract:
; CHECK:       assert.fail:
; CHECK:       call void @klee_abort()
; CHECK:       unreachable
; CHECK:       assert.cont:
; CHECK:       br label %after.contract
