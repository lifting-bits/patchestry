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

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

declare void @hal_usb_write(ptr, ptr, i32)

define i32 @usbd_ep_write_packet(ptr %usb_device, ptr %buffer, i32 %buffer_length) {
entry:
  call void @hal_usb_write(ptr %usb_device, ptr %buffer, i32 %buffer_length)
  ret i32 %buffer_length
}

define void @test_caller() {
entry:
  %r = call i32 @usbd_ep_write_packet(ptr null, ptr null, i32 0), !static_contract !0
  ret void
}

!0 = !{!"usbd_ep_write_packet", !"preconditions=[{kind=relation, target=Arg(0), relation=neq, value=0}, {kind=relation, target=Arg(1), relation=neq, value=0}, {kind=range, target=Arg(2), range=[min=0, max=512]}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=512]}]"}

; --- External stub ---
; CHECK:       define void @hal_usb_write(ptr %0, ptr %1, i32 %2)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret void

; --- Target preserved ---
; CHECK:       define i32 @usbd_ep_write_packet(ptr %usb_device, ptr %buffer, i32 %buffer_length)

; --- Harness main() ---
; CHECK:       define i32 @main()

; Three symbolic args
; CHECK:       call void @klee_make_symbolic(
; CHECK:       call void @klee_make_symbolic(
; CHECK:       call void @klee_make_symbolic(

; Precondition 1: arg0 (pointer) != 0 — treated as nonnull
; CHECK:       icmp ne ptr
; CHECK:       call void @klee_assume(

; Precondition 2: arg1 (pointer) != 0 — treated as nonnull
; CHECK:       icmp ne ptr
; CHECK:       call void @klee_assume(

; Precondition 3: range on arg2 (integer) 0..512
; CHECK:       icmp sge i64
; CHECK:       icmp sle i64 %{{[0-9]+}}, 512
; CHECK:       call void @klee_assume(

; Call target
; CHECK:       call i32 @usbd_ep_write_packet(

; Postcondition: range on return 0..512
; CHECK:       icmp sge i64
; CHECK:       icmp sle i64 %{{[0-9]+}}, 512
; CHECK:       br i1 %{{[0-9]+}}, label %assert.cont, label %assert.fail
; CHECK:       call void @klee_abort()
; CHECK:       unreachable
; CHECK:       assert.cont:
; CHECK:       ret i32 0
