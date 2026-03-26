; RUN: %patchir-klee-verifier %s --target-function bl_usb__send_message -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: Void target function with precondition-only contract (no postconditions).
; Modeled after patchir-transform entrypoint_contract.yaml — nonnull on msg pointer
; at function entrypoint, no postconditions.
;
; Verifies:
;   1. Void-returning target: no klee_assert emitted
;   2. Precondition-only: klee_assume for nonnull on arg0
;   3. No return value captured from void call

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

declare void @usbd_ep_write_packet(ptr)

define void @bl_usb__send_message(ptr %msg) {
entry:
  call void @usbd_ep_write_packet(ptr %msg)
  ret void
}

define void @test_caller() {
entry:
  call void @bl_usb__send_message(ptr null), !static_contract !0
  ret void
}

!0 = !{!"bl_usb__send_message", !"preconditions=[{kind=nonnull, target=Arg(0)}], postconditions=[]"}

; --- External stub (void return — just ret void) ---
; CHECK:       define void @usbd_ep_write_packet(ptr %0)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret void

; --- Target preserved ---
; CHECK:       define void @bl_usb__send_message(ptr %msg)

; --- Harness main() ---
; CHECK:       define i32 @main()
; CHECK:       call void @klee_make_symbolic(
; CHECK:       icmp ne ptr
; CHECK:       call void @klee_assume(

; Void call — no return value captured
; CHECK:       call void @bl_usb__send_message(

; No klee_assert should appear (no postconditions)
; CHECK-NOT:   call void @klee_assert(

; CHECK:       ret i32 0
