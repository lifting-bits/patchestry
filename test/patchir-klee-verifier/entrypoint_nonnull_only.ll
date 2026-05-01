; RUN: %patchir-klee-verifier %s --target-function bl_usb__send_message -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: Void target function with precondition-only contract on the inner
; @usbd_ep_write_packet call (no postconditions). Modeled after the
; patchir-transform entrypoint_contract.yaml — nonnull on the msg pointer.
;
; Verifies:
;   1. Precondition-only: klee_assume for nonnull on Arg(0)
;   2. Void-returning inner call: no postcondition assertion chain emitted
;   3. main() drives the target without contract emission of its own

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

declare void @usbd_ep_write_packet(ptr)

define void @bl_usb__send_message(ptr %msg) {
entry:
  call void @usbd_ep_write_packet(ptr %msg), !static_contract !0
  ret void
}

!0 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}], postconditions=[]"}

; --- External stub (void return — just ret void) ---
; CHECK:       define void @usbd_ep_write_packet(ptr %0)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret void

; --- Target body: nonnull assume before the inner call ---
; CHECK:       define void @bl_usb__send_message(ptr %msg)
; CHECK:       icmp ne ptr
; CHECK:       call void @klee_assume(
; CHECK:       call void @usbd_ep_write_packet(

; --- No postconditions => no klee_abort branch ---
; CHECK-NOT:   call void @klee_abort(

; --- Harness main() ---
; CHECK:       define i32 @main()
; CHECK:       call void @klee_make_symbolic(
; CHECK:       call void @bl_usb__send_message(
; CHECK:       ret i32 0
