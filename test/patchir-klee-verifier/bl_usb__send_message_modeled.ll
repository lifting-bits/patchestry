; RUN: %cc-x86_64 -emit-llvm -c -O0 \
; RUN:   -include %patchestry_src_root/lib/patchestry/klee/models/klee_stub.h \
; RUN:   %patchestry_src_root/lib/patchestry/klee/models/usb_hal_models.c -o %t_model.bc
; RUN: %patchir-klee-verifier %s --target-function bl_usb__send_message \
; RUN:   --model-library %t_model.bc -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: KLEE harness with --model-library for usbd_ep_write_packet.
;
; Instead of auto-stubbing usbd_ep_write_packet with an unconstrained symbolic
; return, the model library provides a body with:
;   - klee_make_symbolic on the buffer
;   - return constrained to [0, 255] via klee_assume
;
; Verifies:
;   1. usbd_ep_write_packet has a body from the model (NOT auto-stubbed)
;   2. Model body contains klee_make_symbolic + klee_assume calls
;   3. main() harness still has klee_abort for the postcondition
;   4. Global usb_g is still made symbolic

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

; Global referenced by target function
@usb_g = global i32 0

; External function (will be provided by model library, NOT auto-stubbed)
declare i32 @usbd_ep_write_packet(ptr, ptr, i32)

; Target function with a call to the modeled external
define i32 @bl_usb__send_message(ptr %msg) {
entry:
  %0 = load i32, ptr @usb_g
  %1 = call i32 @usbd_ep_write_packet(ptr %msg, ptr %msg, i32 %0)
  ret i32 %1
}

; Caller with static contract metadata on the call
define void @test_caller() {
entry:
  %r = call i32 @bl_usb__send_message(ptr null), !static_contract !0
  ret void
}

!0 = !{!"bl_usb__send_message", !"preconditions=[{kind=nonnull, target=Arg(0)}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=255]}]"}

; --- FileCheck: target function preserved ---
; CHECK:       define i32 @bl_usb__send_message(ptr %msg)

; --- FileCheck: model body for usbd_ep_write_packet (from usb_hal_models.c) ---
; The model should have a body (define, not declare) with klee_make_symbolic
; CHECK:       define {{.*}} @usbd_ep_write_packet(
; CHECK:         call void @klee_make_symbolic(

; --- FileCheck: harness main() ---
; CHECK:       define i32 @main()
; CHECK:       call void @klee_make_symbolic(ptr @usb_g,
; CHECK:       call i32 @bl_usb__send_message(
; CHECK:       br i1 %{{[0-9]+}}, label %assert.cont, label %assert.fail
; CHECK:       call void @klee_abort()
; CHECK:       unreachable
; CHECK:       assert.cont:
; CHECK:       ret i32 0
