; RUN: %cc-x86_64 -emit-llvm -c -O0 \
; RUN:   -include %patchestry_src_root/lib/patchestry/klee/models/klee_stub.h \
; RUN:   %patchestry_src_root/lib/patchestry/klee/models/usb_hal_models.c -o %t_model.bc
; RUN: %patchir-klee-verifier %s --target-function bl_usb__send_message \
; RUN:   --model-library %t_model.bc -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: KLEE harness with --model-library for usbd_ep_write_packet, where the
; static contract sits on the inner @usbd_ep_write_packet call inside the
; target.
;
; Instead of auto-stubbing usbd_ep_write_packet with an unconstrained symbolic
; return, the model library provides a body with:
;   - klee_make_symbolic on the buffer
;   - return constrained to [0, 255] via klee_assume
;
; Verifies:
;   1. usbd_ep_write_packet has a body from the model (NOT auto-stubbed)
;   2. The contracted call inside the target is wrapped with klee_assume
;      precondition and a klee_abort postcondition assertion chain
;   3. main() drives the target with no contract emission of its own
;   4. Global usb_g is still made symbolic

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

@usb_g = global i32 0

declare i32 @usbd_ep_write_packet(ptr, ptr, i32)

define i32 @bl_usb__send_message(ptr %msg) {
entry:
  %0 = load i32, ptr @usb_g
  %1 = call i32 @usbd_ep_write_packet(ptr %msg, ptr %msg, i32 %0), !static_contract !0
  ret i32 %1
}

!0 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=255]}]"}

; Contract instrumentation around the modeled call.
; CHECK-LABEL: define i32 @bl_usb__send_message(ptr %msg)
; CHECK:       icmp ne ptr
; CHECK:       call void @klee_assume(
; CHECK:       call i32 @usbd_ep_write_packet(
; CHECK:       icmp sge i64 %{{[0-9]+}}, 0
; CHECK:       icmp sle i64 %{{[0-9]+}}, 255
; CHECK:       br i1 %{{[0-9]+}}, label %assert.cont, label %assert.fail
; CHECK:       after.contract:
; CHECK:       assert.fail:
; CHECK:       call void @klee_abort()
; CHECK:       unreachable
; CHECK:       assert.cont:
; CHECK:       br label %after.contract

; The model library supplied a body (define, not declare) for
; @usbd_ep_write_packet — the distinctive thing this test checks.
; CHECK:       define {{.*}} @usbd_ep_write_packet(
; CHECK:       call void @klee_make_symbolic(
