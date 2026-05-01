; RUN: %patchir-klee-verifier %s --target-function bl_usb__send_message -S -o %t.ll
; RUN: %file-check -check-prefix=HARNESS %s --input-file %t.ll

; Test: KLEE harness generation for bl_usb__send_message with a static
; contract attached to the inner @usbd_ep_write_packet call.
;
; Verifies:
;   1. External function (usbd_ep_write_packet) is stubbed with klee_make_symbolic
;   2. Target body is preserved with klee_assume/klee_assert wrapping the
;      contracted call (nonnull on Arg(0); range on ReturnValue [0, 255])
;   3. main() harness symbolizes the global and the target's arg, then calls
;      the target — main does not emit contract predicates of its own.

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

; --- External stub for @usbd_ep_write_packet ---
; HARNESS:       define i32 @usbd_ep_write_packet(ptr %0, ptr %1, i32 %2)
; HARNESS-NEXT:  entry:
; HARNESS-NEXT:    %3 = alloca i32
; HARNESS-NEXT:    call void @klee_make_symbolic(ptr %3,
; HARNESS:         ret i32

; --- Target body: precondition assume + postcondition assertion around the
; contracted call. Block layout: entry -> after.contract -> assert.fail ->
; assert.cont. ---
; HARNESS:       define i32 @bl_usb__send_message(ptr %msg)
; HARNESS:       icmp ne ptr
; HARNESS:       call void @klee_assume(
; HARNESS:       call i32 @usbd_ep_write_packet(
; HARNESS:       icmp sge i64 %{{[0-9]+}}, 0
; HARNESS:       icmp sle i64 %{{[0-9]+}}, 255
; HARNESS:       br i1 %{{[0-9]+}}, label %assert.cont, label %assert.fail
; HARNESS:       after.contract:
; HARNESS:       ret i32
; HARNESS:       assert.fail:
; HARNESS:       call void @klee_abort()
; HARNESS:       unreachable
; HARNESS:       assert.cont:
; HARNESS:       br label %after.contract

; --- klee_abort declaration emitted early during abort redirection ---
; HARNESS:       declare void @klee_abort()

; --- Harness main(): globals init dispatcher, symbolize msg, call target ---
; HARNESS:       define i32 @main()
; HARNESS:       call void @__klee_init_globals()
; HARNESS:       alloca [256 x i8]
; HARNESS:       call void @klee_make_symbolic(
; HARNESS:       call i32 @bl_usb__send_message(
; HARNESS:       ret i32 0

; --- Per-global wrapper for @usb_g (scalar i32 → trivial-init fast path) ---
; HARNESS:       define internal void @__klee_init_g_usb_g()
; HARNESS:       call void @klee_make_symbolic(ptr @usb_g, i64 4,

; --- Dispatcher is a counted loop over the descriptor table ---
; HARNESS:       define internal void @__klee_init_globals()
; HARNESS:       phi i64
