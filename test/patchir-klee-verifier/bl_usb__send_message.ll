; RUN: %patchir-klee-verifier %s --target-function bl_usb__send_message -S -o %t.ll
; RUN: %file-check -check-prefix=HARNESS %s --input-file %t.ll

; Test: KLEE harness generation for bl_usb__send_message with static contracts.
; Modeled after patchir-transform bl_usb__send_message_before_patch pipeline output.
;
; Verifies:
;   1. External function (usbd_ep_write_packet) is stubbed with klee_make_symbolic
;   2. Target function body is preserved
;   3. main() harness is generated with:
;      - Global (usb_g) made symbolic
;      - Pointer argument allocated and made symbolic
;      - klee_assume for nonnull precondition on arg0
;      - Call to target function
;      - klee_abort for range postcondition on return value [0, 255]

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

; Global referenced by target function
@usb_g = global i32 0

; External function (should be stubbed)
declare i32 @usbd_ep_write_packet(ptr, ptr, i32)

; Target function with a call to an external
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

; --- FileCheck: external stub ---
; HARNESS:       define i32 @usbd_ep_write_packet(ptr %0, ptr %1, i32 %2)
; HARNESS-NEXT:  entry:
; HARNESS-NEXT:    %3 = alloca i32
; HARNESS-NEXT:    call void @klee_make_symbolic(ptr %3,
; HARNESS:         ret i32

; --- FileCheck: target function preserved ---
; HARNESS:       define i32 @bl_usb__send_message(ptr %msg)

; --- FileCheck: harness main() ---
; HARNESS:       define i32 @main()
; HARNESS:       call void @klee_make_symbolic(ptr @usb_g,
; HARNESS:       alloca [256 x i8]
; HARNESS:       call void @klee_make_symbolic(
; HARNESS:       icmp ne ptr
; HARNESS:       call void @klee_assume(
; HARNESS:       call i32 @bl_usb__send_message(
; HARNESS:       icmp sge i64 %{{[0-9]+}}, 0
; HARNESS:       icmp sle i64 %{{[0-9]+}}, 255
; HARNESS:       br i1 %{{[0-9]+}}, label %assert.cont, label %assert.fail
; HARNESS:       assert.fail:
; HARNESS:       call void @klee_abort()
; HARNESS:       unreachable
; HARNESS:       assert.cont:
; HARNESS:       ret i32 0

; --- FileCheck: declarations (after main) ---
; HARNESS:       declare void @klee_assume(
; HARNESS:       declare void @klee_abort()
