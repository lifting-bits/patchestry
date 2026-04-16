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

; --- FileCheck: klee_abort declaration is emitted early during abort
; redirection (rewriteAbortCalls runs before harness codegen) ---
; HARNESS:       declare void @klee_abort()

; --- FileCheck: harness main() ---
; Globals are now initialized via an internally-linked dispatcher function
; (@__klee_init_globals) that main() calls once before argument symbolization
; and target invocation. See buildTypeInitBody in patchir-klee-verifier.
; HARNESS:       define i32 @main()
; HARNESS:       call void @__klee_init_globals()
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

; --- FileCheck: per-global wrapper and per-type init for @usb_g ---
; The wrapper is a trivial forwarder into the per-type init function;
; the per-type init hits the pointer-free fast path (i32 leaf) and emits
; a single klee_make_symbolic of the global's bytes.
; HARNESS:       define internal void @__klee_init_g_usb_g()
; HARNESS:       call void @__klee_init_type_i32(ptr @usb_g, i32 0)
; HARNESS:       define internal void @__klee_init_type_i32(ptr %p, i32 %depth)
; HARNESS:       call void @klee_make_symbolic(ptr %p, i64 4,

; --- FileCheck: the dispatcher is a counted loop over the descriptor table ---
; HARNESS:       define internal void @__klee_init_globals()
; HARNESS:       phi i64

; --- FileCheck: klee_assume declaration is emitted at the end during
; harness codegen ---
; HARNESS:       declare void @klee_assume(
