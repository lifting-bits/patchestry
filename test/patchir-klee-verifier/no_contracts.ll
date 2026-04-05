; RUN: %patchir-klee-verifier %s --target-function measurement_update -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: Target function with no static contract metadata.
; Modeled after patchir-transform measurement_update — a function with
; external calls but no contract YAML applied.
;
; Verifies:
;   1. Harness is generated even without contracts
;   2. Globals made symbolic, args made symbolic
;   3. External functions stubbed
;   4. No klee_assume or klee_assert emitted

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

@sensor_state = global i32 0

declare float @spo2_lookup(float)

define void @measurement_update(float %reading) {
entry:
  %0 = load i32, ptr @sensor_state
  %1 = sitofp i32 %0 to float
  %2 = call float @spo2_lookup(float %1)
  store i32 1, ptr @sensor_state
  ret void
}

; --- External stub (float return — symbolic) ---
; CHECK:       define float @spo2_lookup(float %0)
; CHECK:       call void @klee_make_symbolic(
; CHECK:       ret float

; --- Target preserved ---
; CHECK:       define void @measurement_update(float %reading)

; --- Harness main() ---
; CHECK:       define i32 @main()

; Global made symbolic
; CHECK:       call void @klee_make_symbolic(ptr @sensor_state,

; Float arg made symbolic
; CHECK:       call void @klee_make_symbolic(

; No assumes or asserts (no contracts)
; CHECK-NOT:   call void @klee_assume(
; CHECK-NOT:   call void @klee_abort(

; Void call to target
; CHECK:       call void @measurement_update(

; CHECK:       ret i32 0
