; RUN: %patchir-seahorn-verifier -S %s -o %t.ll
; RUN: %file-check %s < %t.ll

; Test correct handling of different integer widths with proper sign extension

target triple = "arm-unknown-linux-gnueabihf"

declare void @__VERIFIER_assume(i1)
declare void @__VERIFIER_assert(i1)

; =============================================================================
; Test 1: i8 argument with range predicate
; =============================================================================

; CHECK-LABEL: define i32 @i8_range
define i32 @i8_range(i8 %x) {
entry:
  ; CHECK: %[[SEXT:.*]] = sext i8 %x to i64
  ; CHECK-NEXT: %[[GE:.*]] = icmp sge i64 %[[SEXT]], -128
  ; CHECK-NEXT: %[[LE:.*]] = icmp sle i64 %[[SEXT]], 127
  ; CHECK-NEXT: %[[RANGE:.*]] = and i1 %[[GE]], %[[LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE]])
  %result = call i32 @process_i8(i8 %x), !static_contract !0
  ret i32 %result
}

declare i32 @process_i8(i8)

!0 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=-128, max=127]}]"}

; =============================================================================
; Test 2: i16 argument with relation predicate
; =============================================================================

; CHECK-LABEL: define i32 @i16_relation
define i32 @i16_relation(i16 %x) {
entry:
  ; CHECK: %[[CMP:.*]] = icmp sgt i16 %x, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[CMP]])
  %result = call i32 @process_i16(i16 %x), !static_contract !1
  ret i32 %result
}

declare i32 @process_i16(i16)

!1 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=gt, value=0}]"}

; =============================================================================
; Test 3: i32 argument with range predicate
; =============================================================================

; CHECK-LABEL: define i32 @i32_range
define i32 @i32_range(i32 %x) {
entry:
  ; CHECK: %[[SEXT:.*]] = sext i32 %x to i64
  ; CHECK-NEXT: %[[GE:.*]] = icmp sge i64 %[[SEXT]], 0
  ; CHECK-NEXT: %[[LE:.*]] = icmp sle i64 %[[SEXT]], 1000
  ; CHECK-NEXT: %[[RANGE:.*]] = and i1 %[[GE]], %[[LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE]])
  %result = call i32 @process_i32(i32 %x), !static_contract !2
  ret i32 %result
}

declare i32 @process_i32(i32)

!2 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=0, max=1000]}]"}

; =============================================================================
; Test 4: i64 argument (no extension needed)
; =============================================================================

; CHECK-LABEL: define i32 @i64_range
define i32 @i64_range(i64 %x) {
entry:
  ; CHECK-NOT: sext i64
  ; CHECK: %[[GE:.*]] = icmp sge i64 %x, -5000
  ; CHECK-NEXT: %[[LE:.*]] = icmp sle i64 %x, 5000
  ; CHECK-NEXT: %[[RANGE:.*]] = and i1 %[[GE]], %[[LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE]])
  %result = call i32 @process_i64(i64 %x), !static_contract !3
  ret i32 %result
}

declare i32 @process_i64(i64)

!3 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=-5000, max=5000]}]"}

; =============================================================================
; Test 5: Mixed width arguments
; =============================================================================

; CHECK-LABEL: define i32 @mixed_widths
define i32 @mixed_widths(i8 %a, i16 %b, i32 %c, i64 %d) {
entry:
  ; First call: i8 argument
  ; CHECK: %[[SEXT_A:.*]] = sext i8 %a to i64
  ; CHECK-NEXT: %[[GE_A:.*]] = icmp sge i64 %[[SEXT_A]], 0
  ; CHECK-NEXT: %[[LE_A:.*]] = icmp sle i64 %[[SEXT_A]], 10
  ; CHECK-NEXT: %[[RANGE_A:.*]] = and i1 %[[GE_A]], %[[LE_A]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE_A]])
  %r1 = call i32 @func_i8(i8 %a), !static_contract !4

  ; Second call: i16 argument
  ; CHECK: %[[SEXT_B:.*]] = sext i16 %b to i64
  ; CHECK-NEXT: %[[GE_B:.*]] = icmp sge i64 %[[SEXT_B]], 0
  ; CHECK-NEXT: %[[LE_B:.*]] = icmp sle i64 %[[SEXT_B]], 100
  ; CHECK-NEXT: %[[RANGE_B:.*]] = and i1 %[[GE_B]], %[[LE_B]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE_B]])
  %r2 = call i32 @func_i16(i16 %b), !static_contract !5

  ; Third call: i32 argument
  ; CHECK: %[[SEXT_C:.*]] = sext i32 %c to i64
  ; CHECK-NEXT: %[[GE_C:.*]] = icmp sge i64 %[[SEXT_C]], 0
  ; CHECK-NEXT: %[[LE_C:.*]] = icmp sle i64 %[[SEXT_C]], 1000
  ; CHECK-NEXT: %[[RANGE_C:.*]] = and i1 %[[GE_C]], %[[LE_C]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE_C]])
  %r3 = call i32 @func_i32(i32 %c), !static_contract !6

  ; Fourth call: i64 argument (no extension)
  ; CHECK: %[[GE_D:.*]] = icmp sge i64 %d, 0
  ; CHECK-NEXT: %[[LE_D:.*]] = icmp sle i64 %d, 10000
  ; CHECK-NEXT: %[[RANGE_D:.*]] = and i1 %[[GE_D]], %[[LE_D]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE_D]])
  %r4 = call i32 @func_i64(i64 %d), !static_contract !7

  %sum = add i32 %r1, %r2
  %sum2 = add i32 %sum, %r3
  %result = add i32 %sum2, %r4
  ret i32 %result
}

declare i32 @func_i8(i8)
declare i32 @func_i16(i16)
declare i32 @func_i32(i32)
declare i32 @func_i64(i64)

!4 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=0, max=10]}]"}
!5 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=0, max=100]}]"}
!6 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=0, max=1000]}]"}
!7 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=0, max=10000]}]"}

; =============================================================================
; Test 6: i1 (boolean) with relation
; =============================================================================

; CHECK-LABEL: define i32 @i1_relation
define i32 @i1_relation(i1 %flag) {
entry:
  ; CHECK: %[[CMP:.*]] = icmp eq i1 %flag, true
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[CMP]])
  %result = call i32 @process_bool(i1 %flag), !static_contract !8
  ret i32 %result
}

declare i32 @process_bool(i1)

!8 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=eq, value=1}]"}

; =============================================================================
; Test 7: Return value with smaller width (i8 return)
; =============================================================================

; CHECK-LABEL: define i8 @i8_return_range
define i8 @i8_return_range(i32 %x) {
entry:
  %result = call i8 @returns_i8(i32 %x), !static_contract !9
  ; CHECK: %[[RET_SEXT:.*]] = sext i8 %result to i64
  ; CHECK-NEXT: %[[RET_GE:.*]] = icmp sge i64 %[[RET_SEXT]], 0
  ; CHECK-NEXT: %[[RET_LE:.*]] = icmp sle i64 %[[RET_SEXT]], 127
  ; CHECK-NEXT: %[[RET_RANGE:.*]] = and i1 %[[RET_GE]], %[[RET_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assert(i1 %[[RET_RANGE]])
  ret i8 %result
}

declare i8 @returns_i8(i32)

!9 = !{!"static_contract", !"postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=127]}]"}
