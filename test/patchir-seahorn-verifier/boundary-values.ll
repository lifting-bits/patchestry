; RUN: %patchir-seahorn-verifier -S %s -o %t.ll
; RUN: %file-check %s < %t.ll

; Test boundary and extreme values for range and relation predicates

target triple = "arm-unknown-linux-gnueabihf"

declare void @__VERIFIER_assume(i1)
declare void @__VERIFIER_assert(i1)

; =============================================================================
; Test 1: Negative range values
; =============================================================================

; CHECK-LABEL: define i32 @negative_range
define i32 @negative_range(i32 %x) {
entry:
  ; CHECK: %[[SEXT:.*]] = sext i32 %x to i64
  ; CHECK-NEXT: %[[GE:.*]] = icmp sge i64 %[[SEXT]], -100
  ; CHECK-NEXT: %[[LE:.*]] = icmp sle i64 %[[SEXT]], -10
  ; CHECK-NEXT: %[[RANGE:.*]] = and i1 %[[GE]], %[[LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE]])
  %result = call i32 @process_negative(i32 %x), !static_contract !0
  ret i32 %result
}

declare i32 @process_negative(i32)

!0 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=-100, max=-10]}]"}

; =============================================================================
; Test 2: Zero boundaries (min=0, max=0 - single value range)
; =============================================================================

; CHECK-LABEL: define i32 @zero_only_range
define i32 @zero_only_range(i32 %x) {
entry:
  ; CHECK: %[[SEXT:.*]] = sext i32 %x to i64
  ; CHECK-NEXT: %[[GE:.*]] = icmp sge i64 %[[SEXT]], 0
  ; CHECK-NEXT: %[[LE:.*]] = icmp sle i64 %[[SEXT]], 0
  ; CHECK-NEXT: %[[RANGE:.*]] = and i1 %[[GE]], %[[LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE]])
  %result = call i32 @must_be_zero(i32 %x), !static_contract !1
  ret i32 %result
}

declare i32 @must_be_zero(i32)

!1 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=0, max=0]}]"}

; =============================================================================
; Test 3: Maximum int64 value (9223372036854775807)
; =============================================================================

; CHECK-LABEL: define i32 @max_int64_relation
define i32 @max_int64_relation(i64 %x) {
entry:
  ; CHECK: %[[CMP:.*]] = icmp eq i64 %x, 9223372036854775807
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[CMP]])
  %result = call i32 @process_max(i64 %x), !static_contract !2
  ret i32 %result
}

declare i32 @process_max(i64)

!2 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=eq, value=9223372036854775807}]"}

; =============================================================================
; Test 4: Minimum int64 value (-9223372036854775808)
; =============================================================================

; CHECK-LABEL: define i32 @min_int64_relation
define i32 @min_int64_relation(i64 %x) {
entry:
  ; CHECK: %[[CMP:.*]] = icmp sgt i64 %x, -9223372036854775808
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[CMP]])
  %result = call i32 @process_min(i64 %x), !static_contract !3
  ret i32 %result
}

declare i32 @process_min(i64)

!3 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=gt, value=-9223372036854775808}]"}

; =============================================================================
; Test 5: Range spanning from negative to positive
; =============================================================================

; CHECK-LABEL: define i32 @negative_to_positive_range
define i32 @negative_to_positive_range(i32 %x) {
entry:
  ; CHECK: %[[SEXT:.*]] = sext i32 %x to i64
  ; CHECK-NEXT: %[[GE:.*]] = icmp sge i64 %[[SEXT]], -50
  ; CHECK-NEXT: %[[LE:.*]] = icmp sle i64 %[[SEXT]], 50
  ; CHECK-NEXT: %[[RANGE:.*]] = and i1 %[[GE]], %[[LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE]])
  %result = call i32 @process_symmetric(i32 %x), !static_contract !4
  ret i32 %result
}

declare i32 @process_symmetric(i32)

!4 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=-50, max=50]}]"}

; =============================================================================
; Test 6: Large positive range
; =============================================================================

; CHECK-LABEL: define i32 @large_positive_range
define i32 @large_positive_range(i64 %x) {
entry:
  ; CHECK: %[[GE:.*]] = icmp sge i64 %x, 1000000000
  ; CHECK-NEXT: %[[LE:.*]] = icmp sle i64 %x, 9000000000000000000
  ; CHECK-NEXT: %[[RANGE:.*]] = and i1 %[[GE]], %[[LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE]])
  %result = call i32 @process_large(i64 %x), !static_contract !5
  ret i32 %result
}

declare i32 @process_large(i64)

!5 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=1000000000, max=9000000000000000000]}]"}

; =============================================================================
; Test 7: Negative one (common edge case)
; =============================================================================

; CHECK-LABEL: define i32 @not_negative_one
define i32 @not_negative_one(i32 %x) {
entry:
  ; CHECK: %[[CMP:.*]] = icmp ne i32 %x, -1
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[CMP]])
  %result = call i32 @process_not_minus_one(i32 %x), !static_contract !6
  ret i32 %result
}

declare i32 @process_not_minus_one(i32)

!6 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=neq, value=-1}]"}

; =============================================================================
; Test 8: Return value with negative range (postcondition)
; =============================================================================

; CHECK-LABEL: define i32 @negative_return_range
define i32 @negative_return_range(i32 %x) {
entry:
  %result = call i32 @returns_negative(i32 %x), !static_contract !7
  ; CHECK: %[[RET_SEXT:.*]] = sext i32 %result to i64
  ; CHECK-NEXT: %[[RET_GE:.*]] = icmp sge i64 %[[RET_SEXT]], -1000
  ; CHECK-NEXT: %[[RET_LE:.*]] = icmp sle i64 %[[RET_SEXT]], -1
  ; CHECK-NEXT: %[[RET_RANGE:.*]] = and i1 %[[RET_GE]], %[[RET_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assert(i1 %[[RET_RANGE]])
  ret i32 %result
}

declare i32 @returns_negative(i32)

!7 = !{!"static_contract", !"postconditions=[{kind=range, target=ReturnValue, range=[min=-1000, max=-1]}]"}
