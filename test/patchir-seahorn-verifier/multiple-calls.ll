; RUN: %patchir-seahorn-verifier -S %s -o %t.ll
; RUN: %file-check %s < %t.ll

; Test multiple function calls with different contracts in same function

target triple = "arm-unknown-linux-gnueabihf"
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

declare void @__VERIFIER_assume(i1)
declare void @__VERIFIER_assert(i1)

; =============================================================================
; Test 1: Sequential calls with different predicate types
; =============================================================================

; CHECK-LABEL: define i32 @sequential_different_predicates
define i32 @sequential_different_predicates(i32* %ptr, i32 %x, i32 %divisor) {
entry:
  ; First call: nonnull precondition
  ; CHECK: %[[PTR_NULL:.*]] = icmp ne ptr %ptr, null
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[PTR_NULL]])
  ; CHECK-NEXT: %[[R1:.*]] = call i32 @read_value(ptr %ptr)
  %r1 = call i32 @read_value(ptr %ptr), !static_contract !0

  ; Second call: range precondition
  ; CHECK-NEXT: %[[X_SEXT:.*]] = sext i32 %x to i64
  ; CHECK-NEXT: %[[X_GE:.*]] = icmp sge i64 %[[X_SEXT]], 0
  ; CHECK-NEXT: %[[X_LE:.*]] = icmp sle i64 %[[X_SEXT]], 100
  ; CHECK-NEXT: %[[X_RANGE:.*]] = and i1 %[[X_GE]], %[[X_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[X_RANGE]])
  ; CHECK-NEXT: %[[R2:.*]] = call i32 @square(i32 %x)
  %r2 = call i32 @square(i32 %x), !static_contract !1

  ; Third call: relation precondition (division by zero prevention)
  ; CHECK-NEXT: %[[DIV_CMP:.*]] = icmp ne i32 %divisor, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[DIV_CMP]])
  ; CHECK-NEXT: %[[R3:.*]] = call i32 @divide(i32 %[[R2]], i32 %divisor)
  %r3 = call i32 @divide(i32 %r2, i32 %divisor), !static_contract !2

  ; Combine results
  %sum = add i32 %r1, %r3
  ret i32 %sum
}

declare i32 @read_value(i32*)
declare i32 @square(i32)
declare i32 @divide(i32, i32)

!0 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}]"}
!1 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=0, max=100]}]"}
!2 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(1), relation=neq, value=0}]"}

; =============================================================================
; Test 2: Using result from one call as argument to another
; =============================================================================

; CHECK-LABEL: define i32 @chained_calls
define i32 @chained_calls(i32 %input) {
entry:
  ; First call: input must be positive
  ; CHECK: %[[IN_CMP:.*]] = icmp sgt i32 %input, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[IN_CMP]])
  ; CHECK-NEXT: %[[STEP1:.*]] = call i32 @normalize(i32 %input)
  %step1 = call i32 @normalize(i32 %input), !static_contract !3

  ; Postcondition on step1: result in [0, 255]
  ; CHECK-NEXT: %[[S1_SEXT:.*]] = sext i32 %[[STEP1]] to i64
  ; CHECK-NEXT: %[[S1_GE:.*]] = icmp sge i64 %[[S1_SEXT]], 0
  ; CHECK-NEXT: %[[S1_LE:.*]] = icmp sle i64 %[[S1_SEXT]], 255
  ; CHECK-NEXT: %[[S1_RANGE:.*]] = and i1 %[[S1_GE]], %[[S1_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assert(i1 %[[S1_RANGE]])

  ; Second call: use step1 result, must be in [0, 255]
  ; CHECK-NEXT: %[[S1_CHECK_SEXT:.*]] = sext i32 %[[STEP1]] to i64
  ; CHECK-NEXT: %[[S1_CHECK_GE:.*]] = icmp sge i64 %[[S1_CHECK_SEXT]], 0
  ; CHECK-NEXT: %[[S1_CHECK_LE:.*]] = icmp sle i64 %[[S1_CHECK_SEXT]], 255
  ; CHECK-NEXT: %[[S1_CHECK_RANGE:.*]] = and i1 %[[S1_CHECK_GE]], %[[S1_CHECK_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[S1_CHECK_RANGE]])
  ; CHECK-NEXT: %[[STEP2:.*]] = call i32 @process_byte(i32 %[[STEP1]])
  %step2 = call i32 @process_byte(i32 %step1), !static_contract !4

  ret i32 %step2
}

declare i32 @normalize(i32)
declare i32 @process_byte(i32)

!3 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=gt, value=0}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=255]}]"}
!4 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=0, max=255]}]"}

; =============================================================================
; Test 3: Multiple calls with same contract (repeated validation)
; =============================================================================

; CHECK-LABEL: define i32 @repeated_contracts
define i32 @repeated_contracts(i32 %a, i32 %b, i32 %c) {
entry:
  ; First call: a must be positive
  ; CHECK: %[[A_CMP:.*]] = icmp sgt i32 %a, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[A_CMP]])
  ; CHECK-NEXT: %[[R1:.*]] = call i32 @validate_positive(i32 %a)
  %r1 = call i32 @validate_positive(i32 %a), !static_contract !5

  ; Second call: b must be positive (same contract, different argument)
  ; CHECK-NEXT: %[[B_CMP:.*]] = icmp sgt i32 %b, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[B_CMP]])
  ; CHECK-NEXT: %[[R2:.*]] = call i32 @validate_positive(i32 %b)
  %r2 = call i32 @validate_positive(i32 %b), !static_contract !5

  ; Third call: c must be positive (same contract again)
  ; CHECK-NEXT: %[[C_CMP:.*]] = icmp sgt i32 %c, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[C_CMP]])
  ; CHECK-NEXT: %[[R3:.*]] = call i32 @validate_positive(i32 %c)
  %r3 = call i32 @validate_positive(i32 %c), !static_contract !5

  %sum1 = add i32 %r1, %r2
  %sum2 = add i32 %sum1, %r3
  ret i32 %sum2
}

declare i32 @validate_positive(i32)

!5 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=gt, value=0}]"}

; =============================================================================
; Test 4: Calls in different basic blocks (control flow)
; =============================================================================

; CHECK-LABEL: define i32 @calls_in_branches
define i32 @calls_in_branches(i1 %flag, i32* %ptr1, i32* %ptr2) {
entry:
  br i1 %flag, label %then_block, label %else_block

then_block:
  ; Call in then block: ptr1 must be nonnull
  ; CHECK: %[[PTR1_NULL:.*]] = icmp ne ptr %ptr1, null
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[PTR1_NULL]])
  ; CHECK-NEXT: %[[THEN_VAL:.*]] = call i32 @read_value(ptr %ptr1)
  %then_val = call i32 @read_value(ptr %ptr1), !static_contract !6
  br label %merge

else_block:
  ; Call in else block: ptr2 must be nonnull
  ; CHECK: %[[PTR2_NULL:.*]] = icmp ne ptr %ptr2, null
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[PTR2_NULL]])
  ; CHECK-NEXT: %[[ELSE_VAL:.*]] = call i32 @read_value(ptr %ptr2)
  %else_val = call i32 @read_value(ptr %ptr2), !static_contract !6
  br label %merge

merge:
  ; CHECK: %[[RESULT:.*]] = phi i32
  %result = phi i32 [ %then_val, %then_block ], [ %else_val, %else_block ]
  ret i32 %result
}

!6 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}]"}

; =============================================================================
; Test 5: Multiple calls with both pre and postconditions
; =============================================================================

; CHECK-LABEL: define i32 @multiple_pre_post_calls
define i32 @multiple_pre_post_calls(i32 %x, i32 %y) {
entry:
  ; First call with pre and post
  ; CHECK: %[[X_SEXT:.*]] = sext i32 %x to i64
  ; CHECK-NEXT: %[[X_GE:.*]] = icmp sge i64 %[[X_SEXT]], 0
  ; CHECK-NEXT: %[[X_LE:.*]] = icmp sle i64 %[[X_SEXT]], 100
  ; CHECK-NEXT: %[[X_RANGE:.*]] = and i1 %[[X_GE]], %[[X_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[X_RANGE]])
  ; CHECK-NEXT: %[[R1:.*]] = call i32 @compute_bounded(i32 %x)
  %r1 = call i32 @compute_bounded(i32 %x), !static_contract !7
  ; CHECK-NEXT: %[[R1_SEXT:.*]] = sext i32 %[[R1]] to i64
  ; CHECK-NEXT: %[[R1_GE:.*]] = icmp sge i64 %[[R1_SEXT]], 0
  ; CHECK-NEXT: %[[R1_LE:.*]] = icmp sle i64 %[[R1_SEXT]], 1000
  ; CHECK-NEXT: %[[R1_RANGE:.*]] = and i1 %[[R1_GE]], %[[R1_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assert(i1 %[[R1_RANGE]])

  ; Second call with pre and post
  ; CHECK-NEXT: %[[Y_SEXT:.*]] = sext i32 %y to i64
  ; CHECK-NEXT: %[[Y_GE:.*]] = icmp sge i64 %[[Y_SEXT]], 0
  ; CHECK-NEXT: %[[Y_LE:.*]] = icmp sle i64 %[[Y_SEXT]], 100
  ; CHECK-NEXT: %[[Y_RANGE:.*]] = and i1 %[[Y_GE]], %[[Y_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[Y_RANGE]])
  ; CHECK-NEXT: %[[R2:.*]] = call i32 @compute_bounded(i32 %y)
  %r2 = call i32 @compute_bounded(i32 %y), !static_contract !7
  ; CHECK-NEXT: %[[R2_SEXT:.*]] = sext i32 %[[R2]] to i64
  ; CHECK-NEXT: %[[R2_GE:.*]] = icmp sge i64 %[[R2_SEXT]], 0
  ; CHECK-NEXT: %[[R2_LE:.*]] = icmp sle i64 %[[R2_SEXT]], 1000
  ; CHECK-NEXT: %[[R2_RANGE:.*]] = and i1 %[[R2_GE]], %[[R2_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assert(i1 %[[R2_RANGE]])

  %sum = add i32 %r1, %r2
  ret i32 %sum
}

declare i32 @compute_bounded(i32)

!7 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=0, max=100]}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=1000]}]"}

; =============================================================================
; Test 6: Call with complex contract followed by simple call
; =============================================================================

; CHECK-LABEL: define i32 @complex_then_simple
define i32 @complex_then_simple(i32* %buffer, i32 %size) {
entry:
  ; Complex call: multiple preconditions
  ; CHECK: %[[BUF_NULL:.*]] = icmp ne ptr %buffer, null
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[BUF_NULL]])
  ; CHECK-NEXT: %[[BUF_INT:.*]] = ptrtoint ptr %buffer to i32
  ; CHECK-NEXT: %[[BUF_MOD:.*]] = urem i32 %[[BUF_INT]], 4
  ; CHECK-NEXT: %[[BUF_ALIGNED:.*]] = icmp eq i32 %[[BUF_MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[BUF_ALIGNED]])
  ; CHECK-NEXT: %[[SIZE_SEXT:.*]] = sext i32 %size to i64
  ; CHECK-NEXT: %[[SIZE_GE:.*]] = icmp sge i64 %[[SIZE_SEXT]], 1
  ; CHECK-NEXT: %[[SIZE_LE:.*]] = icmp sle i64 %[[SIZE_SEXT]], 1024
  ; CHECK-NEXT: %[[SIZE_RANGE:.*]] = and i1 %[[SIZE_GE]], %[[SIZE_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[SIZE_RANGE]])
  ; CHECK-NEXT: %[[R1:.*]] = call i32 @dma_transfer(ptr %buffer, i32 %size)
  %r1 = call i32 @dma_transfer(ptr %buffer, i32 %size), !static_contract !8

  ; Simple call: just check result is non-negative
  ; CHECK-NEXT: %[[R1_CMP:.*]] = icmp sge i32 %[[R1]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[R1_CMP]])
  ; CHECK-NEXT: %[[R2:.*]] = call i32 @process_result(i32 %[[R1]])
  %r2 = call i32 @process_result(i32 %r1), !static_contract !9

  ret i32 %r2
}

declare i32 @dma_transfer(i32*, i32)
declare i32 @process_result(i32)

!8 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}, {kind=alignment, target=Arg(0), align=4}, {kind=range, target=Arg(1), range=[min=1, max=1024]}]"}
!9 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=gte, value=0}]"}

; =============================================================================
; Test 7: Calls interleaved with other operations
; =============================================================================

; CHECK-LABEL: define i32 @interleaved_operations
define i32 @interleaved_operations(i32 %a, i32 %b) {
entry:
  ; First call
  ; CHECK: %[[A_CMP:.*]] = icmp sgt i32 %a, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[A_CMP]])
  ; CHECK-NEXT: %[[R1:.*]] = call i32 @increment(i32 %a)
  %r1 = call i32 @increment(i32 %a), !static_contract !10

  ; Some arithmetic
  ; CHECK-NEXT: %[[TEMP:.*]] = mul i32 %[[R1]], 2
  %temp = mul i32 %r1, 2

  ; Second call using temp result
  ; CHECK-NEXT: %[[TEMP_CMP:.*]] = icmp sgt i32 %[[TEMP]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[TEMP_CMP]])
  ; CHECK-NEXT: %[[R2:.*]] = call i32 @increment(i32 %[[TEMP]])
  %r2 = call i32 @increment(i32 %temp), !static_contract !10

  ; More arithmetic
  ; CHECK-NEXT: %[[TEMP2:.*]] = add i32 %[[R2]], %b
  %temp2 = add i32 %r2, %b

  ; Third call
  ; CHECK-NEXT: %[[B_CMP:.*]] = icmp ne i32 %b, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[B_CMP]])
  ; CHECK-NEXT: %[[R3:.*]] = call i32 @divide(i32 %[[TEMP2]], i32 %b)
  %r3 = call i32 @divide(i32 %temp2, i32 %b), !static_contract !11

  ret i32 %r3
}

declare i32 @increment(i32)

!10 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=gt, value=0}]"}
!11 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(1), relation=neq, value=0}]"}

; =============================================================================
; Test 8: Five sequential calls (stress test for multiple instrumentation)
; =============================================================================

; CHECK-LABEL: define i32 @five_sequential_calls
define i32 @five_sequential_calls(i32 %v1, i32 %v2, i32 %v3, i32 %v4, i32 %v5) {
entry:
  ; Call 1
  ; CHECK: %[[V1_SEXT:.*]] = sext i32 %v1 to i64
  ; CHECK-NEXT: %[[V1_GE:.*]] = icmp sge i64 %[[V1_SEXT]], 0
  ; CHECK-NEXT: %[[V1_LE:.*]] = icmp sle i64 %[[V1_SEXT]], 10
  ; CHECK-NEXT: %[[V1_RANGE:.*]] = and i1 %[[V1_GE]], %[[V1_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[V1_RANGE]])
  ; CHECK-NEXT: %[[R1:.*]] = call i32 @scale(i32 %v1)
  %result1 = call i32 @scale(i32 %v1), !static_contract !12

  ; Call 2
  ; CHECK-NEXT: %[[V2_SEXT:.*]] = sext i32 %v2 to i64
  ; CHECK-NEXT: %[[V2_GE:.*]] = icmp sge i64 %[[V2_SEXT]], 0
  ; CHECK-NEXT: %[[V2_LE:.*]] = icmp sle i64 %[[V2_SEXT]], 10
  ; CHECK-NEXT: %[[V2_RANGE:.*]] = and i1 %[[V2_GE]], %[[V2_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[V2_RANGE]])
  ; CHECK-NEXT: %[[R2:.*]] = call i32 @scale(i32 %v2)
  %result2 = call i32 @scale(i32 %v2), !static_contract !12

  ; Call 3
  ; CHECK-NEXT: %[[V3_SEXT:.*]] = sext i32 %v3 to i64
  ; CHECK-NEXT: %[[V3_GE:.*]] = icmp sge i64 %[[V3_SEXT]], 0
  ; CHECK-NEXT: %[[V3_LE:.*]] = icmp sle i64 %[[V3_SEXT]], 10
  ; CHECK-NEXT: %[[V3_RANGE:.*]] = and i1 %[[V3_GE]], %[[V3_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[V3_RANGE]])
  ; CHECK-NEXT: %[[R3:.*]] = call i32 @scale(i32 %v3)
  %result3 = call i32 @scale(i32 %v3), !static_contract !12

  ; Call 4
  ; CHECK-NEXT: %[[V4_SEXT:.*]] = sext i32 %v4 to i64
  ; CHECK-NEXT: %[[V4_GE:.*]] = icmp sge i64 %[[V4_SEXT]], 0
  ; CHECK-NEXT: %[[V4_LE:.*]] = icmp sle i64 %[[V4_SEXT]], 10
  ; CHECK-NEXT: %[[V4_RANGE:.*]] = and i1 %[[V4_GE]], %[[V4_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[V4_RANGE]])
  ; CHECK-NEXT: %[[R4:.*]] = call i32 @scale(i32 %v4)
  %result4 = call i32 @scale(i32 %v4), !static_contract !12

  ; Call 5
  ; CHECK-NEXT: %[[V5_SEXT:.*]] = sext i32 %v5 to i64
  ; CHECK-NEXT: %[[V5_GE:.*]] = icmp sge i64 %[[V5_SEXT]], 0
  ; CHECK-NEXT: %[[V5_LE:.*]] = icmp sle i64 %[[V5_SEXT]], 10
  ; CHECK-NEXT: %[[V5_RANGE:.*]] = and i1 %[[V5_GE]], %[[V5_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[V5_RANGE]])
  ; CHECK-NEXT: %[[R5:.*]] = call i32 @scale(i32 %v5)
  %result5 = call i32 @scale(i32 %v5), !static_contract !12

  ; Combine all results
  %sum1 = add i32 %result1, %result2
  %sum2 = add i32 %sum1, %result3
  %sum3 = add i32 %sum2, %result4
  %final = add i32 %sum3, %result5

  ret i32 %final
}

declare i32 @scale(i32)

!12 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=0, max=10]}]"}
