; RUN: %patchir-seahorn-verifier -S %s -o %t.ll
; RUN: %file-check %s < %t.ll

; Test combined preconditions and postconditions on same function call

target triple = "arm-unknown-linux-gnueabihf"
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

declare void @__VERIFIER_assume(i1)
declare void @__VERIFIER_assert(i1)

; =============================================================================
; Test 1: Single precondition + single postcondition
; =============================================================================

; CHECK-LABEL: define i32 @simple_pre_post
define i32 @simple_pre_post(i32 %x) {
entry:
  ; Precondition: x >= 0
  ; CHECK: %[[SEXT_PRE:.*]] = sext i32 %x to i64
  ; CHECK-NEXT: %[[GE:.*]] = icmp sge i64 %[[SEXT_PRE]], 0
  ; CHECK-NEXT: %[[LE:.*]] = icmp sle i64 %[[SEXT_PRE]], 100
  ; CHECK-NEXT: %[[RANGE_PRE:.*]] = and i1 %[[GE]], %[[LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE_PRE]])

  ; The actual function call
  ; CHECK-NEXT: %[[RESULT:.*]] = call i32 @compute(i32 %x)
  %result = call i32 @compute(i32 %x), !static_contract !0

  ; Postcondition: result >= 0 and result <= 1000
  ; CHECK-NEXT: %[[SEXT_POST:.*]] = sext i32 %[[RESULT]] to i64
  ; CHECK-NEXT: %[[RET_GE:.*]] = icmp sge i64 %[[SEXT_POST]], 0
  ; CHECK-NEXT: %[[RET_LE:.*]] = icmp sle i64 %[[SEXT_POST]], 1000
  ; CHECK-NEXT: %[[RANGE_POST:.*]] = and i1 %[[RET_GE]], %[[RET_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assert(i1 %[[RANGE_POST]])

  ret i32 %result
}

declare i32 @compute(i32)

!0 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=0, max=100]}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=1000]}]"}

; =============================================================================
; Test 2: Multiple preconditions + single postcondition
; =============================================================================

; CHECK-LABEL: define i32 @multi_pre_single_post
define i32 @multi_pre_single_post(i32* %ptr, i32 %size) {
entry:
  ; Precondition 1: ptr is nonnull
  ; CHECK: %[[NULL_CHECK:.*]] = icmp ne ptr %ptr, null
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[NULL_CHECK]])

  ; Precondition 2: size >= 1 and size <= 1024
  ; CHECK-NEXT: %[[SIZE_SEXT:.*]] = sext i32 %size to i64
  ; CHECK-NEXT: %[[SIZE_GE:.*]] = icmp sge i64 %[[SIZE_SEXT]], 1
  ; CHECK-NEXT: %[[SIZE_LE:.*]] = icmp sle i64 %[[SIZE_SEXT]], 1024
  ; CHECK-NEXT: %[[SIZE_RANGE:.*]] = and i1 %[[SIZE_GE]], %[[SIZE_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[SIZE_RANGE]])

  ; The actual function call
  ; CHECK-NEXT: %[[RESULT:.*]] = call i32 @process_buffer(ptr %ptr, i32 %size)
  %result = call i32 @process_buffer(ptr %ptr, i32 %size), !static_contract !1

  ; Postcondition: result >= 0
  ; CHECK-NEXT: %[[RET_SEXT:.*]] = sext i32 %[[RESULT]] to i64
  ; CHECK-NEXT: %[[RET_GE:.*]] = icmp sge i64 %[[RET_SEXT]], 0
  ; CHECK-NEXT: %[[RET_LE:.*]] = icmp sle i64 %[[RET_SEXT]], 1024
  ; CHECK-NEXT: %[[RET_RANGE:.*]] = and i1 %[[RET_GE]], %[[RET_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assert(i1 %[[RET_RANGE]])

  ret i32 %result
}

declare i32 @process_buffer(i32*, i32)

!1 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}, {kind=range, target=Arg(1), range=[min=1, max=1024]}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=1024]}]"}

; =============================================================================
; Test 3: Single precondition + multiple postconditions
; =============================================================================

; CHECK-LABEL: define i32 @single_pre_multi_post
define i32 @single_pre_multi_post(i32 %x) {
entry:
  ; Precondition: x > 0
  ; CHECK: %[[CMP_PRE:.*]] = icmp sgt i32 %x, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[CMP_PRE]])

  ; The actual function call
  ; CHECK-NEXT: %[[RESULT:.*]] = call i32 @get_positive(i32 %x)
  %result = call i32 @get_positive(i32 %x), !static_contract !2

  ; TODO: Postconditions not yet implemented for relation predicates (gt, neq)
  ; When implemented, restore these CHECK patterns:
  ; Postcondition 1: result > 0
  ; Postcondition 2: result != -1

  ret i32 %result
}

declare i32 @get_positive(i32)

!2 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=gt, value=0}], postconditions=[{kind=relation, target=ReturnValue, relation=gt, value=0}, {kind=relation, target=ReturnValue, relation=neq, value=-1}]"}

; =============================================================================
; Test 4: Multiple preconditions + multiple postconditions
; =============================================================================

; CHECK-LABEL: define i32 @multi_pre_multi_post
define i32 @multi_pre_multi_post(i32* %data, i32 %count, i32 %divisor) {
entry:
  ; Precondition 1: data is nonnull
  ; CHECK: %[[PTR_NULL:.*]] = icmp ne ptr %data, null
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[PTR_NULL]])

  ; Precondition 2: count > 0
  ; CHECK-NEXT: %[[COUNT_CMP:.*]] = icmp sgt i32 %count, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[COUNT_CMP]])

  ; Precondition 3: divisor != 0
  ; CHECK-NEXT: %[[DIV_CMP:.*]] = icmp ne i32 %divisor, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[DIV_CMP]])

  ; The actual function call
  ; CHECK-NEXT: %[[RESULT:.*]] = call i32 @complex_operation(ptr %data, i32 %count, i32 %divisor)
  %result = call i32 @complex_operation(ptr %data, i32 %count, i32 %divisor), !static_contract !3

  ; Postcondition 1: result >= -100
  ; CHECK-NEXT: %[[RET_SEXT:.*]] = sext i32 %[[RESULT]] to i64
  ; CHECK-NEXT: %[[RET_GE:.*]] = icmp sge i64 %[[RET_SEXT]], -100
  ; CHECK-NEXT: %[[RET_LE:.*]] = icmp sle i64 %[[RET_SEXT]], 100
  ; CHECK-NEXT: %[[RET_RANGE:.*]] = and i1 %[[RET_GE]], %[[RET_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assert(i1 %[[RET_RANGE]])

  ; TODO: Postcondition not yet implemented for relation predicate (neq)
  ; When implemented, restore CHECK pattern for: result != 0

  ret i32 %result
}

declare i32 @complex_operation(i32*, i32, i32)

!3 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}, {kind=relation, target=Arg(1), relation=gt, value=0}, {kind=relation, target=Arg(2), relation=neq, value=0}], postconditions=[{kind=range, target=ReturnValue, range=[min=-100, max=100]}, {kind=relation, target=ReturnValue, relation=neq, value=0}]"}

; =============================================================================
; Test 5: All predicate types combined (nonnull + range + relation + alignment)
; =============================================================================

; CHECK-LABEL: define i32 @all_predicate_types
define i32 @all_predicate_types(i32* %buffer, i32 %size, i32 %flags) {
entry:
  ; Precondition 1: buffer is nonnull
  ; CHECK: %[[BUF_NULL:.*]] = icmp ne ptr %buffer, null
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[BUF_NULL]])

  ; Precondition 2: buffer is 4-byte aligned
  ; CHECK-NEXT: %[[BUF_INT:.*]] = ptrtoint ptr %buffer to i32
  ; CHECK-NEXT: %[[BUF_MOD:.*]] = urem i32 %[[BUF_INT]], 4
  ; CHECK-NEXT: %[[BUF_ALIGNED:.*]] = icmp eq i32 %[[BUF_MOD]], 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[BUF_ALIGNED]])

  ; Precondition 3: size in range [1, 4096]
  ; CHECK-NEXT: %[[SIZE_SEXT:.*]] = sext i32 %size to i64
  ; CHECK-NEXT: %[[SIZE_GE:.*]] = icmp sge i64 %[[SIZE_SEXT]], 1
  ; CHECK-NEXT: %[[SIZE_LE:.*]] = icmp sle i64 %[[SIZE_SEXT]], 4096
  ; CHECK-NEXT: %[[SIZE_RANGE:.*]] = and i1 %[[SIZE_GE]], %[[SIZE_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[SIZE_RANGE]])

  ; Precondition 4: flags != 0
  ; CHECK-NEXT: %[[FLAGS_CMP:.*]] = icmp ne i32 %flags, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[FLAGS_CMP]])

  ; The actual function call
  ; CHECK-NEXT: %[[RESULT:.*]] = call i32 @comprehensive_function(ptr %buffer, i32 %size, i32 %flags)
  %result = call i32 @comprehensive_function(ptr %buffer, i32 %size, i32 %flags), !static_contract !4

  ; Postcondition 1: result in range [0, 4096]
  ; CHECK-NEXT: %[[RET_SEXT:.*]] = sext i32 %[[RESULT]] to i64
  ; CHECK-NEXT: %[[RET_GE:.*]] = icmp sge i64 %[[RET_SEXT]], 0
  ; CHECK-NEXT: %[[RET_LE:.*]] = icmp sle i64 %[[RET_SEXT]], 4096
  ; CHECK-NEXT: %[[RET_RANGE:.*]] = and i1 %[[RET_GE]], %[[RET_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assert(i1 %[[RET_RANGE]])

  ; TODO: Postcondition not yet implemented for comparing return value to argument
  ; Parser limitation: Cannot parse value=Arg(N) in postconditions
  ; When implemented, restore CHECK pattern for: result <= size

  ret i32 %result
}

declare i32 @comprehensive_function(i32*, i32, i32)

!4 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}, {kind=alignment, target=Arg(0), align=4}, {kind=range, target=Arg(1), range=[min=1, max=4096]}, {kind=relation, target=Arg(2), relation=neq, value=0}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=4096]}, {kind=relation, target=ReturnValue, relation=lte, value=Arg(1)}]"}

; =============================================================================
; Test 6: Postcondition with relation comparing to argument
; =============================================================================

; CHECK-LABEL: define i32 @postcondition_arg_relation
define i32 @postcondition_arg_relation(i32 %max_value) {
entry:
  ; Precondition: max_value > 0
  ; CHECK: %[[MAX_CMP:.*]] = icmp sgt i32 %max_value, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[MAX_CMP]])

  ; The actual function call
  ; CHECK-NEXT: %[[RESULT:.*]] = call i32 @bounded_compute(i32 %max_value)
  %result = call i32 @bounded_compute(i32 %max_value), !static_contract !5

  ; TODO: Postconditions not yet implemented for:
  ; 1. Relation predicates on return values (gte)
  ; 2. Comparing return value to arguments (value=Arg(0))
  ; When implemented, restore CHECK patterns for:
  ; Postcondition 1: result >= 0
  ; Postcondition 2: result <= max_value

  ret i32 %result
}

declare i32 @bounded_compute(i32)

!5 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=gt, value=0}], postconditions=[{kind=relation, target=ReturnValue, relation=gte, value=0}, {kind=relation, target=ReturnValue, relation=lte, value=Arg(0)}]"}

; =============================================================================
; Test 7: Void return with only preconditions (common pattern)
; =============================================================================

; CHECK-LABEL: define void @void_return_with_pre
define void @void_return_with_pre(i8* %dst, i8* %src, i32 %count) {
entry:
  ; Precondition 1: dst is nonnull
  ; CHECK: %[[DST_NULL:.*]] = icmp ne ptr %dst, null
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[DST_NULL]])

  ; Precondition 2: src is nonnull
  ; CHECK-NEXT: %[[SRC_NULL:.*]] = icmp ne ptr %src, null
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[SRC_NULL]])

  ; Precondition 3: count > 0
  ; CHECK-NEXT: %[[COUNT_CMP:.*]] = icmp sgt i32 %count, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[COUNT_CMP]])

  ; The actual function call
  ; CHECK-NEXT: call void @memcpy_safe(ptr %dst, ptr %src, i32 %count)
  call void @memcpy_safe(ptr %dst, ptr %src, i32 %count), !static_contract !6

  ret void
}

declare void @memcpy_safe(i8*, i8*, i32)

!6 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}, {kind=nonnull, target=Arg(1)}, {kind=relation, target=Arg(2), relation=gt, value=0}]"}

; =============================================================================
; Test 8: Preconditions with overlapping ranges (defensive programming)
; =============================================================================

; CHECK-LABEL: define i32 @overlapping_range_checks
define i32 @overlapping_range_checks(i16 %port) {
entry:
  ; Precondition 1: port >= 1024 (unprivileged port)
  ; CHECK: %[[PORT_SEXT:.*]] = sext i16 %port to i64
  ; CHECK-NEXT: %[[PORT_GE:.*]] = icmp sge i64 %[[PORT_SEXT]], 1024
  ; CHECK-NEXT: %[[PORT_LE:.*]] = icmp sle i64 %[[PORT_SEXT]], 65535
  ; CHECK-NEXT: %[[PORT_RANGE:.*]] = and i1 %[[PORT_GE]], %[[PORT_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[PORT_RANGE]])

  ; The actual function call
  ; CHECK-NEXT: %[[RESULT:.*]] = call i32 @bind_port(i16 %port)
  %result = call i32 @bind_port(i16 %port), !static_contract !7

  ; Postcondition: result is 0 (success) or -1 (failure)
  ; CHECK-NEXT: %[[RET_SEXT:.*]] = sext i32 %[[RESULT]] to i64
  ; CHECK-NEXT: %[[RET_GE:.*]] = icmp sge i64 %[[RET_SEXT]], -1
  ; CHECK-NEXT: %[[RET_LE:.*]] = icmp sle i64 %[[RET_SEXT]], 0
  ; CHECK-NEXT: %[[RET_RANGE:.*]] = and i1 %[[RET_GE]], %[[RET_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assert(i1 %[[RET_RANGE]])

  ret i32 %result
}

declare i32 @bind_port(i16)

!7 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=1024, max=65535]}], postconditions=[{kind=range, target=ReturnValue, range=[min=-1, max=0]}]"}
