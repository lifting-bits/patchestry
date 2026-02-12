; RUN: %patchir-seahorn-verifier -S %s -o %t.ll
; RUN: %file-check %s < %t.ll

; Test range precondition and postcondition

target triple = "arm-unknown-linux-gnueabihf"

declare void @__VERIFIER_assume(i1)
declare void @__VERIFIER_assert(i1)

; CHECK-LABEL: define i32 @bounded_operation
define i32 @bounded_operation(i32 %x) {
entry:
  ; CHECK: %[[SEXT:.*]] = sext i32 %x to i64
  ; CHECK-NEXT: %[[GE:.*]] = icmp sge i64 %[[SEXT]], 0
  ; CHECK-NEXT: %[[LE:.*]] = icmp sle i64 %[[SEXT]], 100
  ; CHECK-NEXT: %[[RANGE:.*]] = and i1 %[[GE]], %[[LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[RANGE]])
  %result = call i32 @compute(i32 %x), !static_contract !0
  ; CHECK: %[[RET_SEXT:.*]] = sext i32 %result to i64
  ; CHECK-NEXT: %[[RET_GE:.*]] = icmp sge i64 %[[RET_SEXT]], 0
  ; CHECK-NEXT: %[[RET_LE:.*]] = icmp sle i64 %[[RET_SEXT]], 1000
  ; CHECK-NEXT: %[[RET_RANGE:.*]] = and i1 %[[RET_GE]], %[[RET_LE]]
  ; CHECK-NEXT: call void @__VERIFIER_assert(i1 %[[RET_RANGE]])
  ret i32 %result
}

declare i32 @compute(i32)

; Contract: input in [0,100], output in [0,1000]
!0 = !{!"static_contract", !"preconditions=[{kind=range, target=Arg(0), range=[min=0, max=100]}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=1000]}]"}
