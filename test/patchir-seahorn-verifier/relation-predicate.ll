; RUN: %patchir-seahorn-verifier -S %s -o %t.ll
; RUN: %file-check %s < %t.ll

; Test relation predicates (eq, neq, lt, lte, gt, gte)

target triple = "arm-unknown-linux-gnueabihf"

declare void @__VERIFIER_assume(i1)
declare void @__VERIFIER_assert(i1)

; CHECK-LABEL: define i32 @safe_divide
define i32 @safe_divide(i32 %a, i32 %b) {
entry:
  ; CHECK: %[[NEQ:.*]] = icmp ne i32 %b, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[NEQ]])
  %result = call i32 @divide(i32 %a, i32 %b), !static_contract !0
  ret i32 %result
}

; CHECK-LABEL: define i32 @positive_only
define i32 @positive_only(i32 %x) {
entry:
  ; CHECK: %[[GT:.*]] = icmp sgt i32 %x, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[GT]])
  %result = call i32 @process_positive(i32 %x), !static_contract !1
  ret i32 %result
}

declare i32 @divide(i32, i32)
declare i32 @process_positive(i32)

; Contract: divisor must be non-zero
!0 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(1), relation=neq, value=0}]"}

; Contract: input must be positive
!1 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=gt, value=0}]"}
