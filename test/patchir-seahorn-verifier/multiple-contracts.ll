; RUN: %patchir-seahorn-verifier -S %s -o %t.ll
; RUN: %file-check %s < %t.ll

; Test multiple contracts in a single function

target triple = "arm-unknown-linux-gnueabihf"

declare void @__VERIFIER_assume(i1)
declare void @__VERIFIER_assert(i1)

; CHECK-LABEL: define i32 @complex_function
define i32 @complex_function(i32* %buffer, i32 %size, i32 %flags) {
entry:
  ; First call: nonnull buffer, positive size
  ; CHECK: %[[NULL1:.*]] = icmp ne ptr %buffer, null
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[NULL1]])
  ; CHECK: %[[GT1:.*]] = icmp sgt i32 %size, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[GT1]])
  %validated = call i32 @validate_input(i32* %buffer, i32 %size), !static_contract !0

  ; Second call: flags must be non-zero
  ; CHECK: %[[NEQ:.*]] = icmp ne i32 %flags, 0
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[NEQ]])
  %processed = call i32 @process_flags(i32 %flags), !static_contract !1

  %result = add i32 %validated, %processed
  ret i32 %result
}

declare i32 @validate_input(i32*, i32)
declare i32 @process_flags(i32)

; Multiple preconditions in one contract
!0 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}, {kind=relation, target=Arg(1), relation=gt, value=0}]"}

; Single precondition
!1 = !{!"static_contract", !"preconditions=[{kind=relation, target=Arg(0), relation=neq, value=0}]"}
