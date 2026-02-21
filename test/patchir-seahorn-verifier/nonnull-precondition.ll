; RUN: %patchir-seahorn-verifier -S %s -o %t.ll
; RUN: %file-check %s < %t.ll

; Test basic nonnull precondition contract transformation

target triple = "arm-unknown-linux-gnueabihf"

; External verifier functions
declare void @__VERIFIER_assume(i1)
declare void @__VERIFIER_assert(i1)

; CHECK-LABEL: define i32 @safe_dereference
define i32 @safe_dereference(i32* %ptr) {
entry:
  ; CHECK: %[[NULL:.*]] = icmp ne ptr %ptr, null
  ; CHECK-NEXT: call void @__VERIFIER_assume(i1 %[[NULL]])
  %result = call i32 @process_data(i32* %ptr), !static_contract !0
  ret i32 %result
}

; Function with contract
declare i32 @process_data(i32*)

; Contract metadata: nonnull precondition on Arg(0)
!0 = !{!"static_contract", !"preconditions=[{kind=nonnull, target=Arg(0)}]"}
