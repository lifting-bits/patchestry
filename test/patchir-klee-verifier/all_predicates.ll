; RUN: %patchir-klee-verifier %s --target-function bl_device__process_entry -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll

; Test: All predicate kinds — nonnull, alignment, relation (ne), range.
; Modeled after patchir-transform all_predicates.yaml contract on sprintf
; replacement inside bl_device__process_entry.

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

; External called by target (should be stubbed)
declare i32 @snprintf(ptr, i32, ptr)

define i32 @bl_device__process_entry(ptr %dest, i32 %dest_size, ptr %format) {
entry:
  %r = call i32 @snprintf(ptr %dest, i32 %dest_size, ptr %format)
  ret i32 %r
}

; Caller attaching the all-predicates static contract
define void @test_caller() {
entry:
  %r = call i32 @bl_device__process_entry(ptr null, i32 0, ptr null), !static_contract !0
  ret void
}

!0 = !{!"bl_device__process_entry", !"preconditions=[{kind=nonnull, target=Arg(0)}, {kind=alignment, target=Arg(0), align=4}, {kind=relation, target=Arg(1), relation=neq, value=0}], postconditions=[{kind=range, target=ReturnValue, range=[min=0, max=32]}]"}

; --- External stub ---
; CHECK:       define i32 @snprintf(ptr %0, i32 %1, ptr %2)
; CHECK:       call void @klee_make_symbolic(

; --- Target preserved ---
; CHECK:       define i32 @bl_device__process_entry(ptr %dest, i32 %dest_size, ptr %format)

; --- Harness main() ---
; CHECK:       define i32 @main()

; Symbolic args: 3 pointer + int + pointer
; CHECK:       call void @klee_make_symbolic(
; CHECK:       call void @klee_make_symbolic(
; CHECK:       call void @klee_make_symbolic(

; Precondition 1: nonnull on arg0 (icmp ne ptr)
; CHECK:       icmp ne ptr
; CHECK:       call void @klee_assume(

; Precondition 2: alignment on arg0 (ptrtoint + urem + icmp eq)
; CHECK:       ptrtoint ptr
; CHECK:       urem
; CHECK:       icmp eq
; CHECK:       call void @klee_assume(

; Precondition 3: relation neq on arg1 (icmp ne i32)
; CHECK:       icmp ne i32
; CHECK:       call void @klee_assume(

; Call target
; CHECK:       call i32 @bl_device__process_entry(

; Postcondition: range on return [0, 32]
; CHECK:       icmp sge i64
; CHECK:       icmp sle i64 %{{[0-9]+}}, 32
; CHECK:       call void @klee_assert(

; CHECK:       ret i32 0
