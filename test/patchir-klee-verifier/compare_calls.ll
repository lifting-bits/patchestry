; RUN: %patchir-klee-verifier %s --compare-original %S/Inputs/compare_copy.ll.in --target-function copy --reject-return -1 --pure-callee strlen --pure-callee checksum --substitution strcpy:strncpy:2 --model strcpy --model strncpy --model strlen --model unused --keep-target --harness-info %t.json -S -o %t.ll
; RUN: %file-check -check-prefix=CHECK %s --input-file %t.ll
; RUN: %file-check -check-prefix=INFO %s --input-file %t.json
; RUN: not %patchir-klee-verifier %s --compare-original %S/Inputs/compare_copy.ll.in --target-function copy --pure-callee absent --keep-target -S -o %t.ll 2>&1 | %file-check -check-prefix=PURE %s
; RUN: not %patchir-klee-verifier %s --compare-original %S/Inputs/compare_copy.ll.in --target-function copy --substitution strcpy:memcpy_s:2 --keep-target -S -o %t.ll 2>&1 | %file-check -check-prefix=SWAP %s
; RUN: not %patchir-klee-verifier %s --compare-original %S/Inputs/compare_copy.ll.in --target-function copy --pure-callee strlen --substitution strlen:strncpy:1 --keep-target -S -o %t.ll 2>&1 | %file-check -check-prefix=BOTH %s
; RUN: not %patchir-klee-verifier %s --compare-original %S/Inputs/compare_copy.ll.in --target-function copy --substitution strcpy:strncpy --keep-target -S -o %t.ll 2>&1 | %file-check -check-prefix=FORM %s

; Test: declared relaxations of the call comparison.  The patched version
; bounds a copy: it calls strlen (pure), and strncpy where the original calls
; strcpy (a declared substitution, compared on the two leading arguments).

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-none-unknown-eabi"

declare i32 @strlen(ptr)
declare ptr @strncpy(ptr, ptr, i32)
declare i32 @checksum(ptr, i32)

define i32 @copy(ptr %to, ptr %from) {
entry:
  %n = call i32 @strlen(ptr %from)
  %bad = icmp uge i32 %n, 16
  br i1 %bad, label %reject, label %ok
reject:
  ret i32 -1
ok:
  %r = call ptr @strncpy(ptr %to, ptr %from, i32 16)
  %c = call i32 @checksum(ptr %to, i32 4)
  ret i32 %c
}

; The original's callee is recorded and run by its model.
; CHECK: define internal ptr @strcpy(ptr %0, ptr %1)
; CHECK: call void @__patchestry_trace_call(i32 [[#ID:]], i32 2)
; CHECK: call ptr @__patchestry_model_strcpy(ptr %0, ptr %1)

; A pure callee without a model: no record, and a value chosen by its
; integer arguments instead of by the call index.
; CHECK: define internal i32 @checksum(ptr %0, i32 %1)
; CHECK-NOT: __patchestry_trace_call
; CHECK-NOT: __patchestry_calls.checksum
; CHECK: urem i32 %{{.*}}, 8
; CHECK: getelementptr inbounds [8 x i32], ptr @__patchestry_values.checksum

; A pure callee with a model: no record.
; CHECK: define internal i32 @strlen(ptr %0)
; CHECK-NEXT: entry:
; CHECK-NEXT: call i32 @__patchestry_model_strlen(ptr %0)

; The substitute, called by the patched version, is recorded as the function
; it replaces with the two arguments they share; elsewhere as itself.
; CHECK: define internal ptr @strncpy(ptr %0, ptr %1, i32 %2)
; CHECK: as.patched:
; CHECK: call void @__patchestry_trace_call(i32 [[#ID]], i32 2)
; CHECK: as.original:
; CHECK: call void @__patchestry_trace_call(i32 [[#ID+1]], i32 3)
; CHECK: call ptr @__patchestry_model_strncpy(ptr %0, ptr %1, i32 %2)

; INFO-DAG: "substitutes": "strcpy"
; INFO-DAG: "pure": true
; INFO-DAG: "kind": "model"
; INFO-DAG: "compare_args": 2
; PURE: pure callee 'absent' is not called by either version
; SWAP: substitution 'strcpy' to 'memcpy_s' names a function neither version calls
; BOTH: substitution 'strlen' to 'strncpy' names a pure callee
; FORM: --substitution expects ORIGINAL:PATCHED:COUNT
