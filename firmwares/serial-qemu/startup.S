.syntax unified
.cpu cortex-m3
.thumb

.global Reset_Handler
.global Default_Handler

.extern main
.extern _stack_top
.extern _sidata
.extern _sdata
.extern _edata
.extern _sbss
.extern _ebss

.section .vectors, "a", %progbits
.word _stack_top
.word Reset_Handler
.word Default_Handler
.word Default_Handler
.word Default_Handler
.word Default_Handler
.word Default_Handler
.word 0
.word 0
.word 0
.word 0
.word Default_Handler
.word Default_Handler
.word 0
.word Default_Handler
.word Default_Handler

.text
.thumb_func
Reset_Handler:
  ldr r0, =_sidata
  ldr r1, =_sdata
  ldr r2, =_edata

.Lcopy_data:
  cmp r1, r2
  bcs .Lzero_bss
  ldr r3, [r0], #4
  str r3, [r1], #4
  b .Lcopy_data

.Lzero_bss:
  ldr r1, =_sbss
  ldr r2, =_ebss
  movs r3, #0

.Lclear_bss:
  cmp r1, r2
  bcs .Lrun_main
  str r3, [r1], #4
  b .Lclear_bss

.Lrun_main:
  bl main

.Lhang:
  b .Lhang

.thumb_func
Default_Handler:
  b .Lhang
