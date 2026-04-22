.section .patch_arena, "ax", %progbits
.balign 4

/* Reserved executable flash space for runtime patch detours. */
.global qemu_patch_arena
qemu_patch_arena:
  .space 1024
