# Atomics probe firmwares. One freestanding ELF per architecture; every
# public function executes a single synchronization or LL/SC instruction so
# Ghidra exposes the corresponding CALLOTHER userop in the serialized P-Code.
# The resulting JSON ground-truth fixtures live under test/patchir-decomp/.
#
# Targets (one ELF each):
#   atomics_probe_cortexm4.elf - ARMv7-M (arm-none-eabi-gcc + newlib)
#   atomics_probe_aarch64.elf  - AArch64 ELF (aarch64-elf-gcc)
#   atomics_probe_x86_64.elf   - x86-64 ELF (LLVM clang --target=x86_64-linux-gnu)
#
# Tooling expected on PATH (Homebrew names on macOS):
#   - arm-none-eabi-gcc   (formula: arm-none-eabi-gcc)
#   - aarch64-elf-gcc     (formula: aarch64-elf-gcc)
#   - clang               (LLVM with X86 target enabled)
#
# License: same as the rest of patchestry (see LICENSE).

CLANG ?= clang

.PHONY: all clean cortexm4 aarch64 x86_64

all: cortexm4 aarch64 x86_64

# -----------------------------------------------------------------------------
# ARMv7-M (Cortex-M4)
# -----------------------------------------------------------------------------
CORTEXM4_ELF   := atomics_probe_cortexm4.elf
CORTEXM4_OBJ   := atomics_probe_cortexm4.o
CORTEXM4_CFLAGS := -mcpu=cortex-m4 -mthumb -mfloat-abi=soft \
                   -O1 -g -fno-builtin -ffunction-sections -fdata-sections \
                   -Wall -Wextra -std=gnu11
CORTEXM4_LDFLAGS := -mcpu=cortex-m4 -mthumb -mfloat-abi=soft \
                    -nostartfiles -nostdlib \
                    -Wl,-Ttext=0x10000000 -Wl,-e,probe_dmb

cortexm4: $(CORTEXM4_ELF)

$(CORTEXM4_OBJ): atomics_probe_cortexm4.c
	arm-none-eabi-gcc $(CORTEXM4_CFLAGS) -c $< -o $@

$(CORTEXM4_ELF): $(CORTEXM4_OBJ)
	arm-none-eabi-gcc $(CORTEXM4_LDFLAGS) $< -o $@

# -----------------------------------------------------------------------------
# AArch64 and x86-64
# -----------------------------------------------------------------------------
# Both targets share the same clang/lld cross-compile recipe: clang ships
# freestanding stdint.h, lld handles every target the LLVM build was
# configured for, and using clang lets the probes build on any host
# (including Apple Silicon) without an arch-specific gcc/newlib install.

XCOMPILE_CFLAGS := -O1 -g -ffreestanding -fno-builtin \
                   -ffunction-sections -fdata-sections \
                   -Wall -Wextra -std=gnu11
XCOMPILE_LDFLAGS := -nostartfiles -nostdlib -static -fuse-ld=lld -Wl,--no-dynamic-linker

AARCH64_ELF    := atomics_probe_aarch64.elf
AARCH64_OBJ    := atomics_probe_aarch64.o
AARCH64_TARGET := --target=aarch64-linux-gnu
AARCH64_LDADD  := -Wl,--image-base=0x10000000 -Wl,-e,probe_dmb

aarch64: $(AARCH64_ELF)

$(AARCH64_OBJ): atomics_probe_aarch64.c
	$(CLANG) $(AARCH64_TARGET) $(XCOMPILE_CFLAGS) -c $< -o $@

$(AARCH64_ELF): $(AARCH64_OBJ)
	$(CLANG) $(AARCH64_TARGET) $(XCOMPILE_LDFLAGS) $(AARCH64_LDADD) $< -o $@

X86_64_ELF    := atomics_probe_x86_64.elf
X86_64_OBJ    := atomics_probe_x86_64.o
X86_64_TARGET := --target=x86_64-linux-gnu
X86_64_LDADD  := -Wl,--image-base=0x400000 -Wl,-e,probe_mfence

x86_64: $(X86_64_ELF)

$(X86_64_OBJ): atomics_probe_x86_64.c
	$(CLANG) $(X86_64_TARGET) $(XCOMPILE_CFLAGS) -c $< -o $@

$(X86_64_ELF): $(X86_64_OBJ)
	$(CLANG) $(X86_64_TARGET) $(XCOMPILE_LDFLAGS) $(X86_64_LDADD) $< -o $@

clean:
	rm -f $(CORTEXM4_OBJ) $(CORTEXM4_ELF) \
	      $(AARCH64_OBJ) $(AARCH64_ELF) \
	      $(X86_64_OBJ) $(X86_64_ELF)
