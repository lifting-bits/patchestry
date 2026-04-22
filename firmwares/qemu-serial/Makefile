SHELL := /bin/bash

LLVM_PREFIX ?= /Users/artem/projects/patchestry/bin
LD_LLD ?= /opt/homebrew/Cellar/llvm@18/18.1.8/bin/ld.lld
BUILD_DIR := build
TARGET := $(BUILD_DIR)/qemu-serial.elf
MAP := $(BUILD_DIR)/qemu-serial.map
DISASM := $(BUILD_DIR)/qemu-serial.objdump

CLANG := $(LLVM_PREFIX)/clang
OBJDUMP := $(LLVM_PREFIX)/llvm-objdump

CFLAGS := \
	--target=thumbv7m-none-eabi \
	-mcpu=cortex-m3 \
	-mthumb \
	-ffreestanding \
	-fno-builtin \
	-fdata-sections \
	-ffunction-sections \
	-fno-unwind-tables \
	-fno-asynchronous-unwind-tables \
	-fno-stack-protector \
	-O0 \
	-g \
	-Wall \
	-Wextra \
	-Werror \
	-I./src

LDFLAGS := \
	--ld-path=$(LD_LLD) \
	-nostdlib \
	-Wl,-T,linker.ld \
	-Wl,-Map,$(MAP) \
	-Wl,--gc-sections

SOURCES := \
	startup.S \
	src/main.c \
	src/patch_arena.S \
	src/targets.c \
	src/uart.c

.PHONY: all clean check-tools

all: check-tools $(TARGET) $(DISASM)

check-tools:
	@test -x "$(CLANG)" || { \
		echo "Missing clang at $(CLANG). Set LLVM_PREFIX to your patched LLVM bin directory." >&2; \
		exit 1; \
	}
	@test -x "$(LD_LLD)" || { \
		echo "Missing ld.lld at $(LD_LLD). Set LD_LLD to an ELF-capable lld binary." >&2; \
		exit 1; \
	}
	@test -x "$(OBJDUMP)" || { \
		echo "Missing llvm-objdump at $(OBJDUMP). Set LLVM_PREFIX to your patched LLVM bin directory." >&2; \
		exit 1; \
	}

$(TARGET): $(SOURCES) linker.ld | $(BUILD_DIR)
	$(CLANG) $(CFLAGS) $(LDFLAGS) $(SOURCES) -o $(TARGET)

$(DISASM): $(TARGET)
	$(OBJDUMP) -d -r --print-imm-hex $(TARGET) > $(DISASM)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
