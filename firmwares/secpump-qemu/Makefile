# SecPump QEMU re-host — bare-metal Cortex-M4 build for QEMU mps2-an386.
# Layout: src/*.c -> build/secpump.elf + build/secpump.bin
# License: GPLv3 (inherited from upstream SecPump).

CROSS   ?= arm-none-eabi-
CC       = $(CROSS)gcc
OBJCOPY  = $(CROSS)objcopy
SIZE     = $(CROSS)size

BUILD_DIR := build
TARGET    := secpump
ELF       := $(BUILD_DIR)/$(TARGET).elf
BIN       := $(BUILD_DIR)/$(TARGET).bin
MAP       := $(BUILD_DIR)/$(TARGET).map
LDSCRIPT  := mps2-an386.ld

SRCS := $(wildcard src/*.c)
OBJS := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# Cortex-M4, soft-float ABI. The seeded MaliciousMemCpy overflow is calibrated
# against the saved-LR position produced by this exact target/ABI combination;
# changing -mfloat-abi or -mcpu without re-deriving the clobber math will break
# the exploit demo (see scripts/smoke_test.py).
ARCHFLAGS := -mcpu=cortex-m4 -mthumb -mfloat-abi=soft

# -Os keeps saved LR within the 112-byte clobber window of MaliciousMemCpy.
# Other optimization levels are likely to move it; do not change blindly.
OPTFLAGS  := -Os -ffunction-sections -fdata-sections -fno-common

WARNFLAGS := -Wall -Wextra -Wno-unused-parameter

CPPFLAGS  := -Iinc -DSECPUMP_QEMU_REHOST=1
CFLAGS    := $(ARCHFLAGS) $(OPTFLAGS) $(WARNFLAGS) -std=gnu11 -g \
             -fno-builtin -MMD -MP $(CPPFLAGS)

# nano.specs swaps in newlib-nano. We do NOT pass --specs=nosys.specs because
# src/uart.c already provides _write/_read/_sbrk/_close/_fstat/_isatty/_lseek/
# _getpid/_kill/_exit. -Wl,-u,_printf_float forces the float-aware printf
# variant required by [G]:%.4f / [u]:%.4f in InsulinController.c.
LDFLAGS   := $(ARCHFLAGS) -T $(LDSCRIPT) \
             -Wl,--gc-sections -Wl,-Map=$(MAP) -Wl,--build-id=none \
             -nostartfiles --specs=nano.specs \
             -Wl,-u,_printf_float

LDLIBS    := -lc -lm -lnosys

QEMU      ?= qemu-system-arm
# -display none + -serial mon:stdio multiplexes the QEMU monitor and the
# guest UART on stdio (Ctrl-A c switches to the monitor where `quit` exits).
# -semihosting enables BKPT 0xAB so the firmware can self-exit via the
# 'Q' command (semihosting_exit() in src/main.c).
QEMU_ARGS := -M mps2-an386 \
             -display none -serial mon:stdio \
             -semihosting -semihosting-config enable=on,target=native \
             -kernel $(ELF)

.PHONY: all clean run smoke test debug size help
.DEFAULT_GOAL := all

all: $(ELF) $(BIN) size

$(BUILD_DIR):
	@mkdir -p $@

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(ELF): $(OBJS) $(LDSCRIPT) | $(BUILD_DIR)
	$(CC) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(BIN): $(ELF)
	$(OBJCOPY) -O binary $< $@

size: $(ELF)
	@$(SIZE) $(ELF)

run: $(ELF)
	$(QEMU) $(QEMU_ARGS)

smoke: $(ELF)
	./run.sh smoke

test: $(ELF)
	./run.sh test

debug: $(ELF)
	$(QEMU) $(QEMU_ARGS) -s -S

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "secpump-qemu Makefile targets:"
	@echo "  all (default)  build $(ELF) and $(BIN)"
	@echo "  run            qemu-system-arm -M mps2-an386 -nographic -kernel ELF"
	@echo "  smoke          ./run.sh smoke (exploit demo)"
	@echo "  test           ./run.sh test  (protocol coverage + exploit)"
	@echo "  debug          qemu paused, gdb stub on tcp::1234"
	@echo "  size           print arm-none-eabi-size on the ELF"
	@echo "  clean          rm -rf $(BUILD_DIR)"

-include $(DEPS)
