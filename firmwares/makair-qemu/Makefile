# MakAir QEMU re-host — bare-metal Cortex-M4 build for QEMU mps2-an386.
# Layout: shim TUs under src/ + curated upstream .cpp files pulled live from
# the cloned makair-firmware tree at $(MAKAIR_SRC). Output: build/makair.elf.
# License: AGPL-3.0 (inherited from upstream MakAir).

CROSS    ?= arm-none-eabi-
CC        = $(CROSS)gcc
CXX       = $(CROSS)g++
OBJCOPY   = $(CROSS)objcopy
SIZE      = $(CROSS)size

BUILD_DIR := build
TARGET    := makair
ELF       := $(BUILD_DIR)/$(TARGET).elf
BIN       := $(BUILD_DIR)/$(TARGET).bin
MAP       := $(BUILD_DIR)/$(TARGET).map
LDSCRIPT  := mps2-an386.ld

# Path to the upstream MakAir clone (firmwares/build.sh populates this).
# Override with `make MAKAIR_SRC=/abs/path` when the layout differs.
MAKAIR_SRC ?= $(realpath ../repos/makair-firmware)

ifeq ($(wildcard $(MAKAIR_SRC)/srcs/telemetry.cpp),)
$(error MakAir upstream not found at MAKAIR_SRC=$(MAKAIR_SRC) — run firmwares/build.sh first)
endif

# Shim TUs (always compiled).
SHIM_C_SRCS    := $(wildcard src/*.c)
SHIM_CXX_SRCS  := $(wildcard src/*.cpp)

# Curated subset of the upstream .cpp tree — only what the protocol layer needs
# without dragging in HardwareTimer / blower / valve / screen / EEPROM. This
# list is the surgical seam between "verbatim upstream" and "QEMU shim".
#
# v0 scope: just telemetry.cpp (the firmware->host wire). Upstream
# serial_control.cpp transitively references mainController/eolTest globals
# whose stubs aren't landed yet; the QEMU loop handles Heartbeat itself in
# src/main.cpp::serial_control_loop_v0() until those stubs ship.
UPSTREAM_CXX_SRCS := \
    $(MAKAIR_SRC)/srcs/telemetry.cpp

C_OBJS    := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(SHIM_C_SRCS))
CXX_OBJS  := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SHIM_CXX_SRCS))
UP_OBJS   := $(patsubst $(MAKAIR_SRC)/srcs/%.cpp,$(BUILD_DIR)/upstream/%.o,$(UPSTREAM_CXX_SRCS))
OBJS      := $(C_OBJS) $(CXX_OBJS) $(UP_OBJS)
DEPS      := $(OBJS:.o=.d)

# Cortex-M4, soft-float ABI. mps2-an386 has FPU disabled by default; matching
# the secpump-qemu setup keeps the toolchain config homogeneous.
ARCHFLAGS := -mcpu=cortex-m4 -mthumb -mfloat-abi=soft

# -Os to keep code dense; -ffunction-sections + --gc-sections to drop
# anything pulled by the upstream chain we never call (e.g. variants of the
# Arduino classes the shim doesn't define).
OPTFLAGS  := -Os -ffunction-sections -fdata-sections -fno-common

WARNFLAGS := -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
             -Wno-unused-variable -Wno-write-strings

CPPFLAGS  := -Iinc \
             -I$(MAKAIR_SRC)/srcs \
             -I$(MAKAIR_SRC)/includes \
             -DMAKAIR_QEMU=1 \
             -DSIMULATOR=1 \
             -DPROVIDE_BLOWER_STUB=1 \
             -D__STDC_FORMAT_MACROS

CFLAGS    := $(ARCHFLAGS) $(OPTFLAGS) $(WARNFLAGS) -std=gnu11 -g \
             -fno-builtin -MMD -MP $(CPPFLAGS)
CXXFLAGS  := $(ARCHFLAGS) $(OPTFLAGS) $(WARNFLAGS) -std=gnu++14 -g \
             -fno-rtti -fno-exceptions -fno-builtin -MMD -MP $(CPPFLAGS)

# src/uart.c provides newlib syscalls, so we use nano.specs (newlib-nano)
# rather than nosys.specs.
LDFLAGS   := $(ARCHFLAGS) -T $(LDSCRIPT) \
             -Wl,--gc-sections -Wl,-Map=$(MAP) -Wl,--build-id=none \
             -nostartfiles --specs=nano.specs

LDLIBS    := -lc -lm -lnosys -lstdc++

QEMU      ?= qemu-system-arm
QEMU_ARGS := -M mps2-an386 \
             -display none -serial mon:stdio \
             -semihosting -semihosting-config enable=on,target=native \
             -kernel $(ELF)

.PHONY: all clean run smoke test debug size help demo demo-clean test-patched
.DEFAULT_GOAL := all

all: $(ELF) $(BIN) size

$(BUILD_DIR) $(BUILD_DIR)/upstream:
	@mkdir -p $@

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/upstream/%.o: $(MAKAIR_SRC)/srcs/%.cpp | $(BUILD_DIR)/upstream
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ELF): $(OBJS) $(LDSCRIPT) | $(BUILD_DIR)
	$(CXX) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $@

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

demo: $(ELF)
	./scripts/demo_makair.sh --stage=klee

demo-clean:
	rm -rf $(BUILD_DIR)/demo

test-patched:
	./run.sh patched-test

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "makair-qemu Makefile targets:"
	@echo "  all (default)  build $(ELF) and $(BIN)"
	@echo "  run            qemu-system-arm -M mps2-an386 -kernel ELF"
	@echo "  smoke          ./run.sh smoke (boot + first BootMessage)"
	@echo "  test           ./run.sh test (full Telemetry + Control protocol)"
	@echo "  debug          qemu paused, gdb stub on tcp::1234"
	@echo "  size           print arm-none-eabi-size on the ELF"
	@echo "  demo           run patchestry demo through KLEE (no Patcherex2)"
	@echo "  demo-clean     rm -rf $(BUILD_DIR)/demo"
	@echo "  test-patched   protocol tests against build/makair-patched.elf"
	@echo "  clean          rm -rf $(BUILD_DIR)"

-include $(DEPS)
