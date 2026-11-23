#!/usr/bin/env python3
"""Say what kind of function each external is, and where its model should come from.

Reads the JSON of list_externals.py and adds to every external:

| Field | Meaning |
| --- | --- |
| `class` | `libc`, `runtime` (compiler and loader support), `hal`, `rtos`, `internal` (a function of the binary), `import` (a function the binary imports), `unknown` |
| `family` | for `hal` and `rtos`, the library the name belongs to |
| `pattern` | the model shape the name or the signature suggests, when one does: `noreturn`, `allocator`, `free`, `buffer-writer`, `pure`, `formatted-output`, `status-return` |
| `address` | where the function is in the binary, when --binary or --pcode-json has it |
| `source` | where the model should come from, in the order the skill uses: `existing-model`, `klee` (KLEE provides the function, write no model), `known-semantics`, `convention`, `body`, `signature-only` |

The name decides the class.  A binary or a Ghidra P-Code export refines it:
a function they define is `internal` with its address, one they import is
an `import`.  `source` is `body` only when there is something to read: an
address in the binary.  It is never a claim that a body was read.

--binary takes an ELF file (32 or 64 bit, either byte order) and uses its
symbol tables, so a stripped image tells only its imports.  --pcode-json
takes the export of the project's Ghidra script, where a function under an
`EXTERNAL:` key is an import and any other has its entry point.

CLI: classify.py [EXTERNALS.json] [--binary ELF] [--pcode-json FILE]... [--output FILE]
     (reads stdin when no file is given)
Exit 0 on success, 1 when an input cannot be read, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

# name -> pattern.  Only what a model can be written for from the C standard
# or POSIX without reading anything.
LIBC = {
    **dict.fromkeys(("malloc", "calloc", "realloc", "aligned_alloc", "strdup", "strndup"),
                    "allocator"),
    "free": "free",
    **dict.fromkeys(("exit", "_exit", "_Exit", "abort", "__assert_fail", "__assert_func",
                     "longjmp", "pthread_exit"), "noreturn"),
    **dict.fromkeys(("memcpy", "memmove", "memset", "strcpy", "strncpy", "strlcpy", "strcat",
                     "strncat", "strlcat", "stpcpy", "read", "recv", "recvfrom", "fread",
                     "fgets", "gets", "getline", "bzero", "bcopy", "strtok_r"), "buffer-writer"),
    **dict.fromkeys(("sprintf", "snprintf", "vsprintf", "vsnprintf", "asprintf"),
                    "formatted-output"),
    **dict.fromkeys(("strlen", "strnlen", "strcmp", "strncmp", "strcasecmp", "strncasecmp",
                     "memcmp", "strchr", "strrchr", "strstr", "memchr", "strspn", "strcspn",
                     "strpbrk", "atoi", "atol", "atoll", "strtol", "strtoul", "strtoll",
                     "strtoull", "abs", "labs", "isalpha", "isdigit", "isalnum", "isspace",
                     "isupper", "islower", "isxdigit", "isprint", "toupper", "tolower",
                     "htons", "htonl", "ntohs", "ntohl"), "pure"),
    **dict.fromkeys(("printf", "fprintf", "vprintf", "vfprintf", "puts", "fputs", "putchar",
                     "fputc", "putc", "perror", "write", "send", "sendto", "fwrite", "fflush",
                     "open", "open64", "close", "fopen", "fopen64", "fclose", "ioctl", "fcntl",
                     "lseek", "stat", "fstat", "unlink", "rename", "socket", "bind", "listen",
                     "accept", "connect", "select", "poll", "sleep", "usleep", "nanosleep",
                     "time", "gettimeofday", "clock_gettime", "getpid", "fork", "execve",
                     "system", "popen", "pclose", "signal", "sigaction", "kill", "getenv",
                     "setenv", "rand", "srand", "random", "srandom", "qsort", "bsearch",
                     "strtok", "strerror", "sscanf", "scanf", "fscanf", "getchar", "fgetc",
                     "getc", "feof", "ferror", "mmap", "munmap", "lockf", "ftruncate",
                     "ftruncate64", "inet_aton", "inet_ntoa", "inet_addr", "syslog", "openlog",
                     "closelog", "pthread_create", "pthread_join", "pthread_mutex_lock",
                     "pthread_mutex_unlock", "setjmp", "atexit"), None),
}

RUNTIME = (
    (r"__stack_chk_fail$|__cxa_pure_virtual$|__cxa_throw$|_Unwind_Resume$|__cxa_bad_cast$"
     r"|_ZSt9terminatev$|_ZSt17__throw_.*", "noreturn"),
    (r"_Znw[jmy]$|_Zna[jmy]$|__cxa_allocate_exception$", "allocator"),
    (r"_ZdlPv.*|_ZdaPv.*", "free"),
    (r"__aeabi_(mem(cpy|move|set|clr)[48]?)$", "buffer-writer"),
    (r"__aeabi_.*|__(u?div|u?mod|mul|ashl|ashr|lshr|fix|float|add|sub|neg|cmp|eq|ne|lt|le|gt|ge)"
     r"[sdt][ifc]\w*$|__u?divmod[sd]i4$|__clz[sd]i2$|__ctz[sd]i2$|__popcount[sd]i2$", "pure"),
    (r"__stack_chk_guard$|__cxa_.*|__gxx_personality_.*|__libc_start_main$|__gmon_start__$"
     r"|__errno_location$|__errno$|_ITM_.*|__dso_handle$", None),
)

# (class, family, name pattern, pattern the family's functions usually have)
FAMILIES = (
    ("hal", "stm32-hal", r"(__)?HAL_\w+", "status-return"),
    ("hal", "stm32-ll", r"LL_[A-Z0-9]+_\w+", None),
    ("hal", "cmsis-core", r"(__)?NVIC_\w+|SysTick_\w+|SCB_\w+|ITM_\w+|SystemInit$"
                          r"|SystemCoreClockUpdate$|__(enable|disable)_irq$|__(get|set)_[A-Z]+\w*"
                          r"|__WFI$|__WFE$|__NOP$|__BKPT$", None),
    ("rtos", "cmsis-rtos", r"os(Thread|Mutex|Semaphore|MessageQueue|Timer|EventFlags|Kernel|Delay"
                           r"|Signal|Pool|Mail|MemoryPool)\w*", "status-return"),
    ("rtos", "freertos", r"[xvu]?(x|v|ux|pv|pc|e)?(Task|Queue|Semaphore|Timer|EventGroup"
                         r"|StreamBuffer|MessageBuffer|Port)[A-Z]\w*|pvPortMalloc$|vPortFree$",
     None),
    ("rtos", "zephyr", r"k_(thread|sem|mutex|msgq|timer|work|sleep|malloc|free|yield|uptime)\w*"
                       r"|z_impl_\w+", None),
    ("hal", "libopencm3", r"(usbd|rcc|gpio|usart|spi|i2c|timer|flash|nvic|systick|adc|dac|exti"
                          r"|iwdg|pwr|rtc|dma|crc|desig|scb|cm)_[a-z0-9_]+", None),
    ("hal", "nordic-nrfx", r"nrfx?_[a-z0-9]+_\w+|sd_[a-z]+_\w+|app_(timer|uart|usbd)_\w+", None),
    ("hal", "ambiq", r"am_(hal|bsp|util|devices)_\w+", None),
    ("hal", "tivaware", r"(ROM_|MAP_)?(GPIO|UART|SSI|I2C|Timer|SysCtl|Int|ADC|PWM|Flash|EEPROM"
                        r"|USB|Watchdog)[A-Z][A-Za-z0-9]+", None),
    ("hal", "esp-idf", r"(esp|gpio|uart|i2c|spi|ledc|nvs|wifi)_[a-z0-9_]+", None),
    ("hal", "mbed-hal", r"(gpio|serial|spi|i2c|pwmout|analogin|us_ticker|lp_ticker|sleep)_"
                        r"[a-z0-9_]+", None),
    ("hal", "pico-sdk", r"(gpio|uart|spi|i2c|pwm|adc|pio|watchdog|multicore|stdio)_[a-z0-9_]+"
                        r"|sleep_(ms|us)$|time_us_(32|64)$", None),
)
# KLEE handles these itself and patchir-klee-verifier never stubs them
# (kKleeAllocators in tools/patchir-klee-verifier/HarnessGen.cpp).
KLEE_PROVIDED = ("malloc", "calloc", "realloc", "free", "posix_memalign", "valloc", "memalign")
GHIDRA_NAME = re.compile(r"(?:FUN|thunk_FUN|SUB|LAB)_([0-9A-Fa-f]{4,16})$")


class InputError(Exception):
    pass


def elf_symbols(path: Path) -> tuple[dict[str, int], set[str]]:
    """(functions the image defines and their addresses, functions it imports)."""
    data = path.read_bytes()
    if data[:4] != b"\x7fELF":
        raise InputError(f"{path} is not an ELF file")
    wide, order = data[4] == 2, "<" if data[5] == 1 else ">"
    if wide:
        offset, size, count = (struct.unpack_from(order + f, data, at)[0]
                               for f, at in (("Q", 0x28), ("H", 0x3A), ("H", 0x3C)))
    else:
        offset, size, count = (struct.unpack_from(order + f, data, at)[0]
                               for f, at in (("I", 0x20), ("H", 0x2E), ("H", 0x30)))
    sections = []
    for index in range(count):
        at = offset + index * size
        if at + size > len(data):
            break
        if wide:
            _, kind, _, _, start, length, link, _, _, entry = struct.unpack_from(
                order + "IIQQQQIIQQ", data, at)
        else:
            _, kind, _, _, start, length, link, _, _, entry = struct.unpack_from(
                order + "IIIIIIIIII", data, at)
        sections.append((kind, start, length, link, entry))

    defined, imported = {}, set()
    for kind, start, length, link, entry in sections:
        if kind not in (2, 11) or not entry or link >= len(sections):  # SYMTAB, DYNSYM
            continue
        strings = data[sections[link][1]:sections[link][1] + sections[link][2]]
        for at in range(start, min(start + length, len(data)) - entry + 1, entry):
            if wide:
                name, info, _, index, value, _ = struct.unpack_from(order + "IBBHQQ", data, at)
            else:
                name, value, _, info, _, index = struct.unpack_from(order + "IIIBBH", data, at)
            text = strings[name:strings.find(b"\0", name)].decode("utf-8", errors="replace")
            if not text or info & 0xF != 2:  # STT_FUNC
                continue
            # A versioned import is `name@GLIBC_2.4` in some tables.
            text = text.split("@", 1)[0]
            if index == 0:
                imported.add(text)
            else:
                defined.setdefault(text, value & ~1)  # the low bit marks Thumb
    return defined, imported - set(defined)


def pcode_functions(path: Path) -> tuple[dict[str, int], set[str]]:
    try:
        functions = json.loads(path.read_text(encoding="utf-8"))["functions"]
    except (ValueError, KeyError, TypeError) as error:
        raise InputError(f"{path} is not a P-Code export: {error}") from error
    defined, imported = {}, set()
    for key, function in functions.items():
        name = function.get("name")
        if not name:
            continue
        if key.startswith("EXTERNAL:"):
            imported.add(name)
            continue
        match = re.fullmatch(r"ram:([0-9a-fA-F]+)", function.get("entry_point", ""))
        if match:
            defined.setdefault(name, int(match.group(1), 16))
    return defined, imported - set(defined)


def by_name(name: str) -> tuple[str, str | None, str | None]:
    """(class, family, pattern) from the name alone."""
    # glibc's internal aliases: __GI_memcpy is memcpy.
    plain = name[len("__GI_"):] if name.startswith("__GI_") else name
    if plain in LIBC:
        return "libc", None, LIBC[plain]
    if name.startswith("__") and name.endswith("_chk") and name[2:-4] in LIBC:
        return "libc", None, LIBC[name[2:-4]]
    for pattern, shape in RUNTIME:
        if re.match(pattern, name):
            return "runtime", None, shape
    for kind, family, pattern, shape in FAMILIES:
        if re.fullmatch(pattern, name):
            return kind, family, shape
    if GHIDRA_NAME.match(name):
        return "internal", None, None
    return "unknown", None, None


def classify(external: dict, defined: dict[str, int], imported: set[str]) -> dict:
    name = external["name"]
    kind, family, pattern = by_name(name)
    address = defined.get(name)
    ghidra = GHIDRA_NAME.match(name)
    if address is None and ghidra:
        address = int(ghidra.group(1), 16)

    # libc and runtime keep their class even when the image links them
    # statically: what they do is known without reading them.
    if kind in ("unknown", "internal"):
        if name in defined or ghidra:
            kind = "internal"
        elif name in imported:
            kind = "import"
    if external.get("noreturn"):
        pattern = "noreturn"

    if external.get("modelled_in"):
        source = "existing-model"
    elif name in KLEE_PROVIDED:
        source = "klee"
    elif kind in ("libc", "runtime"):
        source = "known-semantics"
    elif kind in ("hal", "rtos"):
        # A vendor function linked into the image can be read when the
        # convention does not say enough.
        source = "convention"
    elif address is not None:
        source = "body"
    else:
        source = "signature-only"
    return {**external, "class": kind, "family": family, "pattern": pattern,
            "address": None if address is None else hex(address),
            "in_binary": name in defined, "imported": name in imported, "source": source}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("externals", nargs="?", type=Path,
                        help="output of list_externals.py (default: stdin)")
    parser.add_argument("--binary", type=Path, help="the ELF image the IR was lifted from")
    parser.add_argument("--pcode-json", type=Path, action="append", default=[],
                        help="a Ghidra P-Code export of the image or of one function")
    parser.add_argument("--output", type=Path, help="write the JSON here instead of stdout")
    args = parser.parse_args(argv)

    try:
        text = args.externals.read_text(encoding="utf-8") if args.externals else sys.stdin.read()
        report = json.loads(text)
        if not isinstance(report, dict) or "externals" not in report:
            raise InputError("the input is not the output of list_externals.py")
        defined, imported = {}, set()
        for load, paths in ((elf_symbols, [args.binary] if args.binary else []),
                            (pcode_functions, args.pcode_json)):
            for path in paths:
                found, missing = load(path)
                for name, address in found.items():
                    defined.setdefault(name, address)
                imported |= missing
        imported -= set(defined)
    except (OSError, ValueError, InputError, struct.error) as error:
        print(f"classify: {error}", file=sys.stderr)
        return 1

    report["externals"] = [classify(item, defined, imported) for item in report["externals"]]
    counts: dict[str, int] = {}
    sources: dict[str, int] = {}
    for item in report["externals"]:
        counts[item["class"]] = counts.get(item["class"], 0) + 1
        sources[item["source"]] = sources.get(item["source"], 0) + 1
    report.setdefault("summary", {}).update(
        classes=dict(sorted(counts.items())), sources=dict(sorted(sources.items())),
        binary=str(args.binary) if args.binary else None,
        pcode_json=[str(path) for path in args.pcode_json])
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
