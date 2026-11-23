# Model patterns

One shape per kind of function. Pick by what the function does to its caller:
what it returns, what memory it writes, whether it comes back. Every model
has the same header comment and the same rules.

## Rules for every model

1. The signature is the IR's, not the header's. Take the return type and the
   parameter types from the `declare` line (`returns` and `parameters` in the
   extractor's JSON) and write them with fixed-width types:

   | IR | C |
   | --- | --- |
   | `i1` | `_Bool` |
   | `i8` | `uint8_t`, or `int8_t` when the JSON says `signext` |
   | `i16` | `uint16_t` / `int16_t` |
   | `i32` | `uint32_t` / `int32_t` |
   | `i64` | `uint64_t` / `int64_t` |
   | `ptr` | `void *`, or a pointer to what the model reads or writes |
   | `void` | `void` |

   `int` happens to be `i32` on x86_64, but `long`, `size_t` and `uintptr_t`
   are `i64` there and `i32` in ARM32 firmware. `check_models.py` reports the
   mismatch as a `type` error. The lifted IR can also disagree with the
   vendor header: libopencm3 declares
   `uint16_t usbd_ep_write_packet(usbd_device *, uint8_t addr, const void *buf, uint16_t len)`
   and the test fixture's IR has `i32 @usbd_ep_write_packet(ptr, ptr, i32)`.
   The IR is what gets linked.
2. Over-approximate unless something read justifies less. `klee_assume` drops
   paths. An assumption the real function does not keep turns a real bug into
   "proven".
3. Never write more bytes than the caller's object has. KLEE reports a write
   past an object as a memory error, which shows up as a failure of the
   function under test.
4. Name each symbolic value after the function and what it stands for
   (`"usart_recv_ret"`). The names appear in KLEE's test cases.
5. No model for `malloc`, `calloc`, `realloc`, `free`, `posix_memalign`,
   `valloc`, `memalign`. KLEE provides them.
6. Do not read through a pointer that is a fixed address of the image.
   Lifted code passes strings and tables of the original binary as
   `inttoptr (i32 441964 to ptr)`, which the extractor shows as
   `address:0x6be6c` in `constant_arguments`. The module does not contain
   that memory, so a model that reads it (a `strcmp` loop over a format or
   delimiter string) stops KLEE with a memory error. When the content
   matters, read the bytes from the binary and write them into the model as
   a constant; otherwise ignore the argument.
7. Start each model with this comment, as `usb_hal_models.c` does:

   ```c
   // Model: <name>
   //   Source: <existing-model | known-semantics | convention | body | signature-only>
   //           and what was read (header path, address range, standard)
   //   Memory effect: <what it writes, or "none">
   //   Return: <range or meaning>
   //   Assumes: <every klee_assume, in words; "nothing" when there is none>
   ```

The file starts with:

```c
#include "klee_stub.h"
#include <stdint.h>
#include <stddef.h>
```

## symbolic-return

A function that computes a value and writes nothing the caller sees: a status
read, a getter, a sensor read. This is also the signature-only fallback for a
function that returns a scalar and has no pointer parameters.

```c
uint32_t adc_read_regular(uint32_t adc) {
    (void) adc;
    uint32_t value;
    klee_make_symbolic(&value, sizeof(value), "adc_read_regular_ret");
    return value;
}
```

With a documented range, add the assumptions and say where the range comes
from. From `lib/patchestry/klee/models/usb_hal_models.c`:

```c
int32_t ret;
klee_make_symbolic(&ret, sizeof(ret), "ep_write_ret");
klee_assume(ret >= 0);
klee_assume(ret <= 255);
return ret;
```

## status-return

A function that reports success or one of a few error codes. The return is
symbolic and restricted to the codes the function can return. Read the enum
from the vendor header or the constants from the body; do not guess them.

```c
int32_t HAL_UART_Transmit(void *huart, void *data, uint16_t size, uint32_t timeout) {
    (void) huart; (void) data; (void) size; (void) timeout;
    int32_t status;
    klee_make_symbolic(&status, sizeof(status), "HAL_UART_Transmit_status");
    klee_assume(status >= 0);
    klee_assume(status <= 3);   // HAL_OK, HAL_ERROR, HAL_BUSY, HAL_TIMEOUT
    return status;
}
```

A caller that never checks the status is then explored on the error paths
too, which is the point.

## buffer-writer

A function that fills memory the caller passed: `read`, `recv`, a HAL
receive, `memset`, `strcpy`. Copy symbolic bytes in, one at a time, up to the
length the caller gave:

```c
#define RECV_MODEL_MAX 64   // bound on the loop; listed in the report

int32_t uart_receive(void *handle, uint8_t *buffer, uint32_t length) {
    (void) handle;
    uint8_t bytes[RECV_MODEL_MAX];
    klee_make_symbolic(bytes, sizeof(bytes), "uart_receive_bytes");
    uint32_t count;
    klee_make_symbolic(&count, sizeof(count), "uart_receive_count");
    klee_assume(count <= length);
    klee_assume(count <= RECV_MODEL_MAX);
    for (uint32_t i = 0; i < count; i++)
        buffer[i] = bytes[i];
    return (int32_t) count;
}
```

Why not `klee_make_symbolic(buffer, length, ...)` directly: KLEE accepts
`klee_make_symbolic` only for a whole object, with the object's exact size.
The existing `usbd_ep_write_packet` model does call it on the caller's buffer
with 256 bytes, which matches the 256-byte objects the verifier allocates for
pointer arguments (`--symbolic-ptr-size`, default 256) and nothing else. The
copy loop works for a buffer of any size, including one inside a struct or in
the middle of an array. This statement about KLEE comes from its documented
behaviour and was not re-run here (KLEE is only in the Docker image); if a
run reports "wrong size given to klee_make_symbolic", this is the cause.

When the length is a constant at every call site (`constant_arguments` in the
extractor's JSON), use it as the bound. Otherwise define a named bound, and
say in the report that writes beyond it are not modelled.

For functions with exact semantics, write the semantics. `memset` and
`memcpy` are loops; they need no symbolic value at all.

## out-parameter

A function that returns a status and a result through a pointer. Write the
result only on the paths where the real function writes it. From
`lib/patchestry/klee/models/libc_models.c`:

```c
char bl_device__match(unsigned int id, void **out_handle) {
    (void) id;
    char ret;
    klee_make_symbolic(&ret, sizeof(ret), "bl_device__match_ret");
    if (ret == 1) {
        *out_handle = malloc(64);
    } else {
        *out_handle = (void *) 0;
    }
    return ret;
}
```

A scalar result is a local symbolic value stored through the pointer:

```c
uint32_t value;
klee_make_symbolic(&value, sizeof(value), "get_config_value");
*out = value;
```

## pointer-return

A function that returns a pointer the caller reads through. Allocate with
`malloc` (KLEE tracks it), fill with symbolic bytes, and return null on some
path if the real function can.

```c
extern void *malloc(size_t size);

void *get_descriptor(uint32_t index) {
    (void) index;
    _Bool found;
    klee_make_symbolic(&found, sizeof(found), "get_descriptor_found");
    if (!found)
        return (void *) 0;
    void *result = malloc(64);
    klee_make_symbolic(result, 64, "get_descriptor_result");
    return result;
}
```

Never return the address of a local. The verifier's own stub for a
pointer-returning function does the same with `--symbolic-ptr-size` bytes
and never returns null, so a model is needed only for the null path, a
different size, or structure in the bytes.

## noreturn

A fatal-error handler, `abort`, `__stack_chk_fail`, a reset. The path ends.

```c
void fatal(const char *message) {
    (void) message;
    klee_abort();
}
```

`klee_abort` ends the path as an error, so reaching the handler is reported.
That is right for `__stack_chk_fail` and for an assertion handler. For a
function that ends the program normally (`exit`), reaching it is not a
defect, and the stub has nothing quieter than `klee_abort`: declare
`void klee_silent_exit(int);` in `klee_stub.h` first, or leave the function
to the verifier's stub and say so in the report.

## formatted-output

`sprintf`, `snprintf` and their kind: the content depends on the format, and
only the length and the terminator matter to most callers. The existing
models in `libc_models.c` are the shape to copy: a symbolic length within a
bound, a terminator written at that length, the length returned. The bounded
form:

```c
size_t writelen = ((size_t) n < size - 1) ? (size_t) n : size - 1;
dest[writelen] = '\0';
return (int) writelen;
```

Note that the existing `vsnprintf` takes `size_t size`, which is `i64` on
x86_64. Against ARM32 IR, where the parameter is `i32`, write it as
`uint32_t`.

A printf-like function with no destination buffer (`printf`, a logger) is a
`symbolic-return`, or an empty body when it returns `void`.

## pure

`strlen`, `strcmp`, `memcmp`, `abs`: no memory effect, and the result is a
function of the arguments. Write the real loop when the caller's behaviour
depends on the result (a length used as an index, a comparison that selects
a branch). A symbolic return is sound but loses the link between argument
and result, and gives false alarms: a symbolic `strlen` can exceed the
buffer.

```c
uint32_t strlen(const char *s) {
    uint32_t n = 0;
    while (s[n])
        n++;
    return n;
}
```

## mmio

A register read or write through a HAL accessor. A read returns a fresh
symbolic value on every call, because hardware state changes between reads.
A write has no effect the function under test can see, unless the caller
reads the same register back and depends on it, in which case keep the value
in a `static` and return it from the read.

## From a body

When the source is `body`, the summary has to answer five questions, each
from an instruction or a line of decompiled C:

1. Which values can it return? Constants loaded before each return
   (`mov r0, #1`), or a value passed through from a callee.
2. Which pointer parameters does it write through, and how far?
3. Which globals or device registers does it write?
4. Can it fail to return (a call to a noreturn function, an infinite loop)?
5. Which of its own callees decide the answer to 1 to 4? If those are
   unknown, the summary inherits the uncertainty: over-approximate.

Then write the smallest pattern that covers the answers. Example, from
`netcgi` `FUN_000198c4` (0x198c4 to 0x19918): it returns 0 for a null
argument, 1 when `strcasecmp(p, <string at 0x75fdc>)` is 0, and 0 after a
logging call otherwise; it writes nothing through `p`. The model is a
symbolic `_Bool`, with the null case kept exact:

```c
// Model: FUN_000198c4
//   Source: body, original image 0x198c4-0x19918 (disassembly)
//   Memory effect: none
//   Return: 0 when the argument is null, otherwise 0 or 1
//   Assumes: nothing beyond the range of _Bool
_Bool FUN_000198c4(const char *name) {
    if (!name)
        return 0;
    _Bool match;
    klee_make_symbolic(&match, sizeof(match), "FUN_000198c4_ret");
    return match;
}
```

Check the return type against the body. Ghidra types some functions `_Bool`
that return 2, 3 or 4, and the lifted IR then carries `i1`. The model has to
use the IR's type to link, so record the disagreement in the report: the
function under test already truncates the value, and no model can restore it.
