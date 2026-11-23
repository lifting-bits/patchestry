# KLEE functions a model may call

Models are compiled against `lib/patchestry/klee/models/klee_stub.h`, not
against a KLEE installation. That header declares exactly this, and a model
that calls anything else does not compile:

```c
void klee_make_symbolic(void *addr, size_t nbytes, const char *name);
void klee_assume(unsigned long long condition);
void klee_abort(void) __attribute__((noreturn));

#define klee_assert(cond) do { if (!(cond)) klee_abort(); } while (0)
```

| Function | Use in a model |
| --- | --- |
| `klee_make_symbolic(addr, n, name)` | the `n` bytes at `addr` become unknown: a return value, an out-parameter, a buffer the callee fills |
| `klee_assume(condition)` | restricts the unknown values to those the real function can produce. Paths where the condition is false are dropped, so a condition that is too strong hides real behaviour |
| `klee_abort()` | ends the path as an error. For a function that never returns, such as a fatal-error handler |
| `klee_assert(condition)` | a precondition of the callee that the caller must meet |

Not available through the stub: `klee_range`, `klee_int`,
`klee_silent_exit`, `klee_prefer_cex`, `klee_warning`. A range is written as a
symbolic value plus two `klee_assume` calls, as `usb_hal_models.c` does. If one
of the others is needed, add its declaration to `klee_stub.h` first.

`list_externals.py` leaves out every function whose name starts with `klee_`,
so these never show up as functions to model.
