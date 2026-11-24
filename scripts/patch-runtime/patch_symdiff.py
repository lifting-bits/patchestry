#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""PatchSymDiff: compare a function in the original binary with the same
function in the rewritten binary using cozy (https://github.com/draperlaboratory/cozy).

This is the comparison the runtime suites use.  It reads neither the lifted
IR nor the patched IR for what the code does: angr executes the bytes of the
original ELF and the bytes of the rewritten ELF, and cozy pairs the final
states of the two runs and finds the memory and registers that can differ.
A difference means some stage between the two images (lift, transformation,
compiler, patcher) changed the behaviour, and does not say which.  A lifting
error in angr's own VEX is invisible to it.

How it works:

1. Both images are loaded.  The function starts at the same address in both;
   in the rewritten image the entry may be a branch to moved code, which is
   followed.
2. Both runs get the same inputs: one shared 32-bit symbol per argument, or,
   for a declared argument buffer, a pointer to a block of shared symbolic
   bytes at a fixed address.  Globals hold the bytes of the image.  The
   registers a function must preserve, r4 to r11, hold one shared symbol
   each, so a function that returns with any other value in one of them
   differs from one that preserves it.  Other memory and registers that
   are read before they are written are zero.
3. cozy runs each side to the function's return.  Callees are followed.  A
   callee named with `--stub` is not: it returns a symbol that depends on
   the callee and on how many times it was called, the same on both sides,
   and writes nothing.  The calls to stubbed callees are compared in
   order: which callee, and with `--stub-args NAME:N` its first N register
   arguments.  An argument that is an address in the function's own frame
   is not compared, since the two frames are laid out differently; what it
   points to is not compared either.  An argument that is an address in a
   read-only part of the image is compared by the bytes there up to the
   first zero, at most 256: recompiled code brings its own copy of a
   string.  A callee whose result is read through is
   named with `--stub-pointer NAME:SIZE`: each call returns a pointer to
   its own SIZE unknown bytes, the same on both sides.
4. cozy pairs every final state of the original with every final state of
   the rewritten function that some input reaches together, and lists the
   written memory and the registers that can differ in the pair.  That is a
   comparison of final states: a byte written twice counts once, with its
   last value.  For a device register that loses what matters, so every
   write to a range given with `--mmio-region` is also recorded, in order,
   and the two sequences are compared write by write.
5. Each listed difference is checked again with the inputs both sides
   accept, and reported with an input that shows it.

A function rejects an input when it returns a value given with
`--reject-return`, calls a function given with `--reject-handler` (the path
ends at that call and the handler is not run), or writes to an address given
with `--reject-write`, such as the store to address 0 a check with nothing to
call uses to fault.  The value returned is r0, with r1 above it when r1 is
among the compared registers (a 64-bit result); a negative reject value
stands for its bit pattern at that width.

A read or write through an address that depends on the input is made at
every address it can take, up to ADDRESS_LIMIT of them.  With more, the
inputs that lead to the others are not explored and the run is `incomplete`.  A store the processor makes
on some inputs of a path only (a conditional instruction) is a device write
or a rejection on those inputs only.  With `--reject-skip NAME`, which also stubs NAME, a
rewritten path whose calls to stubbed callees are those of the original path
without some calls of NAME is a rejection as well.  A patch that adds a check
makes the function reject inputs it used to accept, and that is not a
difference.  The reverse is:

| Difference | Some input |
| --- | --- |
| `accepts-rejected-input` | is rejected by the original and accepted by the rewritten function |
| `memory` | is accepted by both and leaves a different byte outside the dead stack |
| `register` | is accepted by both and leaves a different value in a compared register |
| `device-writes` | is accepted by both and gives different writes to a device range |
| `calls` | is accepted by both and gives different calls to stubbed callees |
| `unmatched-path` | takes a path on one side that no path on the other side shares |

Verdicts:

| Verdict | Meaning |
| --- | --- |
| `proven` | no input shows a difference, and every path of both returned |
| `failed` | the solver found an input; `differences` has it |
| `incomplete` | no input found, but a path stopped at the loop bound or with an error, a path was lost at a jump or call whose target can be anywhere, a read or write went through an address with more possible values than are followed, a write through an address that depends on the input can reach a device range, or the time ran out |

`proven` holds within the assumptions the report lists.  The compared
registers are r0, r4 to r11 and sp unless `--compare-registers` says
otherwise; for a function that returns nothing, leave r0 out.

`--arg-suffix I:HEX` puts fixed bytes after the symbolic bytes of buffer I.

What a stub takes and gives back.  `--stub-args NAME:N` compares the first N
argument registers.  A suite that has the IR the code was compiled from
gives the layout instead (`declaration_from_ir`): the registers the AAPCS
passes each declared parameter in (a 64-bit integer is an even-numbered
pair, so `f(int, long long)` is r0, r2 and r3) and the registers of the
result (r0, or r0 and r1 for a 64-bit integer).  A called stub with a
parameter or result that is not placed in core registers (floating point, an
aggregate, a stack argument, the variable part of a variadic list) makes the
run `incomplete`.  The same IR says what the function itself returns: r0 is
left out of the comparison for void, r1 is added for a 64-bit integer, and
any other type is compared in r0 and makes the run `incomplete`.

A stubbed callee that is one of the read-only string functions (`strncmp`,
`strncasecmp`, `memcmp`, `strcmp`, `strcasecmp`, `strlen`, `strchr`) is
recorded like any other stub, but its result is computed from the bytes it
is given when the path fixes its pointers and its length; `--no-models`
turns that off.  When one call is computed on some paths and left unknown on
others, the differences found are reported as `unconfirmed` and the run is
`incomplete`, because an unknown result can separate two paths that agree.

`--declaration FILE` takes all of the above from a JSON object, as a suite's
case writes it (`DECLARED` lists the keys).

Not handled: arguments on the stack, floating point, images other than
32-bit ARM, a device write through an address that depends on the input, and
imports of a dynamically linked image (angr gives each a fresh result per
side, which shows up as a difference; stub them).  Paths are not merged, so
a function with many branches is slow.  The time limit is checked between
solver calls, not inside one.

Needs its own environment, because cozy pins an older angr than the rest of
this directory uses:

    uv run --python python3.11 --with cozy-re==1.5.4 --with 'bitstring<5' \\
        --with 'pycparser<3' patch_symdiff.py ...

Where unicorn is built from source with CMake 4, the first run also needs
CMAKE_POLICY_VERSION_MINIMUM=3.5 in the environment.

CLI: patch_symdiff.py --original ELF --rewritten ELF
         (--function NAME | --address ADDR --mode arm|thumb)
         [--args N] [--arg-buffer I:SIZE]... [--arg-suffix I:HEX]...
         [--arg-range I:LO:HI]... [--declaration FILE]
         [--reject-return VALUE]... [--reject-handler NAME_OR_ADDR]...
         [--reject-skip NAME_OR_ADDR]...
         [--stub NAME_OR_ADDR]... [--stub-pointer NAME_OR_ADDR:SIZE]...
         [--stub-args NAME_OR_ADDR:N]...
         [--mmio-region LO:HI]... [--reject-write LO:HI]...
         [--compare-registers LIST] [--loop-bound N] [--max-seconds S]
         [--output report.json] [--text report.txt] [--graph graph.json]
Exit 0 when proven, 1 when failed, 3 when incomplete, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

WORD = 32
BUFFER_BASE = 0x60000000
BUFFER_STEP = 0x10000
RESULT_BASE = 0x70000000  # what a `--stub-pointer` callee returns pointers into
RESULT_STEP = 0x10000
RESULT_LIMIT = 0x800
MODEL_BYTES = 64  # the longest string or count a modelled function is computed for
ADDRESS_LIMIT = 256  # addresses a read or write through a symbolic address is spread over
GUARDED_LIMIT = 4096  # pairs of conditional device writes or calls compared exactly
SUSPECT_LIMIT = 8  # positions listed for a difference in writes or calls
SAVED_REGISTERS = ("r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11")
DEFAULT_REGISTERS = ("r0", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "sp")
ARGUMENT_REGISTERS = ("r0", "r1", "r2", "r3")
EXIT = {"proven": 0, "failed": 1, "incomplete": 3}
KINDS = ("accepts-rejected-input", "memory", "register", "device-writes", "calls",
         "unmatched-path")


@dataclass
class Settings:
    args: int = 4
    arg_buffers: dict[int, int] = field(default_factory=dict)
    arg_ranges: dict[int, tuple[int, int]] = field(default_factory=dict)
    arg_suffixes: dict[int, bytes] = field(default_factory=dict)  # bytes after the buffer
    reject_skips: list[str | int] = field(default_factory=list)
    # Per stub, when the IR gives them: the registers its arguments are in,
    # the registers its result comes back in, and what of either is not
    # placed in core registers.
    stub_registers: dict[str | int, tuple[str, ...]] = field(default_factory=dict)
    stub_returns: dict[str | int, tuple[str, ...]] = field(default_factory=dict)
    stub_notes: dict[str | int, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)  # what makes the run incomplete as given
    models: bool = True  # compute the result of a stubbed string function where possible
    reject_returns: list[int] = field(default_factory=list)
    reject_handlers: list[str | int] = field(default_factory=list)
    stubs: list[str | int] = field(default_factory=list)
    stub_pointers: dict[str | int, int] = field(default_factory=dict)
    stub_args: dict[str | int, int] = field(default_factory=dict)
    mmio_regions: list[tuple[int, int]] = field(default_factory=list)
    reject_writes: list[tuple[int, int]] = field(default_factory=list)
    registers: tuple[str, ...] = DEFAULT_REGISTERS
    loop_bound: int | None = None
    max_seconds: float = 300.0

    def check(self) -> None:
        # Leaving out a call can only be seen for a callee that is not run.
        for name in self.reject_skips:
            if name not in self.stubs and name not in self.stub_pointers:
                self.stubs.append(name)
        for index, suffix in self.arg_suffixes.items():
            if index not in self.arg_buffers:
                raise UsageError(f"--arg-suffix {index}: the argument is not a buffer")
            if self.arg_buffers[index] + len(suffix) > BUFFER_STEP:
                raise UsageError(f"--arg-suffix {index}: buffer and suffix are over "
                                 f"{BUFFER_STEP} bytes")
        for table in (self.stub_registers, self.stub_returns, self.stub_notes):
            for stub in table:
                if stub not in self.stubs and stub not in self.stub_pointers:
                    raise UsageError(f"{label_of(stub)}: a layout for a callee that is "
                                     "not stubbed")
        if not 0 <= self.args <= 4:
            raise UsageError("--args must be 0 to 4 (arguments on the stack are not handled)")
        for index, size in self.arg_buffers.items():
            if not 0 <= index < self.args:
                raise UsageError(f"--arg-buffer {index}: the function has {self.args} argument(s)")
            if not 0 < size <= BUFFER_STEP:
                raise UsageError(f"--arg-buffer {index}: size must be 1 to {BUFFER_STEP}")
        for index, (low, high) in self.arg_ranges.items():
            if not 0 <= index < self.args:
                raise UsageError(f"--arg-range {index}: the function has {self.args} argument(s)")
            if index in self.arg_buffers:
                raise UsageError(f"--arg-range {index}: the argument is a buffer")
            if low > high:
                raise UsageError(f"--arg-range {index}: {low:#x} is above {high:#x}")
        for low, high in self.mmio_regions:
            if low > high:
                raise UsageError(f"--mmio-region: {low:#x} is above {high:#x}")
        for low, high in self.reject_writes:
            if low > high:
                raise UsageError(f"--reject-write: {low:#x} is above {high:#x}")
        for stub, size in self.stub_pointers.items():
            if stub in self.stubs:
                raise UsageError(f"--stub-pointer {stub}: also given as --stub")
            if not 0 < size <= RESULT_STEP:
                raise UsageError(f"--stub-pointer {stub}: size must be 1 to {RESULT_STEP}")
        for stub, count in self.stub_args.items():
            if stub not in self.stubs and stub not in self.stub_pointers:
                raise UsageError(f"--stub-args {stub}: not a stubbed callee")
            if not 0 <= count <= 4:
                raise UsageError(f"--stub-args {stub}: 0 to 4 register arguments")
        if self.loop_bound is not None and self.loop_bound < 0:
            raise UsageError("--loop-bound must not be negative")


class UsageError(ValueError):
    """Something wrong with what the caller asked for, not with the comparison."""


class OutOfTime(BaseException):
    """Raised by the alarm.  Not an Exception, so the engine cannot swallow it."""


@contextlib.contextmanager
def time_limit(seconds: float):
    if not seconds or not hasattr(signal, "SIGALRM"):
        yield
        return

    def expire(signum, frame):
        raise OutOfTime()

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def verdict_of(differences: list, incomplete: list) -> str:
    if differences:
        return "failed"
    return "incomplete" if incomplete else "proven"


def number(text: str) -> int:
    return int(text, 0)


def indexed_size(text: str) -> tuple[int, int]:
    index, size = text.split(":")
    return number(index), number(size)


def indexed_range(text: str) -> tuple[int, tuple[int, int]]:
    index, low, high = text.split(":")
    return number(index), (number(low), number(high))


def address_range(text: str) -> tuple[int, int]:
    low, high = text.split(":")
    return number(low), number(high)


def stub_name(text: str) -> str | int:
    try:
        return number(text)
    except ValueError:
        return text


def stub_size(text: str) -> tuple[str | int, int]:
    name, size = text.rsplit(":", 1)
    return stub_name(name), number(size)


def label_of(stub: str | int) -> str:
    return stub if isinstance(stub, str) else hex(stub)


def skipped(rewritten: list, original: list, names: set) -> bool:
    """Whether the calls `rewritten` are `original` without some calls of
    `names`, and fewer."""
    if not names or len(rewritten) >= len(original):
        return False
    at = 0
    for callee in original:
        if at < len(rewritten) and rewritten[at] == callee:
            at += 1
        elif callee not in names:
            return False
    return at == len(rewritten)


def guarded_difference(claripy, first: list, second: list):
    """The condition under which two sequences of events differ, when an
    event may be conditional: a store the compiler made conditional happens
    on some inputs of its path only.

    An event is (label, value, guard); the guard is a condition or None for
    an event that always happens.  The sequences are equal on an input when
    the events that happen on it are the same in number and order, each with
    the same label and value.
    """
    true, false = claripy.BoolV(True), claripy.BoolV(False)

    def happens(event):
        return true if event[2] is None else event[2]

    def same(one, other):
        if one[0] != other[0] or one[1].length != other[1].length:
            return false
        return true if one[1] is other[1] else one[1] == other[1]

    # equal[i][j]: the events from first[i] and second[j] onward agree.
    equal = [[None] * (len(second) + 1) for _ in range(len(first) + 1)]
    for i in range(len(first), -1, -1):
        for j in range(len(second), -1, -1):
            if i == len(first) and j == len(second):
                equal[i][j] = true
            elif i == len(first):
                equal[i][j] = claripy.And(claripy.Not(happens(second[j])), equal[i][j + 1])
            elif j == len(second):
                equal[i][j] = claripy.And(claripy.Not(happens(first[i])), equal[i + 1][j])
            else:
                equal[i][j] = claripy.If(
                    claripy.Not(happens(first[i])), equal[i + 1][j],
                    claripy.If(claripy.Not(happens(second[j])), equal[i][j + 1],
                               claripy.And(same(first[i], second[j]), equal[i + 1][j + 1])))
    return claripy.Not(equal[0][0])


def declared_functions(text: str, keyword: str = "declare") -> dict[str, tuple[str, list[str]]]:
    """Per function an LLVM IR text declares (or, with `define`, defines): its
    return type and its parameters, as written."""
    closing = {"(": ")", "[": "]", "{": "}", "<": ">"}
    found_functions = {}
    for found in re.finditer(rf"^{keyword}\s+(.*?)@([\w.$]+)\(", text, re.MULTILINE):
        parameters, nesting, start = [], [], found.end()
        for position in range(found.end(), len(text)):
            character = text[position]
            if character in closing:
                nesting.append(closing[character])
            elif nesting and character == nesting[-1]:
                nesting.pop()
            elif not nesting and character in ",)":
                parameters.append(text[start:position].strip())
                start = position + 1
                if character == ")":
                    break
        head = found.group(1).strip()
        returned = head.split()[-1] if head else ""
        # An aggregate or vector type has spaces in it and ends with its bracket.
        for opening, close in (("{", "}"), ("[", "]"), ("<", ">")):
            if head.endswith(close) and opening in head:
                returned = head[head.rindex(opening):]
        found_functions[found.group(2)] = (returned, [p for p in parameters if p])
    return found_functions


def argument_layout(parameters: list[str]) -> tuple[tuple, str]:
    """The core registers the AAPCS passes `parameters` in, in order, and the
    first parameter that is not placed in them ("" when all are).

    A pointer and an integer of up to 32 bits take the next register.  An
    integer of up to 64 bits takes two, starting at an even one, so a register
    can be skipped.  Placement stops at a parameter that goes on the stack or
    whose place depends on more than its type here: floating point, an
    aggregate, a vector, a wider integer, a pointer passed by value, and the
    variable part of a variadic list, whose length and types a declaration
    does not give.
    """
    registers: list[str] = []
    following = 0
    for parameter in parameters:
        if parameter == "...":
            return tuple(registers), "... (variable arguments)"
        words = parameter.split()
        integer = re.fullmatch(r"i(\d+)", words[0])
        if words[0] == "ptr" and not any(
                word.startswith(("byval", "inalloca", "preallocated")) for word in words):
            width = 1
        elif integer and int(integer.group(1)) <= 32:
            width = 1
        elif integer and int(integer.group(1)) <= 64:
            width = 2
            following += following % 2
        else:
            return tuple(registers), parameter
        if following + width > len(ARGUMENT_REGISTERS):
            return tuple(registers), f"{parameter} (on the stack)"
        registers.extend(ARGUMENT_REGISTERS[following:following + width])
        following += width
    return tuple(registers), ""


def return_layout(returned: str) -> tuple[tuple, str]:
    """The registers that hold a result of LLVM type `returned`, least
    significant first, and the type again when they do not hold all of it:
    nothing for void, r0 for a pointer or an integer of up to 32 bits, r0 and
    r1 for an integer of up to 64 bits, and r0 with the type reported for
    anything else."""
    if returned == "void":
        return (), ""
    integer = re.fullmatch(r"i(\d+)", returned)
    if returned == "ptr" or (integer and int(integer.group(1)) <= 32):
        return ("r0",), ""
    if integer and int(integer.group(1)) <= 64:
        return ("r0", "r1"), ""
    return ("r0",), returned


def string_models(claripy) -> dict:
    """Functions that read memory and write none: name -> (argument count,
    model).  A model returns None when it cannot compute the result on a
    path, and the call is then a stub like any other."""

    def fixed(state, value):
        # The value when the path fixes it.  A value built from conditions is
        # an expression even when one value is left, so the solver is asked.
        if not state.solver.symbolic(value):
            return state.solver.eval(value)
        found = state.solver.eval_upto(value, 2)
        return found[0] if len(found) == 1 else None

    def bytes_at(state, address, count):
        return [state.memory.load(address + index, 1) for index in range(count)]

    def string_at(state, address):
        # Up to and including the first byte that is zero on every input.
        found = []
        for byte in bytes_at(state, address, MODEL_BYTES):
            found.append(byte)
            if fixed(state, byte) == 0:
                return found
        return None

    def lower(byte):
        return claripy.If(claripy.And(claripy.UGE(byte, 0x41), claripy.ULE(byte, 0x5A)),
                          byte + 0x20, byte)

    def difference(first, second, fold, strings):
        # The difference of the first pair of bytes that differ, zero when
        # none does (or, for strings, when both end first).
        result = claripy.BVV(0, WORD)
        for one, other in reversed(list(zip(first, second))):
            if fold:
                one, other = lower(one), lower(other)
            rest = claripy.If(one == 0, claripy.BVV(0, WORD), result) if strings else result
            result = claripy.If(one != other, one.zero_extend(24) - other.zero_extend(24), rest)
        return result

    def counted(fold, strings):
        def model(state, first, second, count):
            first, second, count = (fixed(state, value) for value in (first, second, count))
            if first is None or second is None or count is None or count > MODEL_BYTES:
                return None
            return difference(bytes_at(state, first, count), bytes_at(state, second, count),
                              fold, strings)
        return model

    def whole(fold):
        def model(state, first, second):
            first, second = fixed(state, first), fixed(state, second)
            if first is None or second is None:
                return None
            ends = [len(found) for found in (string_at(state, first), string_at(state, second))
                    if found is not None]
            if not ends:
                return None
            return difference(bytes_at(state, first, min(ends)),
                              bytes_at(state, second, min(ends)), fold, True)
        return model

    def strlen(state, address):
        address = fixed(state, address)
        found = string_at(state, address) if address is not None else None
        if found is None:
            return None
        result = claripy.BVV(len(found) - 1, WORD)
        for index in reversed(range(len(found) - 1)):
            result = claripy.If(found[index] == 0, claripy.BVV(index, WORD), result)
        return result

    def strchr(state, address, wanted):
        address = fixed(state, address)
        found = string_at(state, address) if address is not None else None
        if found is None:
            return None
        wanted = wanted[7:0]
        last = len(found) - 1
        result = claripy.If(wanted == 0, claripy.BVV(address + last, WORD), claripy.BVV(0, WORD))
        for index in reversed(range(last)):
            result = claripy.If(found[index] == wanted, claripy.BVV(address + index, WORD),
                                claripy.If(found[index] == 0, claripy.BVV(0, WORD), result))
        return result

    return {
        "strncmp": (3, counted(fold=False, strings=True)),
        "strncasecmp": (3, counted(fold=True, strings=True)),
        "memcmp": (3, counted(fold=False, strings=False)),
        "strcmp": (2, whole(fold=False)),
        "strcasecmp": (2, whole(fold=True)),
        "strlen": (1, strlen),
        "strchr": (2, strchr),
    }


DECLARED = {"models", "args", "arg_buffers", "arg_suffixes", "arg_ranges", "reject_returns",
            "reject_handlers", "reject_skips", "reject_writes", "mmio_regions", "stubs",
            "stub_pointers", "stub_args", "stub_registers", "stub_returns", "stub_notes",
            "registers", "notes", "loop_bound", "max_seconds"}


def declaration_from_ir(declared: dict, ir_text: str, function: str,
                        stubs: dict | None = None) -> dict:
    """`declared` with what the IR the code was compiled from says added.

    `function` is the function under comparison; its return type decides
    whether r0 and r1 are compared.  `stubs` maps the IR name of a callee to
    stub to how the image names it (a symbol or an address).  The callees
    of `reject_skips` are stubs too.  Each stub the IR declares gets its
    argument and result registers.
    """
    declared = json.loads(json.dumps(declared))  # a copy, with JSON's types
    notes = declared.setdefault("notes", [])
    defined = declared_functions(ir_text, "define").get(function)
    if defined is not None and "registers" not in declared:
        registers, unsupported = return_layout(defined[0])
        declared["registers"] = [*registers, *[name for name in DEFAULT_REGISTERS
                                                if name != "r0"]]
        if unsupported:
            notes.append(f"the return type {unsupported} is not compared in full: only r0 is")
    wanted = dict(stubs or {})
    for name in declared.get("reject_skips", []):
        wanted.setdefault(name, name)
    prototypes = declared_functions(ir_text)
    listed = declared.setdefault("stubs", [])
    for name, key in wanted.items():
        if key not in listed and str(key) not in declared.get("stub_pointers", {}):
            listed.append(key)
        if name not in prototypes:
            continue
        returned, parameters = prototypes[name]
        registers, unplaced = argument_layout(parameters)
        result, unsupported = return_layout(returned)
        label = label_of(key)
        declared.setdefault("stub_registers", {})[label] = list(registers)
        declared.setdefault("stub_returns", {})[label] = list(result)
        problems = ([f"its parameter {unplaced} is not compared"] if unplaced else []) \
            + ([f"its result of type {unsupported} is not modelled in full"]
               if unsupported else [])
        if problems:
            declared.setdefault("stub_notes", {})[label] = "; ".join(problems)
    return declared


def stubs_from_bindings(bindings: dict, function: str, followed=()) -> dict:
    """The callees to stub for a function of an image that is not built here:
    every function the lifted code is bound to, other than itself and the
    ones in `followed`.  An import is named by its symbol, which the loader
    resolves; any other function by its address, since the image may have no
    symbol for it.  Returns IR name -> symbol or address.
    """
    stubs = {}
    for row in bindings["bindings"]:
        if row["kind"] != "function" or row["name"] == function or row["name"] in followed:
            continue
        if row.get("origin") == "imported":
            stubs[row["name"]] = row.get("symbol", row["name"])
        elif row.get("address"):
            stubs[row["name"]] = int(row["address"], 16)
    return stubs


def settings_from_declaration(declared: dict) -> Settings:
    """The settings a suite's case declares, as JSON: argument indexes and
    stubs are keys, suffixes are hex strings, a stub is a name or a number."""
    unknown = sorted(set(declared) - DECLARED)
    if unknown:
        raise UsageError(f"unknown comparison settings: {', '.join(unknown)}")

    def stub(key):
        return stub_name(key) if isinstance(key, str) else key

    def by_stub(name, convert):
        return {stub(key): convert(value) for key, value in declared.get(name, {}).items()}

    settings = Settings(
        arg_buffers={int(index): int(size)
                     for index, size in declared.get("arg_buffers", {}).items()},
        arg_suffixes={int(index): bytes.fromhex(text)
                      for index, text in declared.get("arg_suffixes", {}).items()},
        arg_ranges={int(index): (int(low), int(high))
                    for index, (low, high) in declared.get("arg_ranges", {}).items()},
        reject_returns=[int(value) for value in declared.get("reject_returns", [])],
        reject_handlers=[stub(key) for key in declared.get("reject_handlers", [])],
        reject_skips=[stub(key) for key in declared.get("reject_skips", [])],
        reject_writes=[(int(low), int(high)) for low, high in declared.get("reject_writes", [])],
        mmio_regions=[(int(low), int(high)) for low, high in declared.get("mmio_regions", [])],
        stubs=[stub(key) for key in declared.get("stubs", [])],
        stub_pointers=by_stub("stub_pointers", int),
        stub_args=by_stub("stub_args", int),
        stub_registers=by_stub("stub_registers", tuple),
        stub_returns=by_stub("stub_returns", tuple),
        stub_notes=by_stub("stub_notes", str),
        notes=list(declared.get("notes", [])),
    )
    if "registers" in declared:
        settings.registers = tuple(declared["registers"])
    for name in ("args", "loop_bound", "max_seconds", "models"):
        if name in declared:
            setattr(settings, name, declared[name])
    return settings


def own_environment() -> list[str]:
    """The command that runs Python with cozy.  cozy pins an older angr than
    the rest of this directory uses, so it cannot share an interpreter with
    the suites.  PATCHESTRY_SYMDIFF_PYTHON names an interpreter that has it;
    otherwise uv builds the environment of scripts/symdiff/requirements.txt.
    """
    interpreter = os.environ.get("PATCHESTRY_SYMDIFF_PYTHON")
    if interpreter:
        return [interpreter]
    if shutil.which("uv") is None:
        raise RuntimeError("the binary comparison needs cozy: set PATCHESTRY_SYMDIFF_PYTHON "
                           "to an interpreter that has it, or install uv")
    requirements = Path(__file__).resolve().parents[1] / "symdiff" / "requirements.txt"
    return ["uv", "run", "--no-project", "--python", "3.11",
            "--with-requirements", str(requirements), "python"]


def compare_in_own_environment(original: Path, rewritten: Path, entry: int, thumb: bool,
                               declared: dict, output: Path) -> dict:
    """Run this script in the environment that has cozy and return its
    report, which is also written to `output`."""
    settings_from_declaration(declared).check()  # a usage error before the engine loads
    declaration = output.with_suffix(".declaration.json")
    declaration.write_text(json.dumps(declared, indent=2) + "\n", encoding="utf-8")
    output.unlink(missing_ok=True)
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("PYTHONPATH", "VIRTUAL_ENV")}
    # unicorn's source build, where there is no wheel, needs this with CMake 4.
    environment.setdefault("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
    done = subprocess.run(
        [*own_environment(), str(Path(__file__).resolve()),
         "--original", str(original), "--rewritten", str(rewritten),
         "--address", hex(entry & ~1), "--mode", "thumb" if thumb else "arm",
         "--declaration", str(declaration), "--output", str(output)],
        env=environment, capture_output=True, text=True, check=False)
    if done.returncode not in EXIT.values() or not output.is_file():
        raise RuntimeError(f"patch_symdiff.py exited with {done.returncode}:\n"
                           f"{done.stdout}{done.stderr}")
    return json.loads(output.read_text(encoding="utf-8"))


def kinds_of(report: dict) -> list[str]:
    """The kinds of difference in a report, each once, in a fixed order."""
    found = {difference["kind"] for difference in report["differences"]}
    return [kind for kind in KINDS if kind in found]


def find_function(image: Path, name: str) -> tuple[int, bool]:
    """The address of `name` in the symbol table of `image`, and whether it is Thumb."""
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection

    with image.open("rb") as handle:
        for section in ELFFile(handle).iter_sections():
            if not isinstance(section, SymbolTableSection):
                continue
            for symbol in section.get_symbol_by_name(name) or ():
                if symbol["st_info"]["type"] == "STT_FUNC":
                    return symbol["st_value"] & ~1, bool(symbol["st_value"] & 1)
    raise UsageError(f"{image} has no function symbol {name}")


def compare(original: Path, rewritten: Path, entry: int, thumb: bool, settings: Settings,
            text: Path | None = None, graph: Path | None = None) -> dict:
    """Run both sides and return the report.  `text` gets cozy's own report,
    `graph` the execution graphs for cozy's viewer."""
    settings.check()

    import logging

    import angr
    import claripy
    import cozy
    from cozy import analysis, execution_graph, side_effect
    from cozy.project import Project
    from cozy.terminal_state import DeadendedState
    from importlib.metadata import version

    cozy.log.set_level(logging.WARNING)
    for noisy in ("angr", "cle", "pyvex", "claripy"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    class SharedResult(angr.SimProcedure):
        """Stands for a callee: the same symbol on both sides for the same call."""

        # Not `returns`: that is SimProcedure's own word for "comes back".
        def run(self, label=None, size=0, taken=(), given=("r0",)):
            called.add(label)
            # What the caller passed.  Its frame is everything from its
            # stack pointer up to where the function under test started.
            passed = []
            for name in taken:
                value = getattr(self.state.regs, name)
                if value.concrete and self.state.regs.sp.concrete and \
                        self.state.regs.sp.concrete_value <= value.concrete_value < stack_top[0]:
                    value = frame_address
                elif value.concrete:
                    value = constant_text(self.project, self.state, value)
                passed.append(value)
            side_effect.perform(self.state, "calls",
                                {"value": claripy.Concat(claripy.BVV(0, 1), *passed)}, label=label)
            calls = dict(self.state.globals["symdiff_calls"]) \
                if "symdiff_calls" in self.state.globals else {}
            index = calls.get(label, 0)
            calls[label] = index + 1
            self.state.globals["symdiff_calls"] = calls
            if label in models:
                count, model = models[label]
                result = model(self.state, *[getattr(self.state.regs, name)
                                             for name in ARGUMENT_REGISTERS[:count]])
                (unknown if result is None else computed).add((label, index))
                if result is not None:
                    return result
            if "r1" in given:  # the upper half of a 64-bit result
                self.state.regs.r1 = claripy.BVS(f"{label}#{index}.r1", WORD,
                                                 explicit_name=True)
            if not size:
                return claripy.BVS(f"{label}#{index}", WORD, explicit_name=True)
            # The same call gets the same block on both sides, whichever
            # side makes it first.
            address = results.setdefault(
                (label, index), RESULT_BASE + len(results) * RESULT_STEP)
            if len(results) > RESULT_LIMIT:
                raise angr.errors.SimProcedureError("too many calls of --stub-pointer callees")
            self.state.memory.store(address, claripy.BVS(
                f"{label}#{index}.bytes", size * 8, explicit_name=True))
            return claripy.BVV(address, WORD)

    class CountedAddresses(
            angr.concretization_strategies.SimConcretizationStrategyUnlimitedRange):
        """Every address an access can take, up to ADDRESS_LIMIT, and a note
        when there are more."""

        def __init__(self, kind):
            super().__init__(ADDRESS_LIMIT)
            self.kind = kind

        def _concretize(self, memory, addr, **kwargs):
            found = self._eval(memory, addr, ADDRESS_LIMIT + 1, **kwargs)
            if len(found) > ADDRESS_LIMIT:
                unrecorded.add(f"a {self.kind} through an address with more than "
                               f"{ADDRESS_LIMIT} possible values: the inputs that lead to "
                               "the others were not explored")
                found = found[:ADDRESS_LIMIT]
            return found

    def constant_text(project, state, value):
        """`value`, or a number for the bytes it points to in read-only memory."""
        segment = project.loader.find_segment_containing(value.concrete_value)
        if segment is None or segment.is_writable:
            return value
        text = bytearray()
        while len(text) < 256:
            byte = state.memory.load(value.concrete_value + len(text), 1)
            if not byte.concrete or byte.concrete_value == 0:
                break
            text.append(byte.concrete_value)
        if not text:
            return value
        return claripy.BVV(zlib.crc32(bytes(text)), WORD)

    class Reject(angr.SimProcedure):
        """Stands for a reject handler: the path ends here, marked as rejected."""

        NO_RET = True

        def run(self):
            self.state.globals["symdiff_rejected"] = True
            self.exit(0)

    def device_write(state):
        address = state.inspect.mem_write_address
        if isinstance(address, int):
            address = claripy.BVV(address, WORD)
        if address is None:
            return
        if not address.concrete:
            # Where the write goes depends on the input.  It is not recorded,
            # so when it can land in a range that is, the record is short.
            for kind, regions in (("device", settings.mmio_regions),
                                  ("reject-write", settings.reject_writes)):
                for low, high in regions:
                    if state.solver.satisfiable(extra_constraints=[
                            claripy.UGE(address, low), claripy.ULE(address, high)]):
                        unrecorded.add(
                            f"a write through an address that depends on the input can reach "
                            f"the {kind} range {low:#x}:{high:#x} and is not compared")
            return
        if any(low <= address.concrete_value <= high for low, high in settings.mmio_regions):
            body = {"value": state.inspect.mem_write_expr}
            # A store the compiler made conditional happens on some inputs of
            # the path only.  One bit, since a side effect holds values.
            if state.inspect.mem_write_condition is not None:
                body["guard"] = claripy.If(state.inspect.mem_write_condition,
                                           claripy.BVV(1, 1), claripy.BVV(0, 1))
            side_effect.perform(state, "device-writes", body,
                                label=hex(address.concrete_value))
        # A write that faults on the target (a deliberate store to address 0
        # is how a check with no handler to call refuses its input).  The
        # path is followed to its end and counts as a rejection.
        if any(low <= address.concrete_value <= high for low, high in settings.reject_writes):
            # A store the compiler made conditional rejects only the inputs
            # that take it.
            condition = state.inspect.mem_write_condition
            if condition is None:
                state.globals["symdiff_rejected"] = True
            else:
                state.globals["symdiff_rejected_when"] = claripy.Or(
                    state.globals.get("symdiff_rejected_when", claripy.BoolV(False)), condition)

    results: dict[tuple[str, int], int] = {}
    called: set[str] = set()  # labels of the stubs some path called
    unrecorded: set[str] = set()  # writes that were made and not recorded
    models = string_models(claripy) if settings.models else {}
    # Calls of modelled functions, as (callee, call index): where the result
    # was computed and where it was left unknown.
    computed: set[tuple] = set()
    unknown: set[tuple] = set()
    unconfirmed: list[dict] = []
    skip_names = {label_of(stub) for stub in settings.reject_skips}

    def layout(stub) -> dict:
        return {"taken": settings.stub_registers.get(
                    stub, ARGUMENT_REGISTERS[:settings.stub_args.get(stub, 0)]),
                "given": settings.stub_returns.get(stub, ("r0",))}

    frame_address = claripy.BVS("frame-address", WORD, explicit_name=True)
    stack_top = [0]
    symbols = {}
    for index in range(settings.args):
        if index in settings.arg_buffers:
            symbols[f"buf{index}"] = claripy.BVS(
                f"buf{index}", settings.arg_buffers[index] * 8, explicit_name=True)
        else:
            symbols[f"arg{index}"] = claripy.BVS(f"arg{index}", WORD, explicit_name=True)

    saved = {name: claripy.BVS(f"entry_{name}", WORD, explicit_name=True)
             for name in SAVED_REGISTERS}

    def run(image: Path):
        project = Project(str(image), auto_load_libs=False)
        if project.arch.name not in ("ARMEL", "ARMHF", "ARMCortexM"):
            raise UsageError(f"{image}: {project.arch.name} is not 32-bit ARM")
        for stub in [*settings.stubs, *settings.stub_pointers, *settings.reject_handlers]:
            if isinstance(stub, str) and project.try_find_symbol_addr(stub) is None:
                raise UsageError(f"{image} has no symbol {stub}")
        # Not cozy's hook_symbol: it drops the keyword arguments, and every
        # stub would then share one name.
        for stub in settings.stubs:
            project.angr_proj.hook_symbol(stub, SharedResult(
                project=project.angr_proj, label=label_of(stub), **layout(stub)), replace=True)
        for stub, size in settings.stub_pointers.items():
            project.angr_proj.hook_symbol(stub, SharedResult(
                project=project.angr_proj, label=label_of(stub), size=size,
                **layout(stub)), replace=True)
        for handler in settings.reject_handlers:
            project.hook_symbol(handler, Reject, replace=True)
        session = project.session(entry | 1 if thumb else entry)
        # cozy spreads an access through a symbolic address over the first
        # 128 addresses it can take and says nothing when there are more.
        for kind, strategies in (("read", session.state.memory.read_strategies),
                                 ("write", session.state.memory.write_strategies)):
            strategies[0] = CountedAddresses(kind)
        stack_top[0] = session.state.regs.sp.concrete_value + 1
        if settings.mmio_regions or settings.reject_writes:
            session.state.inspect.b("mem_write", when=angr.BP_AFTER, action=device_write)
        for name, symbol in saved.items():
            setattr(session.state.regs, name, symbol)
        arguments = []
        for index in range(settings.args):
            if index in settings.arg_buffers:
                address = BUFFER_BASE + index * BUFFER_STEP
                session.store(address, symbols[f"buf{index}"])
                if settings.arg_suffixes.get(index):
                    session.store(address + settings.arg_buffers[index],
                                  claripy.BVV(settings.arg_suffixes[index]))
                arguments.append(claripy.BVV(address, WORD))
            else:
                arguments.append(symbols[f"arg{index}"])
        for index, (low, high) in settings.arg_ranges.items():
            session.add_constraints(claripy.UGE(symbols[f"arg{index}"], low),
                                    claripy.ULE(symbols[f"arg{index}"], high))
        # A jump or call whose target can be anywhere (a call through a
        # pointer taken from the input) has no successor to follow.  angr
        # drops such a path unless asked to keep it, and cozy does not ask.
        factory = project.angr_proj.factory
        make_manager, managers = factory.simulation_manager, []

        def keeping_lost_paths(*positional, **named):
            named.setdefault("save_unconstrained", True)
            managers.append(make_manager(*positional, **named))
            return managers[-1]

        factory.simulation_manager = keeping_lost_paths
        result = session.run(arguments, cache_intermediate_info=graph is not None,
                             loop_bound=settings.loop_bound)
        lost[len(lost_sides)] = sum(len(manager.stashes.get("unconstrained", ()))
                                    for manager in managers)
        lost_sides.append(image)
        return project, result, arguments

    lost: dict[int, int] = {}  # per run, in order: paths with nowhere to go
    lost_sides: list = []
    differences: list[dict] = []
    incomplete: list[str] = list(settings.notes)
    states: dict = {}
    rejections = 0
    names = list(symbols)

    def ask(solver, condition, values=()):
        """None when no input satisfies `condition`; otherwise that input and
        the `values` under it."""
        try:
            if not solver.satisfiable(extra_constraints=[condition]):
                return None
            model = solver.batch_eval([symbols[name] for name in names] + list(values), 1,
                                      extra_constraints=[condition])[0]
        except claripy.errors.ClaripyError as error:
            incomplete.append(f"the solver gave no answer for one comparison: {error}")
            return None
        found = {}
        for name, value in zip(names, model):
            width = symbols[name].length // 4
            found[name] = f"{value:0{width}x}" if name.startswith("buf") else hex(value)
        return found, model[len(names):]

    started = time.time()
    stage = "running the original"
    # claripy walks expressions recursively, and a read spread over
    # ADDRESS_LIMIT addresses is that many levels deep.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    try:
        with time_limit(settings.max_seconds):
            original_project, original_result, arguments = run(original)
            stage = "running the rewritten function"
            rewritten_project, rewritten_result, _ = run(rewritten)

            for run_index, (side, result) in enumerate((("original", original_result),
                                                        ("rewritten", rewritten_result))):
                states[side] = {"returned": len(result.deadended),
                                "errored": len(result.errored),
                                "loop-bound": len(result.spinning),
                                "lost": lost.get(run_index, 0)}
                if lost.get(run_index):
                    incomplete.append(f"{side}: {lost[run_index]} path(s) ended at a jump or "
                                      "call whose target can be anywhere and were not followed")
                for state in result.errored:
                    incomplete.append(f"{side}: a path stopped with an error: {state.error}")
                if result.spinning:
                    incomplete.append(f"{side}: {len(result.spinning)} path(s) stopped at the "
                                      f"loop bound of {settings.loop_bound}")
                if not result.deadended:
                    incomplete.append(f"{side}: no path returned")

            stage = "pairing the final states"
            comparison = analysis.Comparison(original_result, rewritten_result)
            states["pairs"] = len(comparison.pairs)
            states["unmatched"] = {"original": len(comparison.orphans_left),
                                   "rewritten": len(comparison.orphans_right)}

            stage = "checking the differences"
            true = claripy.BoolV(True)
            for pair in comparison:
                left, right = pair.state_left, pair.state_right
                if not (isinstance(left, DeadendedState) and isinstance(right, DeadendedState)):
                    continue  # counted above, per side
                solver = claripy.Solver()
                solver.add(left.state.solver.constraints)
                solver.add(right.state.solver.constraints)
                # The value returned: r0, with r1 above it when the result is
                # 64 bits wide.  A reject value that does not fit is never
                # returned.
                results = tuple(
                    claripy.Concat(side.state.regs.r1, side.state.regs.r0)
                    if "r1" in settings.registers else side.state.regs.r0
                    for side in (left, right))
                # A negative reject value is its bit pattern at this width.
                width = results[0].length
                rejects = [reject % (1 << width) for reject in settings.reject_returns
                           if -(1 << (width - 1)) <= reject < 1 << width]
                accepts = [claripy.BoolV(False) if "symdiff_rejected" in side.state.globals
                           else claripy.And(
                               claripy.Not(side.state.globals.get("symdiff_rejected_when",
                                                                  claripy.BoolV(False))),
                               *(value != claripy.BVV(reject, value.length)
                                 for reject in rejects))
                           for side, value in zip((left, right), results)]
                # A rewritten path that leaves out a call the case names has
                # rejected its input.
                made = [[effect.label for effect in side.side_effects.get("calls", [])]
                        for side in (left, right)]
                if skipped(made[1], made[0], skip_names):
                    accepts[1] = claripy.BoolV(False)
                both = claripy.And(*accepts)
                paths = {"original": left.state_id, "rewritten": right.state_id}

                def record(kind, condition, values, **details):
                    answer = ask(solver, condition, values)
                    if answer is not None:
                        # Memory is shown as its bytes in address order.
                        shown = {side: f"{value:0{details['size'] * 2}x}" if kind == "memory"
                                 else hex(value)
                                 for side, value in zip(("original", "rewritten"), answer[1])}
                        differences.append({"kind": kind, **details, **shown,
                                            "input": answer[0], "paths": paths})

                if settings.reject_returns or settings.reject_handlers \
                        or settings.reject_writes or settings.reject_skips:
                    record("accepts-rejected-input",
                           claripy.And(claripy.Not(accepts[0]), accepts[1]), results)
                    if ask(solver, claripy.And(accepts[0], claripy.Not(accepts[1]))) is not None:
                        rejections += 1
                for addresses, (before, after) in sorted(
                        pair.mem_diff.items(), key=lambda item: item[0].start):
                    record("memory", claripy.And(both, before != after), (before, after),
                           address=hex(addresses.start), size=len(addresses))
                for register in settings.registers:
                    if register in pair.reg_diff:
                        before, after = pair.reg_diff[register]
                        record("register", claripy.And(both, before != after), (before, after),
                               register=register)
                # The two sequences of device writes, compared here and not
                # with pair.side_effect_diff: the alignment in cozy 1.5.4
                # drops the writes the original makes after the last one of
                # the rewritten function, so a removed write would be missed.
                # The calls to stubbed callees are compared the same way.
                for kind, count in (("device-writes", "writes"), ("calls", "calls")):
                    writes = [side.side_effects.get(kind, []) for side in (left, right)]
                    if any("guard" in effect.body for side in writes for effect in side):
                        # Some event happens on part of its path only.
                        if len(writes[0]) * len(writes[1]) > GUARDED_LIMIT:
                            incomplete.append(f"too many conditional {kind} on one path "
                                              "to compare")
                            continue
                        events = [[(effect.label, effect.body["value"],
                                    effect.body["guard"] == 1 if "guard" in effect.body
                                    else None) for effect in side] for side in writes]
                        record(kind, claripy.And(both, guarded_difference(claripy, *events)),
                               (), **{count: {"original": len(writes[0]),
                                              "rewritten": len(writes[1])},
                                      "suspects": [], "conditional": True})
                        continue
                    unequal = [] if len(writes[0]) == len(writes[1]) else [true]
                    # Where the two sequences can differ, for the reader: the
                    # position, the two labels and what is not the same.
                    suspects = []
                    for index, (before, after) in enumerate(zip(*writes)):
                        if before.label != after.label:
                            unequal.append(true)
                            what = "target"
                        elif before.body["value"].length != after.body["value"].length:
                            unequal.append(true)
                            what = "width"
                        elif before.body["value"] is not after.body["value"]:
                            unequal.append(before.body["value"] != after.body["value"])
                            what = "value"
                        else:
                            continue
                        suspects.append({"index": index, "what": what,
                                         "original": str(before.label),
                                         "rewritten": str(after.label)})
                    if unequal:
                        record(kind, claripy.And(both, claripy.Or(*unequal)), (),
                               **{count: {"original": len(writes[0]),
                                          "rewritten": len(writes[1])},
                                  "suspects": suspects[:SUSPECT_LIMIT]})

            for side, orphans in (("original", comparison.orphans_left),
                                  ("rewritten", comparison.orphans_right)):
                for orphan in sorted(orphans, key=lambda state: state.state_id):
                    solver = claripy.Solver()
                    solver.add(orphan.state.solver.constraints)
                    answer = ask(solver, true)
                    differences.append({"kind": "unmatched-path", "side": side,
                                        "input": answer[0] if answer else None,
                                        "paths": {side: orphan.state_id}})

            incomplete.extend(sorted(unrecorded))
            for stub, note in settings.stub_notes.items():
                if label_of(stub) in called:
                    incomplete.append(f"the stubbed callee {label_of(stub)}: {note}")

            # A call whose result is computed on one path and unknown on
            # another can separate two paths that agree, so what was found
            # is not a failure.
            mixed = computed & unknown
            if mixed:
                incomplete.append("the result of " + ", ".join(sorted({c for c, _ in mixed}))
                                  + " was computed on some paths and left unknown on others")
                unconfirmed.extend(differences)
                differences.clear()

            stage = "writing cozy's outputs"
            if text is not None:
                try:
                    text.write_text(comparison.report(arguments), encoding="utf-8")
                except Exception as error:  # cozy 1.5.4 cannot print an unmatched path
                    text.write_text(f"cozy could not write its report: {error!r}\n",
                                    encoding="utf-8")
            if graph is not None:
                execution_graph.dump_comparison(
                    original_project, rewritten_project, original_result, rewritten_result,
                    comparison, output_file=str(graph), args=arguments)
    except OutOfTime:
        incomplete.append(f"the time limit of {settings.max_seconds:g}s ran out while {stage}")
    except RecursionError:
        # An expression nested deeper than the interpreter follows, as a read
        # spread over many addresses and then spread again makes.
        incomplete.append(f"an expression grew too deep to handle while {stage}")

    return {
        "tool": "PatchSymDiff",
        "engine": f"cozy {version('cozy-re')} / angr {angr.__version__}",
        "original": str(original),
        "rewritten": str(rewritten),
        "entry": hex(entry),
        "mode": "thumb" if thumb else "arm",
        "verdict": verdict_of(differences, incomplete),
        "differences": differences,
        "unconfirmed": unconfirmed,
        "incomplete": incomplete,
        "rejections": rejections,
        "states": states,
        "seconds": round(time.time() - started, 1),
        "assumptions": {
            "arguments": settings.args,
            "arg_buffers": {str(index): size for index, size in settings.arg_buffers.items()},
            "arg_ranges": {str(index): [hex(low), hex(high)]
                           for index, (low, high) in settings.arg_ranges.items()},
            "arg_suffixes": {str(index): suffix.hex()
                             for index, suffix in settings.arg_suffixes.items()},
            "reject_skips": [label_of(stub) for stub in settings.reject_skips],
            "stub_registers": {label_of(stub): list(registers)
                               for stub, registers in settings.stub_registers.items()},
            "stub_returns": {label_of(stub): list(registers)
                             for stub, registers in settings.stub_returns.items()},
            "globals": "image",
            "unwritten_memory": "zero",
            "saved_registers_at_entry": "unknown, the same on both sides",
            "callees": "followed",
            "modelled_callees": sorted({callee for callee, _ in computed}),
            "stubbed_callees": [label_of(stub)
                                for stub in [*settings.stubs, *settings.stub_pointers]],
            "stub_pointers": {label_of(stub): size
                              for stub, size in settings.stub_pointers.items()},
            "stub_args": {label_of(stub): count for stub, count in settings.stub_args.items()},
            "stubs": "write no memory; the result depends on the callee and the call index; "
                     "calls are compared in order, with the arguments counted in stub_args, "
                     "except addresses in the function's own frame; an address of read-only "
                     "bytes is compared by those bytes up to the first zero",
            "reject_returns": [hex(value) for value in settings.reject_returns],
            "reject_handlers": [handler if isinstance(handler, str) else hex(handler)
                                for handler in settings.reject_handlers],
            "mmio_regions": [[hex(low), hex(high)] for low, high in settings.mmio_regions],
            "reject_writes": [[hex(low), hex(high)] for low, high in settings.reject_writes],
            "memory": "final contents only, except writes to an mmio region, which are "
                      "compared in order",
            "compared_registers": list(settings.registers),
            "stack": "memory below the final stack pointer is not compared",
            "limits": {"loop_bound": settings.loop_bound, "seconds": settings.max_seconds},
        },
    }


def describe(difference: dict) -> str:
    where = {"memory": lambda d: f"{d['size']} byte(s) at {d['address']}",
             "register": lambda d: d["register"],
             "device-writes": lambda d: f"{d['writes']['original']} originally, "
                                        f"{d['writes']['rewritten']} rewritten",
             "calls": lambda d: f"{d['calls']['original']} originally, "
                                f"{d['calls']['rewritten']} rewritten",
             "unmatched-path": lambda d: f"{d['side']} only",
             "accepts-rejected-input": lambda d: "return value"}[difference["kind"]](difference)
    values = (f": {difference['original']} originally, {difference['rewritten']} rewritten"
              if "original" in difference else "")
    return f"{difference['kind']} ({where}){values}, input {difference['input']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--rewritten", type=Path, required=True)
    parser.add_argument("--function")
    parser.add_argument("--address", type=number)
    parser.add_argument("--mode", choices=("arm", "thumb"))
    parser.add_argument("--args", type=int, default=4,
                        help="number of register arguments (default 4)")
    parser.add_argument("--arg-buffer", type=indexed_size, action="append", default=[],
                        metavar="I:SIZE", help="argument I points to SIZE symbolic bytes")
    parser.add_argument("--arg-range", type=indexed_range, action="append", default=[],
                        metavar="I:LO:HI", help="argument I is between LO and HI, unsigned")
    parser.add_argument("--arg-suffix", action="append", default=[], metavar="I:HEX",
                        help="these bytes follow the symbolic bytes of buffer I")
    parser.add_argument("--reject-skip", type=stub_name, action="append", default=[],
                        metavar="NAME_OR_ADDR",
                        help="stub this callee; leaving out a call to it rejects the input")
    parser.add_argument("--declaration", type=Path,
                        help="JSON object with the settings, in place of the options for them")
    parser.add_argument("--reject-return", type=number, action="append", default=[],
                        metavar="VALUE", help="a return value that means the input was rejected")
    parser.add_argument("--reject-handler", type=stub_name, action="append", default=[],
                        metavar="NAME_OR_ADDR", help="a call to this rejects the input")
    parser.add_argument("--stub", type=stub_name, action="append", default=[],
                        metavar="NAME_OR_ADDR", help="do not run this callee")
    parser.add_argument("--stub-pointer", type=stub_size, action="append", default=[],
                        metavar="NAME_OR_ADDR:SIZE",
                        help="do not run this callee; it returns a pointer to SIZE unknown bytes")
    parser.add_argument("--stub-args", type=stub_size, action="append", default=[],
                        metavar="NAME_OR_ADDR:N",
                        help="compare the first N register arguments of calls to this stub")
    parser.add_argument("--reject-write", type=address_range, action="append", default=[],
                        metavar="LO:HI", help="a write to these addresses rejects the input")
    parser.add_argument("--mmio-region", type=address_range, action="append", default=[],
                        metavar="LO:HI", help="writes to these addresses are compared in order")
    parser.add_argument("--compare-registers", default=",".join(DEFAULT_REGISTERS),
                        metavar="LIST", help="comma-separated; default %(default)s")
    parser.add_argument("--no-models", action="store_true",
                        help="leave stubbed string functions uninterpreted")
    parser.add_argument("--loop-bound", type=int)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, help="the report, as JSON")
    parser.add_argument("--text", type=Path, help="cozy's own report")
    parser.add_argument("--graph", type=Path, help="execution graphs for cozy's viewer")
    args = parser.parse_args(argv)

    try:
        for image in (args.original, args.rewritten):
            if not image.is_file():
                raise UsageError(f"{image} is not a file")
        settings = settings_from_declaration(
            json.loads(args.declaration.read_text(encoding="utf-8"))
        ) if args.declaration else Settings(
            args=args.args, arg_buffers=dict(args.arg_buffer), arg_ranges=dict(args.arg_range),
            arg_suffixes={number(index): bytes.fromhex(text) for index, text in
                          (item.split(":") for item in args.arg_suffix)},
            reject_skips=args.reject_skip,
            reject_returns=args.reject_return, reject_handlers=args.reject_handler,
            stubs=args.stub, stub_pointers=dict(args.stub_pointer),
            stub_args=dict(args.stub_args), mmio_regions=args.mmio_region,
            reject_writes=args.reject_write,
            registers=tuple(name for name in args.compare_registers.split(",") if name),
            loop_bound=args.loop_bound, max_seconds=args.max_seconds,
            models=not args.no_models,
        )
        entry, thumb = args.address, args.mode == "thumb"
        if entry is None:
            if not args.function:
                raise UsageError("give --function or --address")
            entry, found_thumb = find_function(args.original, args.function)
            if args.mode is None:
                thumb = found_thumb
        elif args.mode is None:
            raise UsageError("--address needs --mode")
        report = compare(args.original, args.rewritten, entry & ~1, thumb, settings,
                         text=args.text, graph=args.graph)
    except UsageError as error:
        print(f"patch_symdiff: {error}", file=sys.stderr)
        return 2

    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['verdict']}: {len(report['differences'])} difference(s), "
          f"{report['states'].get('pairs', 0)} path pair(s), {report['seconds']}s "
          f"({report['engine']})")
    for difference in report["differences"][:5]:
        print(f"  {describe(difference)}")
    for note in report["incomplete"][:5]:
        print(f"  incomplete: {note}")
    return EXIT[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
