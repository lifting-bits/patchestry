#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""PatchSymDiff: compare a function in the original binary with the same
function in the rewritten binary using cozy (https://github.com/draperlaboratory/cozy).

A second opinion on `check_binary_equivalence.py`.  Both execute the machine
code of the two images with angr, so a lifting error in VEX is invisible to
both.  What differs is everything after execution: cozy pairs the final
states of the two runs and finds the memory and registers that can differ,
where the other script has its own path pairing and its own checks.

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
call uses to fault.  A patch that adds a check
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
| `incomplete` | no input found, but a path stopped at the loop bound or with an error, or the time ran out |

`proven` holds within the assumptions the report lists.  The compared
registers are r0, r4 to r11 and sp unless `--compare-registers` says
otherwise; for a function that returns nothing, leave r0 out.

Not handled: arguments on the stack, floating point, images other than
32-bit ARM, a device write through an address that depends on the input, a
patch that rejects by leaving out a call (`--reject-skip` of the other
script), and imports of a dynamically linked image (angr gives each a fresh
result per side, which shows up as a difference; stub them).  Paths are
not merged, so a function with many branches is slower here than in
`check_binary_equivalence.py`.  The time limit is checked between solver
calls, not inside one.

Needs its own environment, because cozy pins an older angr than the rest of
this directory uses:

    uv run --python python3.11 --with cozy-re==1.5.4 --with 'bitstring<5' \\
        --with 'pycparser<3' patch_symdiff.py ...

Where unicorn is built from source with CMake 4, the first run also needs
CMAKE_POLICY_VERSION_MINIMUM=3.5 in the environment.

CLI: patch_symdiff.py --original ELF --rewritten ELF
         (--function NAME | --address ADDR --mode arm|thumb)
         [--args N] [--arg-buffer I:SIZE]... [--arg-range I:LO:HI]...
         [--reject-return VALUE]... [--reject-handler NAME_OR_ADDR]...
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
import signal
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
SUSPECT_LIMIT = 8  # positions listed for a difference in writes or calls
SAVED_REGISTERS = ("r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11")
DEFAULT_REGISTERS = ("r0", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "sp")
EXIT = {"proven": 0, "failed": 1, "incomplete": 3}


@dataclass
class Settings:
    args: int = 4
    arg_buffers: dict[int, int] = field(default_factory=dict)
    arg_ranges: dict[int, tuple[int, int]] = field(default_factory=dict)
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

        def run(self, label=None, size=0, args=0):
            # What the caller passed.  Its frame is everything from its
            # stack pointer up to where the function under test started.
            passed = []
            for name in ("r0", "r1", "r2", "r3")[:args]:
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
        if address is None or not address.concrete:
            return
        if any(low <= address.concrete_value <= high for low, high in settings.mmio_regions):
            side_effect.perform(state, "device-writes", {"value": state.inspect.mem_write_expr},
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
                project=project.angr_proj, label=label_of(stub),
                args=settings.stub_args.get(stub, 0)), replace=True)
        for stub, size in settings.stub_pointers.items():
            project.angr_proj.hook_symbol(stub, SharedResult(
                project=project.angr_proj, label=label_of(stub), size=size,
                args=settings.stub_args.get(stub, 0)), replace=True)
        for handler in settings.reject_handlers:
            project.hook_symbol(handler, Reject, replace=True)
        session = project.session(entry | 1 if thumb else entry)
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
                arguments.append(claripy.BVV(address, WORD))
            else:
                arguments.append(symbols[f"arg{index}"])
        for index, (low, high) in settings.arg_ranges.items():
            session.add_constraints(claripy.UGE(symbols[f"arg{index}"], low),
                                    claripy.ULE(symbols[f"arg{index}"], high))
        result = session.run(arguments, cache_intermediate_info=graph is not None,
                             loop_bound=settings.loop_bound)
        return project, result, arguments

    differences: list[dict] = []
    incomplete: list[str] = []
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
    try:
        with time_limit(settings.max_seconds):
            original_project, original_result, arguments = run(original)
            stage = "running the rewritten function"
            rewritten_project, rewritten_result, _ = run(rewritten)

            for side, result in (("original", original_result), ("rewritten", rewritten_result)):
                states[side] = {"returned": len(result.deadended),
                                "errored": len(result.errored),
                                "loop-bound": len(result.spinning)}
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
                results = (left.state.regs.r0, right.state.regs.r0)
                accepts = [claripy.BoolV(False) if "symdiff_rejected" in side.state.globals
                           else claripy.And(
                               claripy.Not(side.state.globals.get("symdiff_rejected_when",
                                                                  claripy.BoolV(False))),
                               *(value != reject for reject in settings.reject_returns))
                           for side, value in zip((left, right), results)]
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
                        or settings.reject_writes:
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

    return {
        "tool": "PatchSymDiff",
        "engine": f"cozy {version('cozy-re')} / angr {angr.__version__}",
        "original": str(original),
        "rewritten": str(rewritten),
        "entry": hex(entry),
        "mode": "thumb" if thumb else "arm",
        "verdict": verdict_of(differences, incomplete),
        "differences": differences,
        "incomplete": incomplete,
        "rejections": rejections,
        "states": states,
        "seconds": round(time.time() - started, 1),
        "assumptions": {
            "arguments": settings.args,
            "arg_buffers": {str(index): size for index, size in settings.arg_buffers.items()},
            "arg_ranges": {str(index): [hex(low), hex(high)]
                           for index, (low, high) in settings.arg_ranges.items()},
            "globals": "image",
            "unwritten_memory": "zero",
            "saved_registers_at_entry": "unknown, the same on both sides",
            "callees": "followed",
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
        settings = Settings(
            args=args.args, arg_buffers=dict(args.arg_buffer), arg_ranges=dict(args.arg_range),
            reject_returns=args.reject_return, reject_handlers=args.reject_handler,
            stubs=args.stub, stub_pointers=dict(args.stub_pointer),
            stub_args=dict(args.stub_args), mmio_regions=args.mmio_region,
            reject_writes=args.reject_write,
            registers=tuple(name for name in args.compare_registers.split(",") if name),
            loop_bound=args.loop_bound, max_seconds=args.max_seconds,
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
