#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Compare a function in the original binary with the same function in the
rewritten binary, by symbolic execution of the machine code of both.

Prototype.  Every other comparison in this directory has lifted code on at
least one side, so a lifting error that changes both sides the same way is
invisible to it.  This one reads neither the lifted IR nor the patched IR:
angr executes the bytes of the original ELF and the bytes of the rewritten
ELF.  A difference means some stage between the two (lift, transformation,
compiler, patcher) changed the behaviour, and does not say which.

How it works:

1. Both images are loaded.  The function starts at the same address in both;
   in the rewritten image the entry may be a branch to moved code, which is
   simply followed.
2. Both start from the same state: the argument registers r0 to r3 hold
   shared symbols, a declared argument buffer is a block of shared symbolic
   bytes at a fixed address, and the writable segments of the image hold
   shared symbols (or the image's bytes, `--globals image`).  Everything else
   that is read before it is written is zero.
3. Each side is explored path by path.  A call is not followed: it is
   recorded with its arguments and returns a symbol that depends on the
   callee and on how many times it was called, the same on both sides.  An
   argument that is the address of a local is recorded as "a local", and
   one that is the address of read-only data as the bytes found there,
   because the two images lay both out differently.  A
   call into code the rewrite added is followed, since that code is part of
   the patch.  Writes outside the stack are recorded; writes to a declared
   device range are recorded in order.  A write through an address that
   depends on the input is made to every address it can reach (at most
   SYMBOLIC_WRITE_SPAN of them), each under its own condition, and the whole
   range is recorded.
4. For every pair of a patched path and an original path that one input can
   take, the solver is asked for an input that separates them:

   | Check | Violated when some input |
   | --- | --- |
   | P1 | is accepted by the patched function and rejected by the original |
   | P2 | is accepted by both and leaves a different byte outside the stack |
   | P3 | is accepted by both and gives different return values |
   | P4 | is accepted by both and gives different calls or call arguments |
   | mmio-order | is accepted by both and gives different device writes |

A function rejects an input when it calls a declared reject handler or
returns a declared reject value.  With `--reject-skip NAME`, a patched path
whose calls are those of the original path without some calls of NAME is a
rejection as well.

Verdicts, per check and for the run (the weakest):

| Verdict | Meaning |
| --- | --- |
| `proven` | no input separates the two, and every path of both was explored |
| `failed` | the solver found an input; `violations` has it |
| `incomplete` | no input found, but a limit was reached or a path could not be followed |
| `not-run` | nothing to compare (P3 of a function that returns nothing) |

`proven` holds within what the run assumed, and the report lists it: the
buffer sizes, that callees are uninterpreted and write no memory, that
unwritten memory is zero, how many arguments each callee takes, and the
limits on paths and steps.

Not handled: arguments on the stack, floating-point arguments and results,
callees that write through their pointer arguments, indirect jumps the engine
cannot resolve, functions with more paths than the limit, and a read or write
whose address can take more values than the engine enumerates (the run is
then `incomplete`).

CLI: check_binary_equivalence.py --original ELF --rewritten ELF
     --function NAME | --address ADDR [--mode arm|thumb]
     [--prototypes IR] [--arg-buffer INDEX:SIZE] [--arg-range INDEX:LOW:HIGH]
     [--reject-handler NAME]
     [--reject-return VALUE] [--reject-skip NAME] [--pure-callee NAME]
     [--mmio-region LOW:HIGH] [--globals symbolic|image]
     [--max-paths N] [--max-steps N] [--max-seconds N] [--output FILE]
Exit 0 when proven, 1 when failed, 3 when incomplete, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import angr
import claripy
from angr.concretization_strategies import (
    SimConcretizationStrategyAny,
    SimConcretizationStrategyMax,
    SimConcretizationStrategyRange,
)

CHECKS = ("P1", "P2", "P3", "P4", "mmio-order")
RETURN_ADDRESS = 0xDEAD0000
BUFFER_BASE = 0x60000000
BUFFER_STEP = 0x10000
STACK_LOW, STACK_HIGH = 0x7FF00000, 0x80000000
# A writable segment larger than this keeps the image's bytes.
SYMBOLIC_SEGMENT_LIMIT = 0x100000
ARGUMENT_REGISTERS = ("r0", "r1", "r2", "r3")
# A write address that depends on the input is followed to this many places.
SYMBOLIC_WRITE_SPAN = 256


@dataclass
class Settings:
    arg_buffers: dict = field(default_factory=dict)  # index -> size
    arg_ranges: dict = field(default_factory=dict)  # index -> (low, high), unsigned, inclusive
    reject_handlers: tuple = ()
    reject_returns: tuple = ()
    reject_skips: tuple = ()
    pure_callees: tuple = ()
    mmio_regions: tuple = ()  # (low, high), inclusive
    symbolic_globals: bool = True
    prototypes: dict = field(default_factory=dict)  # callee -> argument count
    returns_value: bool = True
    default_args: int = 4
    max_paths: int = 256
    max_steps: int = 4000
    max_seconds: float = 300.0  # per side, and again for the comparison


@dataclass
class Path_:
    """One explored path of one side."""
    kind: str  # "returned" or "handler"
    state: object
    constraint: object
    result: object
    calls: tuple
    device: tuple
    writes: frozenset


class Shared:
    """Symbols both sides start from and get back from callees."""

    def __init__(self):
        self.arguments = [claripy.BVS(f"arg{index}", 32, explicit_name=True)
                          for index in range(len(ARGUMENT_REGISTERS))]
        self.buffers: dict[int, object] = {}
        self.segments: dict[int, object] = {}
        self.returns: dict[tuple, object] = {}

    def buffer(self, index: int, size: int):
        if index not in self.buffers:
            self.buffers[index] = claripy.BVS(f"buffer{index}", size * 8, explicit_name=True)
        return self.buffers[index]

    def segment(self, address: int, size: int):
        if address not in self.segments:
            self.segments[address] = claripy.BVS(f"global_{address:x}", size * 8,
                                                 explicit_name=True)
        return self.segments[address]

    def returned(self, callee: str, index: int):
        key = (callee, index)
        if key not in self.returns:
            self.returns[key] = claripy.BVS(f"ret_{callee}_{index}", 32, explicit_name=True)
        return self.returns[key]


def read_prototypes(path: Path, function: str) -> tuple[dict, bool]:
    """Argument counts of the declared functions of an LLVM IR file, and
    whether `function` returns a value there."""
    counts, returns_value = {}, True
    text = path.read_text(encoding="utf-8")
    for found in re.finditer(r"^declare\s+.*?@([\w.$]+)\((.*?)\)", text, re.MULTILINE):
        parts = [part.strip() for part in found.group(2).split(",") if part.strip()]
        counts[found.group(1)] = min(len([part for part in parts if part != "..."]),
                                     len(ARGUMENT_REGISTERS))
    defined = re.search(rf"^define\s+(.*?)@{re.escape(function)}\(", text, re.MULTILINE)
    if defined:
        returns_value = defined.group(1).split()[-1] != "void"
    return counts, returns_value


class Side:
    """One image and the exploration of the function in it."""

    def __init__(self, image: Path, entry: int, thumb: bool, settings: Settings,
                 shared: Shared, known: "Side | None" = None):
        self.project = angr.Project(str(image), auto_load_libs=False)
        self.entry, self.thumb = entry, thumb
        self.settings, self.shared = settings, shared
        self.main = self.project.loader.main_object
        # Code the other image does not have was added by the rewrite.
        self.known = known or self
        # Function symbols and PLT stubs; ARM mapping symbols ($t, $a, $d)
        # share addresses with them and name nothing.
        self.names = {symbol.rebased_addr & ~1: symbol.name for symbol in self.main.symbols
                      if symbol.is_function and symbol.name and not symbol.name.startswith("$")}
        self.names.update({address & ~1: name
                           for address, name in getattr(self.main, "reverse_plt", {}).items()})
        self.notes: set[str] = set()
        self.paths: list[Path_] = []
        self.steps = 0

    def mapped(self, address: int) -> bool:
        return any(segment.vaddr <= address < segment.vaddr + segment.memsize
                   for segment in self.main.segments)

    def name(self, address: int) -> str:
        """The callee at `address`, named from the original image."""
        return self.known.names.get(address, f"sub_{address:x}")

    def initial(self):
        options = angr.options
        state = self.project.factory.blank_state(
            addr=self.entry | (1 if self.thumb else 0),
            add_options={options.ZERO_FILL_UNCONSTRAINED_MEMORY,
                         options.ZERO_FILL_UNCONSTRAINED_REGISTERS},
            remove_options={options.SYMBOL_FILL_UNCONSTRAINED_MEMORY,
                            options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS},
        )
        state.regs.sp = STACK_HIGH - 0x10000
        state.regs.lr = RETURN_ADDRESS
        for index, register in enumerate(ARGUMENT_REGISTERS):
            setattr(state.regs, register, self.shared.arguments[index])
        for index, size in sorted(self.settings.arg_buffers.items()):
            address = BUFFER_BASE + index * BUFFER_STEP
            state.memory.store(address, self.shared.buffer(index, size))
            setattr(state.regs, ARGUMENT_REGISTERS[index], claripy.BVV(address, 32))
        for index, (low, high) in sorted(self.settings.arg_ranges.items()):
            state.solver.add(claripy.UGE(self.shared.arguments[index], low),
                             claripy.ULE(self.shared.arguments[index], high))
        if self.settings.symbolic_globals:
            for segment in self.known.main.segments:
                if not segment.is_writable or segment.memsize == 0:
                    continue
                if segment.memsize > SYMBOLIC_SEGMENT_LIMIT:
                    self.notes.add(f"segment at {segment.vaddr:#x} is too large to make "
                                   "symbolic; it holds the image's bytes")
                    continue
                state.memory.store(segment.vaddr,
                                   self.shared.segment(segment.vaddr, segment.memsize))
        state.globals["calls"] = ()
        state.globals["counts"] = ()
        state.globals["device"] = ()
        state.globals["writes"] = frozenset()
        # angr's default fixes a write address that depends on the input to
        # its largest value.  Enumerate a bounded range first; `pinned` notes
        # an address the engine still fixed.
        state.memory.write_strategies.insert(
            0, SimConcretizationStrategyRange(SYMBOLIC_WRITE_SPAN))
        state.inspect.b("mem_write", when=angr.BP_BEFORE, action=self.wrote)
        state.inspect.b("address_concretization", when=angr.BP_AFTER, action=self.pinned)
        return state

    def pinned(self, state) -> None:
        strategy = state.inspect.address_concretization_strategy
        if state.inspect.address_concretization_result is not None and isinstance(
                strategy, (SimConcretizationStrategyAny, SimConcretizationStrategyMax)):
            action = "read" if state.inspect.address_concretization_action == "load" else "write"
            self.notes.add(f"a {action} through an address with too many possible values; "
                           "one was taken")

    def wrote_anywhere(self, state, address, size: int) -> None:
        """A write whose address depends on the input: record every byte it
        can reach, so that the final memory there is compared."""
        low, high = state.solver.min(address), state.solver.max(address)
        if STACK_LOW <= low and high + size <= STACK_HIGH:
            return
        if low < STACK_HIGH and high + size > STACK_LOW:
            self.notes.add("a write through an address that can be inside or outside the stack")
            return
        if any(low <= last and high + size > first
               for first, last in self.settings.mmio_regions):
            self.notes.add("a device write through an address that depends on the input")
            return
        if high - low >= SYMBOLIC_WRITE_SPAN:
            self.notes.add(f"a write through an address with more than {SYMBOLIC_WRITE_SPAN} "
                           "possible values")
            return
        state.globals["writes"] = state.globals["writes"] | {(low, high - low + size)}

    def wrote(self, state) -> None:
        address = state.inspect.mem_write_address
        size = state.inspect.mem_write_length
        size = state.solver.eval(size) if size is not None else (
            state.inspect.mem_write_expr.size() // 8)
        if state.solver.symbolic(address):
            self.wrote_anywhere(state, address, size)
            return
        address = state.solver.eval(address)
        if STACK_LOW <= address < STACK_HIGH:
            return
        if any(low <= address <= high for low, high in self.settings.mmio_regions):
            state.globals["device"] += ((address, size, state.inspect.mem_write_expr),)
        else:
            state.globals["writes"] = state.globals["writes"] | {(address, size)}

    def finish(self, state, kind: str) -> None:
        self.paths.append(Path_(
            kind=kind, state=state, constraint=claripy.And(*state.solver.constraints)
            if state.solver.constraints else claripy.true(),
            result=state.regs.r0, calls=state.globals["calls"],
            device=state.globals["device"], writes=state.globals["writes"],
        ))

    def argument(self, state, value):
        """A call argument as it is compared.

        The two images place locals and constant data at different addresses.
        The address of a local is compared as "a local", the address of
        read-only data by the bytes found there (to the first zero byte, at
        most 64), and anything else as a number.
        """
        if state.solver.symbolic(value):
            return value
        address = state.solver.eval(value)
        if STACK_LOW <= address < STACK_HIGH:
            return ("local",)
        for segment in self.main.segments:
            if (segment.vaddr <= address < segment.vaddr + segment.memsize
                    and not segment.is_writable):
                size = min(64, segment.vaddr + segment.memsize - address)
                data = state.solver.eval(state.memory.load(address, size), cast_to=bytes)
                return ("constant", data.split(b"\0")[0])
        return value

    def call(self, state, target: int) -> bool:
        """Replace a call by its record and result; False when the path ends."""
        callee = self.name(target)
        count = self.settings.prototypes.get(callee, self.settings.default_args)
        arguments = tuple(self.argument(state, getattr(state.regs, register))
                          for register in ARGUMENT_REGISTERS[:count])
        if callee in self.settings.reject_handlers:
            self.finish(state, "handler")
            return False
        if callee not in self.settings.pure_callees:
            state.globals["calls"] += ((callee, arguments),)
        counts = dict(state.globals["counts"])
        index = counts.get(callee, 0)
        counts[callee] = index + 1
        state.globals["counts"] = tuple(sorted(counts.items()))
        state.regs.r0 = self.shared.returned(callee, index)
        state.regs.pc = state.regs.lr
        return True

    def explore(self) -> None:
        pending = [(self.initial(), 0)]
        deadline = time.time() + self.settings.max_seconds
        while pending:
            state, depth = pending.pop()
            if time.time() > deadline:
                self.notes.add(f"exploration took more than {self.settings.max_seconds:.0f} "
                               f"seconds ({len(pending) + 1} paths unfinished)")
                return
            if len(self.paths) + len(pending) >= self.settings.max_paths:
                self.notes.add(f"more than {self.settings.max_paths} paths")
                return
            if depth >= self.settings.max_steps:
                self.notes.add(f"a path longer than {self.settings.max_steps} blocks")
                continue
            self.steps += 1
            try:
                successors = self.project.factory.successors(state)
            except (angr.errors.AngrError, claripy.errors.ClaripyError) as error:
                self.notes.add(f"a path stopped: {type(error).__name__}: {str(error)[:120]}")
                continue
            if successors.unconstrained_successors:
                self.notes.add("a jump to an address that depends on the input")
            for after in successors.flat_successors:
                target = after.solver.eval(after.regs.pc) & ~1
                if target == RETURN_ADDRESS:
                    self.finish(after, "returned")
                elif after.history.jumpkind == "Ijk_Call" and self.known.mapped(target):
                    if self.call(after, target):
                        pending.append((after, depth + 1))
                elif after.history.jumpkind.startswith("Ijk_Sys"):
                    self.notes.add("a system call")
                else:
                    pending.append((after, depth + 1))


def rejects(path: Path_, settings: Settings):
    if path.kind == "handler":
        return claripy.true()
    if not settings.reject_returns:
        return claripy.false()
    return claripy.Or(*[path.result == claripy.BVV(value & 0xFFFFFFFF, 32)
                        for value in settings.reject_returns])


def skipped(patched: tuple, original: tuple, names: tuple) -> bool:
    """Whether `patched` is `original` without some calls of `names`, and shorter."""
    if not names or len(patched) >= len(original):
        return False
    at = 0
    for callee, _ in original:
        if at < len(patched) and patched[at][0] == callee:
            at += 1
        elif callee not in names:
            return False
    return at == len(patched)


def sequences_differ(first: tuple, second: tuple, expressions):
    """A condition under which two recorded sequences differ; True when they
    differ on every input."""
    if len(first) != len(second):
        return claripy.true()
    terms = []
    for one, other in zip(first, second):
        left, right = expressions(one), expressions(other)
        if left[0] != right[0] or len(left[1]) != len(right[1]):
            return claripy.true()
        for a, b in zip(left[1], right[1]):
            if isinstance(a, tuple) or isinstance(b, tuple):
                if a != b:
                    return claripy.true()
            else:
                terms.append(a != b)
    return claripy.Or(*terms) if terms else claripy.false()


def compare(patched: Side, original: Side, settings: Settings, shared: Shared,
            notes: list[str]) -> list[dict]:
    violations: list[dict] = []
    deadline = time.time() + settings.max_seconds
    pairs = 0
    found: set[tuple] = set()

    def ask(check: str, kind: str, conditions: list, detail: dict) -> None:
        if (check, kind) in found and len([v for v in violations if v["check"] == check]) >= 3:
            return
        solver = claripy.Solver()
        solver.add(conditions)
        if not solver.satisfiable():
            return
        witness = {f"arg{index}": f"{solver.eval(symbol, 1)[0]:#x}"
                   for index, symbol in enumerate(shared.arguments)
                   if index not in shared.buffers}
        for index, symbol in shared.buffers.items():
            witness[f"buffer{index}"] = f"{solver.eval(symbol, 1)[0]:0{symbol.size() // 4}x}"
        found.add((check, kind))
        violations.append({"check": check, "kind": kind, "input": witness, **detail})

    for p in patched.paths:
        for o in original.paths:
            if time.time() > deadline:
                notes.append(f"comparison took more than {settings.max_seconds:.0f} seconds "
                             f"({pairs} of {len(patched.paths) * len(original.paths)} "
                             "path pairs compared)")
                return violations
            pairs += 1
            base = [p.constraint, o.constraint]
            reject_p, reject_o = rejects(p, settings), rejects(o, settings)
            ask("P1", "original-rejects", base + [claripy.Not(reject_p), reject_o], {})
            both = base + [claripy.Not(reject_p), claripy.Not(reject_o)]
            solver = claripy.Solver()
            solver.add(both)
            if not solver.satisfiable():
                continue
            if skipped(p.calls, o.calls, settings.reject_skips):
                continue  # the patched function rejected by not calling
            names = {"patched": [c for c, _ in p.calls], "original": [c for c, _ in o.calls]}
            ask("P4", "calls-differ",
                both + [sequences_differ(p.calls, o.calls, lambda call: (call[0], call[1]))],
                {"calls": names})
            ask("mmio-order", "writes-differ",
                both + [sequences_differ(
                    p.device, o.device, lambda write: ((write[0], write[1]), (write[2],)))],
                {"writes": {"patched": [f"{a:#x}" for a, _, _ in p.device],
                            "original": [f"{a:#x}" for a, _, _ in o.device]}})
            if settings.returns_value and p.kind == o.kind == "returned":
                ask("P3", "return-differs", both + [p.result != o.result], {})
            for address, size in sorted(p.writes | o.writes):
                ask("P2", "memory-differs",
                    both + [p.state.memory.load(address, size)
                            != o.state.memory.load(address, size)],
                    {"address": f"{address:#x}", "size": size})
    return violations


def check_binary_equivalence(original: Path, rewritten: Path, entry: int, thumb: bool,
                             settings: Settings) -> dict:
    started = time.time()
    shared = Shared()
    before = Side(original, entry, thumb, settings, shared)
    after = Side(rewritten, entry, thumb, settings, shared, known=before)
    before.explore()
    after.explore()
    notes = sorted({f"original: {note}" for note in before.notes}
                   | {f"rewritten: {note}" for note in after.notes})
    violations = compare(after, before, settings, shared, notes)
    if not before.paths or not after.paths:
        notes.append("a side has no finished path")
    failed = {row["check"] for row in violations}
    declared = {"P3": settings.returns_value, "mmio-order": bool(settings.mmio_regions)}
    checks = {}
    for name in CHECKS:
        if name in failed:
            checks[name] = "failed"
        elif not declared.get(name, True):
            checks[name] = "not-run"
        else:
            checks[name] = "incomplete" if notes else "proven"
    verdict = "failed" if failed else ("incomplete" if notes else "proven")
    return {
        "engine": f"angr {angr.__version__}",
        "original": str(original),
        "rewritten": str(rewritten),
        "entry": f"{entry:#x}",
        "mode": "thumb" if thumb else "arm",
        "verdict": verdict,
        "checks": checks,
        "violations": violations,
        "incomplete": notes,
        "paths": {"original": len(before.paths), "rewritten": len(after.paths)},
        "blocks": {"original": before.steps, "rewritten": after.steps},
        "seconds": round(time.time() - started, 1),
        "assumptions": {
            "arg_buffers": {str(index): size for index, size in settings.arg_buffers.items()},
            "arg_ranges": {str(index): list(bounds)
                           for index, bounds in settings.arg_ranges.items()},
            "globals": "symbolic" if settings.symbolic_globals else "image",
            "callees": "uninterpreted; they write no memory; the result depends on the "
                       "callee and the call index",
            "callee_arguments": {"known": settings.prototypes,
                                 "otherwise": settings.default_args},
            "unwritten_memory": "zero",
            "reject_handlers": list(settings.reject_handlers),
            "reject_returns": list(settings.reject_returns),
            "reject_skips": list(settings.reject_skips),
            "pure_callees": list(settings.pure_callees),
            "mmio_regions": [[low, high] for low, high in settings.mmio_regions],
            "limits": {"paths": settings.max_paths, "blocks_per_path": settings.max_steps,
                       "seconds_per_side": settings.max_seconds,
                       "addresses_per_write": SYMBOLIC_WRITE_SPAN},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--rewritten", type=Path, required=True)
    parser.add_argument("--function", help="symbol of the function in the original image")
    parser.add_argument("--address", type=lambda text: int(text, 0))
    parser.add_argument("--mode", choices=("arm", "thumb"))
    parser.add_argument("--prototypes", type=Path,
                        help="LLVM IR whose declarations give each callee's argument count")
    parser.add_argument("--prototype-function", help="name of the function in --prototypes")
    parser.add_argument("--arg-buffer", action="append", default=[], metavar="INDEX:SIZE")
    parser.add_argument("--arg-range", action="append", default=[], metavar="INDEX:LOW:HIGH",
                        help="an integer argument takes values LOW to HIGH inclusive")
    parser.add_argument("--reject-handler", action="append", default=[])
    parser.add_argument("--reject-return", action="append", default=[],
                        type=lambda text: int(text, 0))
    parser.add_argument("--reject-skip", action="append", default=[])
    parser.add_argument("--pure-callee", action="append", default=[])
    parser.add_argument("--mmio-region", action="append", default=[], metavar="LOW:HIGH")
    parser.add_argument("--globals", choices=("symbolic", "image"), default="symbolic")
    parser.add_argument("--default-args", type=int, default=4)
    parser.add_argument("--max-paths", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-seconds", type=float, default=300.0,
                        help="time to explore each side")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.getLogger("angr").setLevel(logging.ERROR)
    logging.getLogger("cle").setLevel(logging.ERROR)
    logging.getLogger("pyvex").setLevel(logging.ERROR)
    try:
        settings = Settings(
            arg_buffers={int(index): int(size, 0) for index, size in
                         (text.split(":") for text in args.arg_buffer)},
            arg_ranges={int(index): (int(low, 0), int(high, 0)) for index, low, high in
                        (text.split(":") for text in args.arg_range)},
            reject_handlers=tuple(args.reject_handler),
            reject_returns=tuple(args.reject_return),
            reject_skips=tuple(args.reject_skip),
            pure_callees=tuple(args.pure_callee),
            mmio_regions=tuple(tuple(int(part, 0) for part in text.split(":"))
                               for text in args.mmio_region),
            symbolic_globals=args.globals == "symbolic",
            default_args=args.default_args, max_paths=args.max_paths,
            max_steps=args.max_steps, max_seconds=args.max_seconds,
        )
        entry, thumb = args.address, args.mode == "thumb"
        if entry is None:
            if not args.function:
                raise ValueError("give --function or --address")
            symbol = angr.Project(str(args.original), auto_load_libs=False) \
                .loader.find_symbol(args.function)
            if symbol is None:
                raise ValueError(f"{args.original} has no symbol {args.function}")
            entry = symbol.rebased_addr & ~1
            if args.mode is None:
                thumb = bool(symbol.rebased_addr & 1)
        elif args.mode is None:
            raise ValueError("--address needs --mode")
        if args.prototypes:
            settings.prototypes, settings.returns_value = read_prototypes(
                args.prototypes, args.prototype_function or args.function or "")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    report = check_binary_equivalence(args.original, args.rewritten, entry, thumb, settings)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(f"{report['verdict']}: {report['checks']} paths {report['paths']} "
          f"in {report['seconds']}s")
    for row in report["violations"][:5]:
        print(f"  {row['check']} {row['kind']} input {row['input']}")
    for note in report["incomplete"][:5]:
        print(f"  incomplete: {note}")
    return {"proven": 0, "failed": 1, "incomplete": 3}[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
