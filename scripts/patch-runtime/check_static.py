#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Static checks on the machine code a reinsertion added to an image.

The other checks say where the new bytes are and that they are the intended
ones.  These look at what the new code does to the machine state every
function shares with its caller, read from the disassembly of the rewritten
image, with no trust in the compiler that produced it.

Per inserted function, the code is followed from its entry along every direct
branch.  Along each path the analysis tracks how far the stack pointer is
below its value at entry, which callee-saved registers no longer hold their
entry value, and which stack slots hold a saved entry value.

Problems, any of which fails the check:

| Problem | Meaning |
| --- | --- |
| `callee-saved-clobbered` | at a return or tail call, one of r4-r11 or d8-d15 does not hold the value it had at entry |
| `unbalanced-stack` | at a return or tail call, the stack pointer is not where it was at entry |
| `return-address-lost` | the function returns to something other than the lr it was entered with |
| `code-size-over-budget` | the inserted functions together are larger than the code budget |
| `stack-over-budget` | their frames together are deeper than the stack budget |

A function the analysis cannot follow is `not-run`, with the reason, and is
never counted as passed:

| Reason | Meaning |
| --- | --- |
| `indirect-jump` | a computed jump: a jump table, or a branch through a register other than a return |
| `untracked-stack-pointer` | the stack pointer is set to something that is not a constant distance from its entry value (a variable-sized frame) |
| `conditional-stack-change` | a conditionally executed instruction moves the stack pointer |
| `inconsistent-stack` | two paths reach the same instruction with the stack pointer at different depths |
| `runs-past-end` | execution continues past the last byte of the function, other than after a call |
| `undecodable` | a reachable address does not decode as an instruction |

Reported, not judged:

- `stack_bytes`: the sum of the inserted functions' own frames, which bounds
  the stack they hold at once.  Frames of the image's own functions that the
  new code calls are not included.
- `original`: the same analysis of the function that was replaced, and
  `stack_growth`, the difference between the new body's frame and the old.
- `inlined_copies`: places in the original image whose instructions repeat a
  run of the replaced function's body at least half the body long (and at
  least 12 instructions).  A compiler that inlined the function left a copy
  the reinsertion does not reach.  Operands are compared as written, so a copy
  that uses other registers is not found.  Shorter runs are not reported: real
  code repeats its call sequences too often for them to mean anything.
- Budgets are compared only when given.  With none, the measured sizes are
  reported and the budget verdicts are `not-judged`.

Not modelled: a store through a pointer into a save slot, and a saved value
moved through another register before it is restored (reported as clobbered).

CLI: check_static.py --record patch_record.json --elf original.elf
     --rewritten rewritten.elf [--max-code-bytes N] [--max-stack-bytes N]
     [--output report.json]
Exit 0 when nothing fails, 1 when something does, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import capstone
from capstone import arm
from elftools.elf.elffile import ELFFile

import check_trampoline
from check_bindings import PF_X

PROBLEMS = (
    "callee-saved-clobbered",
    "unbalanced-stack",
    "return-address-lost",
    "code-size-over-budget",
    "stack-over-budget",
)

REGISTER_NAMES = {"sb": "r9", "sl": "r10", "fp": "r11", "ip": "r12"}
CALLEE_SAVED = {f"r{n}" for n in range(4, 12)} | {f"d{n}" for n in range(8, 16)}
TRACKED = CALLEE_SAVED | {"lr"}
UNCONDITIONAL = (arm.ARM_CC_AL, arm.ARM_CC_INVALID)

PUSH_ONE = re.compile(r"^(\w+), \[sp, #-4\]!$")
POP_ONE = re.compile(r"^(\w+), \[sp\], #4$")


class NotRun(Exception):
    def __init__(self, reason: str, address: int, detail: str = ""):
        super().__init__(reason)
        self.reason, self.address, self.detail = reason, address, detail


class State:
    """What is known at one instruction, relative to the function's entry."""

    def __init__(self, depth=0, alias=None, clobbered=frozenset(), slots=None):
        self.depth = depth  # bytes the stack pointer is below its entry value
        self.alias = dict(alias or {})  # register -> entry sp minus its value
        self.clobbered = frozenset(clobbered)  # tracked registers not holding their entry value
        self.slots = dict(slots or {})  # depth of a stack slot -> register whose entry value it holds

    def copy(self) -> "State":
        return State(self.depth, self.alias, self.clobbered, self.slots)

    def key(self):
        return (self.depth, tuple(sorted(self.alias.items())), self.clobbered,
                tuple(sorted(self.slots.items())))

    def merge(self, other: "State") -> "State":
        return State(
            self.depth,
            {r: v for r, v in self.alias.items() if other.alias.get(r) == v},
            self.clobbered | other.clobbered,
            {k: v for k, v in self.slots.items() if other.slots.get(k) == v},
        )


def _name(instruction, register: int) -> str:
    name = instruction.reg_name(register)
    return REGISTER_NAMES.get(name, name)


def _registers(instruction) -> list[str]:
    return [
        _name(instruction, operand.reg)
        for operand in instruction.operands
        if operand.type == arm.ARM_OP_REG
    ]


def _push(state: State, registers: list[str], width: int) -> None:
    count = len(registers)
    state.depth += width * count
    for index, register in enumerate(registers):
        slot = state.depth - width * index
        if register in TRACKED and register not in state.clobbered:
            state.slots[slot] = register
        else:
            state.slots.pop(slot, None)


def _pop(state: State, registers: list[str], width: int) -> str | None:
    """Apply a pop; return what the slot loaded into pc held, if pc was popped."""
    returned = None
    clobbered = set(state.clobbered)
    for index, register in enumerate(registers):
        held = state.slots.pop(state.depth - width * index, None)
        if register == "pc":
            returned = held or "unknown"
        elif register in TRACKED:
            if held == register:
                clobbered.discard(register)
            else:
                clobbered.add(register)
        state.alias.pop(register, None)
    state.depth -= width * len(registers)
    state.clobbered = frozenset(clobbered)
    return returned


def _stack_pointer_write(instruction, state: State) -> bool:
    """Apply an instruction that sets sp by a constant; False if it is not one."""
    operands = instruction.operands
    kinds = [operand.type for operand in operands]
    if instruction.id in (arm.ARM_INS_SUB, arm.ARM_INS_ADD):
        sign = 1 if instruction.id == arm.ARM_INS_SUB else -1
        if kinds == [arm.ARM_OP_REG, arm.ARM_OP_IMM]:
            state.depth += sign * operands[1].imm
            return True
        if kinds == [arm.ARM_OP_REG, arm.ARM_OP_REG, arm.ARM_OP_IMM]:
            source = _name(instruction, operands[1].reg)
            if source == "sp":
                state.depth += sign * operands[2].imm
                return True
            if source in state.alias:
                state.depth = state.alias[source] + sign * operands[2].imm
                return True
    if instruction.id == arm.ARM_INS_MOV and kinds == [arm.ARM_OP_REG, arm.ARM_OP_REG]:
        source = _name(instruction, operands[1].reg)
        if source in state.alias:
            state.depth = state.alias[source]
            return True
    return False


def _new_alias(instruction, state: State) -> tuple[str, int] | None:
    """A register this instruction sets to sp plus a constant, with its distance."""
    operands = instruction.operands
    kinds = [operand.type for operand in operands]
    if instruction.id == arm.ARM_INS_MOV and kinds == [arm.ARM_OP_REG, arm.ARM_OP_REG]:
        if _name(instruction, operands[1].reg) == "sp":
            return _name(instruction, operands[0].reg), state.depth
    if instruction.id in (arm.ARM_INS_ADD, arm.ARM_INS_SUB) and kinds == [
        arm.ARM_OP_REG, arm.ARM_OP_REG, arm.ARM_OP_IMM
    ]:
        if _name(instruction, operands[1].reg) == "sp":
            sign = -1 if instruction.id == arm.ARM_INS_ADD else 1
            return _name(instruction, operands[0].reg), state.depth + sign * operands[2].imm
    return None


def analyze_function(memory: check_trampoline.Memory, start: int, size: int, mode: str) -> dict:
    """Follow the function at `start` and report what it does to shared state."""
    end = start + size
    decoder = capstone.Cs(
        capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM if mode == "arm" else capstone.CS_MODE_THUMB
    )
    decoder.detail = True

    problems: dict[str, list] = {}
    not_run: dict[str, list] = {}
    exits: list[dict] = []
    seen_exits: set[tuple] = set()
    states: dict[int, State] = {}
    deepest = 0
    work: list[tuple[int, State]] = [(start, State())]

    def leave(address: int, kind: str, state: State, returns_to: str) -> None:
        key = (address, kind, state.key(), returns_to)
        if key in seen_exits:
            return
        seen_exits.add(key)
        if not any(row["address"] == f"0x{address:x}" for row in exits):
            exits.append({"address": f"0x{address:x}", "kind": kind})
        lost = sorted(state.clobbered & CALLEE_SAVED, key=lambda r: (r[0], int(r[1:])))
        if lost:
            problems.setdefault("callee-saved-clobbered", []).append(
                {"exit": f"0x{address:x}", "registers": lost}
            )
        if state.depth != 0:
            problems.setdefault("unbalanced-stack", []).append(
                {"exit": f"0x{address:x}", "depth": state.depth}
            )
        if returns_to != "lr":
            problems.setdefault("return-address-lost", []).append(
                {"exit": f"0x{address:x}", "returns_to": returns_to}
            )

    def step(address: int, state: State) -> list[tuple[int, State]]:
        """Successors of the instruction at `address` entered in `state`."""
        nonlocal deepest
        data = memory.read(address, min(4, end - address)) or memory.read(address, 2)
        instruction = next(decoder.disasm(data, address, 1), None) if data else None
        if instruction is None:
            raise NotRun("undecodable", address)
        following = address + instruction.size
        conditional = instruction.cc not in UNCONDITIONAL
        untouched = state.copy()
        base = check_trampoline.branch_base(instruction.mnemonic)
        text = instruction.op_str
        try:
            written = {_name(instruction, r) for r in instruction.regs_access()[1]}
        except capstone.CsError:
            raise NotRun("undecodable", address, f"{instruction.mnemonic} {text}") from None

        def lr_or_lost(current: State) -> str:
            return "overwritten lr" if "lr" in current.clobbered else "lr"

        # Branches.
        if instruction.id in (arm.ARM_INS_B, arm.ARM_INS_CBZ, arm.ARM_INS_CBNZ):
            target = check_trampoline.direct_target(instruction.mnemonic, text)
            if target is None:
                raise NotRun("indirect-jump", address, f"{instruction.mnemonic} {text}")
            successors = []
            if start <= target < end:
                successors.append((target, state.copy()))
            else:
                leave(address, "tail-call", state, lr_or_lost(state))
            if conditional or instruction.id != arm.ARM_INS_B:
                successors.append((following, untouched))
            return successors
        if base in ("bl", "blx"):
            state.clobbered |= {"lr"}
            for register in ("r0", "r1", "r2", "r3", "r12", "lr"):
                state.alias.pop(register, None)
            return [(following, state)]
        if base == "bx" or (instruction.id == arm.ARM_INS_MOV and text == "pc, lr"):
            if base == "bx" and _registers(instruction) != ["lr"]:
                raise NotRun("indirect-jump", address, f"{instruction.mnemonic} {text}")
            leave(address, "return", state, lr_or_lost(state))
            return [(following, untouched)] if conditional else []

        # Stack pushes and pops.
        wide = instruction.id in (arm.ARM_INS_VPUSH, arm.ARM_INS_VPOP)
        popped = None
        if instruction.id in (arm.ARM_INS_PUSH, arm.ARM_INS_VPUSH):
            popped = False
            _push(state, _registers(instruction), 8 if wide else 4)
        elif instruction.id in (arm.ARM_INS_POP, arm.ARM_INS_VPOP):
            popped = _registers(instruction)
        elif (
            instruction.id in (arm.ARM_INS_LDM, arm.ARM_INS_STMDB)
            and instruction.writeback
            and _registers(instruction)[:1] == ["sp"]
        ):
            # Push and pop spelled as a multiple store or load through sp.
            if instruction.id == arm.ARM_INS_STMDB:
                popped = False
                _push(state, _registers(instruction)[1:], 4)
            else:
                popped = _registers(instruction)[1:]
        elif instruction.id == arm.ARM_INS_STR and PUSH_ONE.match(text):
            popped = False
            _push(state, [REGISTER_NAMES.get(PUSH_ONE.match(text)[1], PUSH_ONE.match(text)[1])], 4)
        elif instruction.id == arm.ARM_INS_LDR and POP_ONE.match(text):
            popped = [REGISTER_NAMES.get(POP_ONE.match(text)[1], POP_ONE.match(text)[1])]
        if popped is not None:
            if conditional and (popped is False or "pc" not in popped):
                raise NotRun("conditional-stack-change", address, f"{instruction.mnemonic} {text}")
            if popped:
                returned = _pop(state, popped, 8 if wide else 4)
                if returned is not None:
                    leave(address, "return", state, returned)
                    return [(following, untouched)] if conditional else []
            deepest = max(deepest, state.depth)
            return [(following, state)]

        # Everything else.
        if "pc" in written:
            raise NotRun("indirect-jump", address, f"{instruction.mnemonic} {text}")
        if instruction.id in (arm.ARM_INS_TBB, arm.ARM_INS_TBH):
            raise NotRun("indirect-jump", address, f"{instruction.mnemonic} {text}")
        if "sp" in written:
            if conditional:
                raise NotRun("conditional-stack-change", address, f"{instruction.mnemonic} {text}")
            if not _stack_pointer_write(instruction, state):
                raise NotRun("untracked-stack-pointer", address, f"{instruction.mnemonic} {text}")
            deepest = max(deepest, state.depth)
        alias = _new_alias(instruction, state)
        for register in written:
            state.alias.pop(register, None)
        if alias is not None and not conditional:
            state.alias[alias[0]] = alias[1]
        state.clobbered |= written & TRACKED
        return [(following, state)]

    after_call: set[int] = set()
    while work:
        address, state = work.pop()
        if not start <= address < end:
            if address not in after_call:
                not_run.setdefault("runs-past-end", []).append(f"0x{address:x}")
            continue
        known = states.get(address)
        if known is not None:
            if known.depth != state.depth:
                not_run.setdefault("inconsistent-stack", []).append(
                    f"0x{address:x}: depth {known.depth} and {state.depth}"
                )
                continue
            merged = known.merge(state)
            if merged.key() == known.key():
                continue
            state = merged
        states[address] = state.copy()
        try:
            successors = step(address, state)
        except NotRun as stop:
            row = f"0x{stop.address:x}" + (f": {stop.detail}" if stop.detail else "")
            not_run.setdefault(stop.reason, []).append(row)
            continue
        data = memory.read(address, 4) or b""
        call = next(decoder.disasm(data, address, 1), None)
        if call is not None and check_trampoline.branch_base(call.mnemonic) in ("bl", "blx"):
            # A call that does not return can be the last instruction.
            after_call.update(target for target, _ in successors)
        work.extend(successors)

    verdict = "fail" if problems else "not-run" if not_run else "pass"
    return {
        "address": f"0x{start:x}",
        "size": size,
        "mode": mode,
        "instructions": len(states),
        "stack_depth": None if not_run else deepest,
        "exits": sorted(exits, key=lambda row: int(row["address"], 16)),
        "problems": problems,
        "not_run": not_run,
        "verdict": verdict,
    }


def _tokens(memory: check_trampoline.Memory, low: int, high: int, mode: str):
    """(address, token) per instruction of [low, high), decoded linearly.

    A token is the instruction as written, with what depends on where it sits
    taken out: a branch names its distance, a literal load only that it is one.
    """
    decoder = capstone.Cs(
        capstone.CS_ARCH_ARM, capstone.CS_MODE_ARM if mode == "arm" else capstone.CS_MODE_THUMB
    )
    step = 4 if mode == "arm" else 2
    address = low
    data = memory.read(low, high - low)
    if data is None:
        return
    while address < high:
        decoded = False
        for where, size, mnemonic, operands in decoder.disasm_lite(data[address - low:], address):
            decoded = True
            base = check_trampoline.branch_base(mnemonic)
            if base in ("b", "cbz", "cbnz"):
                target = check_trampoline.direct_target(mnemonic, operands)
                if target is not None:
                    operands = operands.rsplit("#", 1)[0] + f"{target - where:+#x}"
            operands = re.sub(r"\[pc, #-?(0x)?[0-9a-f]+\]", "[pc]", operands)
            yield where, f"{mnemonic} {operands}"
            address = where + size
        if not decoded or address < high:
            address += step  # data between functions; resume after it


def inlined_copies(
    image: Path, start: int, size: int, mode: str, minimum: int = 12, share: float = 0.5
) -> list[dict]:
    """Runs elsewhere in `image` that repeat a large part of the body.

    A run counts when it is at least `minimum` instructions and at least
    `share` of the body's instructions.
    """
    memory = check_trampoline.Memory(image)
    body = [token for _, token in _tokens(memory, start, start + size, mode)]
    minimum = max(minimum, int(len(body) * share + 0.999))
    if len(body) < minimum:
        return []
    grams: dict[tuple, list[int]] = {}
    for index in range(len(body) - minimum + 1):
        grams.setdefault(tuple(body[index:index + minimum]), []).append(index)

    copies = []
    with image.open("rb") as handle:
        executable = [
            (int(s["p_vaddr"]), int(s["p_vaddr"]) + int(s["p_filesz"]))
            for s in ELFFile(handle).iter_segments()
            if s["p_type"] == "PT_LOAD" and s["p_flags"] & PF_X and s["p_filesz"]
        ]
    for low, high in executable:
        code = [row for row in _tokens(memory, low, high, mode)
                if not start <= row[0] < start + size]
        tokens = [token for _, token in code]
        covered = -1
        for position in range(len(tokens) - minimum + 1):
            if position <= covered:
                continue
            best = 0, 0
            for index in grams.get(tuple(tokens[position:position + minimum]), ()):
                length = minimum
                while (
                    position + length < len(tokens)
                    and index + length < len(body)
                    and tokens[position + length] == body[index + length]
                    # Stay within one contiguous run of code.
                    and code[position + length][0] > code[position + length - 1][0]
                ):
                    length += 1
                best = max(best, (length, index))
            if best[0]:
                copies.append({
                    "address": f"0x{code[position][0]:x}",
                    "instructions": best[0],
                    "body_instruction": best[1],
                })
                covered = position + best[0] - 1
    return copies


def functions_from_record(record: dict) -> tuple[list[dict], dict]:
    """The inserted functions and the replaced target, as `check_static` takes them."""
    mode = record["target"].get("mode", "thumb")
    functions = [
        {"name": row["name"], "address": int(row["address"], 16), "size": row["size"],
         "mode": mode}
        for row in record["functions"]
        if row["address"] is not None
    ]
    target = {
        "name": record["target"]["name"],
        "address": int(record["target"]["address"], 16),
        "size": record["target"]["size"],
        "mode": mode,
    }
    return functions, target


def check_static(
    original: Path,
    rewritten: Path,
    functions: list[dict],
    target: dict,
    budgets: dict | None = None,
    minimum_copy: int = 12,
) -> dict:
    """Analyze each inserted function in `rewritten` and the target in `original`.

    `functions` and `target` are dicts with `name`, `address`, `size`, `mode`.
    `budgets` may give `code_bytes` and `stack_bytes`.
    """
    budgets = budgets or {}
    after = check_trampoline.Memory(rewritten)
    rows = [
        {"name": function["name"],
         **analyze_function(after, function["address"], function["size"], function["mode"])}
        for function in functions
    ]
    before = analyze_function(
        check_trampoline.Memory(original), target["address"], target["size"], target["mode"]
    )
    code_bytes = sum(function["size"] for function in functions)
    depths = [row["stack_depth"] for row in rows]
    stack_bytes = None if None in depths else sum(depths)

    def judged(limit, measured, problem: str) -> dict:
        if limit is None:
            verdict = "not-judged"
        elif measured is None:
            verdict = "not-run"
        else:
            verdict = "pass" if measured <= limit else problem
        return {"limit": limit, "measured": measured, "verdict": verdict}

    replacement = next((row for row in rows if row["name"] == target["name"]), None)
    growth = None
    if replacement and replacement["stack_depth"] is not None \
            and before["stack_depth"] is not None:
        growth = replacement["stack_depth"] - before["stack_depth"]
    return {
        "original_image": str(original),
        "rewritten_image": str(rewritten),
        "functions": rows,
        "original": {"name": target["name"], **before},
        "code_bytes": code_bytes,
        "stack_bytes": stack_bytes,
        "stack_growth": growth,
        "budgets": {
            "code_bytes": judged(budgets.get("code_bytes"), code_bytes, "code-size-over-budget"),
            "stack_bytes": judged(budgets.get("stack_bytes"), stack_bytes, "stack-over-budget"),
        },
        "inlined_copies": inlined_copies(
            original, target["address"], target["size"], target["mode"], minimum_copy
        ),
    }


def failures(report: dict) -> list[str]:
    """The problems in the report, each once, in a fixed order."""
    found = {problem for row in report["functions"] for problem in row["problems"]}
    found.update(
        budget["verdict"] for budget in report["budgets"].values()
        if budget["verdict"] in PROBLEMS
    )
    return [problem for problem in PROBLEMS if problem in found]


def not_run(report: dict) -> list[str]:
    """Why some inserted function could not be analyzed, each reason once."""
    return sorted({reason for row in report["functions"] for reason in row["not_run"]})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--elf", type=Path, required=True, help="original image")
    parser.add_argument("--rewritten", type=Path, required=True)
    parser.add_argument("--max-code-bytes", type=int)
    parser.add_argument("--max-stack-bytes", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
        functions, target = functions_from_record(record)
        report = check_static(
            args.elf, args.rewritten, functions, target,
            {"code_bytes": args.max_code_bytes, "stack_bytes": args.max_stack_bytes},
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    failed, skipped = failures(report), not_run(report)
    if failed:
        print(f"static checks: {', '.join(failed)}", file=sys.stderr)
    if skipped:
        print(f"static checks not run: {', '.join(skipped)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
