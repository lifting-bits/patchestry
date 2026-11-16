#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Check how a replaced function was reinserted into a Thumb firmware image.

The rewriter either writes the new body over the old one, or writes a branch
at the old entry and puts the body somewhere else.  It does not check that the
function is as long as the branch, or that nothing else jumps into the bytes
it overwrote.  This script compares the original ELF with the rewritten one
and checks both.

Verdict:

  in-place               only bytes of the function changed and its entry
                         does not branch away
  trampoline             the entry is an unconditional Thumb branch to code
                         the rewriter added

Problems, any of which fails the check:

  unchanged              no byte of the function differs; nothing was patched
  unknown-extent         the function symbol has no size, so the bytes that
                         belong to it are not known
  not-thumb              the function symbol is not a Thumb function, or the
                         new entry switches mode (blx)
  span-outside-function  a changed byte of the original image is not part of
                         the function
  bad-destination        the branch target is not in an executable segment of
                         the rewritten image, lies in the original image, or
                         does not decode as Thumb code
  incoming-branch        a direct branch in another function targets an
                         address inside the function other than its entry
  alternate-entry        another function symbol starts inside the function,
                         or the Ghidra function record has more than one
                         address range or an entry that is not its lowest
                         address

A trampoline normally ends in the middle of the old body.  What is left is
dead code, reported as `tail_split` when the cut falls inside an instruction;
it only matters if something branches there, which is what incoming-branch
looks for.

Not checked: indirect branches and stored pointers to addresses inside the
function.  Branches are looked for in Thumb code as the mapping symbols mark
it, except in the gap between the end of a sized symbol and the next symbol,
which is taken to be linker padding.  Only ARM images whose code is Thumb are supported; the image must
also pass the static-image policy of check_bindings.py.

Exit status is 1 when the report lists a problem, 2 when the image or the
function cannot be checked at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import capstone
from elftools.elf.elffile import ELFFile

from check_bindings import PF_X, ElfSymbol, Image, UnsupportedImage, load_image

# Order in which problems are reported.
PROBLEMS = (
    "unchanged",
    "unknown-extent",
    "not-thumb",
    "span-outside-function",
    "bad-destination",
    "incoming-branch",
    "alternate-entry",
)

DIRECT_BRANCHES = ("b", "bl", "blx", "cbz", "cbnz")


class Memory:
    """File-backed bytes of an ELF's loaded segments, by virtual address."""

    def __init__(self, path: Path):
        self.chunks: list[tuple[int, bytes]] = []
        with path.open("rb") as handle:
            elf = ELFFile(handle)
            if elf["e_machine"] != "EM_ARM":
                raise UnsupportedImage(
                    f"{path}: machine is {elf['e_machine']}; only ARM Thumb images are supported"
                )
            for segment in elf.iter_segments():
                if segment["p_type"] == "PT_LOAD" and segment["p_filesz"]:
                    self.chunks.append((int(segment["p_vaddr"]), segment.data()))

    def read(self, address: int, size: int) -> bytes | None:
        for start, data in self.chunks:
            if start <= address and address + size <= start + len(data):
                return data[address - start:address - start + size]
        return None

    def byte(self, address: int) -> int | None:
        data = self.read(address, 1)
        return data[0] if data else None


def changed_runs(original: Memory, rewritten: Memory) -> list[tuple[int, int]]:
    """Half-open address ranges of the original image whose bytes differ."""
    runs: list[list[int]] = []
    for start, data in original.chunks:
        for offset, value in enumerate(data):
            address = start + offset
            if rewritten.byte(address) == value:
                continue
            if runs and runs[-1][1] == address:
                runs[-1][1] = address + 1
            else:
                runs.append([address, address + 1])
    return [(low, high) for low, high in runs]


def thumb_runs(image: Image) -> list[tuple[int, int]]:
    """Half-open ranges of executable segments that hold Thumb code.

    ARM mapping symbols ($t, $a, $d) mark where each kind of content starts.
    An image without them is taken to be Thumb code throughout.
    """
    executable = [s for s in image.segments if s.flags & PF_X]
    marks = sorted(
        (symbol.addr, name[:2])
        for name, group in image.symbols.items()
        if name[:2] in ("$t", "$a", "$d") and (len(name) == 2 or name[2] == ".")
        for symbol in group
    )
    if not marks:
        return [(s.start, s.end) for s in executable]
    runs = []
    for index, (address, kind) in enumerate(marks):
        if kind != "$t":
            continue
        segment = next((s for s in executable if s.start <= address < s.end), None)
        if segment is None:
            continue
        end = segment.end
        if index + 1 < len(marks):
            end = min(end, max(marks[index + 1][0], address))
        if end > address:
            runs.append((address, end))
    return runs


def padding_gaps(image: Image) -> list[tuple[int, int]]:
    """Ranges between the end of a sized symbol and the next symbol.

    The linker fills alignment gaps with a trap pattern (lld uses 0xd4 bytes,
    which read as `bmi`).  That is not code and must not be decoded as a
    branch.  An unsized symbol, such as an assembly entry point, has no known
    end and so no gap after it.
    """
    starts = sorted({s.addr for group in image.symbols.values() for s in group})
    gaps = []
    for symbol in image.by_address:
        if not symbol.size:
            continue
        end = symbol.addr + symbol.size
        following = next((address for address in starts if address >= end), None)
        if following is not None and following > end:
            gaps.append((end, following))
    return gaps


def code_runs(image: Image) -> list[tuple[int, int]]:
    """Thumb runs with the padding between symbols taken out."""
    runs = thumb_runs(image)
    for gap_low, gap_high in padding_gaps(image):
        kept = []
        for low, high in runs:
            if gap_high <= low or high <= gap_low:
                kept.append((low, high))
                continue
            if low < gap_low:
                kept.append((low, gap_low))
            if gap_high < high:
                kept.append((gap_high, high))
        runs = kept
    return sorted(runs)


def decode_thumb(memory: Memory, start: int, end: int):
    """Yield (address, size, mnemonic, operands) for Thumb code in [start, end)."""
    decoder = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB)
    address = start
    while address < end:
        data = memory.read(address, min(4, end - address))
        if data is None:
            # Past the file-backed part of the segment (or a 2-byte tail).
            data = memory.read(address, 2) if end - address >= 2 else None
            if data is None:
                return
        instruction = next(decoder.disasm(data, address, 1), None)
        if instruction is None:
            address += 2  # not an instruction; resynchronize on the next halfword
            continue
        yield address, instruction.size, instruction.mnemonic, instruction.op_str
        address += instruction.size


def branch_base(mnemonic: str) -> str:
    """Strip the width suffix and condition code: `bne.w` -> `b`."""
    name = mnemonic.split(".")[0]
    for base in ("cbnz", "cbz", "blx", "bl", "bx"):
        if name == base:
            return base
    if name == "b" or (name.startswith("b") and len(name) == 3):
        return "b"
    return name


def direct_target(mnemonic: str, operands: str) -> int | None:
    """The target of a direct branch, or None for anything else."""
    if branch_base(mnemonic) not in DIRECT_BRANCHES:
        return None
    last = operands.split(",")[-1].strip()
    if not last.startswith("#"):
        return None
    return int(last[1:], 0)


def is_unconditional_branch(mnemonic: str) -> bool:
    return mnemonic.split(".")[0] == "b"


def owner(image: Image, address: int) -> ElfSymbol | None:
    """The sized symbol whose bytes include `address`."""
    for symbol in image.by_address:
        if symbol.size and symbol.addr <= address < symbol.addr + symbol.size:
            return symbol
    return None


def parse_ghidra_address(text: str) -> int:
    return int(text.rsplit(":", 1)[-1], 16)


def ghidra_entries(function_json: dict, name: str) -> list[str]:
    """Reasons the Ghidra record of `name` describes more than one entry."""
    reasons = []
    for key, function in function_json.get("functions", {}).items():
        if function.get("name") != name:
            continue
        ranges = function.get("address_ranges") or []
        if len(ranges) > 1:
            reasons.append(f"{len(ranges)} address ranges")
        if ranges:
            entry = parse_ghidra_address(function.get("entry_point") or key)
            lowest = min(parse_ghidra_address(r["start"]) for r in ranges)
            if entry != lowest:
                reasons.append(f"entry 0x{entry:x} is not the lowest address 0x{lowest:x}")
    return reasons


def check_trampoline(
    original: Path, rewritten: Path, function: str, function_json: dict | None = None
) -> dict:
    """Compare `function` in the two images and return the report.

    Raises UnsupportedImage when either ELF is not a static ARM image and
    LookupError when the original has no single function symbol of that name.
    """
    image = load_image(original)
    after_image = load_image(rewritten)
    before = Memory(original)
    after = Memory(rewritten)

    candidates = [s for s in image.symbols.get(function, []) if s.kind == "STT_FUNC"]
    if len(candidates) != 1:
        raise LookupError(
            f"{original}: expected one function symbol named '{function}', "
            f"found {len(candidates)}"
        )
    symbol = candidates[0]
    start, end = symbol.addr, symbol.addr + symbol.size

    problems: dict[str, list] = {}
    report: dict = {
        "original": str(original),
        "rewritten": str(rewritten),
        "function": function,
        "address": f"0x{start:x}",
        "size": symbol.size,
    }

    if not symbol.size:
        problems["unknown-extent"] = []
    if not symbol.thumb:
        problems["not-thumb"] = ["the function symbol does not have the Thumb bit set"]

    runs = changed_runs(before, after)
    inside = [(low, high) for low, high in runs if start <= low and high <= end]
    outside = [run for run in runs if run not in inside]
    report["changed"] = [[f"0x{low:x}", f"0x{high:x}"] for low, high in runs]
    if outside:
        rows = []
        for low, high in outside:
            # The part of the run that is not the function's own bytes.
            spill = low if not start <= low < end else end
            other = owner(image, spill)
            rows.append({
                "range": [f"0x{low:x}", f"0x{high:x}"],
                "belongs_to": other.name if other else None,
            })
        problems["span-outside-function"] = rows
    touched = [run for run in runs if run[0] < max(end, start + 1) and run[1] > start]
    if not touched:
        problems["unchanged"] = []

    # What the entry does now.
    verdict = "in-place"
    entry = next(decode_thumb(after, start, start + 4), None)
    span_end = max((high for _, high in touched), default=start)
    if entry is not None:
        _, size, mnemonic, operands = entry
        report["entry"] = {
            "before": (before.read(start, size) or b"").hex(),
            "after": (after.read(start, size) or b"").hex(),
            "instruction": f"{mnemonic} {operands}".strip(),
        }
        target = direct_target(mnemonic, operands)
        if branch_base(mnemonic) == "blx":
            problems.setdefault("not-thumb", []).append(
                f"the entry is '{mnemonic} {operands}', which switches to ARM mode"
            )
        elif is_unconditional_branch(mnemonic) and target is not None and touched \
                and not start <= target < end:
            verdict = "trampoline"
            # The branch is the whole instruction even when some of its bytes
            # equal the ones they replaced.
            span_end = max(span_end, start + size)
            report["destination"] = f"0x{target:x}"
            report["span"] = [f"0x{start:x}", f"0x{span_end:x}"]
            segment = after_image.segment_at(target)
            reason = None
            if segment is None:
                reason = "not in a loaded segment of the rewritten image"
            elif not segment.flags & PF_X:
                reason = f"in a segment with permissions {segment.permissions()}"
            elif image.segment_at(target) is not None:
                reason = "inside the original image"
            elif next(decode_thumb(after, target, target + 4), None) is None:
                reason = "does not decode as Thumb code"
            if reason:
                problems["bad-destination"] = [f"0x{target:x}: {reason}"]

            # Where the cut falls in the old body.
            boundaries = {start}
            for address, size, _, _ in decode_thumb(before, start, end):
                boundaries.add(address + size)
            report["tail_split"] = span_end < end and span_end not in boundaries

    # Direct branches into the body.  After a replacement the old interior
    # addresses hold either new code or the dead remainder of the old body.
    incoming = []
    for low, high in code_runs(image):
        for address, _, mnemonic, operands in decode_thumb(before, low, high):
            target = direct_target(mnemonic, operands)
            if target is None or not start < target < end:
                continue
            source = owner(image, address)
            incoming.append({
                "from": f"0x{address:x}",
                "in": source.name if source else None,
                "to": f"0x{target:x}",
                "instruction": f"{mnemonic} {operands}",
                "internal": start <= address < end,
            })
    report["incoming"] = incoming
    external = [row for row in incoming if not row["internal"]]
    if external:
        problems["incoming-branch"] = external

    entries = [
        f"{other.name} at 0x{other.addr:x}"
        for other in image.by_address
        if other.kind == "STT_FUNC" and start < other.addr < end
    ]
    if function_json is not None:
        entries += ghidra_entries(function_json, function)
    if entries:
        problems["alternate-entry"] = entries

    report["verdict"] = verdict
    report["problems"] = {name: problems[name] for name in PROBLEMS if name in problems}
    return report


def failures(report: dict) -> list[str]:
    return list(report["problems"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--elf", type=Path, required=True, help="ELF before reinsertion")
    parser.add_argument("--rewritten", type=Path, required=True, help="ELF after reinsertion")
    parser.add_argument("--function", required=True, help="name of the replaced function")
    parser.add_argument("--function-json", type=Path,
                        help="Ghidra JSON the function was lifted from (for address ranges)")
    parser.add_argument("--output", type=Path, help="write the report here (default: stdout)")
    args = parser.parse_args()

    function_json = None
    if args.function_json:
        function_json = json.loads(args.function_json.read_text(encoding="utf-8"))
    try:
        report = check_trampoline(args.elf, args.rewritten, args.function, function_json)
    except (UnsupportedImage, LookupError) as error:
        print(f"check-trampoline: {error}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    failed = failures(report)
    print(
        f"check-trampoline: {args.function} at {report['address']}: {report['verdict']}"
        + (f" to {report['destination']}" if "destination" in report else "")
        + (f"; problems: {', '.join(failed)}" if failed else ""),
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
