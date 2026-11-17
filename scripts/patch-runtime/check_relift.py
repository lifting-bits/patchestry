#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Lift the patched function back from the rewritten image and compare it
with the code that was meant to be inserted.

Every other check of the inserted code starts from the patched IR.  This one
starts from the bytes: Ghidra lifts the rewritten ELF again, the lifted
function is lowered to LLVM IR the same way the original was, and
`check_equivalence` compares it with the intended IR.  Nothing is declared
for that comparison: no reject path, no pure callee, no substitution.  The
two must agree on every input.  Device register ranges and balanced pairs may
be given; they add things to compare and relax nothing.

Steps, each with its file in the output directory:

1. `relift/all.json`: every function of the rewritten image, one Ghidra run.
   With `--single-function` only the target is lifted and step 2 is skipped;
   that is enough when the patch inserted no helper, and the only practical
   way on a large image.
2. `relift/function.json`: the target and, transitively, the functions it
   calls that have no symbol (Ghidra names them `FUN_<address>`): those are
   the helpers the patch inserted.  Every other function keeps its
   declaration only.
3. `relift/function.cir`, `relift/function.ll`: `patchir-decomp`, then
   `patchir-cir2llvm`.
4. `relift/equivalence.json`: the comparison, intended IR as the original
   and the re-lifted IR as the patched version.

The report (`relift.json`) has one verdict:

| Verdict | Meaning |
| --- | --- |
| `tested` | the two agreed on every input that ran to a conclusion |
| `failed` | a property differs; `violations` says which and on which input |
| `not-run` | a step failed or no property could be compared; `reason` says why |

What a `failed` means: the bytes in the image, as Ghidra reads them, do not
behave like the intended code.  The cause is the inserted bytes or the lift;
the report cannot tell which.  A device write Ghidra's decompiler removes as
a dead store fails here although the bytes are right.

The check is as independent as its parts: it shares Ghidra, `patchir-decomp`
and the comparison harness with the rest of the pipeline, and shares neither
the compiler nor Patcherex2.

CLI: check_relift.py --rewritten ELF --function NAME --intended IR
     --output-dir DIR --decompile-script PATH --patchir-decomp PATH
     --patchir-cir2llvm PATH [--mmio-region LOW:HIGH] [--balance A:R]
     [--arg-buffer INDEX:SIZE] [--inputs N] [--single-function]
     [--address ADDRESS]
Exit 0 when tested, 1 when failed, 3 when not run, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import check_equivalence

# The fields of a lifted function that hold its body.
BODY_KEYS = ("basic_blocks", "entry_block", "region")
# Verdicts of these must not be `failed`; P2 may be `not-run` (no memory).
REQUIRED = ("P1", "P3", "P4")


@dataclass(frozen=True)
class LiftTools:
    decompile_script: Path
    patchir_decomp: Path
    patchir_cir2llvm: Path


def locate(lifted: dict, function: str, address: int | None) -> str:
    """The key of the target among the lifted functions, named `function` on return.

    Ghidra names the function by its symbol when the old entry still leads to
    it.  When the entry is a branch to moved code in an image without
    symbols, the body is a function of its own at the destination; `address`
    finds it, and it is given the name the intended code uses.
    """
    functions = lifted["functions"]
    found = [key for key, row in functions.items()
             if row["name"] == function and row.get("basic_blocks")]
    if not found and address is not None:
        found = [key for key, row in functions.items()
                 if key.rsplit(":", 1)[-1].lower() == f"{address:08x}" and row.get("basic_blocks")]
    if len(found) != 1:
        raise ValueError(f"{len(found)} lifted functions are {function}"
                         + (f" or at {address:#x}" if address is not None else ""))
    functions[found[0]]["name"] = function
    if "display_name" in functions[found[0]]:
        functions[found[0]]["display_name"] = function
    return found[0]


def select(lifted: dict, start: str) -> tuple[dict, list[str]]:
    """`lifted` with bodies only for the function at `start` and the unnamed ones it reaches."""
    functions = lifted["functions"]
    kept, pending = [], [start]
    while pending:
        key = pending.pop()
        if key in kept:
            continue
        kept.append(key)
        body = json.dumps({name: functions[key].get(name) for name in BODY_KEYS})
        for other, row in functions.items():
            if (other not in kept and row["name"].startswith("FUN_")
                    and row.get("basic_blocks") and f'"{other}"' in body):
                pending.append(other)
    chosen = dict(lifted)
    chosen["functions"] = {
        key: row if key in kept else {k: v for k, v in row.items() if k not in BODY_KEYS}
        for key, row in functions.items()
    }
    return chosen, [functions[key]["name"] for key in kept]


def check_relift(
    rewritten: Path, function: str, intended: Path, out: Path,
    tools: check_equivalence.Tools, lift: LiftTools, inputs: int = 200,
    mmio_regions=(), balance=(), arg_buffers=None, whole_image: bool = True,
    address: int | None = None,
) -> dict:
    work = out / "relift"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    report = {
        "function": function,
        "rewritten": str(rewritten),
        "intended": str(intended),
        "lifted": [],
        "verdict": "not-run",
        "reason": None,
        "properties": {},
        "violations": [],
        "equivalence": None,
    }

    def step(name: str, command: list[str]) -> bool:
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        (work / f"{name}.log").write_text(done.stdout + done.stderr, encoding="utf-8")
        if done.returncode != 0:
            report["reason"] = f"{name} failed (exit {done.returncode}); see {work / (name + '.log')}"
        return done.returncode == 0

    # The script mounts the output directory into its container and reads the
    # image from there.
    image = work / rewritten.name
    shutil.copyfile(rewritten, image)
    lift_command = ["bash", str(lift.decompile_script), "--input", str(image),
                    "--output", str(work / "all.json"), "--ci", str(work.resolve())]
    if not whole_image:
        lift_command += ["--function", function if address is None else f"0x{address:08x}"]
    if not step("ghidra", lift_command):
        return report
    try:
        lifted = json.loads((work / "all.json").read_text(encoding="utf-8"))
        start = locate(lifted, function, address)
        chosen, names = select(lifted, start) if whole_image else (lifted, [function])
    except (OSError, ValueError, KeyError) as error:
        report["reason"] = f"the lifted image cannot be used: {error}"
        return report
    report["lifted"] = names
    (work / "function.json").write_text(json.dumps(chosen), encoding="utf-8")
    if not step("decomp", [str(lift.patchir_decomp), "-input", str(work / "function.json"),
                           "-emit-cir", "-output", str(work / "function")]):
        return report
    if not step("cir2llvm", [str(lift.patchir_cir2llvm), "-S", str(work / "function.cir"),
                             "-o", str(work / "function.ll")]):
        return report

    compared = check_equivalence.check_equivalence(
        intended, work / "function.ll", function, work, tools, inputs=inputs,
        arg_buffers=arg_buffers, mmio_regions=mmio_regions, balance=balance,
    )
    (work / "equivalence.json").write_text(json.dumps(compared, indent=2) + "\n", encoding="utf-8")
    report["equivalence"] = str(work / "equivalence.json")
    report["properties"] = compared["properties"]
    report["violations"] = compared["violations"][:20]
    report["outcomes"] = {key: count for key, count in compared["outcomes"].items() if count}
    failed = check_equivalence.failures(compared)
    if failed:
        report["verdict"] = "failed"
        report["reason"] = "differs from the intended code: " + ", ".join(failed)
        return report
    untested = [name for name in REQUIRED if compared["properties"][name]["verdict"] != "tested"]
    if untested:
        report["reason"] = "; ".join(
            f"{name}: {compared['properties'][name]['reason']}" for name in untested
        )
        return report
    report["verdict"] = "tested"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rewritten", type=Path, required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--intended", type=Path, required=True, help="patched LLVM IR")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decompile-script", type=Path, required=True)
    parser.add_argument("--patchir-decomp", type=Path, required=True)
    parser.add_argument("--patchir-cir2llvm", type=Path, required=True)
    parser.add_argument("--mmio-region", action="append", default=[], metavar="LOW:HIGH")
    parser.add_argument("--balance", action="append", default=[], metavar="ACQUIRE:RELEASE")
    parser.add_argument("--arg-buffer", action="append", default=[], metavar="INDEX:SIZE")
    parser.add_argument("--inputs", type=int, default=200)
    parser.add_argument("--single-function", action="store_true",
                        help="lift only the target, not the helpers it calls")
    parser.add_argument("--address", type=lambda text: int(text, 0),
                        help="where the patched body is, when the image has no symbol for it "
                             "(the destination in trampoline.json)")
    args = parser.parse_args()
    try:
        tools = check_equivalence.Tools.from_environment()
        regions = [tuple(int(part, 0) for part in text.split(":")) for text in args.mmio_region]
        pairs = [tuple(text.split(":")) for text in args.balance]
        buffers = {int(i): int(size, 0) for i, size in
                   (text.split(":") for text in args.arg_buffer)}
        if any(len(row) != 2 for row in regions + pairs):
            raise ValueError("--mmio-region and --balance take two parts")
    except (check_equivalence.MissingTool, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = check_relift(
        args.rewritten, args.function, args.intended, args.output_dir, tools,
        LiftTools(args.decompile_script, args.patchir_decomp, args.patchir_cir2llvm),
        inputs=args.inputs, mmio_regions=regions, balance=pairs, arg_buffers=buffers or None,
        whole_image=not args.single_function, address=args.address,
    )
    (args.output_dir / "relift.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")
    print(f"{report['verdict']}: {report['function']}"
          + (f": {report['reason']}" if report["reason"] else ""))
    return {"tested": 0, "failed": 1}.get(report["verdict"], 3)


if __name__ == "__main__":
    sys.exit(main())
