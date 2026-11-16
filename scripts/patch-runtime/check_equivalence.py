#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Compare a lifted function with its patched version on concrete inputs.

`patchir-klee-verifier --compare-original` builds a product harness: one
module with both versions of the function, each with its own globals, and a
main() that runs them on the same input and ends every path with a code.  This
compiles that harness for ARM, links it with a small bare-metal runtime
(equivalence_runtime/), and runs it under `qemu-system-arm` on a series of
inputs derived from seeds.  The modules keep their 32-bit ARM layout: nothing
is retargeted.

Two properties are judged per input:

| Property | Holds when |
| --- | --- |
| P1 | an input the patched version accepts is one the original accepts |
| P3 | when both accept, they return the same value |

A version rejects an input when it calls a declared reject handler or returns
a declared reject value.  Violations:

| Property | Kind | Meaning |
| --- | --- | --- |
| P1 | `original-rejects` | the patched version accepted an input the original rejects |
| P1 | `original-faults`, `original-aborts` | the patched version accepted an input the original crashes on |
| P3 | `return-differs` | both accepted, with different return values |
| P3 | `patched-faults`, `patched-aborts` | the patched version crashes on an input the original accepts |

Verdicts: `failed` when a violation was seen; `tested` when none was seen and
at least one input reached the comparison (both versions ran and accepted);
`not-run` otherwise.  `tested` covers the inputs that were run and no others.
`proven` needs the same harness explored by KLEE, which this does not do.

Not compared here: memory the versions write (P2) and the calls they make
(P4).  An external function returns the same value at the same call index on
both sides, whatever its arguments.

CLI: check_equivalence.py --original original.ll --patched patched.ll
     --function NAME --output-dir DIR [--reject-handler NAME]...
     [--reject-return VALUE]... [--arg-buffer INDEX:SIZE]...
     [--arg-range INDEX:LOW:HIGH]... [--inputs N]
Tools come from PATCHESTRY_LLVM_PREFIX, PATCHESTRY_LD_LLD, QEMU_SYSTEM_ARM and
PATCHESTRY_KLEE_VERIFIER (default: the Debug build in this repository).
Exit 0 when no property failed, 1 when one did, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

RUNTIME = Path(__file__).with_name("equivalence_runtime")
TARGET_FLAGS = ["--target=armv7a-none-eabi", "-mcpu=cortex-a15", "-mfloat-abi=soft"]
BATCH = 50  # inputs per boot
BATCH_TIMEOUT = 5  # seconds; a boot of 50 inputs takes a fraction of one
MAX_TIMEOUTS = 8  # inputs that never end, before the run is cut short

# Path-end codes of the harness (tools/patchir-klee-verifier/Compare.cpp).
ACCEPTED, PATCHED_REJECTED, P1_VIOLATION, P3_VIOLATION, UNMODELLED = 0, 1, 10, 11, 20
ORIGINAL_ALONE_ACCEPTED, ORIGINAL_ALONE_REJECTED = 30, 31
ORIGINAL, PATCHED = 1, 2

OUTCOMES = (
    "accepted", "patched-rejected", "p1-violation", "p3-violation", "unmodelled",
    "both-fail", "discarded", "timeout", "unknown",
)


class MissingTool(RuntimeError):
    pass


@dataclass(frozen=True)
class Tools:
    clang: Path
    lld: Path
    qemu: Path
    verifier: Path

    @classmethod
    def from_environment(cls) -> "Tools":
        prefix = os.environ.get("PATCHESTRY_LLVM_PREFIX")
        if not prefix:
            raise MissingTool("PATCHESTRY_LLVM_PREFIX is not set")
        repo = Path(__file__).resolve().parents[2]
        found = {
            "clang": Path(prefix) / "clang",
            "lld": Path(os.environ.get("PATCHESTRY_LD_LLD") or Path(prefix) / "ld.lld"),
            "qemu": Path(os.environ.get("QEMU_SYSTEM_ARM")
                         or shutil.which("qemu-system-arm") or "qemu-system-arm"),
            "verifier": Path(
                os.environ.get("PATCHESTRY_KLEE_VERIFIER")
                or repo / "builds/default/tools/patchir-klee-verifier/Debug/patchir-klee-verifier"
            ),
        }
        for name, path in found.items():
            if not path.is_file():
                raise MissingTool(f"{name} not found at {path}")
        return cls(**found)


def _run(command: list[str], **options) -> subprocess.CompletedProcess:
    return subprocess.run([str(part) for part in command], capture_output=True, text=True,
                          check=False, **options)


def build_harness(
    original: Path, patched: Path, function: str, out: Path, tools: Tools,
    reject_handlers=(), reject_returns=(), arg_buffers=None, pointer_size: int = 64,
    arg_ranges=None,
) -> tuple[dict | None, str]:
    """Write `harness.ll` and `harness.json` into `out`; (info, error text)."""
    command = [
        tools.verifier, patched, "--compare-original", original, "--target-function", function,
        "--keep-target", "--symbolic-ptr-size", str(pointer_size),
        "--harness-info", out / "harness.json", "-S", "-o", out / "harness.ll",
    ]
    for handler in reject_handlers:
        command += ["--reject-handler", handler]
    for value in reject_returns:
        command += [f"--reject-return={value}"]
    for index, size in sorted((arg_buffers or {}).items()):
        command += ["--arg-buffer", f"{index}:{size}"]
    for index, (low, high) in sorted((arg_ranges or {}).items()):
        command += [f"--arg-range={index}:{low}:{high}"]
    result = _run(command)
    if result.returncode != 0:
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        return None, (lines[-1] if lines else "patchir-klee-verifier failed").split(") ", 1)[-1]
    return json.loads((out / "harness.json").read_text(encoding="utf-8")), ""


def link_image(out: Path, tools: Tools, first_seed: int, count: int) -> str:
    """Build `harness.elf` for seeds [first_seed, first_seed + count); error text."""
    compile_flags = [*TARGET_FLAGS, "-ffreestanding", "-fno-builtin", "-w"]
    steps = [
        [tools.clang, *compile_flags, "-O0", "-c", out / "harness.ll", "-o", out / "harness.o"],
        [tools.clang, *compile_flags, "-O1", f"-DFIRST_SEED={first_seed}",
         f"-DSEED_COUNT={count}", "-c", RUNTIME / "runtime.c", "-o", out / "runtime.o"],
        [tools.clang, *compile_flags, "-c", RUNTIME / "start.S", "-o", out / "start.o"],
        [tools.lld, "-T", RUNTIME / "link.ld", out / "start.o", out / "runtime.o",
         out / "harness.o", "-o", out / "harness.elf"],
    ]
    if (out / "harness.o").exists() and first_seed != 0:
        steps = steps[1:]  # only the seeds changed
    for step in steps:
        result = _run(step)
        if result.returncode != 0:
            lines = [line for line in result.stderr.splitlines() if "error" in line]
            return lines[0] if lines else f"{Path(str(step[0])).name} failed"
    return ""


def run_batch(out: Path, tools: Tools, first_seed: int, count: int) -> tuple[list[tuple], bool]:
    """Run one boot; ([(seed, kind, detail)...], finished)."""
    command = [
        str(tools.qemu), "-M", "virt", "-cpu", "cortex-a15", "-m", "32", "-nographic",
        "-monitor", "none", "-semihosting-config", "enable=on,target=native",
        "-kernel", str(out / "harness.elf"),
    ]
    try:
        # The semihosting console is the emulator's stderr.
        text = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, check=False, stdin=subprocess.DEVNULL,
                              timeout=BATCH_TIMEOUT).stdout
    except subprocess.TimeoutExpired as expired:
        text = expired.stdout or ""
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
    lines = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "I" and parts[1].isdigit() and parts[3].isdigit():
            lines.append((int(parts[1]), parts[2], int(parts[3])))
    return lines, "DONE" in text.split()


def classify(events: list[tuple[str, int]]) -> tuple[str, dict | None]:
    """Outcome and violation of one input, from its lines in order."""
    kind, detail = events[0]
    if kind == "end":
        if detail == ACCEPTED:
            return "accepted", None
        if detail == PATCHED_REJECTED:
            return "patched-rejected", None
        if detail == P1_VIOLATION:
            return "p1-violation", {"property": "P1", "kind": "original-rejects"}
        if detail == P3_VIOLATION:
            return "p3-violation", {"property": "P3", "kind": "return-differs"}
        if detail == UNMODELLED:
            return "unmodelled", None
        return "unknown", None
    if kind == "discard":
        return "discarded", None
    if kind in ("fault", "abort"):
        if detail == ORIGINAL:
            # The patched version ran to the end and accepted; the original died.
            return "p1-violation", {"property": "P1", "kind": f"original-{kind}s"}
        if detail == PATCHED and len(events) > 1:
            # The original was then run alone on the same input.
            if events[1] == ("end", ORIGINAL_ALONE_ACCEPTED):
                return "p3-violation", {"property": "P3", "kind": f"patched-{kind}s"}
            return "both-fail", None
    return "unknown", None


def _verdict(failed: bool, reached: int, outcomes: dict, what: str) -> dict:
    if failed:
        return {"verdict": "failed", "reason": None}
    if reached:
        return {"verdict": "tested", "reason": None}
    seen = ", ".join(f"{count} {name}" for name, count in outcomes.items() if count)
    return {"verdict": "not-run",
            "reason": f"no input {what} ({seen or 'no input was run'})"}


def check_equivalence(
    original: Path, patched: Path, function: str, out: Path, tools: Tools,
    reject_handlers=(), reject_returns=(), arg_buffers=None, inputs: int = 200,
    pointer_size: int = 64, arg_ranges=None,
) -> dict:
    """Build, run and judge; the report is also what `equivalence.json` holds."""
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("harness.o", "harness.elf", "harness.json", "harness.ll"):
        (out / stale).unlink(missing_ok=True)
    report = {
        "engine": "concrete",
        "runner": "qemu-system-arm -M virt -cpu cortex-a15, bare metal",
        "function": function,
        "original": str(original),
        "patched": str(patched),
        "reject_handlers": list(reject_handlers),
        "reject_returns": list(reject_returns),
        "inputs": 0,
        "outcomes": {name: 0 for name in OUTCOMES},
        "violations": [],
        "bounds": {"inputs": inputs, "pointer_size": pointer_size, "seeds": [0, inputs]},
        "harness": None,
        "cut_short": None,
    }

    def not_run(reason: str) -> dict:
        report["properties"] = {
            name: {"verdict": "not-run", "reason": reason} for name in ("P1", "P3")
        }
        return report

    info, error = build_harness(
        original, patched, function, out, tools, reject_handlers, reject_returns, arg_buffers,
        pointer_size, arg_ranges,
    )
    if info is None:
        return not_run(f"the product harness could not be built: {error}")
    report["harness"] = info
    report["bounds"]["stub_values"] = info["stub_values"]

    seed = 0
    while seed < inputs:
        count = min(BATCH, inputs - seed)
        error = link_image(out, tools, seed, count)
        if error:
            return not_run(f"the harness could not be compiled for ARM: {error}")
        lines, finished = run_batch(out, tools, seed, count)
        by_seed: dict[int, list] = {}
        for number, kind, detail in lines:
            by_seed.setdefault(number, []).append((kind, detail))
        done = seed
        for number in range(seed, seed + count):
            if number not in by_seed:
                break
            outcome, violation = classify(by_seed[number])
            report["outcomes"][outcome] += 1
            if violation:
                report["violations"].append({"seed": number, **violation})
            done = number + 1
        if not finished and done < seed + count:
            # The boot stopped inside this input (a loop that never ends, or
            # a crash of the emulator); skip it and go on.
            report["outcomes"]["timeout"] += 1
            done += 1
            if report["outcomes"]["timeout"] >= MAX_TIMEOUTS:
                report["cut_short"] = (
                    f"{MAX_TIMEOUTS} inputs did not end within {BATCH_TIMEOUT} seconds; "
                    f"stopped after seed {done - 1}"
                )
                break
        seed = done
    report["inputs"] = sum(report["outcomes"].values())

    outcomes = report["outcomes"]
    kinds = {row["property"] for row in report["violations"]}
    compared = outcomes["accepted"] + sum(
        1 for row in report["violations"] if row["kind"] in ("original-rejects", "return-differs")
    )
    report["properties"] = {
        "P1": _verdict("P1" in kinds, compared, outcomes,
                       "was accepted by the patched version and run through the original"),
        "P3": _verdict("P3" in kinds, outcomes["accepted"], outcomes,
                       "was accepted by both versions"),
    }
    report["reject_exercised"] = outcomes["patched-rejected"] > 0
    return report


def failures(report: dict) -> list[str]:
    return [name for name, row in report["properties"].items() if row["verdict"] == "failed"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--original", type=Path, required=True, help="lifted original, LLVM IR")
    parser.add_argument("--patched", type=Path, required=True, help="patched version, LLVM IR")
    parser.add_argument("--function", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reject-handler", action="append", default=[])
    parser.add_argument("--reject-return", action="append", type=int, default=[])
    parser.add_argument("--arg-buffer", action="append", default=[], metavar="INDEX:SIZE")
    parser.add_argument("--arg-range", action="append", default=[], metavar="INDEX:LOW:HIGH")
    parser.add_argument("--inputs", type=int, default=200)
    parser.add_argument("--pointer-size", type=int, default=64)
    args = parser.parse_args()
    try:
        buffers = {int(index): int(size, 0) for index, size in
                   (text.split(":", 1) for text in args.arg_buffer)}
        ranges = {int(index): (int(low, 0), int(high, 0)) for index, low, high in
                  (text.split(":", 2) for text in args.arg_range)}
        tools = Tools.from_environment()
    except (ValueError, MissingTool) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    report = check_equivalence(
        args.original, args.patched, args.function, args.output_dir, tools,
        args.reject_handler, args.reject_return, buffers, args.inputs, args.pointer_size,
        ranges,
    )
    (args.output_dir / "equivalence.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for name, row in report["properties"].items():
        print(f"{name}: {row['verdict']}" + (f" ({row['reason']})" if row["reason"] else ""),
              file=sys.stderr)
    return 1 if failures(report) else 0


if __name__ == "__main__":
    sys.exit(main())
