#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Lift a function, reinsert it unchanged, and check the rewritten binary.

No patch is applied: the function is lifted from its P-Code JSON, lowered to
LLVM IR, compiled and put back in place of the original.  Every difference
between the two binaries is then the work of the lift and the reinsertion, so
this is the baseline a real patch is judged against.

Each case names a binary, the JSON of one function lifted from it, and the
function.  The binary is a dynamically linked, fixed-address ARM executable;
it may be stripped and its code may be ARM-mode.  Per case, in order:

| Check | What it establishes |
| --- | --- |
| `bindings` | every name the lifted code refers to is at the address Ghidra recovered; an import is reached through its own PLT stub |
| `dependencies` | the code to insert references nothing the binary does not provide and owns no writable state |
| `abi` | the code to insert is compiled for the CPU architecture and floating-point convention the binary declares |
| `reinsertion` | only the function's bytes changed, its entry leads to the new body, nothing else branches into the overwritten bytes |
| `link-addresses` | the rewriter linked each reference against the address the binding check confirmed |
| `patch-record` | the record of what was written agrees with the image; entry point and original segments are unchanged |
| `applied-bytes` | the new code equals a fresh compile and link of the same IR; no other loaded byte differs |
| `rewritten-bindings` | in the rewritten binary every binding still gets the verdict it had |
| `static` | the new code returns with the stack pointer and the callee-saved registers as it found them, to the address it was called from; its size and frame are measured |

A case ends at the first check that fails, with the status
`rejected:<problem>`.  A case can declare the rejection it expects.  Two
rejections come from before any check of the result: `unsupported-image` (a
shared object or position-independent executable) and `rewriter-no-function`
(the rewriter's own analysis has no function at the address).

Each case ends with `validation.json`, the evidence built by evidence.py from
the reports the case wrote: a verdict per check, the assumptions, the reports.
The driver's status and the evidence must agree, or the case is unexpected.

What this does not establish, and the evidence says so as `not-run`:

- That the reinserted function behaves like the original.  The checks above
  cover where the bytes are and what they link to, not what they compute.
- Anything that needs the patch specification's `validation` block, which is
  not designed yet: declared reject paths, strict callees, device regions,
  budgets.
- Behaviour at run time.  The binary is not executed.

CLI: identity_reinsertion.py --cases cases.json --binary-root DIR
     --output-dir DIR [--build-type Debug] [--only NAME]
Exit 0 when every case ends as it declares, 1 when one does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from patcherex2 import ModifyFunctionPatch, Patcherex
from patcherex2.targets.elf_arm_linux import ElfArmLinux

import check_abi
import check_applied_bytes
import check_bindings
import check_equivalence
import check_static
import check_trampoline
import evidence
import patch_dependencies
import patch_record
import qemu_firmware_runtime as runtime

CHECKS = (
    "bindings",
    "dependencies",
    "abi",
    "reinsertion",
    "link-addresses",
    "patch-record",
    "applied-bytes",
    "rewritten-bindings",
    "static",
)

# The driver's name for a check, where the evidence names it otherwise.
EVIDENCE_CHECK = {"static": "calling-convention"}


@dataclass(frozen=True)
class Case:
    name: str
    binary: str  # relative to --binary-root
    sha256: str
    fixture: str  # relative to the repository root
    function: str
    expect: str = "pass"
    # Patched versions of the lifted function to compare with the original:
    # {"pointer_size", "arg_buffers", "arg_ranges", "inputs", "variants": [...]}.
    equivalence: dict | None = None


class Rejected(Exception):
    """A check failed; `check` names it and `problem` is the status suffix."""

    def __init__(self, check: str, problem: str, detail: str):
        super().__init__(detail)
        self.check, self.problem, self.detail = check, problem, detail


def load_fixture(path: Path) -> dict:
    """The function JSON; lit fixtures carry `//` comment lines before it."""
    text = path.read_text(encoding="utf-8")
    return json.loads(re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def reinsert(
    p: Patcherex,
    snippets: dict[str, str],
    function: str,
    address: int,
    symbols: dict[str, int],
    flags: list[str],
    rewritten: Path,
) -> tuple[list[patch_record.Applied], list[tuple[int, bytes]]]:
    """Replace the function at `address`; return what was written."""
    # The names the lifted code uses are Ghidra's.  The rewriter has its own
    # for a stripped image, so give it the checked address of each.
    p.symbols.update(symbols)
    recorder = patch_record.Recorder()
    patch = ModifyFunctionPatch(
        address,
        snippets[function],
        compile_opts={"extension": ".ll", "extra_compiler_flags": list(flags)},
    )
    recorder.watch(patch, function, "target", snippets[function])
    p.patches.append(patch)
    p.apply_patches()
    p.save_binary(str(rewritten))
    return recorder.applied, recorder.unattributed(p)


def new_summary(case: Case) -> dict:
    return {
        "case": case.name,
        "binary": case.binary,
        "function": case.function,
        "patch": "none: the lifted function is reinserted unchanged",
        "checks": {},
    }


def run_case(case: Case, args, repo_root: Path, summary: dict) -> None:
    """Run one case, filling in `summary`; raises Rejected at the first failing check."""
    out = args.output_dir / case.name
    binary = args.binary_root / case.binary
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if digest != case.sha256:
        raise RuntimeError(
            f"{binary} is not the binary the fixture was lifted from "
            f"(sha256 {digest}, expected {case.sha256})"
        )
    summary["binary_sha256"] = digest

    document = load_fixture(repo_root / case.fixture)
    function_json = out / "function.json"
    write_json(function_json, document)
    stem = out / case.function
    bindings_path = out / "bindings.manifest.json"
    runtime.run(
        [str(runtime.repo_tool(repo_root, args.build_type, "patchir-decomp")),
         "-input", str(function_json), "-emit-cir", "-print-tu",
         "-emit-bindings", str(bindings_path), "-output", str(stem)],
        capture_output=True,
    )
    module = out / f"{case.function}.ll"
    runtime.run(
        [str(runtime.repo_tool(repo_root, args.build_type, "patchir-cir2llvm")),
         "-S", f"{stem}.cir", "-o", str(module)],
        capture_output=True,
    )

    def passed(check: str, **facts) -> None:
        summary["checks"][check] = {"verdict": "pass", **facts}

    def reject(check: str, problem: str, detail: str):
        summary["checks"][check] = {"verdict": "fail", "problem": problem, "detail": detail}
        raise Rejected(check, problem, detail)

    # 1. Names the lifted code refers to.
    manifest = json.loads(bindings_path.read_text(encoding="utf-8"))
    stubs = check_bindings.stubs_from_function_json(document)
    try:
        bindings = check_bindings.check_bindings(manifest, binary, None, True, stubs)
    except check_bindings.UnsupportedImage as error:
        reject("bindings", "unsupported-image", str(error).split(": ", 1)[-1])
    write_json(out / "bindings.json", bindings)
    failed = check_bindings.failures(bindings, strict=False)
    if failed:
        reject("bindings", failed[0]["verdict"], ", ".join(
            f"{row['verdict']} {row['kind']} '{row['name']}'" for row in failed
        ))
    passed("bindings", verdicts=bindings["summary"],
           unresolved_calls=len(bindings["unresolved_calls"]))
    addresses = {name: int(value, 16) for name, value in bindings["absolute_symbols"].items()}
    target_address = next(
        int(row["address"], 16) for row in bindings["bindings"]
        if row["name"] == case.function and row["kind"] == "function"
    )
    # What the patch link can resolve: the image's own function symbols, when
    # it has any, and every name given to it by address.
    image = check_bindings.load_image(binary, allow_dynamic=True)
    provided = set(addresses) | {
        name for name, group in image.symbols.items()
        if any(symbol.kind == "STT_FUNC" for symbol in group)
    }

    # 2. What the code to insert needs.  It is compiled for the instruction
    # set of the function it replaces and for the ABI the binary declares.
    rewriter = Patcherex(
        str(binary), target_cls=ElfArmLinux, target_opts={"compiler": "clang19"}
    )
    if rewriter.binary_analyzer.get_function(target_address) is None:
        # Its own analysis has to agree that a function starts here: it takes
        # the size of the function, and so where it may write, from there.
        reject("reinsertion", "rewriter-no-function",
               f"the rewriter's analysis has no function at 0x{target_address:x}")
    mode = "thumb" if rewriter.binary_analyzer.is_thumb(target_address) else "arm"
    mode_flag = "-mthumb" if mode == "thumb" else "-mno-thumb"
    declared = check_abi.abi_attributes(binary)
    flags = [*runtime.patch_compiler_flags(args.patch_opt_level),
             *check_abi.compiler_flags(declared)]
    snippets, problems = runtime.verify_dependencies(
        module, provided, [*flags, mode_flag], case.function, out / "dependencies.json"
    )
    for problem, detail in problems.items():
        reject("dependencies", problem, detail)
    passed("dependencies", functions=list(snippets))

    obj = out / f"{case.function}.o"
    patch_dependencies.compile_snippet(
        snippets[case.function], obj, patch_dependencies.Tools.from_environment(),
        [*flags, mode_flag],
    )
    abi = check_abi.abi_problems(binary, obj)
    write_json(out / "abi.json", {
        "image": declared, "object": check_abi.abi_attributes(obj),
        "flags": check_abi.compiler_flags(declared), "problems": abi,
    })
    for problem, detail in abi.items():
        reject("abi", problem, detail)
    passed("abi", flags=check_abi.compiler_flags(declared))

    # 3. Reinsert, then look at what changed.
    rewritten = out / "rewritten.elf"
    applied, unattributed = reinsert(
        rewriter, snippets, case.function, target_address, addresses, flags, rewritten
    )
    trampoline = check_trampoline.check_trampoline(
        binary, rewritten, case.function, document, mode=mode, allow_dynamic=True
    )
    write_json(out / "trampoline.json", trampoline)
    if trampoline["problems"]:
        reject("reinsertion", next(iter(trampoline["problems"])),
               ", ".join(trampoline["problems"]))
    passed("reinsertion", placement=trampoline["verdict"],
           destination=trampoline.get("destination"), size=trampoline["size"])

    # 4. The record of what was written.
    dependencies = json.loads((out / "dependencies.json").read_text(encoding="utf-8"))
    record = patch_record.build_record(
        binary, rewritten, case.function, applied,
        runtime.patch_compiler_info(flags, thumb=mode == "thumb"),
        {row["function"]: row["undefined"] for row in dependencies["functions"]},
        unattributed,
        target_extent=(target_address, trampoline["size"]),
        mode=mode,
    )
    write_json(out / "patch_record.json", record)
    mismatched = patch_record.link_address_problems(record, addresses)
    record_problems = patch_record.check_record(record, rewritten, trampoline)
    write_json(out / "patch_record_check.json",
               {"problems": record_problems, "link_addresses": mismatched})
    if mismatched:
        reject("link-addresses", "link-address-mismatch", "; ".join(mismatched))
    passed("link-addresses",
           references=sum(len(row["symbols"]) for row in record["functions"]))
    for problem, detail in record_problems.items():
        reject("patch-record", problem, detail)
    passed("patch-record", new_size=record["functions"][0]["size"])

    # 5. The bytes.
    byte_problems = runtime.verify_applied_bytes(
        out / "patch_record.json", binary, rewritten, snippets, out / "applied_bytes.json"
    )
    if byte_problems:
        reject("applied-bytes", byte_problems[0], ", ".join(byte_problems))
    passed("applied-bytes")

    # 6. Nothing the lifted code is bound to moved.
    after = check_bindings.check_bindings(manifest, rewritten, None, True, stubs)
    write_json(out / "bindings.rewritten.json", after)
    moved = [
        f"{old['name']}: {old['verdict']} -> {new['verdict']}"
        for old, new in zip(bindings["bindings"], after["bindings"])
        if (old["verdict"], old.get("address"), old.get("stub"))
        != (new["verdict"], new.get("address"), new.get("stub"))
    ]
    if moved:
        reject("rewritten-bindings", "moved", ", ".join(moved))
    passed("rewritten-bindings")

    # 7. What the new machine code does to the state it shares with its caller.
    functions, target = check_static.functions_from_record(record)
    static = check_static.check_static(binary, rewritten, functions, target)
    write_json(out / "static.json", static)
    measured = {
        "code_bytes": static["code_bytes"],
        "original_code_bytes": target["size"],
        "stack_bytes": static["stack_bytes"],
        "original_stack_bytes": static["original"]["stack_depth"],
        "inlined_copies": len(static["inlined_copies"]),
    }
    failed = check_static.failures(static)
    if failed:
        reject("static", failed[0], ", ".join(failed))
    skipped = check_static.not_run(static)
    if skipped:
        # Not a pass and not a failure: the case ends with what stopped it.
        summary["checks"]["static"] = {"verdict": "not-run", "reasons": skipped, **measured}
        raise Rejected("static", f"not-run:{skipped[0]}", ", ".join(skipped))
    passed("static", **measured)


def run_variants(case: Case, args, repo_root: Path, digest: str | None) -> list[tuple]:
    """Compare patched versions of the case's lifted function with the original.

    A variant is C written from the lifted C, with a patch in it.  It is not
    reinserted: its evidence has the equivalence checks and nothing else.
    Returns summary rows (name, status, expected, detail).
    """
    settings = case.equivalence
    case_dir = args.output_dir / case.name
    lifted = case_dir / f"{case.function}.ll"
    declared = check_abi.abi_attributes(args.binary_root / case.binary)
    tools = check_equivalence.Tools.from_environment()
    rows = []
    for variant in settings["variants"]:
        name = f"{case.name}--{variant['name']}"
        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        for stale in ("case.json", "validation.json", *evidence.REPORT_FILES):
            (out / stale).unlink(missing_ok=True)

        def compiled(defines: list[str], stem: str) -> Path:
            if variant["source"] == "@lifted":
                # The lifted IR with lines replaced: a patch at the level the
                # patcher works at.  Each `find` must name exactly one place.
                text = lifted.read_text(encoding="utf-8")
                for edit in variant["edits"]:
                    if text.count(edit["find"]) != 1:
                        raise ValueError(
                            f"{name}: edit {edit['find']!r} matches "
                            f"{text.count(edit['find'])} places in {lifted.name}"
                        )
                    text = text.replace(edit["find"], edit["replace"])
                module = out / f"{stem}.ll"
                module.write_text(text, encoding="utf-8")
                return module
            # "@printed" is the C the decompiler printed for the case.
            source = (case_dir / f"{case.function}.c" if variant["source"] == "@printed"
                      else repo_root / variant["source"])
            module = out / f"{stem}.ll"
            runtime.run(
                [str(tools.clang), "--target=armv7a-none-eabi",
                 *check_abi.compiler_flags(declared), "-O0", "-w", "-S", "-emit-llvm",
                 *[f"-D{define}" for define in defines], str(source), "-o", str(module)],
                capture_output=True,
            )
            return module

        patched = compiled(variant.get("defines", []), "patched")
        # The original is the lifted function, unless the variant compares
        # two patched versions with each other.
        original = (compiled(variant["original_defines"], "original")
                    if "original_defines" in variant else lifted)
        report = check_equivalence.check_equivalence(
            original, patched, case.function, out, tools,
            reject_handlers=variant.get("reject_handlers", []),
            reject_returns=variant.get("reject_returns", []),
            arg_buffers={int(k): v for k, v in settings.get("arg_buffers", {}).items()},
            arg_ranges={int(k): tuple(v) for k, v in settings.get("arg_ranges", {}).items()},
            inputs=settings.get("inputs", 200),
            pointer_size=settings.get("pointer_size", 64),
            pure_callees=variant.get("pure_callees", []),
            substitutions=variant.get("substitutions", []),
            strict_callees=variant.get("strict_callees", []),
        )
        write_json(out / "equivalence.json", report)
        properties = check_equivalence.PROPERTIES
        status = ",".join(
            f"{key}={report['properties'][key]['verdict']}" for key in properties
        )
        expected = ",".join(f"{key}={variant['expect'][key]}" for key in properties)
        write_json(out / "case.json", {
            "suite": "identity-reinsertion",
            "case": name,
            "function": case.function,
            "binary": case.binary,
            "binary_sha256": digest,
            "patch": variant["description"],
            "fixture": {"path": case.fixture, "hand_written": False, "regenerated": False},
            "runtime": {"verdict": "not-run",
                        "detail": "the patched version was not reinserted or executed"},
            "status": status,
            "expected": expected,
        })
        built = evidence.write_evidence(out)
        detail = "; ".join(sorted({
            f"{row['property']} {row['kind']}"
            + "".join(f" {row[key]}" for key in ("region", "call") if row.get(key) is not None)
            for row in report["violations"]
        })) or (report["properties"]["P1"]["reason"] or "")
        if [built["checks"][key]["verdict"] for key in properties] != [
            report["properties"][key]["verdict"] for key in properties
        ]:
            status, detail = "error", "evidence disagrees with equivalence.json"
        rows.append((name, status, expected, detail))
        print(f"{name}: {status}" + ("" if status == expected else f" (expected {expected})"),
              file=sys.stderr)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of cases")
    parser.add_argument("--binary-root", type=Path, required=True,
                        help="directory the cases' binary paths are relative to")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-type", default="Debug")
    parser.add_argument("--patch-opt-level", default="O0")
    parser.add_argument("--only", action="append", help="run only the named case")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cases = [Case(**row) for row in json.loads(args.cases.read_text(encoding="utf-8"))]
    if args.only:
        cases = [case for case in cases if case.name in args.only]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, unexpected = [], 0
    for case in cases:
        out = args.output_dir / case.name
        out.mkdir(parents=True, exist_ok=True)
        summary = new_summary(case)
        detail = ""
        stopped = None
        for stale in ("case.json", "validation.json", *evidence.REPORT_FILES):
            (out / stale).unlink(missing_ok=True)
        try:
            run_case(case, args, repo_root, summary)
            status = "pass"
        except Rejected as rejected:
            status = f"rejected:{rejected.problem}"
            detail = rejected.detail
            if not rejected.problem.startswith("not-run:"):
                stopped = {"check": EVIDENCE_CHECK.get(rejected.check, rejected.check),
                           "problem": rejected.problem, "detail": rejected.detail}
        except Exception as error:  # the case could not be run at all
            status = "error"
            detail = f"{type(error).__name__}: {error}"
            (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        # What the suite knows and no report holds; the evidence is built from
        # this and the reports the case got to.
        context = {
            "suite": "identity-reinsertion",
            "case": case.name,
            "function": case.function,
            "binary": case.binary,
            "binary_sha256": summary.get("binary_sha256"),
            "patch": summary["patch"],
            "fixture": {"path": case.fixture, "hand_written": False, "regenerated": False},
            "runtime": {"verdict": "not-run", "detail": "the binary was not executed"},
            "status": status,
            "expected": case.expect,
        }
        if stopped:
            context["stopped"] = stopped
        write_json(out / "case.json", context)
        built = evidence.write_evidence(out)
        failed = [name for name, row in built["checks"].items() if row["verdict"] == "failed"]
        # A pass is every integrity check and the calling convention proven
        # from the reports.  Undeclared budgets stay not-run and do not count.
        established = (
            built["groups"]["integrity"] == "proven"
            and built["checks"]["calling-convention"]["verdict"] == "proven"
        )
        ok = status == case.expect
        if (status == "pass") != (not failed and established) or (
            stopped and stopped["check"] not in failed
        ):
            # The driver and the reports disagree about the case.
            ok = False
            detail = (f"evidence disagrees: failed {failed or 'nothing'}, "
                      f"groups {built['groups']}")
        unexpected += not ok
        rows.append((case.name, status, case.expect, detail))
        if case.equivalence and status == "pass":
            try:
                variants = run_variants(case, args, repo_root, summary.get("binary_sha256"))
            except Exception as error:  # the comparison could not be run at all
                variants = [(f"{case.name}--variants", "error", "",
                             f"{type(error).__name__}: {error}")]
                (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            rows.extend(variants)
            unexpected += sum(1 for row in variants if row[1] != row[2])
        print(f"{case.name}: {status}" + ("" if ok else f" (expected {case.expect})"),
              file=sys.stderr)

    lines = ["case\tstatus\texpected\tdetail"] + ["\t".join(row) for row in rows]
    (args.output_dir / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
