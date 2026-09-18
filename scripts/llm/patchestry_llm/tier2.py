# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
"""Tier 2: the model writes the structured C, the decompiler checks it.

Flow: print the flat goto C with `-emit-flat-baseline -print-tu`, ask the
model to rewrite one function at a time, splice the reply into the unit
and run `patchir-decomp -from-c ... -validate-pcode=report`.  A reply is
kept when its function parses, lowers and validates; otherwise the
findings go back to the model for another attempt, and after the last
attempt the flat body stays.  The structuring engine never runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decomp import DecompError, DecompResult, PrintedUnit, run_from_c, run_patchir_decomp
from .model import Program
from .prompt import (
    TIER2_SYSTEM_PROMPT,
    build_tier2_prompt,
    build_tier2_retry,
    instruction_lines_with_pcode,
)
from .providers import Provider, ProviderError

DIAG_ERROR = re.compile(r"Diag (?:Error|Fatal): \[(?:ERROR|FATAL)\] (?P<text>.*?)\s*$")
FROM_C_ERROR = re.compile(r"from-c: (?P<text>.*?)\s*$")
FENCE = re.compile(r"^```[A-Za-z]*\s*\n(.*?)\n```\s*$", re.DOTALL)
MARKER_LINE = re.compile(r"^\s*// patchestry:")


@dataclass
class Tier2Options:
    functions: list[str] | None = None
    max_functions: int | None = None
    attempts: int = 3
    accept_warnings: bool = True
    with_instructions: bool = False
    prompt_dir: Path | None = None
    stop_on_provider_error: bool = False
    emit_cir: bool = False
    emit_llvm: bool = False


@dataclass
class Attempt:
    number: int
    text: str
    verdict: str | None = None
    flags: list[dict[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    accepted: bool = False

    def findings(self) -> list[str]:
        out = [f"{f.get('code')} ({f.get('severity')}): {f.get('detail')}" for f in self.flags]
        out.extend(f"compiler: {e}" for e in self.errors)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "verdict": self.verdict,
            "flags": list(self.flags),
            "errors": list(self.errors),
            "accepted": self.accepted,
            "lines": len(self.text.splitlines()),
        }


@dataclass
class FunctionOutcome:
    key: str
    name: str
    status: str  # accepted | kept-flat | provider-error | missing
    attempts: list[Attempt] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "name": self.name,
            "status": self.status,
            "attempts": [a.as_dict() for a in self.attempts],
        }
        if self.error:
            out["error"] = self.error
        return out


@dataclass
class Tier2Result:
    unit_text: str
    outcomes: list[FunctionOutcome]
    final: DecompResult | None = None
    final_validation: dict[str, Any] | None = None

    @property
    def verified(self) -> bool:
        return self.final is not None and self.final.ok

    def report(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.status] = counts.get(outcome.status, 0) + 1
        return {
            "tier": 2,
            "summary": counts,
            "functions": [o.as_dict() for o in self.outcomes],
            "final": None
            if self.final is None
            else {
                "returncode": self.final.returncode,
                "command": self.final.command,
                "validation": (self.final_validation or {}).get("summary"),
            },
        }


def render_unit(unit: PrintedUnit, replacements: dict[str, str] | None = None) -> str:
    """The translation unit text with some function bodies replaced."""
    replacements = replacements or {}
    lines: list[str] = []
    if unit.header:
        head = "// patchestry:tu"
        for key, value in unit.header.items():
            head += f" {key}={value}"
        lines.append(head)
    if unit.preamble:
        lines.append(unit.preamble.rstrip("\n"))
    for key, function in unit.functions.items():
        lines.append(f"// patchestry:function-begin {key} name={function.name} symbol={function.symbol}")
        lines.append(replacements.get(key, function.text).rstrip("\n"))
        lines.append(f"// patchestry:function-end {key}")
    return "\n".join(lines) + "\n"


def clean_definition(text: str, name: str) -> str | None:
    """The function definition in a model reply, or None when there is none."""
    stripped = text.strip()
    fence = FENCE.match(stripped)
    if fence:
        stripped = fence.group(1)
    else:
        # Prose around a fenced block.
        inner = re.search(r"```[A-Za-z]*\s*\n(.*?)\n```", stripped, re.DOTALL)
        if inner:
            stripped = inner.group(1)
    lines = [line for line in stripped.splitlines() if not MARKER_LINE.match(line)]
    start = next((i for i, line in enumerate(lines) if re.search(rf"\b{re.escape(name)}\s*\(", line)), None)
    if start is None:
        return None
    end = max((i for i, line in enumerate(lines) if line.rstrip().endswith("}")), default=-1)
    if end < start:
        return None
    return "\n".join(lines[start : end + 1]).rstrip() + "\n"


def parse_errors(stderr: str) -> list[str]:
    found: list[str] = []
    for line in stderr.splitlines():
        match = DIAG_ERROR.search(line)
        if match:
            found.append(match.group("text"))
            continue
        match = FROM_C_ERROR.search(line)
        if match and not match.group("text").startswith("cannot read"):
            found.append(match.group("text"))
    return found


def _read_validation(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_tier2(
    program: Program,
    *,
    binary: Path,
    provider: Provider,
    options: Tier2Options,
    workdir: Path,
) -> Tier2Result:
    workdir.mkdir(parents=True, exist_ok=True)
    input_json = workdir / "input.json"
    input_json.write_text(json.dumps(program))
    flat = run_patchir_decomp(binary, input_json, workdir / "flat", print_tu=True, flat=True)
    if not flat.ok or not flat.output(".c").exists():
        raise DecompError(
            f"patchir-decomp failed on the flat lift (exit {flat.returncode}):\n"
            + "\n".join(flat.stderr.strip().splitlines()[-12:])
        )
    unit = flat.printed_unit()

    keys = list(unit.functions)
    outcomes: list[FunctionOutcome] = []
    if options.functions:
        wanted = list(dict.fromkeys(options.functions))
        for key in wanted:
            if key not in unit.functions:
                outcomes.append(FunctionOutcome(key=key, name="", status="missing",
                                                error="not among the printed function definitions"))
        keys = [key for key in keys if key in set(wanted)]
    if options.max_functions is not None:
        keys = keys[: options.max_functions]

    if options.prompt_dir is not None:
        options.prompt_dir.mkdir(parents=True, exist_ok=True)
        (options.prompt_dir / "system.txt").write_text(TIER2_SYSTEM_PROMPT)

    replacements: dict[str, str] = {}
    for key in keys:
        printed = unit.functions[key]
        function_json = program.get("functions", {}).get(key, {})
        instructions = instruction_lines_with_pcode(function_json) if options.with_instructions else None
        base_prompt = build_tier2_prompt(
            printed, unit.preamble, unit.header, program=program, instructions=instructions
        )
        safe = key.replace(":", "_").replace("/", "_")
        outcome = FunctionOutcome(key=key, name=printed.name, status="kept-flat")
        prompt = base_prompt
        for number in range(1, max(1, options.attempts) + 1):
            if options.prompt_dir is not None:
                (options.prompt_dir / f"{safe}.attempt{number}.prompt.txt").write_text(prompt)
            try:
                completion = provider.complete(TIER2_SYSTEM_PROMPT, prompt)
            except ProviderError as error:
                outcome.status, outcome.error = "provider-error", str(error)
                if options.stop_on_provider_error:
                    raise
                break
            if options.prompt_dir is not None:
                (options.prompt_dir / f"{safe}.attempt{number}.reply.txt").write_text(completion.text)
            attempt = Attempt(number=number, text=completion.text)
            outcome.attempts.append(attempt)
            definition = clean_definition(completion.text, printed.name)
            if definition is None:
                attempt.errors.append(f"the reply holds no definition of {printed.name}")
                prompt = build_tier2_retry(base_prompt, number, attempt.findings())
                continue
            attempt.text = definition
            candidate = render_unit(unit, {**replacements, key: definition})
            candidate_path = workdir / f"{safe}.attempt{number}.c"
            candidate_path.write_text(candidate)
            check = run_from_c(
                binary, candidate_path, input_json, workdir / f"{safe}.attempt{number}",
                validate=True, report_only=True, emit_cir=True,
            )
            if not check.ok:
                attempt.errors.extend(parse_errors(check.stderr) or [f"patchir-decomp exited {check.returncode}"])
            report = _read_validation(check.output(".validation.json")) if check.ok else None
            entry = ((report or {}).get("functions") or {}).get(key)
            if isinstance(entry, dict):
                attempt.verdict = str(entry.get("verdict"))
                attempt.flags = [
                    {"code": str(f.get("code")), "severity": str(f.get("severity")), "detail": str(f.get("detail"))}
                    for f in entry.get("flags", []) or []
                    if isinstance(f, dict)
                ]
            elif check.ok:
                attempt.errors.append("the validation report has no entry for this function")
            accepted = check.ok and attempt.verdict in {"pass", "warn"} and (
                attempt.verdict == "pass" or options.accept_warnings
            )
            if accepted:
                attempt.accepted = True
                replacements[key] = definition
                outcome.status = "accepted"
                break
            prompt = build_tier2_retry(base_prompt, number, attempt.findings() or ["rejected"])
        outcomes.append(outcome)

    unit_text = render_unit(unit, replacements)
    final_c = workdir / "final.c"
    final_c.write_text(unit_text)
    final = run_from_c(
        binary, final_c, input_json, workdir / "final",
        validate=True, report_only=True, emit_cir=True, print_tu=False,
        extra_args=["-emit-llvm"] if options.emit_llvm else (),
    )
    validation = _read_validation(final.output(".validation.json")) if final.ok else None
    return Tier2Result(unit_text=unit_text, outcomes=outcomes, final=final, final_validation=validation)
