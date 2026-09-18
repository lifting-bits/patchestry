# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
"""Tier 1: names, comments and types, applied into a refined JSON copy.

Flow: print the input with patchir-decomp, prompt the model once per
function, validate and apply each reply through `Refiner`, stamp the
result with provenance, then lift the refined JSON once more so the
lifter's `refine:` warnings and any hard failure surface here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .decomp import DecompError, DecompResult, run_patchir_decomp
from .model import Program, function_inventory
from .prompt import SYSTEM_PROMPT, build_tier1_prompt, instruction_lines
from .proposal import ApplyReport, Refiner, extract_json
from .providers import Provider, ProviderError


@dataclass
class Tier1Options:
    functions: list[str] | None = None
    lean: bool = False
    with_instructions: bool = False
    max_functions: int | None = None
    verify: bool = True
    prompt_dir: Path | None = None
    stop_on_provider_error: bool = False


@dataclass
class FunctionOutcome:
    key: str
    name: str
    status: str  # applied | unchanged | rejected | unparsable | provider-error | missing
    accepted: list[str] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "name": self.name,
            "status": self.status,
            "accepted": list(self.accepted),
            "rejected": list(self.rejected),
        }
        if self.error:
            out["error"] = self.error
        if self.input_tokens is not None or self.output_tokens is not None:
            out["usage"] = {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}
        return out


@dataclass
class Tier1Result:
    program: Program
    outcomes: list[FunctionOutcome]
    verify: DecompResult | None = None

    @property
    def warnings(self) -> list[str]:
        return self.verify.warnings if self.verify is not None else []

    @property
    def verified(self) -> bool:
        return self.verify is None or self.verify.ok

    def report(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.status] = counts.get(outcome.status, 0) + 1
        return {
            "refinement": self.program.get("refinement", {}),
            "summary": counts,
            "functions": [outcome.as_dict() for outcome in self.outcomes],
            "verify": None
            if self.verify is None
            else {
                "returncode": self.verify.returncode,
                "warnings": self.verify.warnings,
                "command": self.verify.command,
            },
        }


def _tail(text: str, lines: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def _with_provenance(program: Program, provenance: dict[str, Any]) -> Program:
    """The program with `refinement` placed after the header keys."""
    out: Program = {}
    inserted = False
    for key, value in program.items():
        if key == "refinement":
            continue
        out[key] = value
        if key == "format" and not inserted:
            out["refinement"] = provenance
            inserted = True
    if not inserted:
        out = {"refinement": provenance, **out}
    return out


def run_tier1(
    program: Program,
    *,
    binary: Path,
    provider: Provider,
    options: Tier1Options,
    workdir: Path,
    source_name: str = "",
) -> Tier1Result:
    workdir.mkdir(parents=True, exist_ok=True)
    input_json = workdir / "input.json"
    input_json.write_text(json.dumps(program))
    printed = run_patchir_decomp(
        binary, input_json, workdir / "input", print_tu=True, flat=options.lean
    )
    if not printed.ok or not printed.output(".c").exists():
        raise DecompError(
            f"patchir-decomp failed on the input (exit {printed.returncode}):\n{_tail(printed.stderr)}"
        )
    unit = printed.printed_unit()

    refiner = Refiner(program)
    keys = [key for key in unit.functions if key in refiner.program["functions"]]
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
        (options.prompt_dir / "system.txt").write_text(SYSTEM_PROMPT)

    for key in keys:
        printed_function = unit.functions[key]
        inventory = function_inventory(refiner.program, key)
        function_json = refiner.program["functions"][key]
        instructions = instruction_lines(function_json) if options.with_instructions else None
        user = build_tier1_prompt(
            refiner.program, inventory, printed_function, unit.header, instructions=instructions
        )
        if options.prompt_dir is not None:
            safe = key.replace(":", "_").replace("/", "_")
            (options.prompt_dir / f"{safe}.prompt.txt").write_text(user)
        outcome = FunctionOutcome(key=key, name=inventory.display_name, status="unchanged")
        try:
            completion = provider.complete(SYSTEM_PROMPT, user)
        except ProviderError as error:
            outcome.status, outcome.error = "provider-error", str(error)
            outcomes.append(outcome)
            if options.stop_on_provider_error:
                raise
            continue
        outcome.input_tokens = completion.input_tokens
        outcome.output_tokens = completion.output_tokens
        if options.prompt_dir is not None:
            (options.prompt_dir / f"{safe}.reply.txt").write_text(completion.text)
        try:
            proposal = extract_json(completion.text)
        except ValueError as error:
            outcome.status, outcome.error = "unparsable", str(error)
            outcomes.append(outcome)
            continue
        report: ApplyReport = refiner.apply(proposal, only_function=key)
        outcome.accepted = list(report.accepted)
        outcome.rejected = [r.as_dict() for r in report.rejected]
        if report.accepted:
            outcome.status = "applied"
        elif report.rejected:
            outcome.status = "rejected"
        outcomes.append(outcome)

    provenance = {
        "tool": f"patchestry-refine {__version__}",
        "tier": 1,
        "provider": provider.name,
        "model": provider.model,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": source_name,
        "functions": {
            o.key: {"status": o.status, "accepted": len(o.accepted), "rejected": len(o.rejected)}
            for o in outcomes
        },
    }
    refined = _with_provenance(refiner.program, provenance)

    verify: DecompResult | None = None
    if options.verify:
        refined_json = workdir / "refined.json"
        refined_json.write_text(json.dumps(refined))
        verify = run_patchir_decomp(
            binary, refined_json, workdir / "refined", print_tu=True, emit_cir=True, flat=options.lean
        )
    return Tier1Result(program=refined, outcomes=outcomes, verify=verify)
