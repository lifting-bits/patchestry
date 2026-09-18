# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
"""`patchestry-refine`: drive the refinement tiers from the command line."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .decomp import DecompError, find_patchir_decomp
from .providers import PROVIDER_NAMES, ProviderError, make_provider
from .tier1 import Tier1Options, run_tier1
from .tier2 import Tier2Options, run_tier2

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_VERIFY = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="patchestry-refine",
        description="LLM refinement stage for patchir-decomp output.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tier1 = sub.add_parser(
        "tier1",
        help="rename, comment and retype through a refined copy of the Ghidra JSON",
    )
    tier1.add_argument("--input", required=True, type=Path, help="Ghidra export JSON")
    tier1.add_argument("--output", required=True, type=Path, help="refined JSON to write")
    tier1.add_argument("--provider", choices=PROVIDER_NAMES, default="anthropic")
    tier1.add_argument("--model", help="model id (provider default when omitted)")
    tier1.add_argument("--max-tokens", type=int, default=16000, help="reply token cap (anthropic)")
    tier1.add_argument("--fake-responses", type=Path, help="JSON replies for --provider fake")
    tier1.add_argument("--patchir-decomp", type=Path, help="decompiler binary (else $PATCHIR_DECOMP, PATH, build tree)")
    tier1.add_argument("--function", action="append", dest="functions", metavar="KEY",
                       help="refine only this function key (repeatable)")
    tier1.add_argument("--max-functions", type=int, help="stop after this many functions")
    tier1.add_argument("--lean", action="store_true", help="prompt with the flat goto C (-emit-flat-baseline)")
    tier1.add_argument("--with-instructions", action="store_true",
                       help="include the --emit-instructions disassembly in prompts when present")
    tier1.add_argument("--no-verify", action="store_true", help="do not re-lift the refined JSON")
    tier1.add_argument("--strict", action="store_true", help="fail when the re-lift emits refine: warnings")
    tier1.add_argument("--prompt-dir", type=Path, help="save every prompt and reply here")
    tier1.add_argument("--workdir", type=Path, help="keep intermediate files here (default: temporary)")
    tier1.add_argument("--report", type=Path, help="JSON report path (default: <output>.report.json)")
    tier1.add_argument("--pretty", action="store_true", help="indent the refined JSON")
    tier1.add_argument("--stop-on-provider-error", action="store_true")
    tier1.set_defaults(func=_run_tier1)

    tier2 = sub.add_parser(
        "tier2",
        help="have the model write structured C from the flat lift, checked with -validate-pcode",
    )
    tier2.add_argument("--input", required=True, type=Path, help="Ghidra export JSON (refined or not)")
    tier2.add_argument("--output", required=True, type=Path, help="output prefix: <prefix>.c, .validation.json, .report.json")
    tier2.add_argument("--provider", choices=PROVIDER_NAMES, default="anthropic")
    tier2.add_argument("--model", help="model id (provider default when omitted)")
    tier2.add_argument("--max-tokens", type=int, default=16000, help="reply token cap (anthropic)")
    tier2.add_argument("--fake-responses", type=Path, help="JSON replies for --provider fake")
    tier2.add_argument("--patchir-decomp", type=Path, help="decompiler binary (else $PATCHIR_DECOMP, PATH, build tree)")
    tier2.add_argument("--function", action="append", dest="functions", metavar="KEY",
                       help="rewrite only this function key (repeatable)")
    tier2.add_argument("--max-functions", type=int, help="stop after this many functions")
    tier2.add_argument("--attempts", type=int, default=3, help="replies per function before keeping the flat body")
    tier2.add_argument("--no-accept-warnings", action="store_true", help="require verdict pass, not warn")
    tier2.add_argument("--with-instructions", action="store_true",
                       help="include --emit-instructions disassembly and raw P-Code in prompts when present")
    tier2.add_argument("--emit-cir", action="store_true", help="also write <prefix>.cir")
    tier2.add_argument("--emit-llvm", action="store_true", help="also write <prefix>.ll")
    tier2.add_argument("--strict", action="store_true", help="fail when any function kept its flat body")
    tier2.add_argument("--prompt-dir", type=Path, help="save every prompt and reply here")
    tier2.add_argument("--workdir", type=Path, help="keep intermediate files here (default: temporary)")
    tier2.add_argument("--report", type=Path, help="JSON report path (default: <prefix>.report.json)")
    tier2.add_argument("--stop-on-provider-error", action="store_true")
    tier2.set_defaults(func=_run_tier2)
    return parser


def _setup(args: argparse.Namespace):
    """(binary, program, provider) or an exit code."""
    try:
        binary = find_patchir_decomp(args.patchir_decomp)
    except DecompError as error:
        print(f"patchestry-refine: {error}", file=sys.stderr)
        return EXIT_ERROR
    try:
        program = json.loads(args.input.read_text())
    except (OSError, ValueError) as error:
        print(f"patchestry-refine: cannot read {args.input}: {error}", file=sys.stderr)
        return EXIT_ERROR
    if not isinstance(program, dict) or "functions" not in program:
        print(f"patchestry-refine: {args.input} is not a Ghidra export", file=sys.stderr)
        return EXIT_ERROR

    provider_options: dict = {}
    if args.provider == "anthropic":
        provider_options["max_tokens"] = args.max_tokens
    if args.provider == "fake":
        if args.fake_responses is None:
            print("patchestry-refine: --provider fake needs --fake-responses", file=sys.stderr)
            return EXIT_ERROR
        provider_options["responses_file"] = args.fake_responses
    try:
        provider = make_provider(args.provider, args.model, **provider_options)
    except (ProviderError, ImportError, OSError) as error:
        print(f"patchestry-refine: cannot start provider {args.provider}: {error}", file=sys.stderr)
        return EXIT_ERROR
    return binary, program, provider


def _run_tier1(args: argparse.Namespace) -> int:
    setup = _setup(args)
    if isinstance(setup, int):
        return setup
    binary, program, provider = setup

    options = Tier1Options(
        functions=args.functions,
        lean=args.lean,
        with_instructions=args.with_instructions,
        max_functions=args.max_functions,
        verify=not args.no_verify,
        prompt_dir=args.prompt_dir,
        stop_on_provider_error=args.stop_on_provider_error,
    )

    def run(workdir: Path) -> int:
        try:
            result = run_tier1(
                program, binary=binary, provider=provider, options=options,
                workdir=workdir, source_name=args.input.name,
            )
        except (DecompError, ProviderError) as error:
            print(f"patchestry-refine: {error}", file=sys.stderr)
            return EXIT_ERROR
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.program, indent=2 if args.pretty else None) + "\n")
        report_path = args.report or Path(str(args.output) + ".report.json")
        report_path.write_text(json.dumps(result.report(), indent=2) + "\n")

        for outcome in result.outcomes:
            line = f"  {outcome.key} {outcome.name}: {outcome.status}"
            if outcome.accepted or outcome.rejected:
                line += f" ({len(outcome.accepted)} accepted, {len(outcome.rejected)} rejected)"
            if outcome.error:
                line += f": {outcome.error}"
            print(line, file=sys.stderr)
        for warning in result.warnings:
            print(f"  refine warning: {warning}", file=sys.stderr)
        print(f"patchestry-refine: wrote {args.output} and {report_path}", file=sys.stderr)

        if result.verify is not None and not result.verify.ok:
            print(
                f"patchestry-refine: the refined JSON does not lift (exit {result.verify.returncode}); "
                f"see {report_path}",
                file=sys.stderr,
            )
            return EXIT_VERIFY
        if args.strict and result.warnings:
            print("patchestry-refine: refine warnings with --strict", file=sys.stderr)
            return EXIT_VERIFY
        return EXIT_OK

    if args.workdir is not None:
        return run(args.workdir)
    with tempfile.TemporaryDirectory(prefix="patchestry-refine-") as tmp:
        return run(Path(tmp))


def _run_tier2(args: argparse.Namespace) -> int:
    setup = _setup(args)
    if isinstance(setup, int):
        return setup
    binary, program, provider = setup

    options = Tier2Options(
        functions=args.functions,
        max_functions=args.max_functions,
        attempts=args.attempts,
        accept_warnings=not args.no_accept_warnings,
        with_instructions=args.with_instructions,
        prompt_dir=args.prompt_dir,
        stop_on_provider_error=args.stop_on_provider_error,
        emit_cir=args.emit_cir,
        emit_llvm=args.emit_llvm,
    )

    def run(workdir: Path) -> int:
        try:
            result = run_tier2(program, binary=binary, provider=provider, options=options, workdir=workdir)
        except (DecompError, ProviderError) as error:
            print(f"patchestry-refine: {error}", file=sys.stderr)
            return EXIT_ERROR
        prefix = args.output
        prefix.parent.mkdir(parents=True, exist_ok=True)
        c_path = Path(str(prefix) + ".c")
        c_path.write_text(result.unit_text)
        written = [c_path]
        if result.final is not None:
            for suffix, wanted in ((".validation.json", True), (".cir", args.emit_cir), (".ll", args.emit_llvm)):
                source = result.final.output(suffix)
                if wanted and source.exists():
                    target = Path(str(prefix) + suffix)
                    target.write_bytes(source.read_bytes())
                    written.append(target)
        report_path = args.report or Path(str(prefix) + ".report.json")
        report_path.write_text(json.dumps(result.report(), indent=2) + "\n")
        written.append(report_path)

        for outcome in result.outcomes:
            line = f"  {outcome.key} {outcome.name}: {outcome.status}"
            if outcome.attempts:
                line += f" after {len(outcome.attempts)} attempt(s)"
                last = outcome.attempts[-1]
                if not last.accepted and last.findings():
                    line += "; last: " + "; ".join(last.findings()[:3])
            if outcome.error:
                line += f": {outcome.error}"
            print(line, file=sys.stderr)
        print("patchestry-refine: wrote " + ", ".join(str(p) for p in written), file=sys.stderr)

        if result.final is None or not result.final.ok:
            code = result.final.returncode if result.final is not None else "?"
            print(f"patchestry-refine: the final unit does not lift (exit {code}); see {report_path}", file=sys.stderr)
            return EXIT_VERIFY
        if args.strict and any(o.status != "accepted" for o in result.outcomes):
            print("patchestry-refine: a function kept its flat body with --strict", file=sys.stderr)
            return EXIT_VERIFY
        return EXIT_OK

    if args.workdir is not None:
        return run(args.workdir)
    with tempfile.TemporaryDirectory(prefix="patchestry-refine-") as tmp:
        return run(Path(tmp))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
