# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
"""Drive patchir-decomp and read back what it prints.

The stage runs the binary, reads the printed C with its
`// patchestry:` markers and collects the `refine:` warnings the lifter
emits when a refined JSON disagrees with itself.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

MARKER_TU = re.compile(r"^// patchestry:tu format=(?P<format>\d+)(?P<rest>(?: \w+=\S*)*)\s*$")
MARKER_BEGIN = re.compile(
    r"^// patchestry:function-begin (?P<key>\S+) name=(?P<name>\S+) symbol=(?P<symbol>\S+)\s*$"
)
MARKER_END = re.compile(r"^// patchestry:function-end (?P<key>\S+)\s*$")
REFINE_WARNING = re.compile(r"refine: (?P<text>.*?)\s*$")

REPO_BUILD_CANDIDATES = (
    "builds/default/tools/patchir-decomp/Debug/patchir-decomp",
    "builds/default/tools/patchir-decomp/Release/patchir-decomp",
    "builds/default/tools/patchir-decomp/patchir-decomp",
)


class DecompError(RuntimeError):
    """patchir-decomp could not be found or run."""


@dataclass
class PrintedFunction:
    key: str
    name: str
    symbol: str
    text: str


@dataclass
class PrintedUnit:
    header: dict[str, str] = field(default_factory=dict)
    preamble: str = ""
    functions: dict[str, PrintedFunction] = field(default_factory=dict)


def parse_printed_unit(text: str) -> PrintedUnit:
    """Split a -print-tu file into its header, preamble and marked functions."""
    unit = PrintedUnit()
    preamble: list[str] = []
    current: PrintedFunction | None = None
    body: list[str] = []
    for line in text.splitlines():
        if current is None:
            match = MARKER_TU.match(line)
            if match and not unit.header:
                unit.header["format"] = match.group("format")
                for item in match.group("rest").split():
                    key, _, value = item.partition("=")
                    unit.header[key] = value
                continue
            match = MARKER_BEGIN.match(line)
            if match:
                current = PrintedFunction(
                    key=match.group("key"),
                    name=match.group("name"),
                    symbol=match.group("symbol"),
                    text="",
                )
                body = []
                continue
            preamble.append(line)
            continue
        match = MARKER_END.match(line)
        if match:
            current.text = "\n".join(body).rstrip() + "\n"
            unit.functions[current.key] = current
            current = None
            continue
        body.append(line)
    if current is not None:
        raise ValueError(f"unterminated function marker for {current.key}")
    unit.preamble = "\n".join(preamble).rstrip() + ("\n" if preamble else "")
    return unit


def refine_warnings(stderr: str) -> list[str]:
    """The `refine:` warning texts in a patchir-decomp stderr, in order."""
    found: list[str] = []
    for line in stderr.splitlines():
        match = REFINE_WARNING.search(line)
        if match:
            found.append(match.group("text"))
    return found


def find_patchir_decomp(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate the decompiler: explicit path, $PATCHIR_DECOMP, $PATH, then the
    repository build tree relative to this file."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("PATCHIR_DECOMP")
    if env:
        candidates.append(Path(env))
    on_path = shutil.which("patchir-decomp")
    if on_path:
        candidates.append(Path(on_path))
    repo = Path(__file__).resolve().parents[3]
    for rel in REPO_BUILD_CANDIDATES:
        candidates.append(repo / rel)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise DecompError(
        "patchir-decomp not found; pass --patchir-decomp, set PATCHIR_DECOMP, "
        "or build the repository (cmake --build --preset debug)"
    )


@dataclass
class DecompResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    output_prefix: Path

    @property
    def warnings(self) -> list[str]:
        return refine_warnings(self.stderr)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def output(self, suffix: str) -> Path:
        return Path(str(self.output_prefix) + suffix)

    def printed_unit(self) -> PrintedUnit:
        return parse_printed_unit(self.output(".c").read_text())


def run_patchir_decomp(
    binary: Path,
    json_path: Path,
    output_prefix: Path,
    *,
    print_tu: bool = True,
    emit_cir: bool = False,
    emit_llvm: bool = False,
    flat: bool = False,
    extra_args: Sequence[str] = (),
    timeout: float | None = 600.0,
) -> DecompResult:
    """Run `patchir-decomp -input <json> ... -output <prefix>`."""
    command = [str(binary), "-input", str(json_path), "-output", str(output_prefix)]
    if print_tu:
        command.append("-print-tu")
    if emit_cir:
        command.append("-emit-cir")
    if emit_llvm:
        command.append("-emit-llvm")
    if flat:
        command.append("-emit-flat-baseline")
    command.extend(extra_args)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as error:
        raise DecompError(f"cannot run {binary}: {error}") from error
    except subprocess.TimeoutExpired as error:
        raise DecompError(f"patchir-decomp timed out after {timeout}s") from error
    return DecompResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output_prefix=output_prefix,
    )


def run_from_c(
    binary: Path,
    c_path: Path,
    json_path: Path,
    output_prefix: Path,
    *,
    validate: bool = True,
    report_only: bool = False,
    emit_cir: bool = True,
    print_tu: bool = False,
    extra_args: Sequence[str] = (),
    timeout: float | None = 600.0,
) -> DecompResult:
    """Run `patchir-decomp -from-c <c> -input <json> [-validate-pcode] ...`.

    Tier 2 uses this to have the tool check LLM-written C against the
    P-Code model; the JSON report lands at `<prefix>.validation.json`.
    """
    command = [
        str(binary), "-from-c", str(c_path), "-input", str(json_path),
        "-output", str(output_prefix),
    ]
    if validate:
        command.append("-validate-pcode=report" if report_only else "-validate-pcode")
    if emit_cir:
        command.append("-emit-cir")
    if print_tu:
        command.append("-print-tu")
    command.extend(extra_args)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as error:
        raise DecompError(f"cannot run {binary}: {error}") from error
    except subprocess.TimeoutExpired as error:
        raise DecompError(f"patchir-decomp timed out after {timeout}s") from error
    return DecompResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output_prefix=output_prefix,
    )
