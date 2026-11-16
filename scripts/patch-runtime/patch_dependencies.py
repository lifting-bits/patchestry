#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Dependencies of the code that is inserted into a firmware image.

The patched LLVM IR module is reinserted one function at a time, and each
function is compiled and linked on its own.  Two things follow.

`split_module` cuts the module into one snippet per defined function.  A
snippet defines that function, carries the constants it reaches (through
initializers too, so a table brings the strings it points to) as local
definitions, and declares everything else: the other functions and every
writable global.  Writable globals are not copied, because a copy per snippet
would give each function its own variable.  The split is done by llvm-extract
and opt on the parsed module, not on its text.

`check_snippets` compiles each snippet the way the reinsertion will and lists
the symbols the object leaves undefined.  Each of them must be provided by the
original image or by another snippet.  What is left over (a compiler runtime
helper such as __aeabi_uldivmod that the firmware was never linked with, a
writable global of the patch that nothing places) is reported, because the
patch linker would otherwise give it an invented address.

`insertion_order` gives the order to insert the functions in.  They are linked
one at a time against what is already placed, so a helper has to go in before
its caller.  Functions that reference each other have no such order.

Not covered: whether a provided symbol has the type or size the snippet
expects, and references the compiler resolves inside the object.

CLI: patch_dependencies.py --module patched.ll --elf firmware.elf
     [--symbol name=0xaddr ...] [--output report.json]
Exit 0 when nothing is unresolved, 1 when something is, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from elftools.elf.elffile import ELFFile
from elftools.elf.enums import ENUM_RELOC_TYPE_ARM
from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import SymbolTableSection

# The reinsertion compiles for this triple (Patcherex2's clang invocation).
PATCH_TARGET = "arm-linux-gnueabihf"


@dataclass(frozen=True)
class Tools:
    clang: Path
    llvm_extract: Path
    opt: Path
    llvm_as: Path
    llvm_nm: Path

    @classmethod
    def from_prefix(cls, prefix: Path) -> "Tools":
        return cls(
            clang=prefix / "clang",
            llvm_extract=prefix / "llvm-extract",
            opt=prefix / "opt",
            llvm_as=prefix / "llvm-as",
            llvm_nm=prefix / "llvm-nm",
        )

    @classmethod
    def from_environment(cls) -> "Tools":
        prefix = os.environ.get("PATCHESTRY_LLVM_PREFIX")
        if not prefix:
            raise RuntimeError("PATCHESTRY_LLVM_PREFIX is not set (the LLVM bin directory)")
        tools = cls.from_prefix(Path(prefix))
        if tools.missing():
            raise RuntimeError(f"{prefix} has no {', '.join(tools.missing())}")
        return tools

    def missing(self) -> list[str]:
        return [
            path.name
            for path in (self.clang, self.llvm_extract, self.opt, self.llvm_as, self.llvm_nm)
            if not path.exists()
        ]


def _run(command: list[str], stdin: bytes | None = None) -> bytes:
    result = subprocess.run(command, input=stdin, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed ({result.returncode}):\n"
            f"{result.stderr.decode('utf-8', 'replace')}"
        )
    return result.stdout


def _symbols(bitcode: bytes, tools: Tools) -> list[tuple[str, str]]:
    """(type letter, name) for every symbol of a bitcode module, as llvm-nm sees it."""
    listing = _run([str(tools.llvm_nm), "--format=posix", "-"], stdin=bitcode)
    rows = []
    for line in listing.decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            rows.append((fields[1], fields[0]))
    return rows


def _module_symbols(module: Path, tools: Tools) -> list[tuple[str, str]]:
    return _symbols(_run([str(tools.llvm_as), str(module), "-o", "-"]), tools)


def defined_functions(module: Path, tools: Tools) -> list[str]:
    return sorted(name for kind, name in _module_symbols(module, tools) if kind in "tT")


def writable_definitions(module: Path, tools: Tools) -> list[str]:
    """Writable globals the module defines and its functions use.

    This is state the inserted code needs placed somewhere.  llvm-nm does not
    tell a constant from a variable in bitcode, so the list is taken from the
    split: a global the module defines and a snippet only declares.
    """
    defined = {name for kind, name in _module_symbols(module, tools) if kind in "dDbBC"}
    declared = set()
    for snippet in split_module(module, tools).values():
        bitcode = _run([str(tools.llvm_as), "-", "-o", "-"], stdin=snippet.encode("utf-8"))
        declared |= {name for kind, name in _symbols(bitcode, tools) if kind == "U"}
    return sorted(defined & declared)


def split_module(module: Path, tools: Tools) -> dict[str, str]:
    """Function name -> a self-contained LLVM IR module for that function."""
    snippets = {}
    for name in defined_functions(module, tools):
        # --keep-const-init keeps every constant's initializer; without it the
        # constants the function uses become bare declarations.
        extracted = _run(
            [str(tools.llvm_extract), f"--func={name}", "--keep-const-init", str(module), "-o", "-"]
        )
        # llvm-extract gives the kept constants external linkage and keeps the
        # ones the function does not reach.  Make them local again, then drop
        # what nothing references.
        snippet = _run(
            [
                str(tools.opt),
                "-passes=internalize,globaldce",
                f"--internalize-public-api-list={name}",
                "-S",
                "-",
                "-o",
                "-",
            ],
            stdin=extracted,
        )
        snippets[name] = snippet.decode("utf-8")
    return snippets


def undefined_symbols(obj: Path) -> list[str]:
    """Names an ELF object needs from elsewhere.

    A name counts when the object does not define it and a relocation that
    changes bytes refers to it.  R_ARM_NONE relocations are left out: the ARM
    unwind index uses one to name its personality routine
    (__aeabi_unwind_cpp_pr0), and nothing in the inserted code depends on it.
    """
    names = set()
    with obj.open("rb") as stream:
        elf = ELFFile(stream)
        none_type = ENUM_RELOC_TYPE_ARM["R_ARM_NONE"]
        for section in elf.iter_sections():
            if not isinstance(section, RelocationSection):
                continue
            symbols = elf.get_section(section["sh_link"])
            for relocation in section.iter_relocations():
                if relocation["r_info_type"] == none_type:
                    continue
                symbol = symbols.get_symbol(relocation["r_info_sym"])
                if symbol.name and symbol["st_shndx"] == "SHN_UNDEF":
                    names.add(symbol.name)
    return sorted(names)


def unresolved(obj: Path, provided: Mapping[str, int] | Iterable[str]) -> list[str]:
    """Undefined symbols of the object that `provided` has no entry for."""
    known = set(provided)
    return [name for name in undefined_symbols(obj) if name not in known]


def compile_snippet(
    snippet: str, obj: Path, tools: Tools, compiler_flags: Iterable[str] = ()
) -> None:
    source = obj.with_suffix(".ll")
    source.write_text(snippet, encoding="utf-8")
    _run(
        [
            str(tools.clang),
            "-target",
            PATCH_TARGET,
            "-mthumb",
            *compiler_flags,
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
    )


def check_snippets(
    snippets: Mapping[str, str],
    provided: Mapping[str, int] | Iterable[str],
    tools: Tools,
    compiler_flags: Iterable[str] = (),
) -> dict:
    """Compile every snippet and report what each one leaves unresolved.

    `provided` is what the original image supplies; the snippets also supply
    each other.
    """
    known = set(provided) | set(snippets)
    rows = []
    with tempfile.TemporaryDirectory(prefix="patch-dependencies-") as directory:
        for index, (name, snippet) in enumerate(sorted(snippets.items())):
            obj = Path(directory) / f"snippet{index}.o"
            compile_snippet(snippet, obj, tools, compiler_flags)
            undefined = undefined_symbols(obj)
            rows.append(
                {
                    "function": name,
                    "undefined": undefined,
                    "unresolved": [symbol for symbol in undefined if symbol not in known],
                }
            )
    return {"functions": rows}


def insertion_order(report: dict, target: str) -> tuple[list[str], list[str]]:
    """Order in which to insert the functions other than `target`.

    The reinsertion links one function at a time against the image and the
    functions it has already placed, so a function must come after every
    inserted function it references.  `target` replaces a function the image
    already has: it goes last and is not part of the order, and a reference to
    it resolves to its existing address.

    Returns the order and the functions that cannot be ordered because they
    reference each other.
    """
    inserted = {row["function"] for row in report["functions"]} - {target}
    pending = {
        row["function"]: (set(row["undefined"]) & inserted) - {row["function"]}
        for row in report["functions"]
        if row["function"] != target
    }
    order: list[str] = []
    while True:
        ready = sorted(name for name, needs in pending.items() if needs <= set(order))
        if not ready:
            return order, sorted(pending)
        order.append(ready[0])
        del pending[ready[0]]


def failures(report: dict) -> list[str]:
    """Every unresolved symbol in the report, once."""
    return sorted({symbol for row in report["functions"] for symbol in row["unresolved"]})


def elf_symbol_names(elf: Path) -> set[str]:
    with elf.open("rb") as stream:
        names = set()
        for section in ELFFile(stream).iter_sections():
            if not isinstance(section, SymbolTableSection):
                continue
            for symbol in section.iter_symbols():
                if symbol.name and symbol["st_shndx"] != "SHN_UNDEF":
                    names.add(symbol.name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--module", type=Path, required=True, help="patched LLVM IR (.ll)")
    parser.add_argument("--elf", type=Path, required=True, help="original firmware ELF")
    parser.add_argument("--symbol", action="append", default=[], metavar="NAME=ADDR")
    parser.add_argument("--compiler-flag", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        tools = Tools.from_environment()
        provided = elf_symbol_names(args.elf) | {entry.split("=")[0] for entry in args.symbol}
        report = check_snippets(
            split_module(args.module, tools), provided, tools, args.compiler_flag
        )
        report["writable_definitions"] = writable_definitions(args.module, tools)
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    missing = failures(report)
    if missing:
        print(f"unresolved: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
