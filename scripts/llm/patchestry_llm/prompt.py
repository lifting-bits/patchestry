# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
"""Tier 1 prompt: one function, its declared names and the types in reach."""

from __future__ import annotations

from typing import Any

from .decomp import PrintedFunction
from .model import FunctionInventory, Program, referenced_type_keys, type_size, type_spelling

SYSTEM_PROMPT = """\
You refine C that a decompiler lifted from machine code.  You propose better
names, a one-paragraph comment and, when the code makes it obvious, struct
layouts.  You never change what the code does: the decompiler re-lifts your
proposal and rejects anything that does not fit its model of the binary.

Rules:
- Use only the function key, parameter indices, local op keys, global keys
  and type keys listed in the request.  Do not invent keys.
- Names must be C identifiers, unique within the function, not C keywords,
  and must not reuse a global, function or typedef name.
- A parameter, local or return type may only be changed to a type of the
  same byte size.  New struct/union/typedef/pointer/array types may be added
  under new keys and used in the same reply; give every struct field a name,
  an existing type key and a byte offset, and give the struct its total size.
- Keep a name when you have no better one.  Omit fields you do not change.
- Reply with a single JSON object and nothing else, in this shape:

{
  "functions": {
    "<function key>": {
      "display_name": "identifier",
      "comment": "what the function does, one paragraph",
      "parameters": {"<index>": {"name": "identifier", "type": "<type key>"}},
      "locals": {"<op key>": {"name": "identifier", "type": "<type key>"}},
      "return_type": "<type key>"
    }
  },
  "globals": {"<global key>": {"name": "identifier"}},
  "types": {
    "<new type key>": {"kind": "struct", "name": "identifier", "size": 8,
                       "fields": [{"name": "identifier", "type": "<type key>", "offset": 0}]}
  }
}
"""


def _fmt_size(size: int | None) -> str:
    return "?" if size is None else str(size)


def build_tier1_prompt(
    program: Program,
    inventory: FunctionInventory,
    printed: PrintedFunction,
    header: dict[str, str],
    *,
    instructions: list[str] | None = None,
    max_types: int = 80,
) -> str:
    types = program.get("types", {})
    globals_ = program.get("globals", {})
    lines: list[str] = []
    target = header.get("target") or program.get("id") or "unknown"
    arch = header.get("arch") or program.get("architecture") or "unknown"
    lines.append(f"Target: {target} ({arch})")
    lines.append(f"Function key: {inventory.key}")
    lines.append(f"Linker symbol: {inventory.name}")
    lines.append(f"Current C name: {inventory.display_name}")
    if inventory.is_variadic:
        lines.append("Prototype: variadic")
    lines.append("")
    lines.append("Current C:")
    lines.append("```c")
    lines.append(printed.text.rstrip())
    lines.append("```")
    lines.append("")

    lines.append("Parameters (index | name | type key | C type | bytes):")
    if inventory.parameters:
        for decl in inventory.parameters:
            lines.append(
                f"  {decl.index} | {decl.name} | {decl.type_key} | "
                f"{type_spelling(types, decl.type_key)} | {_fmt_size(type_size(types, decl.type_key))}"
            )
    else:
        lines.append("  (none)")
    lines.append(
        f"Return type: {inventory.return_type} | {type_spelling(types, inventory.return_type)} | "
        f"{_fmt_size(type_size(types, inventory.return_type))} bytes"
    )
    lines.append("")

    lines.append("Locals (op key | name | type key | C type | bytes):")
    if inventory.locals:
        for decl in inventory.locals:
            lines.append(
                f"  {decl.op_key} | {decl.name} | {decl.type_key} | "
                f"{type_spelling(types, decl.type_key)} | {_fmt_size(type_size(types, decl.type_key))}"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    if inventory.globals_used:
        lines.append("Globals referenced (global key | name | C type):")
        for gkey in inventory.globals_used:
            entry = globals_.get(gkey, {})
            lines.append(
                f"  {gkey} | {entry.get('name', '?')} | {type_spelling(types, str(entry.get('type', '')))}"
            )
        lines.append("")
    if inventory.callees:
        lines.append("Callees: " + ", ".join(inventory.callees))
        lines.append("")

    type_keys = referenced_type_keys(program, inventory)
    lines.append("Type keys in reach (key | C spelling | bytes):")
    for tkey in type_keys[:max_types]:
        lines.append(f"  {tkey} | {type_spelling(types, tkey)} | {_fmt_size(type_size(types, tkey))}")
    if len(type_keys) > max_types:
        lines.append(f"  ... {len(type_keys) - max_types} more omitted")
    lines.append("")

    if instructions:
        lines.append("Instructions (address | disassembly):")
        lines.extend(f"  {line}" for line in instructions)
        lines.append("")

    lines.append("Reply with the JSON object only.")
    return "\n".join(lines) + "\n"


def instruction_lines(function: dict[str, Any], limit: int = 400) -> list[str]:
    """`address | text` lines from a `--emit-instructions` export."""
    export = function.get("instructions")
    if not isinstance(export, dict):
        return []
    lines = [f"{address} | {entry.get('text', '')}" for address, entry in export.items() if isinstance(entry, dict)]
    if len(lines) > limit:
        lines = lines[:limit] + [f"... {len(lines) - limit} more omitted"]
    return lines


# ---------------------------------------------------------------- Tier 2

TIER2_SYSTEM_PROMPT = """\
You rewrite one function of decompiler output into clean, well-structured C
that behaves exactly like the original.  The decompiler then re-parses the
whole translation unit with your function spliced in and checks it against
the binary's P-Code: calls, string literals, global accesses, stores,
returns, switch cases and the signature.  Anything it cannot match is
rejected and comes back to you with the reason.

Keep, exactly:
- the first line (return type, name, parameter types and names);
- every call, its callee name, argument values and their order, and the
  relative order of calls, stores and returns on every path;
- every string literal, byte for byte;
- every read and write of a global, and every store through a pointer;
- every returned value on every path, and the set of switch case values.

You may:
- replace goto/label control flow with while/for/do loops, if/else chains,
  break, continue and early returns;
- drop temporaries whose value is never used (for example `__call_ret_N`
  holding an ignored return value), fold single-use temporaries into the
  expression that uses them, and remove casts that do not change a value;
- keep local declarations at the top of the function.

Do not rename parameters, globals or callees.  Do not add calls, literals or
comments that are not in the original.  Reply with the complete function
definition only: no markers, no prose, no code fences.
"""

C_TYPE_WORDS = frozenset(
    "void char short int long float double signed unsigned struct union enum typedef "
    "extern static const volatile return if else while for do goto break continue switch "
    "case default sizeof".split()
)
_IDENT = __import__("re").compile(r"[A-Za-z_][A-Za-z0-9_]*")


def body_identifiers(text: str) -> set[str]:
    return {tok for tok in _IDENT.findall(text) if len(tok) >= 3 and tok not in C_TYPE_WORDS}


def select_declarations(preamble: str, identifiers: set[str], *, max_lines: int = 300) -> str:
    """The preamble lines a function needs: everything when short, else the
    declarations and type blocks that share an identifier with the body."""
    lines = preamble.splitlines()
    if len(lines) <= max_lines:
        return preamble.rstrip() + "\n" if preamble.strip() else ""
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("//"):
            index += 1
            continue
        if stripped.endswith("{"):
            block = [line]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                if lines[index].strip().startswith("}"):
                    break
                index += 1
            index += 1
            if identifiers & body_identifiers(block[0]) or identifiers & body_identifiers(block[-1]):
                kept.extend(block)
            continue
        if identifiers & body_identifiers(line):
            kept.append(line)
        index += 1
    return "\n".join(kept).rstrip() + "\n" if kept else ""


def instruction_lines_with_pcode(function: dict[str, Any], limit: int = 400) -> list[str]:
    """`address | text | pcode ; pcode` lines from a `--emit-instructions` export."""
    export = function.get("instructions")
    if not isinstance(export, dict):
        return []
    lines: list[str] = []
    for address, entry in export.items():
        if not isinstance(entry, dict):
            continue
        pcode = " ; ".join(str(op) for op in entry.get("pcode", []) or [])
        lines.append(f"{address} | {entry.get('text', '')} | {pcode}")
    if len(lines) > limit:
        lines = lines[:limit] + [f"... {len(lines) - limit} more omitted"]
    return lines


def build_tier2_prompt(
    printed: PrintedFunction,
    preamble: str,
    header: dict[str, str],
    *,
    program: Program | None = None,
    instructions: list[str] | None = None,
) -> str:
    lines: list[str] = []
    target = header.get("target") or (program or {}).get("id") or "unknown"
    arch = header.get("arch") or (program or {}).get("architecture") or "unknown"
    lines.append(f"Target: {target} ({arch})")
    lines.append(f"Function key: {printed.key}")
    lines.append(f"Function: {printed.name}")
    lines.append("")
    declarations = select_declarations(preamble, body_identifiers(printed.text))
    if declarations:
        lines.append("Declarations in scope (do not repeat them in your reply):")
        lines.append("```c")
        lines.append(declarations.rstrip())
        lines.append("```")
        lines.append("")
    lines.append("Function to rewrite:")
    lines.append("```c")
    lines.append(printed.text.rstrip())
    lines.append("```")
    lines.append("")
    if instructions:
        lines.append("Instructions (address | disassembly | raw P-Code):")
        lines.extend(f"  {line}" for line in instructions)
        lines.append("")
    lines.append("Reply with the complete rewritten function definition only.")
    return "\n".join(lines) + "\n"


def build_tier2_retry(base_prompt: str, attempt: int, findings: list[str]) -> str:
    lines = [base_prompt.rstrip(), "", f"Attempt {attempt} was rejected by the decompiler:"]
    lines.extend(f"- {finding}" for finding in findings)
    lines.append("")
    lines.append("Fix these and reply with the complete corrected function definition only.")
    return "\n".join(lines) + "\n"
