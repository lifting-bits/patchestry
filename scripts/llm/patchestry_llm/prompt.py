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
