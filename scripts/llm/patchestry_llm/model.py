# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
"""Read-only views over the Ghidra export JSON.

The schema is the one `include/patchestry/Ghidra/PcodeOperations.hpp`
loads: `functions`, `globals` and `types` maps, DECLARE_PARAMETER and
DECLARE_LOCAL ops in the entry block, CALL targets and `global` varnodes
inside `basic_blocks`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

Program = dict[str, Any]

SCALAR_KINDS = {"integer", "float", "boolean", "void", "undefined", "wchar", "char"}


def type_size(types: dict[str, Any], key: str) -> int | None:
    """The byte size of a type, or None when the entry has none."""
    entry = types.get(key)
    if not isinstance(entry, dict):
        return None
    size = entry.get("size")
    if size is None:
        return None
    try:
        return int(size)
    except (TypeError, ValueError):
        return None


def type_spelling(types: dict[str, Any], key: str, depth: int = 0) -> str:
    """A C-like spelling for a type key, for prompts and reports only."""
    entry = types.get(key)
    if not isinstance(entry, dict) or depth > 16:
        return key
    kind = entry.get("kind", "")
    name = entry.get("name")
    if kind == "pointer":
        return type_spelling(types, entry.get("element_type", ""), depth + 1) + " *"
    if kind == "array":
        count = entry.get("num_elements", "")
        return f"{type_spelling(types, entry.get('element_type', ''), depth + 1)}[{count}]"
    if kind in ("struct", "union"):
        return f"{kind} {name or key}"
    if kind == "enum":
        return f"enum {name or key}"
    if kind == "function":
        return "function"
    return str(name or key)


@dataclass
class Declared:
    """A DECLARE_PARAMETER or DECLARE_LOCAL op."""

    op_key: str
    name: str
    type_key: str
    index: int | None = None


@dataclass
class FunctionInventory:
    key: str
    name: str
    display_name: str
    return_type: str
    parameter_types: list[str]
    is_variadic: bool
    parameters: list[Declared] = field(default_factory=list)
    locals: list[Declared] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)
    globals_used: list[str] = field(default_factory=list)
    has_body: bool = False


def _iter_ops(function: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for block in function.get("basic_blocks", {}).values():
        if not isinstance(block, dict):
            continue
        ops = block.get("operations", {})
        order = block.get("ordered_operations") or list(ops.keys())
        for op_key in order:
            op = ops.get(op_key)
            if isinstance(op, dict):
                yield op_key, op


def function_inventory(program: Program, key: str) -> FunctionInventory:
    function = program["functions"][key]
    prototype = function.get("type", {}) or {}
    inventory = FunctionInventory(
        key=key,
        name=str(function.get("name", "")),
        display_name=str(function.get("display_name") or function.get("name", "")),
        return_type=str(prototype.get("return_type", "")),
        parameter_types=[str(t) for t in prototype.get("parameter_types", []) or []],
        is_variadic=bool(prototype.get("is_variadic", False)),
        has_body=bool(function.get("basic_blocks")),
    )
    callees: dict[str, None] = {}
    globals_used: dict[str, None] = {}
    functions = program.get("functions", {})
    for op_key, op in _iter_ops(function):
        mnemonic = op.get("mnemonic")
        if mnemonic == "DECLARE_PARAMETER":
            inventory.parameters.append(
                Declared(
                    op_key=op_key,
                    name=str(op.get("name", "")),
                    type_key=str(op.get("type", "")),
                    index=int(op.get("index", len(inventory.parameters))),
                )
            )
        elif mnemonic == "DECLARE_LOCAL":
            inventory.locals.append(
                Declared(op_key=op_key, name=str(op.get("name", "")), type_key=str(op.get("type", "")))
            )
        target = op.get("target")
        if isinstance(target, dict) and target.get("function"):
            callee = functions.get(target["function"], {})
            callees[str(callee.get("display_name") or callee.get("name") or target["function"])] = None
        for varnode in op.get("inputs", []) or []:
            if isinstance(varnode, dict) and varnode.get("kind") == "global" and varnode.get("global"):
                globals_used[str(varnode["global"])] = None
        output = op.get("output")
        if isinstance(output, dict) and output.get("kind") == "global" and output.get("global"):
            globals_used[str(output["global"])] = None
    inventory.parameters.sort(key=lambda d: (d.index if d.index is not None else 0))
    inventory.callees = list(callees)
    inventory.globals_used = list(globals_used)
    return inventory


def functions_with_bodies(program: Program) -> list[str]:
    return [key for key, fn in program.get("functions", {}).items() if isinstance(fn, dict) and fn.get("basic_blocks")]


def referenced_type_keys(program: Program, inventory: FunctionInventory) -> list[str]:
    """Type keys a function mentions, with their transitive components."""
    types = program.get("types", {})
    seen: dict[str, None] = {}
    stack: list[str] = [inventory.return_type, *inventory.parameter_types]
    stack.extend(d.type_key for d in inventory.parameters)
    stack.extend(d.type_key for d in inventory.locals)
    for gkey in inventory.globals_used:
        gtype = program.get("globals", {}).get(gkey, {}).get("type")
        if gtype:
            stack.append(str(gtype))
    while stack:
        key = stack.pop()
        if not key or key in seen or key not in types:
            continue
        seen[key] = None
        entry = types[key]
        for child in ("element_type", "base_type", "return_type"):
            if entry.get(child):
                stack.append(str(entry[child]))
        for fld in entry.get("fields", []) or []:
            if isinstance(fld, dict) and fld.get("type"):
                stack.append(str(fld["type"]))
        for ptype in entry.get("parameter_types", []) or []:
            stack.append(str(ptype))
    return list(seen)
