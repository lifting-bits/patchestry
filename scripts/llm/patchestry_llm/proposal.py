# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
"""Tier 1 proposals: what the model may change and how it is applied.

A proposal is a JSON object::

    {
      "functions": {
        "<function key>": {
          "display_name": "identifier",
          "comment": "text",
          "parameters": {"<index>": {"name": "...", "type": "<type key>"}},
          "locals": {"<DECLARE_LOCAL op key>": {"name": "...", "type": "<type key>"}},
          "return_type": "<type key>"
        }
      },
      "globals": {"<global key>": {"name": "identifier"}},
      "types": {"<new type key>": {"kind": "struct", "name": "...", "size": 8,
                                   "fields": [{"name": "...", "type": "...", "offset": 0}]}}
    }

Every edit is checked before it lands in a deep copy of the program:
identifiers must be valid C and unique in their namespace, retypes must
keep the byte size, new types must reference existing types and have a
consistent layout.  Rejected edits are reported, never applied; the
lifter's own `refine:` warnings remain the last line of defence.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .model import Program, function_inventory, type_size

C_KEYWORDS = frozenset(
    """
    alignas alignof auto bool break case char const constexpr continue default do
    double else enum extern false float for goto if inline int long nullptr register
    restrict return short signed sizeof static static_assert struct switch thread_local
    true typedef typeof typeof_unqual union unsigned void volatile while
    _Alignas _Alignof _Atomic _BitInt _Bool _Complex _Decimal128 _Decimal32 _Decimal64
    _Generic _Imaginary _Noreturn _Static_assert _Thread_local
    asm typeof __asm__ __attribute__ __inline __restrict __typeof__ __volatile__
    """.split()
)
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_COMMENT = 2000
NEW_TYPE_KINDS = ("struct", "union", "typedef", "pointer", "array")


def is_identifier(name: Any) -> bool:
    return isinstance(name, str) and bool(IDENTIFIER.match(name)) and name not in C_KEYWORDS


def extract_json(text: str) -> dict[str, Any]:
    """The first JSON object in a model reply, code fences tolerated."""
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("the reply's JSON is not an object")
    return parsed


@dataclass
class Rejection:
    where: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"where": self.where, "reason": self.reason}


@dataclass
class ApplyReport:
    accepted: list[str] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    def reject(self, where: str, reason: str) -> None:
        self.rejected.append(Rejection(where, reason))

    def accept(self, what: str) -> None:
        self.accepted.append(what)


class Refiner:
    """Applies validated proposals to a private copy of a program."""

    def __init__(self, program: Program):
        self.program: Program = copy.deepcopy(program)
        self.program.setdefault("functions", {})
        self.program.setdefault("globals", {})
        self.program.setdefault("types", {})

    # ----- namespaces -------------------------------------------------

    def _function_names(self, except_key: str | None = None) -> set[str]:
        names: set[str] = set()
        for key, fn in self.program["functions"].items():
            if key == except_key or not isinstance(fn, dict):
                continue
            for field_name in ("name", "display_name"):
                if fn.get(field_name):
                    names.add(str(fn[field_name]))
        return names

    def _global_names(self, except_key: str | None = None) -> set[str]:
        return {
            str(g["name"])
            for key, g in self.program["globals"].items()
            if key != except_key and isinstance(g, dict) and g.get("name")
        }

    def _typedef_names(self) -> set[str]:
        return {
            str(t["name"])
            for t in self.program["types"].values()
            if isinstance(t, dict) and t.get("kind") == "typedef" and t.get("name")
        }

    def _tag_names(self) -> set[str]:
        return {
            str(t["name"])
            for t in self.program["types"].values()
            if isinstance(t, dict) and t.get("kind") in ("struct", "union", "enum") and t.get("name")
        }

    def _ordinary_names(self, *, except_function: str | None = None, except_global: str | None = None) -> set[str]:
        return (
            self._function_names(except_function)
            | self._global_names(except_global)
            | self._typedef_names()
        )

    # ----- entry point -----------------------------------------------

    def apply(self, proposal: dict[str, Any], *, only_function: str | None = None) -> ApplyReport:
        report = ApplyReport()
        if not isinstance(proposal, dict):
            report.reject("proposal", "not a JSON object")
            return report
        types = proposal.get("types") or {}
        if not isinstance(types, dict):
            report.reject("types", "must be an object keyed by new type key")
        else:
            for key, spec in types.items():
                self._add_type(report, str(key), spec)
        globals_ = proposal.get("globals") or {}
        if not isinstance(globals_, dict):
            report.reject("globals", "must be an object keyed by global key")
        else:
            for key, spec in globals_.items():
                self._rename_global(report, str(key), spec)
        functions = proposal.get("functions") or {}
        if not isinstance(functions, dict):
            report.reject("functions", "must be an object keyed by function key")
        else:
            for key, spec in functions.items():
                if only_function is not None and key != only_function:
                    report.reject(f"functions[{key}]", f"only {only_function} may be edited in this reply")
                    continue
                self._refine_function(report, str(key), spec)
        return report

    # ----- types -------------------------------------------------------

    def _add_type(self, report: ApplyReport, key: str, spec: Any) -> None:
        where = f"types[{key}]"
        types = self.program["types"]
        if not key or key in types:
            report.reject(where, "type key is empty or already exists")
            return
        if not isinstance(spec, dict):
            report.reject(where, "type entry must be an object")
            return
        kind = spec.get("kind")
        if kind not in NEW_TYPE_KINDS:
            report.reject(where, f"kind must be one of {', '.join(NEW_TYPE_KINDS)}")
            return
        size = spec.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            report.reject(where, "size must be a positive integer")
            return
        entry: dict[str, Any] = {"kind": kind, "size": size}
        name = spec.get("name")
        if kind in ("struct", "union", "typedef"):
            if not is_identifier(name):
                report.reject(where, "name must be a C identifier")
                return
            taken = self._typedef_names() if kind == "typedef" else self._tag_names()
            if name in taken or (kind == "typedef" and name in self._ordinary_names()):
                report.reject(where, f"name {name!r} is already taken")
                return
            entry["name"] = name
        if kind == "typedef":
            base = spec.get("base_type")
            if not isinstance(base, str) or base not in types:
                report.reject(where, "base_type must be an existing type key")
                return
            base_size = type_size(types, base)
            if base_size is not None and base_size != size:
                report.reject(where, f"size {size} differs from base type size {base_size}")
                return
            entry["base_type"] = base
        elif kind == "pointer":
            element = spec.get("element_type")
            if not isinstance(element, str) or element not in types:
                report.reject(where, "element_type must be an existing type key")
                return
            entry["element_type"] = element
        elif kind == "array":
            element = spec.get("element_type")
            count = spec.get("num_elements")
            if not isinstance(element, str) or element not in types:
                report.reject(where, "element_type must be an existing type key")
                return
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                report.reject(where, "num_elements must be a positive integer")
                return
            element_size = type_size(types, element)
            if element_size is not None and element_size * count != size:
                report.reject(where, f"size {size} is not {count} x {element_size}")
                return
            entry["element_type"] = element
            entry["num_elements"] = count
        else:  # struct / union
            fields = spec.get("fields")
            if not isinstance(fields, list) or not fields:
                report.reject(where, "fields must be a non-empty list")
                return
            seen_names: set[str] = set()
            out_fields: list[dict[str, Any]] = []
            for index, fld in enumerate(fields):
                fwhere = f"{where}.fields[{index}]"
                if not isinstance(fld, dict):
                    report.reject(fwhere, "field must be an object")
                    return
                fname, ftype, foffset = fld.get("name"), fld.get("type"), fld.get("offset", 0)
                if not is_identifier(fname) or fname in seen_names:
                    report.reject(fwhere, "field name must be a unique C identifier")
                    return
                if not isinstance(ftype, str) or ftype not in types:
                    report.reject(fwhere, "field type must be an existing type key")
                    return
                if not isinstance(foffset, int) or isinstance(foffset, bool) or foffset < 0:
                    report.reject(fwhere, "offset must be a non-negative integer")
                    return
                fsize = type_size(types, ftype) or 0
                if foffset + fsize > size:
                    report.reject(fwhere, f"field ends at {foffset + fsize}, past size {size}")
                    return
                seen_names.add(fname)
                out_fields.append({"name": fname, "type": ftype, "offset": foffset})
            if kind == "struct":
                ordered = sorted(out_fields, key=lambda f: f["offset"])
                end = 0
                for fld in ordered:
                    if fld["offset"] < end:
                        report.reject(where, f"field {fld['name']} overlaps the previous field")
                        return
                    end = fld["offset"] + (type_size(types, fld["type"]) or 0)
                out_fields = ordered
            else:
                for fld in out_fields:
                    if fld["offset"] != 0:
                        report.reject(where, "union fields must be at offset 0")
                        return
            entry["fields"] = out_fields
        types[key] = entry
        report.accept(f"{where}: new {kind} {entry.get('name', '')}".rstrip())

    # ----- globals -----------------------------------------------------

    def _rename_global(self, report: ApplyReport, key: str, spec: Any) -> None:
        where = f"globals[{key}]"
        entry = self.program["globals"].get(key)
        if not isinstance(entry, dict):
            report.reject(where, "unknown global key")
            return
        if not isinstance(spec, dict):
            report.reject(where, "global entry must be an object")
            return
        name = spec.get("name")
        if name is None:
            return
        if not is_identifier(name):
            report.reject(where, "name must be a C identifier")
            return
        if name in self._ordinary_names(except_global=key):
            report.reject(where, f"name {name!r} is already taken")
            return
        old = entry.get("name")
        if old != name:
            entry["name"] = name
            report.accept(f"{where}.name: {old} -> {name}")

    # ----- functions ---------------------------------------------------

    def _refine_function(self, report: ApplyReport, key: str, spec: Any) -> None:
        where = f"functions[{key}]"
        function = self.program["functions"].get(key)
        if not isinstance(function, dict):
            report.reject(where, "unknown function key")
            return
        if not isinstance(spec, dict):
            report.reject(where, "function entry must be an object")
            return
        types = self.program["types"]
        inventory = function_inventory(self.program, key)

        display_name = spec.get("display_name")
        if display_name is not None:
            if not is_identifier(display_name):
                report.reject(f"{where}.display_name", "must be a C identifier")
            elif display_name in self._ordinary_names(except_function=key):
                report.reject(f"{where}.display_name", f"{display_name!r} is already taken")
            else:
                old = function.get("display_name") or function.get("name")
                if old != display_name:
                    function["display_name"] = display_name
                    report.accept(f"{where}.display_name: {old} -> {display_name}")

        comment = spec.get("comment")
        if comment is not None:
            if not isinstance(comment, str):
                report.reject(f"{where}.comment", "must be a string")
            else:
                text = " ".join(comment.split())
                if not text:
                    report.reject(f"{where}.comment", "is empty")
                else:
                    if len(text) > MAX_COMMENT:
                        text = text[:MAX_COMMENT].rstrip() + " ..."
                    function["comment"] = text
                    report.accept(f"{where}.comment: set ({len(text)} chars)")

        # Names in scope: every declared parameter and local, updated as we go.
        declared = {d.name for d in inventory.parameters} | {d.name for d in inventory.locals}
        outside = self._ordinary_names(except_function=key) | {
            str(function.get("display_name") or function.get("name", ""))
        }
        by_index = {d.index: d for d in inventory.parameters}
        by_op = {d.op_key: d for d in inventory.locals}
        ops = self._declared_ops(function)

        def rename(dwhere: str, decl, new_name: Any) -> None:
            if new_name is None or new_name == decl.name:
                return
            if not is_identifier(new_name):
                report.reject(dwhere, "name must be a C identifier")
                return
            if new_name in declared or new_name in outside:
                report.reject(dwhere, f"name {new_name!r} is already in scope")
                return
            declared.discard(decl.name)
            declared.add(new_name)
            report.accept(f"{dwhere}.name: {decl.name} -> {new_name}")
            ops[decl.op_key]["name"] = new_name
            decl.name = new_name

        def retype(dwhere: str, decl, new_type: Any, *, parameter_index: int | None = None) -> None:
            if new_type is None or new_type == decl.type_key:
                return
            if not isinstance(new_type, str) or new_type not in types:
                report.reject(dwhere, "type must be an existing type key")
                return
            old_size, new_size = type_size(types, decl.type_key), type_size(types, new_type)
            if old_size is None or new_size is None or old_size != new_size:
                report.reject(dwhere, f"type size must stay {old_size}, {new_type} is {new_size}")
                return
            ops[decl.op_key]["type"] = new_type
            if parameter_index is not None:
                proto = function.setdefault("type", {})
                params = proto.setdefault("parameter_types", [])
                if parameter_index < len(params):
                    params[parameter_index] = new_type
            report.accept(f"{dwhere}.type: {decl.type_key} -> {new_type}")
            decl.type_key = new_type

        parameters = spec.get("parameters")
        if parameters is not None:
            items = self._indexed_items(parameters)
            if items is None:
                report.reject(f"{where}.parameters", "must be an object keyed by index or a list")
            else:
                for index, pspec in items:
                    pwhere = f"{where}.parameters[{index}]"
                    decl = by_index.get(index)
                    if decl is None:
                        report.reject(pwhere, "no DECLARE_PARAMETER with that index")
                        continue
                    if not isinstance(pspec, dict):
                        report.reject(pwhere, "must be an object")
                        continue
                    rename(pwhere, decl, pspec.get("name"))
                    retype(pwhere, decl, pspec.get("type"), parameter_index=index)

        locals_ = spec.get("locals")
        if locals_ is not None:
            if not isinstance(locals_, dict):
                report.reject(f"{where}.locals", "must be an object keyed by DECLARE_LOCAL op key")
            else:
                for op_key, lspec in locals_.items():
                    lwhere = f"{where}.locals[{op_key}]"
                    decl = by_op.get(str(op_key))
                    if decl is None:
                        report.reject(lwhere, "no DECLARE_LOCAL with that op key")
                        continue
                    if not isinstance(lspec, dict):
                        report.reject(lwhere, "must be an object")
                        continue
                    rename(lwhere, decl, lspec.get("name"))
                    retype(lwhere, decl, lspec.get("type"))

        return_type = spec.get("return_type")
        if return_type is not None and return_type != inventory.return_type:
            rwhere = f"{where}.return_type"
            if not isinstance(return_type, str) or return_type not in types:
                report.reject(rwhere, "must be an existing type key")
            else:
                old_size, new_size = type_size(types, inventory.return_type), type_size(types, return_type)
                if old_size is None or new_size is None or old_size != new_size:
                    report.reject(rwhere, f"type size must stay {old_size}, {return_type} is {new_size}")
                else:
                    function.setdefault("type", {})["return_type"] = return_type
                    report.accept(f"{rwhere}: {inventory.return_type} -> {return_type}")

    @staticmethod
    def _declared_ops(function: dict[str, Any]) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for block in function.get("basic_blocks", {}).values():
            if not isinstance(block, dict):
                continue
            for op_key, op in (block.get("operations") or {}).items():
                if isinstance(op, dict) and op.get("mnemonic") in ("DECLARE_PARAMETER", "DECLARE_LOCAL"):
                    found[op_key] = op
        return found

    @staticmethod
    def _indexed_items(parameters: Any) -> list[tuple[int, Any]] | None:
        if isinstance(parameters, list):
            return list(enumerate(parameters))
        if isinstance(parameters, dict):
            items: list[tuple[int, Any]] = []
            for key, value in parameters.items():
                try:
                    items.append((int(key), value))
                except (TypeError, ValueError):
                    items.append((-1, value))
            return items
        return None
